# SER Module — Adaptive VR Quay Crane Training

Speech Emotion Recognition pipeline: MFCC feature extraction + CNN-LSTM classifier,
trained on RAVDESS, SAVEE, and (optionally) IEMOCAP.

## 1. Get the datasets

| Dataset | Access | Notes |
|---|---|---|
| **RAVDESS** (speech) | Free, instant. Download `Audio_Speech_Actors_01-24.zip` from Zenodo: https://zenodo.org/record/1188976 | ~215MB, 24 actors |
| **SAVEE** | Free, instant. http://kahlan.eps.surrey.ac.uk/savee/Download.html | 4 male actors |
| **IEMOCAP** | **Requires a signed license request** to USC SAIL: https://sail.usc.edu/iemocap/iemocap_release.htm | Approval can take days–weeks. Start this today. Build/test on RAVDESS+SAVEE while you wait — the code below treats IEMOCAP as optional. |

Unzip everything into a `data/` folder like this:

```
data/
  RAVDESS/Actor_01/03-01-05-01-01-01-01.wav ...
  SAVEE/ALL/a01.wav ...
  IEMOCAP/Session1/... (once you get access)
```

## 2. Environment (Google Colab recommended)

In a Colab notebook:

```python
!pip install librosa tensorflow scikit-learn pandas soundfile
```

Then upload this `ser_module/` folder to Colab (or `git clone` if you push it to
a repo), mount Google Drive for the `data/` folder so you don't re-upload
every session:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 3. Emotion label mapping (READ THIS — you need to justify this in your report)

Your project targets 3 classes: **Neutral, Stressed, Confident** (matching
Section 3.2 of your Interim Report). None of RAVDESS/SAVEE/IEMOCAP label
emotions this way natively, so `label_map.py` defines an explicit proxy
mapping (documented there). The default is:

- Neutral ← neutral, calm
- Stressed ← fearful/fear
- Confident ← happy/happiness (proxy — flag this as a limitation in your report)

Everything else — sad, surprised, angry, disgust, and IEMOCAP's native
"frustration" label — is dropped (not mapped to any of your 3 classes).
Note this noticeably shrinks how much of IEMOCAP is usable, since a chunk of
its utterances are angry/frustrated/disgust, which are no longer targets.
You can edit `label_map.py` to change any of this — it's the single source
of truth, everything downstream (`preprocess.py`, `train.py`) reads from it
automatically.

**Report consistency note:** your Abstract/Objectives previously referenced
4 classes (including Frustrated) while Section 3.2 said 3 — make sure both
are updated to say 3 classes consistently now that the code matches 3.2.

## 4. Run the pipeline

```bash
python src/build_manifest.py --data_root data/ --out manifest.csv
python src/preprocess.py --manifest manifest.csv --out features/
python src/train.py --features features/ --out models/
```

Or run each script's `main()` directly inside Colab cells — they're written
to work either way.

## 5. What you get

- `features/X.npy`, `features/y.npy` — padded MFCC sequences + integer labels, cached so you don't re-extract every run
- `models/ser_cnn_lstm.h5` — trained Keras model
- `models/label_encoder.pkl` — maps class indices back to emotion names (needed by the real-time server, section 6)
- Confusion matrix + classification report printed at the end of training, plus a `models/training_history.png`

## 6. Real-time inference + Unity (WebSocket)

Pieces: `src/inference.py` (rolling 3 s buffer -> MFCC -> model -> threshold),
`src/server.py` (WebSocket middleware), `src/stream_client.py` (test client, no
Unity needed), `unity/SERClient.cs` (mic streaming + avatar colour).

**A. Copy three trained artifacts next to the code** (they are gitignored, so
`git clone` will not bring them). From Drive:

```
models/ser_cnn_lstm.h5
models/label_encoder.pkl
features/norm_stats.npy      <- from the SAME features/ folder used for training
```

Do not use `features_augmented/norm_stats.npy`; the model was trained on the
non-augmented features and needs its exact mean/std.

**B. Run the server locally** (real inference needs TensorFlow installed locally):

```bash
cd src
python server.py --model ../models/ser_cnn_lstm.h5 \
    --norm ../features/norm_stats.npy --labels ../models/label_encoder.pkl
```

`python server.py --dummy` runs without a model and cycles fake emotions,
which is useful for testing the Unity side first.

**C. Test the whole path without Unity**

```bash
python stream_client.py --wav path/to/some_ravdess_clip.wav   # streams in real time
python stream_client.py --mic                                  # laptop microphone
```

**D. Unity**

1. Package Manager -> Add from git URL: `https://github.com/endel/NativeWebSocket.git#upm`
2. Add `unity/SERClient.cs` to a GameObject, drag the avatar's Renderer into it.
3. Server URL: `ws://localhost:8765` in the Editor on the same PC; for a
   standalone Quest use `ws://<PC LAN IP>:8765`, on the same network, with the
   port allowed through the PC firewall. Shared/campus Wi-Fi sometimes blocks
   device-to-device traffic; if the Quest cannot connect, try a phone hotspot,
   your own router, or Quest Link (then localhost works).
4. Hook the adaptive engine to the `onEmotionChanged` event.

**Measure your microphone first:**

```bash
python mic_check.py
```

It records quiet and talking, then suggests a `--silence_rms` value. Rerun it
for each new mic (the Quest headset mic will differ from your laptop).

**Server settings:**

| Flag | Default | What it does |
|---|---|---|
| `--silence_rms` | 0.02 | No prediction below this level. Recorded RAVDESS clips are quiet, so use `0.002` for `--wav` tests. |
| `--warmup` | 1.0 s | Audio ignored after a client connects, to skip the mic start-up glitch. |
| `--smooth` | 3 | Windows averaged before deciding. Stops one-second spikes changing the state; adds about 1 s of delay. `1` turns it off. |
| `--threshold` | 0.6 | Minimum smoothed probability needed to change state. |

**Known limits from live testing (Oct 2026, laptop mic):**

- Confident never appeared (top probability 0.05), matching the 39% recall
  on the test set plus the studio-to-mic domain gap.
- The model is overconfident (0.99+ on an unseen voice), which is why the
  threshold alone did not stop flicker and smoothing was added.
- Changes register about 1-2 s after the speech that caused them (3 s window
  plus smoothing).

**What has been verified:** the server/client protocol, warm-up, the
DC-proof silence gate, smoothing, 16 kHz and 48 kHz input, and loading a real
`.h5` through `load_artifacts` were run end to end, plus a live laptop-mic run
with the trained model. `SERClient.cs` has NOT been compiled or run in Unity; test it in
the Editor with `--dummy` first.