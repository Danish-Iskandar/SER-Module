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

Your project targets 4 classes: **Neutral, Stressed/Anxious, Frustrated, Confident**.
None of RAVDESS/SAVEE/IEMOCAP label emotions this way natively, so `label_map.py`
defines an explicit proxy mapping (documented there). The default is:

- Neutral ← neutral, calm
- Stressed/Anxious ← fearful/fear
- Frustrated ← angry, disgust (+ IEMOCAP's native "frustration" label when available)
- Confident ← happy/happiness (proxy — flag this as a limitation in your report)

Sad and surprised are dropped by default (not mapped to any of your 4 classes).
You can edit `label_map.py` to change this — it's the single source of truth,
everything downstream reads from it.

**Also flag for yourself:** your Interim Report's Methodology section (3.2) says
Phase 1 trains on only **three** classes (Neutral, Stressed/Anxious, Confident),
while your Abstract/Objectives say **four** (including Frustrated). Pick one and
make the docs consistent — the code here defaults to 4-class since that matches
your stated objectives, but it's a one-line change to drop Frustrated.

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
- `models/label_encoder.pkl` — maps class indices back to emotion names (needed for the real-time inference / WebSocket step later)
- Confusion matrix + classification report printed at the end of training, plus a `models/training_history.png`

## 6. Next steps (not in this pipeline yet)

- Confidence-threshold tuning (mentioned in your Methodology) — I'd do this by
  looking at the softmax output distribution on your validation set once the
  model is trained; happy to build that next.
- Real-time inference script (mic → frames → MFCC → model → JSON over WebSocket) for the Unity/Socket.io integration.
