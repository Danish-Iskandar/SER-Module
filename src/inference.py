"""
Real-time inference engine for the SER module.

Keeps a rolling 3-second buffer of 16 kHz mono audio (the same window length
the model was trained on), converts it to MFCCs using the *exact* same
function as training (preprocess.mfcc_from_waveform), and normalises with the
saved training statistics.

Four safeguards sit between the raw model output and the emotion state:

1. Warm-up: the first `warmup_sec` of audio after a client connects is
   discarded. Microphones often emit a burst of garbage (clipping, large DC
   offset) when the stream opens; without this the model classifies the glitch.

2. Silence gate: if the recent audio's energy is below `silence_rms`, no
   prediction is made and the current state is kept. Energy is measured with
   the DC offset removed (standard deviation, not raw RMS), so a shifted but
   silent signal cannot pass as speech.

3. Smoothing: the class probabilities of the last `smooth_windows` non-silent
   windows are averaged before deciding. The model is overconfident (0.99+
   even on unseen voices), so a single one-second spike would otherwise flip
   the state. With 3 windows, a class needs to win roughly 2 of the last 3.

4. Confidence threshold (from the Methodology): the state only changes when
   the smoothed top probability reaches `threshold`; otherwise the previous
   state is retained.
"""
import pickle
from collections import deque

import numpy as np

from preprocess import mfcc_from_waveform, SAMPLE_RATE, DURATION_SEC

WINDOW_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)


def load_artifacts(model_path, norm_path, labels_path):
    """Load the trained Keras model, normalisation stats and class names."""
    from tensorflow.keras.models import load_model  # lazy: keeps import light

    model = load_model(model_path, compile=False)
    mean, std = np.load(norm_path)
    with open(labels_path, "rb") as f:
        classes = pickle.load(f)
    return model, float(mean), float(std), list(classes)


class EmotionPredictor:
    def __init__(self, model, mean, std, classes, threshold=0.6,
                 silence_rms=0.02, smooth_windows=3, warmup_sec=1.0,
                 recent_sec=1.0, initial_state="Neutral"):
        self.model = model
        self.mean, self.std = mean, std
        self.classes = classes
        self.threshold = threshold
        self.silence_rms = silence_rms
        self.recent_samples = int(SAMPLE_RATE * recent_sec)
        self.warmup_remaining = int(SAMPLE_RATE * warmup_sec)
        self.history = deque(maxlen=max(1, smooth_windows))
        self.state = initial_state
        self.buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)

    @property
    def warming_up(self):
        return self.warmup_remaining > 0

    def push_audio(self, samples):
        """Append float32 mono 16 kHz samples in [-1, 1] to the rolling buffer.

        Samples received during the warm-up period are discarded.
        """
        samples = np.asarray(samples, dtype=np.float32)
        if self.warmup_remaining > 0:
            drop = min(self.warmup_remaining, len(samples))
            self.warmup_remaining -= drop
            samples = samples[drop:]
            if len(samples) == 0:
                return
        if len(samples) >= WINDOW_SAMPLES:
            self.buffer = samples[-WINDOW_SAMPLES:].copy()
        else:
            self.buffer = np.concatenate([self.buffer[len(samples):], samples])

    def _result(self, **kw):
        base = {
            "emotion": self.state, "raw_emotion": None, "confidence": 0.0,
            "probabilities": {}, "smoothed": {}, "smoothed_confidence": 0.0,
            "state_changed": False, "silent": False, "warming_up": False,
            "rms": 0.0,
        }
        base.update(kw)
        return base

    def predict(self):
        """Run one prediction on the current buffer and return a result dict."""
        if self.warming_up:
            return self._result(silent=True, warming_up=True)

        recent = self.buffer[-self.recent_samples:]
        # std = RMS with the DC offset removed
        rms = float(np.std(recent))

        if rms < self.silence_rms:
            return self._result(silent=True, rms=rms)

        # Remove DC offset, then identical to training:
        # MFCC -> pad to MAX_LEN -> global normalisation.
        window = self.buffer - np.mean(self.buffer)
        feat = mfcc_from_waveform(window, SAMPLE_RATE)
        feat = (feat - self.mean) / (self.std + 1e-8)
        probs = np.asarray(self.model.predict(feat[np.newaxis], verbose=0))[0]

        raw_idx = int(np.argmax(probs))
        self.history.append(probs)
        smoothed = np.mean(np.stack(self.history), axis=0)
        sm_idx = int(np.argmax(smoothed))
        sm_conf = float(smoothed[sm_idx])

        previous = self.state
        if sm_conf >= self.threshold:
            self.state = self.classes[sm_idx]

        return self._result(
            raw_emotion=self.classes[raw_idx],
            confidence=float(probs[raw_idx]),
            probabilities={c: float(p) for c, p in zip(self.classes, probs)},
            smoothed={c: float(p) for c, p in zip(self.classes, smoothed)},
            smoothed_confidence=sm_conf,
            state_changed=self.state != previous,
            rms=rms,
        )
