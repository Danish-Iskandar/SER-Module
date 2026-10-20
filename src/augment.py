"""
Audio augmentation utilities for the SER pipeline.

Each function takes a raw waveform (numpy array, as loaded by librosa) and
returns a modified version. Used by preprocess.py to generate extra training
variety, especially for under-represented classes like Confident.

These are deliberately mild transformations (small pitch/speed changes, low
noise) — the goal is realistic variation, not distortion that changes the
actual emotional content of the clip.
"""
import numpy as np
import librosa


def pitch_shift(y, sr, n_steps=None):
    """Shift pitch by n_steps semitones. Random in [-2, 2] if not given."""
    if n_steps is None:
        n_steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def time_stretch(y, rate=None):
    """Speed up/slow down without changing pitch. Random in [0.9, 1.1] if not given."""
    if rate is None:
        rate = np.random.uniform(0.9, 1.1)
    return librosa.effects.time_stretch(y, rate=rate)


def add_noise(y, noise_factor=None):
    """Add light Gaussian noise. Random in [0.001, 0.005] amplitude if not given."""
    if noise_factor is None:
        noise_factor = np.random.uniform(0.001, 0.005)
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


AUGMENTATIONS = {
    "pitch_shift": pitch_shift,
    "time_stretch": time_stretch,
    "add_noise": add_noise,
}


def random_augment(y, sr):
    """Apply one randomly-chosen augmentation. Used for single-variant augmentation."""
    name = np.random.choice(list(AUGMENTATIONS.keys()))
    fn = AUGMENTATIONS[name]
    if name == "time_stretch":
        return fn(y)
    return fn(y, sr) if name == "pitch_shift" else fn(y)


def generate_variants(y, sr, n_variants=2):
    """
    Generate n_variants augmented versions of a clip, each using a different
    augmentation so the variants are meaningfully distinct from each other,
    not just repeated random noise.
    """
    names = list(AUGMENTATIONS.keys())
    chosen = np.random.choice(names, size=min(n_variants, len(names)), replace=False)
    variants = []
    for name in chosen:
        fn = AUGMENTATIONS[name]
        if name == "pitch_shift":
            variants.append(fn(y, sr))
        elif name == "time_stretch":
            variants.append(fn(y))
        else:
            variants.append(fn(y))
    return variants
