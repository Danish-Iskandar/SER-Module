"""
Central emotion label mapping for the SER module.

RAVDESS, SAVEE, and IEMOCAP each use their own native emotion vocabularies.
None of them contain a "Confident" class. This file defines the proxy mapping
from native labels -> the project's 3 target classes:

    Neutral, Stressed, Confident

This mapping is a modeling assumption, not ground truth — document it in your
report (Limitations section). Edit TARGET_CLASSES and the *_MAP dicts below
to change it; everything downstream (preprocess.py, train.py) reads from
here so you only have to change it in one place.

Any native label not present in a mapping dict is dropped (not used).
"""

# Final 3 classes, in a fixed order used for one-hot / integer encoding.
TARGET_CLASSES = ["Neutral", "Stressed", "Confident"]

# --- RAVDESS -----------------------------------------------------------
# Filename format: 03-01-EE-II-SS-RR-AA.wav, EE = emotion code (03 field)
# 01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful 07=disgust 08=surprised
RAVDESS_CODE_TO_NATIVE = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# --- SAVEE ---------------------------------------------------------------
# Filename prefix encodes emotion: a=anger, d=disgust, f=fear, h=happiness,
# n=neutral, sa=sadness, su=surprise
SAVEE_PREFIX_TO_NATIVE = {
    "a": "angry",
    "d": "disgust",
    "f": "fearful",
    "h": "happy",
    "n": "neutral",
    "sa": "sad",
    "su": "surprised",
}

# --- IEMOCAP ---------------------------------------------------------------
# Native categorical labels used in IEMOCAP's evaluation .txt annotations.
IEMOCAP_NATIVE_ALIASES = {
    "neu": "neutral",
    "hap": "happy",
    "exc": "happy",       # excited folded into happy/confident proxy
    "sad": "sad",
    "ang": "angry",
    "fea": "fearful",
    "fru": "frustrated",  # no target class in the 3-class scheme; dropped
    "sur": "surprised",
    "dis": "disgust",
}

# --- Native label -> target class (the actual proxy mapping) --------------
NATIVE_TO_TARGET = {
    "neutral": "Neutral",
    "calm": "Neutral",
    "fearful": "Stressed",
    "happy": "Confident",
    # dropped (no target class in the 3-class scheme):
    # "sad", "surprised", "angry", "disgust", "frustrated"
}


def native_to_target(native_label: str):
    """Return the target class name for a native label, or None if dropped."""
    return NATIVE_TO_TARGET.get(native_label.lower())


def ravdess_code_to_target(code: str):
    native = RAVDESS_CODE_TO_NATIVE.get(code)
    return native_to_target(native) if native else None


def savee_prefix_to_target(prefix: str):
    native = SAVEE_PREFIX_TO_NATIVE.get(prefix.lower())
    return native_to_target(native) if native else None


def iemocap_tag_to_target(tag: str):
    native = IEMOCAP_NATIVE_ALIASES.get(tag.lower())
    return native_to_target(native) if native else None