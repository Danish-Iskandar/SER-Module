"""
Extracts MFCC features for every file in manifest.csv and saves padded
sequences as X.npy / y.npy for fast reuse (extraction is the slow part).

Run:
    python preprocess.py --manifest manifest.csv --out features/
"""
import argparse
import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from label_map import TARGET_CLASSES

SAMPLE_RATE = 16000
DURATION_SEC = 3.0          # pad/truncate every clip to this length
N_MFCC = 40
MAX_LEN = 130                # ~time steps for 3s audio at default hop_length


def extract_mfcc(filepath):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION_SEC)
    target_len = int(SAMPLE_RATE * DURATION_SEC)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)  # (n_mfcc, time)
    mfcc = mfcc.T  # (time, n_mfcc) - time-major for the CNN-LSTM input

    if mfcc.shape[0] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN, :]

    return mfcc.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--out", default="features/")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.manifest)

    label_to_idx = {label: i for i, label in enumerate(TARGET_CLASSES)}

    X, y = [], []
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting MFCCs"):
        try:
            feat = extract_mfcc(row["filepath"])
        except Exception as e:
            skipped += 1
            continue
        X.append(feat)
        y.append(label_to_idx[row["target_label"]])

    X = np.stack(X)          # (N, MAX_LEN, N_MFCC)
    y = np.array(y)

    # Per-sample normalization (zero mean, unit variance) using train-time
    # statistics computed here; store them so real-time inference can reuse.
    mean = X.mean()
    std = X.std()
    X_norm = (X - mean) / (std + 1e-8)

    np.save(os.path.join(args.out, "X.npy"), X_norm)
    np.save(os.path.join(args.out, "y.npy"), y)
    np.save(os.path.join(args.out, "norm_stats.npy"), np.array([mean, std]))

    print(f"Saved {X_norm.shape[0]} samples, shape {X_norm.shape}. Skipped {skipped} unreadable files.")
    print("Class distribution:", {c: int((y == i).sum()) for c, i in label_to_idx.items()})


if __name__ == "__main__":
    main()
