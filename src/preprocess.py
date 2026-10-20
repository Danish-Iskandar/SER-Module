"""
Extracts MFCC features for every file in manifest.csv and saves padded
sequences as X.npy / y.npy for fast reuse (extraction is the slow part).

Run:
    python preprocess.py --manifest manifest.csv --out features/

With augmentation (balances under-represented classes up to the largest
class's count by generating pitch-shift/time-stretch/noise variants):
    python preprocess.py --manifest manifest.csv --out features/ --augment
"""
import argparse
import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from label_map import TARGET_CLASSES
from augment import generate_variants

SAMPLE_RATE = 16000
DURATION_SEC = 3.0          # pad/truncate every clip to this length
N_MFCC = 40
MAX_LEN = 130                # ~time steps for 3s audio at default hop_length


def load_waveform(filepath):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION_SEC)
    target_len = int(SAMPLE_RATE * DURATION_SEC)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y, sr


def mfcc_from_waveform(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)  # (n_mfcc, time)
    mfcc = mfcc.T  # (time, n_mfcc) - time-major for the CNN-LSTM input

    if mfcc.shape[0] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN, :]

    return mfcc.astype(np.float32)


def extract_mfcc(filepath):
    y, sr = load_waveform(filepath)
    return mfcc_from_waveform(y, sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--out", default="features/")
    ap.add_argument("--augment", action="store_true",
                     help="Generate augmented variants for under-represented "
                          "classes, balancing them up toward the largest class.")
    ap.add_argument("--variants_per_sample", type=int, default=2,
                     help="Max augmented variants to generate per original "
                          "sample in an under-represented class (default: 2).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.manifest)

    label_to_idx = {label: i for i, label in enumerate(TARGET_CLASSES)}

    # Original (non-augmented) extraction — always happens first.
    X, y = [], []
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting MFCCs"):
        try:
            feat = extract_mfcc(row["filepath"])
        except Exception:
            skipped += 1
            continue
        X.append(feat)
        y.append(label_to_idx[row["target_label"]])

    if args.augment:
        # Balance under-represented classes up toward the largest class's
        # count by generating pitch/time/noise variants of existing clips.
        counts = {i: y.count(i) for i in set(y)}
        max_count = max(counts.values())
        print(f"\nAugmenting under-represented classes toward {max_count} samples each...")

        # Group filepaths by class so we can sample from them to augment.
        rows_by_class = {}
        for _, row in df.iterrows():
            idx = label_to_idx.get(row["target_label"])
            if idx is not None:
                rows_by_class.setdefault(idx, []).append(row["filepath"])

        for class_idx, current_count in counts.items():
            deficit = max_count - current_count
            if deficit <= 0:
                continue
            class_name = TARGET_CLASSES[class_idx]
            filepaths = rows_by_class[class_idx]
            print(f"  {class_name}: {current_count} -> generating {deficit} augmented samples")

            generated = 0
            pbar = tqdm(total=deficit, desc=f"Augmenting {class_name}")
            while generated < deficit:
                # Cycle through original files, generating variants_per_sample
                # variants each, until the deficit is filled.
                for fp in filepaths:
                    if generated >= deficit:
                        break
                    try:
                        wav, sr = load_waveform(fp)
                        variants = generate_variants(wav, sr, n_variants=args.variants_per_sample)
                        for v in variants:
                            if generated >= deficit:
                                break
                            X.append(mfcc_from_waveform(v, sr))
                            y.append(class_idx)
                            generated += 1
                            pbar.update(1)
                    except Exception:
                        continue
            pbar.close()

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