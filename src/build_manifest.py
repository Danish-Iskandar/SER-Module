"""
Scans data_root/{RAVDESS,SAVEE,IEMOCAP} and builds a single manifest.csv with
columns: filepath, dataset, native_label, target_label

Run:
    python build_manifest.py --data_root data/ --out manifest.csv
"""
import argparse
import os
import re
import pandas as pd

from label_map import (
    ravdess_code_to_target,
    savee_prefix_to_target,
    iemocap_tag_to_target,
)


def scan_ravdess(root):
    rows = []
    if not os.path.isdir(root):
        return rows
    for actor_dir in os.listdir(root):
        actor_path = os.path.join(root, actor_dir)
        if not os.path.isdir(actor_path):
            continue
        for fname in os.listdir(actor_path):
            if not fname.endswith(".wav"):
                continue
            parts = fname.replace(".wav", "").split("-")
            if len(parts) < 3:
                continue
            emotion_code = parts[2]
            target = ravdess_code_to_target(emotion_code)
            if target is None:
                continue
            rows.append({
                "filepath": os.path.join(actor_path, fname),
                "dataset": "RAVDESS",
                "target_label": target,
            })
    return rows


def scan_savee(root):
    rows = []
    if not os.path.isdir(root):
        return rows
    # SAVEE files are typically named like "DC_a01.wav", "JE_sa03.wav" etc.
    # Prefix letters (before digits) encode the emotion.
    pattern = re.compile(r"([a-zA-Z]+)\d+\.wav$")
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.endswith(".wav"):
                continue
            m = pattern.search(fname)
            if not m:
                continue
            prefix = m.group(1).lower()
            # strip speaker-initial prefixes like "dc_" if present
            prefix = prefix.split("_")[-1]
            target = savee_prefix_to_target(prefix)
            if target is None:
                continue
            rows.append({
                "filepath": os.path.join(dirpath, fname),
                "dataset": "SAVEE",
                "target_label": target,
            })
    return rows


def scan_iemocap(root):
    """
    IEMOCAP ships emotion labels in EmoEvaluation/*.txt files, not filenames.
    Each relevant line looks like:
        [6.2901 - 8.2357]  Ses01F_impro01_F000  neu  [2.5000, 2.5000, 2.5000]
    We parse those to get {utterance_id: native_tag}, then locate the matching
    .wav under sentences/wav/.
    """
    rows = []
    if not os.path.isdir(root):
        return rows
    line_re = re.compile(r"^\[.*?\]\s+(\S+)\s+(\w+)\s+\[")
    for session in os.listdir(root):
        eval_dir = os.path.join(root, session, "dialog", "EmoEvaluation")
        wav_root = os.path.join(root, session, "sentences", "wav")
        if not os.path.isdir(eval_dir):
            continue
        for txt_file in os.listdir(eval_dir):
            if not txt_file.endswith(".txt"):
                continue
            with open(os.path.join(eval_dir, txt_file), "r", errors="ignore") as f:
                for line in f:
                    m = line_re.match(line)
                    if not m:
                        continue
                    utt_id, tag = m.group(1), m.group(2)
                    target = iemocap_tag_to_target(tag)
                    if target is None:
                        continue
                    # wav lives at wav_root/<dialog_name>/<utt_id>.wav
                    dialog_name = utt_id.rsplit("_", 1)[0]
                    wav_path = os.path.join(wav_root, dialog_name, utt_id + ".wav")
                    if os.path.exists(wav_path):
                        rows.append({
                            "filepath": wav_path,
                            "dataset": "IEMOCAP",
                            "target_label": target,
                        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/")
    ap.add_argument("--out", default="manifest.csv")
    args = ap.parse_args()

    rows = []
    rows += scan_ravdess(os.path.join(args.data_root, "RAVDESS"))
    rows += scan_savee(os.path.join(args.data_root, "SAVEE"))
    rows += scan_iemocap(os.path.join(args.data_root, "IEMOCAP"))

    if not rows:
        print("No files found. Check --data_root and folder names "
              "(expects RAVDESS/, SAVEE/, IEMOCAP/ subfolders).")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print(df.groupby(["dataset", "target_label"]).size())


if __name__ == "__main__":
    main()