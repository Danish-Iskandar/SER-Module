"""
Re-evaluate a trained SER model on the held-out test set without retraining.

Recreates exactly the same 70/15/15 stratified split as train.py
(random_state=42), loads a saved model and writes the results the
dissertation needs:

    <out>/classification_report.txt   precision / recall / F1 per class
    <out>/metrics.json                same numbers in machine-readable form
    <out>/confusion_matrix.csv        raw counts
    <out>/confusion_matrix.png        300 dpi figure for Chapter 4

Usage (Colab, from SER-Module/src):
    python evaluate.py \
        --features /content/drive/MyDrive/features/ \
        --model    /content/drive/MyDrive/models/ser_cnn_lstm.h5 \
        --out      /content/drive/MyDrive/eval_original/ \
        --label    "Original model"
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from label_map import TARGET_CLASSES

ORIGINAL_DATASET_SIZE = 912  # RAVDESS + SAVEE after label mapping, no augmentation


def split_like_train(X, y):
    """Identical to train.py: 70% train, 15% val, 15% test, stratified, seed 42."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def plot_confusion(cm, labels, title, path):
    """Counts with row percentages; one sequential hue, darker = more samples."""
    plt.rcParams["font.family"] = "serif"
    row_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(4.6, 3.9), dpi=300)
    im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            dark = row_pct[i, j] > 55
            ax.text(j, i, f"{cm[i, j]}\n({row_pct[i, j]:.0f}%)", ha="center", va="center",
                    fontsize=9, color="white" if dark else "#222222")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row percentage (recall)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features/")
    ap.add_argument("--model", default="models/ser_cnn_lstm.h5")
    ap.add_argument("--out", default="eval/")
    ap.add_argument("--label", default="CNN-LSTM model")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import tensorflow as tf  # imported late so --help works without TensorFlow

    X = np.load(os.path.join(args.features, "X.npy"))
    y = np.load(os.path.join(args.features, "y.npy"))
    counts = {TARGET_CLASSES[i]: int((y == i).sum()) for i in range(len(TARGET_CLASSES))}
    print(f"Features: {X.shape[0]} samples, shape {X.shape[1:]}, class counts {counts}")
    if X.shape[0] != ORIGINAL_DATASET_SIZE:
        print(f"\n*** WARNING: expected {ORIGINAL_DATASET_SIZE} samples for the original dataset, "
              f"found {X.shape[0]}.\n*** This folder probably holds AUGMENTED features. Augmented "
              "copies were created before the split, so variants of training clips can appear "
              "in this test set and the scores will be optimistic.\n")

    _, (X_val, y_val), (X_test, y_test) = split_like_train(X, y)
    test_counts = {TARGET_CLASSES[i]: int((y_test == i).sum()) for i in range(len(TARGET_CLASSES))}
    print(f"Test set: {len(y_test)} samples {test_counts}")

    model = tf.keras.models.load_model(args.model, compile=False)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    val_acc = accuracy_score(y_val, np.argmax(model.predict(X_val, verbose=0), axis=1))

    labels_idx = list(range(len(TARGET_CLASSES)))
    report_txt = classification_report(y_test, y_pred, labels=labels_idx,
                                       target_names=TARGET_CLASSES, digits=3, zero_division=0)
    report = classification_report(y_test, y_pred, labels=labels_idx,
                                   target_names=TARGET_CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels_idx)
    majority = max(test_counts.values()) / len(y_test)

    metrics = {
        "label": args.label,
        "model_path": args.model,
        "features_path": args.features,
        "n_total": int(X.shape[0]),
        "class_counts_total": counts,
        "n_test": int(len(y_test)),
        "class_counts_test": test_counts,
        "test_accuracy": round(float(report["accuracy"]), 4),
        "val_accuracy": round(float(val_acc), 4),
        "majority_baseline": round(float(majority), 4),
        "per_class": {c: {k: round(float(report[c][k]), 4) for k in ("precision", "recall", "f1-score")}
                      for c in TARGET_CLASSES},
        "macro_avg": {k: round(float(report["macro avg"][k]), 4) for k in ("precision", "recall", "f1-score")},
        "confusion_matrix": {"rows_true_cols_pred": TARGET_CLASSES, "counts": cm.tolist()},
    }

    with open(os.path.join(args.out, "classification_report.txt"), "w") as f:
        f.write(f"{args.label}\n{report_txt}\n")
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savetxt(os.path.join(args.out, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",",
               header=",".join(TARGET_CLASSES), comments="true\\pred,")
    plot_confusion(cm, TARGET_CLASSES, f"{args.label}: confusion matrix (test set, n={len(y_test)})",
                   os.path.join(args.out, "confusion_matrix.png"))

    print(f"\n{args.label}")
    print(report_txt)
    print("Confusion matrix (rows = true, columns = predicted):")
    print("            " + "  ".join(f"{c:>9}" for c in TARGET_CLASSES))
    for c, row in zip(TARGET_CLASSES, cm):
        print(f"{c:>10}  " + "  ".join(f"{v:>9d}" for v in row))
    print(f"\nTest accuracy {metrics['test_accuracy']:.3f} | validation accuracy {metrics['val_accuracy']:.3f}"
          f" | majority-class baseline {majority:.3f}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
