"""
Trains the CNN-LSTM SER model on cached MFCC features.

Run:
    python train.py --features features/ --out models/
"""
import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from label_map import TARGET_CLASSES
from model import build_cnn_lstm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features/")
    ap.add_argument("--out", default="models/")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    X = np.load(os.path.join(args.features, "X.npy"))
    y = np.load(os.path.join(args.features, "y.npy"))
    num_classes = len(TARGET_CLASSES)

    # Stratified split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # Class weights — RAVDESS/SAVEE/IEMOCAP won't be balanced across your
    # 4 proxy classes, so this matters more than it would in a toy dataset.
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    model = build_cnn_lstm(input_shape=X.shape[1:], num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    checkpoint_path = os.path.join(args.out, "ser_cnn_lstm.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    # --- Evaluation ---
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, target_names=TARGET_CLASSES))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    # Save label encoder (index -> class name) for downstream inference
    with open(os.path.join(args.out, "label_encoder.pkl"), "wb") as f:
        pickle.dump(TARGET_CLASSES, f)

    # Save training curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "training_history.png"))

    print(f"\nModel saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
