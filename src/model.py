"""
CNN-LSTM architecture for MFCC-based speech emotion classification.

Input:  (time_steps, n_mfcc)  e.g. (130, 40)
Output: softmax over TARGET_CLASSES

Design: Conv1D layers extract local spectral patterns across the MFCC
coefficient axis at each time step; stacked LSTM layers then model how
those patterns evolve over time (vocal stress/frustration tends to build
progressively rather than appear instantly) — matching the architecture
description in your Literature Review / System Architecture sections.
"""
from tensorflow.keras import layers, models


def build_cnn_lstm(input_shape, num_classes, dropout=0.3):
    inputs = layers.Input(shape=input_shape)  # (time_steps, n_mfcc)

    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv1D(128, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="ser_cnn_lstm")
    return model