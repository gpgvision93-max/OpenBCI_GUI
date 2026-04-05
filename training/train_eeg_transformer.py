"""
EEG Transformer training script.

Trains a lightweight transformer-based classifier on synthetic EEG data,
exports the model to TensorFlow Lite format, and saves the feature scaler.

Outputs
-------
training/models/eeg_transformer.tflite
training/models/eeg_scaler.pkl
"""
import os
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHANNELS = 8
WINDOW_SIZE = 250
NUM_CLASSES = 4
SAMPLES_PER_CLASS = 200
BATCH_SIZE = 32
EPOCHS = 10
RANDOM_SEED = 42

LABELS = ["LEFT", "RIGHT", "UP", "DOWN"]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")
TFLITE_PATH = os.path.join(OUTPUT_DIR, "eeg_transformer.tflite")
SCALER_PATH = os.path.join(OUTPUT_DIR, "eeg_scaler.pkl")


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_eeg(n_per_class: int, channels: int, window: int, seed: int):
    """Return (X, y) with shape (N, window, channels) and (N,)."""
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for cls in range(NUM_CLASSES):
        freqs = [4.0 + cls * 3.0]  # slightly different dominant frequency per class
        t = np.linspace(0, 1, window)
        for _ in range(n_per_class):
            signals = np.zeros((window, channels))
            for ch in range(channels):
                for f in freqs:
                    signals[:, ch] += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
                signals[:, ch] += rng.normal(0, 0.2, window)
            X_list.append(signals)
            y_list.append(cls)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)


def build_transformer(window: int, channels: int, num_classes: int,
                      d_model: int = 32, num_heads: int = 4,
                      dff: int = 64, dropout_rate: float = 0.1):
    inputs = keras.Input(shape=(window, channels), name="eeg_input")

    # Linear projection to d_model
    x = keras.layers.Dense(d_model)(inputs)

    # Add positional encoding
    pos_enc = positional_encoding(window, d_model)
    x = x + pos_enc

    # Transformer encoder block
    attn_output = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads
    )(x, x)
    attn_output = keras.layers.Dropout(dropout_rate)(attn_output)
    x = keras.layers.LayerNormalization()(x + attn_output)

    ffn = keras.layers.Dense(dff, activation="relu")(x)
    ffn = keras.layers.Dense(d_model)(ffn)
    ffn = keras.layers.Dropout(dropout_rate)(ffn)
    x = keras.layers.LayerNormalization()(x + ffn)

    # Global average pooling → classification head
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="logits")(x)

    return keras.Model(inputs, outputs, name="eeg_transformer")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating synthetic EEG data...")
    X, y = generate_synthetic_eeg(SAMPLES_PER_CLASS, CHANNELS, WINDOW_SIZE, RANDOM_SEED)

    # Scale features using per-sample mean/std (flatten → fit scaler → reshape)
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)
    scaler = StandardScaler()
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(n_samples, WINDOW_SIZE, CHANNELS).astype(np.float32)

    split = int(0.8 * n_samples)
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    model = build_transformer(WINDOW_SIZE, CHANNELS, NUM_CLASSES)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
    )

    # Export to TFLite
    print(f"Converting model to TFLite: {TFLITE_PATH}")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved ({len(tflite_model):,} bytes).")

    # Save scaler
    print(f"Saving scaler: {SCALER_PATH}")
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler saved.")

    print("Training complete.")


if __name__ == "__main__":
    main()
