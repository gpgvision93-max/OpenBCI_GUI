"""
EEG Transformer Training Script

Trains a lightweight transformer model on synthetic EEG data and exports:
  - training/models/eeg_transformer.tflite  (TensorFlow Lite model)
  - training/models/eeg_scaler.pkl          (fitted StandardScaler)
"""

import os
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_SAMPLES = 1000
N_CHANNELS = 8        # EEG channels (e.g. OpenBCI Cyton)
N_TIMESTEPS = 128     # samples per window
N_CLASSES = 3         # mental states: rest, left-hand MI, right-hand MI
D_MODEL = 32          # transformer embedding dimension
N_HEADS = 4           # multi-head attention heads
FFN_DIM = 64          # feed-forward network inner dimension
N_BLOCKS = 2          # number of transformer encoder blocks
EPOCHS = 10
BATCH_SIZE = 32
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
def generate_synthetic_eeg(n_samples: int, n_channels: int, n_timesteps: int,
                             n_classes: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) with X shape (n_samples, n_timesteps, n_channels)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_timesteps, n_channels)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n_samples)
    # Add class-specific signal
    for cls in range(n_classes):
        mask = y == cls
        freq = (cls + 1) * 4.0  # 4 Hz, 8 Hz, 12 Hz
        t = np.linspace(0, 1, n_timesteps, dtype=np.float32)
        sine = np.sin(2 * np.pi * freq * t)
        X[mask] += sine[None, :, None] * 0.5
    return X, y.astype(np.int32)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def fit_and_apply_scaler(X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on flattened training data, return scaled arrays."""
    n_train, t, c = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, c)
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(n_train, t, c)
    n_val = X_val.shape[0]
    X_val_scaled = scaler.transform(X_val.reshape(-1, c)).reshape(n_val, t, c)
    return X_train_scaled, X_val_scaled, scaler


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encoding."""
    positions = np.arange(seq_len)[:, None]
    dims = np.arange(d_model)[None, :]
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return angles.astype(np.float32)[None, :, :]  # (1, seq_len, d_model)


def build_eeg_transformer(n_timesteps: int, n_channels: int, n_classes: int,
                           d_model: int, n_heads: int, ffn_dim: int,
                           n_blocks: int) -> keras.Model:
    """Build a small transformer encoder for EEG classification."""
    inputs = keras.Input(shape=(n_timesteps, n_channels), name="eeg_input")

    # Linear projection to d_model
    x = layers.Dense(d_model, name="input_projection")(inputs)

    # Add positional encoding (non-trainable constant)
    pos_enc = positional_encoding(n_timesteps, d_model)
    pos_enc_layer = layers.Lambda(
        lambda t: t + tf.constant(pos_enc, dtype=tf.float32),
        name="positional_encoding"
    )
    x = pos_enc_layer(x)

    # Transformer encoder blocks
    for block_idx in range(n_blocks):
        # Multi-head self-attention
        attn_output = layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            name=f"mha_{block_idx}"
        )(x, x)
        x = layers.LayerNormalization(name=f"ln1_{block_idx}")(x + attn_output)

        # Feed-forward network
        ffn = layers.Dense(ffn_dim, activation="relu",
                           name=f"ffn_expand_{block_idx}")(x)
        ffn = layers.Dense(d_model, name=f"ffn_project_{block_idx}")(ffn)
        x = layers.LayerNormalization(name=f"ln2_{block_idx}")(x + ffn)

    # Global average pooling + classification head
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    outputs = layers.Dense(n_classes, activation="softmax",
                           name="classifier")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="EEGTransformer")


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def export_tflite(keras_model: keras.Model, output_path: str) -> None:
    """Convert Keras model to TFLite and write to disk."""
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to: {output_path} ({len(tflite_model):,} bytes)")


def export_scaler(scaler: StandardScaler, output_path: str) -> None:
    """Serialize the fitted scaler with pickle."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def main() -> None:
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("Generating synthetic EEG data ...")
    X, y = generate_synthetic_eeg(N_SAMPLES, N_CHANNELS, N_TIMESTEPS,
                                   N_CLASSES, seed=RANDOM_SEED)

    split = int(N_SAMPLES * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print("Fitting scaler ...")
    X_train, X_val, scaler = fit_and_apply_scaler(X_train, X_val)

    print("Building model ...")
    model = build_eeg_transformer(
        n_timesteps=N_TIMESTEPS,
        n_channels=N_CHANNELS,
        n_classes=N_CLASSES,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        ffn_dim=FFN_DIM,
        n_blocks=N_BLOCKS,
    )
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Training ...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    tflite_path = os.path.join(MODELS_DIR, "eeg_transformer.tflite")
    scaler_path = os.path.join(MODELS_DIR, "eeg_scaler.pkl")

    print("Exporting model ...")
    export_tflite(model, tflite_path)
    export_scaler(scaler, scaler_path)

    print("Done.")


if __name__ == "__main__":
    main()
