"""Model definitions (spec §6).

M0: the previous TÜBİTAK prototype's CNN-LSTM (notebook "Action Detection Refined",
`build_cnn_lstm_model`), re-implemented layer for layer. Only the input shape changes
(48 x 100 instead of 30 x 1662). See docs/decisions.md D-10.
"""
import keras
from keras import layers


def build_m0(seq_len, n_features, n_classes):
    return keras.Sequential([
        keras.Input((seq_len, n_features)),
        # Convolutional layers for local spatio-temporal patterns
        layers.Conv1D(64, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1D(64, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        layers.Conv1D(128, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1D(128, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        # Bidirectional LSTMs for temporal modelling
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.3),
        # Classifier
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ], name="m0_cnn_lstm")


MODELS = {"m0": build_m0}


def build_model(name, seq_len, n_features, n_classes):
    if name not in MODELS:
        raise SystemExit(f"Unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](seq_len, n_features, n_classes)
