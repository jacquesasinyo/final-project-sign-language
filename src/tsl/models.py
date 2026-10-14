"""Model definitions (spec §6).

M1: lightweight BiGRU + attention pooling (MOD-02, target ≤ 1/3 of M0's parameters).

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


@keras.saving.register_keras_serializable(package="tsl")
class AttentionPool(layers.Layer):
    """Temporal attention pooling: score each frame, softmax over time, weighted sum of frames."""

    def build(self, input_shape):
        self.score = layers.Dense(1)
        self.score.build(input_shape)

    def call(self, x):
        w = keras.ops.softmax(self.score(x), axis=1)  # (batch, T, 1), sums to 1 over time
        return keras.ops.sum(w * x, axis=1)

    def weights_over_time(self, x):
        return keras.ops.softmax(self.score(x), axis=1)[..., 0]


def build_m1(seq_len, n_features, n_classes, hidden=64, dropout=0.3):
    """M1 efficient model (MOD-02): BiGRU + dropout + temporal attention pooling + softmax."""
    return keras.Sequential([
        keras.Input((seq_len, n_features)),
        layers.Bidirectional(layers.GRU(hidden, return_sequences=True)),
        layers.Dropout(dropout),
        AttentionPool(),
        layers.Dropout(dropout),
        layers.Dense(n_classes, activation="softmax"),
    ], name=f"m1_bigru{hidden}_att")


MODELS = {"m0": build_m0, "m1": build_m1}


def build_model(name, seq_len, n_features, n_classes, **hp):
    """hp: model hyperparameters (M1 only: hidden, dropout). M0 is the fixed previous architecture."""
    if name not in MODELS:
        raise SystemExit(f"Unknown model {name!r}; choose from {sorted(MODELS)}")
    if name == "m0" and hp:
        raise SystemExit("M0 is the fixed previous architecture (D-10); only --lr may change.")
    return MODELS[name](seq_len, n_features, n_classes, **hp)
