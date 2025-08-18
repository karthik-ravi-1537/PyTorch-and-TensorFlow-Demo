"""
TensorFlow Text Classification Module

Production-ready text classification implementation with TensorFlow.
"""

import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

logger = logging.getLogger(__name__)


class LSTMTextClassifier:
    """LSTM-based text classifier using TensorFlow."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 2,
        max_length: int = 100,
        dropout: float = 0.3,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_length = max_length
        self.dropout = dropout
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """Build the LSTM model."""
        model = models.Sequential(
            [
                layers.Embedding(self.vocab_size, self.embed_dim, input_length=self.max_length, mask_zero=True),
                layers.Bidirectional(
                    layers.LSTM(self.hidden_dim, dropout=self.dropout, recurrent_dropout=self.dropout)
                ),
                layers.Dropout(self.dropout),
                layers.Dense(64, activation="relu"),
                layers.Dropout(self.dropout),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> tf.keras.callbacks.History:
        """Train the model."""
        validation_data = (X_val, y_val) if X_val is not None else None

        return self.model.fit(
            X_train, y_train, validation_data=validation_data, epochs=epochs, batch_size=batch_size, verbose=1
        )

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Evaluate model performance."""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return {"loss": loss, "accuracy": accuracy}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X, verbose=0)


class TextClassificationPipeline:
    """Complete text classification pipeline."""

    def __init__(self, classifier: LSTMTextClassifier):
        self.classifier = classifier

    def create_dataset(self, sequences: list[list[int]], labels: list[int], batch_size: int = 32) -> tf.data.Dataset:
        """Create optimized tf.data.Dataset."""
        dataset = tf.data.Dataset.from_tensor_slices(
            (np.array(sequences, dtype=np.int32), np.array(labels, dtype=np.int32))
        )

        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def train_with_pipeline(
        self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset | None = None, epochs: int = 10
    ) -> tf.keras.callbacks.History:
        """Train using tf.data pipeline."""
        return self.classifier.model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=1)
