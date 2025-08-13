"""
TensorFlow MLP Classifier Module

Production-ready multi-layer perceptron for tabular data.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MLPClassifier:
    """Multi-layer perceptron classifier using TensorFlow."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.model = self._build_model()
    
    def _build_model(self) -> tf.keras.Model:
        """Build the MLP model."""
        model_layers = [layers.Input(shape=(self.input_dim,))]
        
        for hidden_dim in self.hidden_dims:
            model_layers.extend([
                layers.Dense(hidden_dim, activation='relu'),
                layers.Dropout(self.dropout)
            ])
        
        model_layers.append(
            layers.Dense(self.num_classes, activation='softmax')
        )
        
        model = models.Sequential(model_layers)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32
    ) -> tf.keras.callbacks.History:
        """Train the model."""
        validation_data = (X_val, y_val) if X_val is not None else None
        
        return self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return {'loss': loss, 'accuracy': accuracy}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X, verbose=0)


class TabularPipeline:
    """Complete tabular data pipeline."""
    
    def __init__(self, classifier: MLPClassifier):
        self.classifier = classifier
    
    def create_dataset(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> tf.data.Dataset:
        """Create optimized tf.data.Dataset."""
        dataset = tf.data.Dataset.from_tensor_slices((
            features.astype(np.float32),
            labels.astype(np.int32)
        ))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        return (
            dataset
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
    
    def train_with_pipeline(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        epochs: int = 50
    ) -> tf.keras.callbacks.History:
        """Train using tf.data pipeline."""
        return self.classifier.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            verbose=1
        )