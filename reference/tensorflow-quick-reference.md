# TensorFlow Quick Reference

## Essential Imports
```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
```

## Tensor Operations
```python
# Create tensors
x = tf.constant([1, 2, 3])
x = tf.zeros((3, 4))
x = tf.random.normal((2, 3))

# Basic operations
y = x + 1
z = tf.matmul(x, y)
```

## Neural Networks (Keras)
```python
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## Training
```python
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_val, y_val)
)
```

## tf.data Pipeline
```python
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = (dataset
    .shuffle(buffer_size=1000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)
```

## Common Patterns
- Use `tf.function` decorator for performance optimization
- Use `tf.GradientTape()` for custom training loops
- Save models with `model.save('model_path')`
- Use `tf.data` for efficient data pipelines