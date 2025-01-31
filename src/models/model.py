"""Model creation and training functions."""

import tensorflow as tf
from tensorflow.keras import layers, Model

from src.config import IMG_SIZE, MODEL_CONFIG


def create_model() -> Model:
    """Create and compile the model.

    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Convolutional layers
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(
        MODEL_CONFIG["num_classes"], activation="sigmoid"
    )(x)

    # Create model
    model = Model(inputs, outputs)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    return model
