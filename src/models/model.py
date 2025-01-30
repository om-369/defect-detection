"""Model definition for defect detection."""

import tensorflow as tf

from src.config import IMG_SIZE, MODEL_CONFIG


def create_model(num_classes: int = MODEL_CONFIG["num_classes"]) -> tf.keras.Model:
    """Create and return the model architecture.

    Args:
        num_classes: Number of output classes.

    Returns:
        Compiled model ready for training.
    """
    # Create input layer with correct shape
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Create base model
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        input_tensor=inputs
    )
    base_model.trainable = False

    # Create model with custom head
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

    # Create and compile model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
    )

    return model
