"""Model architecture definition for defect detection."""

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import IMG_SIZE
from config import MODEL_CONFIG


def create_model(num_classes: int = MODEL_CONFIG["num_classes"]) -> tf.keras.Model:
    """Create and return the model architecture."""
    # Load pre-trained MobileNetV2 instead of ResNet50 (lighter model)
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)
    )

    # Freeze only the first 100 layers
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # Create new model on top
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Pass inputs through base model
    x = base_model(x)

    # Add custom layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Add final classification layer
    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Create the model
    model = tf.keras.Model(inputs, outputs)

    return model


def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    """Compile the model with appropriate optimizer and loss function."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=MODEL_CONFIG["learning_rate"])

    if MODEL_CONFIG["num_classes"] == 2:
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
        ]
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = ["accuracy"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def get_callbacks() -> list:
    """Get list of callbacks for model training."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/best_model.h5", monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1),
    ]
    return callbacks
