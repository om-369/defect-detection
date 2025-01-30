"""Main training script for the defect detection model."""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    MODEL_CONFIG,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from src.data.preprocessing import create_dataset, load_image, preprocess_image
from src.models.model import compile_model, create_model, get_callbacks
from src.utils.visualization import plot_training_history


def load_data():
    """Load and split the dataset."""
    processed_dir = DATA_DIR / "labeled"

    # Load image paths and labels
    image_paths = []
    labels = []

    for class_dir in processed_dir.iterdir():
        if class_dir.is_dir():
            label = 1 if class_dir.name == "defect" else 0
            for img_path in class_dir.glob("*.jpg"):
                image_paths.append(str(img_path))
                labels.append(label)

    return image_paths, labels


def split_data(image_paths, labels):
    """Split data into train, validation, and test sets."""
    # First split: separate test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=TEST_SPLIT, stratify=labels, random_state=42
    )

    # Second split: separate train and validation sets
    val_size = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_size,
        stratify=train_val_labels,
        random_state=42,
    )

    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
    )


def train_model():
    """Train the defect detection model."""
    print("Loading data...")
    image_paths, labels = load_data()

    print("Splitting data...")
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = (
        split_data(image_paths, labels)
    )

    print(f"Train samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")

    print("Creating datasets...")
    train_dataset = create_dataset(
        train_paths, train_labels, batch_size=BATCH_SIZE, augment=True
    )
    val_dataset = create_dataset(
        val_paths, val_labels, batch_size=BATCH_SIZE, augment=False
    )
    test_dataset = create_dataset(
        test_paths, test_labels, batch_size=BATCH_SIZE, augment=False
    )

    print("Creating model...")
    model = create_model()
    model = compile_model(model)

    print("Training model...")
    callbacks = get_callbacks()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=MODEL_CONFIG["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    print("Saving model...")
    model.save("models/latest")

    print("Plotting training history...")
    plot_training_history(history)

    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    print(f"\nTest accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    train_model()
