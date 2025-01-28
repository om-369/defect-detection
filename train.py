"""Main training script for the defect detection model."""

import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR,
    BATCH_SIZE,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    MODEL_CONFIG,
)
from src.data.preprocessing import create_dataset
from src.models.model import create_model, compile_model, get_callbacks
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
    """Main training function."""
    print("Loading and preparing data...")

    # Load and split data
    image_paths, labels = load_data()
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = (
        split_data(image_paths, labels)
    )

    print(f"\nDataset statistics:")
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")

    # Create datasets with smaller batch size for small dataset
    train_dataset = create_dataset(
        train_paths, train_labels, batch_size=4, augment=True
    )
    val_dataset = create_dataset(val_paths, val_labels, batch_size=4)
    test_dataset = create_dataset(test_paths, test_labels, batch_size=4)

    print("\nCreating and compiling model...")
    # Create and compile model
    model = create_model()
    model = compile_model(model)

    # Get callbacks
    callbacks = get_callbacks()

    print("\nStarting training...")
    # Train model with class weights to handle imbalance
    history = model.fit(
        train_dataset,
        epochs=MODEL_CONFIG["epochs"],
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate model
    print("\nEvaluating model on test set...")
    test_results = model.evaluate(test_dataset, verbose=1)
    print("\nTest Results:")
    for metric_name, value in zip(model.metrics_names, test_results):
        print(f"{metric_name}: {value:.4f}")

    # Plot training history
    plot_training_history(history)

    # Save final model
    model.save("models/final_model.h5")
    print("\nTraining completed! Model saved as 'models/final_model.h5'")


if __name__ == "__main__":
    train_model()
