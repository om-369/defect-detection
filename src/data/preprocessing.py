"""Data preprocessing utilities for the defect detection project."""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_DIR, IMG_SIZE


def load_image(image_path: str) -> np.ndarray:
    """Load and preprocess a single image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_image(
    image: np.ndarray, target_size: Tuple[int, int] = IMG_SIZE
) -> np.ndarray:
    """Resize image to target size."""
    return cv2.resize(image, target_size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image pixels to [0, 1] range."""
    return image.astype(np.float32) / 255.0


def create_dataset(
    image_paths: List[str],
    labels: List[int],
    batch_size: int = 32,
    augment: bool = False,
) -> tf.data.Dataset:
    """Create a TensorFlow dataset from image paths and labels."""

    def process_path(image_path, label):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def augment_image(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply data augmentation to an image."""
    # Basic TensorFlow augmentations instead of albumentations
    # Random flip left/right
    image = tf.image.random_flip_left_right(image)

    # Random flip up/down
    image = tf.image.random_flip_up_down(image)

    # Random brightness
    image = tf.image.random_brightness(image, 0.2)

    # Random contrast
    image = tf.image.random_contrast(image, 0.8, 1.2)

    # Random rotation
    image = tf.image.rot90(
        image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    )

    # Ensure the image values are still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def preprocess_dataset():
    """Main function to preprocess the entire dataset."""
    raw_dir = DATA_DIR / "raw"
    processed_dir = DATA_DIR / "processed"

    # Create processed directory if it doesn't exist
    processed_dir.mkdir(exist_ok=True)

    # Process all images in raw directory
    for img_path in raw_dir.glob("*.jpg"):
        img = load_image(str(img_path))
        img = resize_image(img)
        img = normalize_image(img)

        # Save processed image
        output_path = processed_dir / img_path.name
        cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    preprocess_dataset()
