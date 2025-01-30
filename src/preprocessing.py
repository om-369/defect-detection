"""Image preprocessing utilities for defect detection."""

import os
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import IMG_SIZE


def preprocess_image(image_path: Union[str, Path]) -> tf.Tensor:
    """Load and preprocess a single image.

    Args:
        image_path: Path to the image file

    Returns:
        Preprocessed image tensor
    """
    # Load image
    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image, channels=3)

    # Resize
    image = resize_image(image)

    # Normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    return image


def resize_image(image: tf.Tensor) -> tf.Tensor:
    """Resize image to target size.

    Args:
        image: Input image tensor

    Returns:
        Resized image tensor
    """
    return tf.image.resize(image, [IMG_SIZE, IMG_SIZE])


def create_dataset(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
) -> tf.data.Dataset:
    """Create a TensorFlow dataset from directory of images.

    Args:
        data_dir: Directory containing the images
        batch_size: Number of images per batch
        shuffle: Whether to shuffle the dataset
        augment: Whether to apply data augmentation

    Returns:
        TensorFlow dataset
    """
    data_dir = Path(data_dir)

    # Get all image paths
    image_paths = [
        str(p)
        for p in data_dir.glob("**/*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if not image_paths:
        raise ValueError(f"No images found in {data_dir}")

    # Create dataset from paths
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))

    # Map preprocessing function
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def augment_image(image: tf.Tensor) -> tf.Tensor:
    """Apply random augmentations to the image.

    Args:
        image: Input image tensor

    Returns:
        Augmented image tensor
    """
    # Random flip
    image = tf.image.random_flip_left_right(image)

    # Random rotation
    image = tf.image.rot90(
        image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    )

    # Random brightness
    image = tf.image.random_brightness(image, 0.2)

    # Random contrast
    image = tf.image.random_contrast(image, 0.8, 1.2)

    # Ensure values are in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image
