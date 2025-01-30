"""Image preprocessing utilities."""

from pathlib import Path
from typing import Union

import tensorflow as tf
import os

from src.config import IMG_SIZE


def preprocess_image(image_path):
    """Preprocess image for model input.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed image tensor
    """
    # Read image file
    image = tf.io.read_file(image_path)
    
    # Decode image
    image = tf.io.decode_jpeg(image, channels=3)
    
    # Resize image
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    # Normalize pixel values
    image = tf.cast(image, tf.float32) / 255.0
    
    return image


def process_path(file_path):
    """Process a file path into an (image, label) pair.
    
    Args:
        file_path: Path to image file
        
    Returns:
        Tuple of (preprocessed image, label)
    """
    # Get label from parent directory name
    parts = tf.strings.split(file_path, os.sep)
    label = parts[-2] == "defect"
    
    # Load and preprocess the image
    image = preprocess_image(file_path)
    
    return image, label


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

    # Get image paths and labels
    defect_dir = data_dir / "defect"
    no_defect_dir = data_dir / "no_defect"

    defect_paths = list(map(str, defect_dir.glob("*.jpg")))
    no_defect_paths = list(map(str, no_defect_dir.glob("*.jpg")))

    if not defect_paths and not no_defect_paths:
        raise ValueError(f"No images found in {data_dir}")

    # Create dataset
    dataset = tf.data.Dataset.list_files(str(Path(data_dir) / "*/*"))

    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(list(data_dir.glob("*/*"))))

    # Map preprocessing function
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
