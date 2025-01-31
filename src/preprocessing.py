"""Image preprocessing functions."""

import os
from pathlib import Path
from typing import Tuple, Union

import tensorflow as tf

from src.config import IMG_SIZE


def read_yolo_label(label_path: str) -> Tuple[float, float, float, float]:
    """Read YOLO format label file.
    
    Args:
        label_path: Path to label file
        
    Returns:
        Tuple of (x_center, y_center, width, height) normalized coordinates
    """
    with open(label_path, 'r') as f:
        line = f.readline().strip().split()
        # YOLO format: class x_center y_center width height
        return tuple(map(float, line[1:]))


def preprocess_image(image_path: str) -> tf.Tensor:
    """Preprocess image for model input.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed image tensor
    """
    # Read and decode image
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    
    # Resize image
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    # Normalize pixel values
    image = tf.cast(image, tf.float32) / 255.0
    
    return image


def create_dataset(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True
) -> tf.data.Dataset:
    """Create a TensorFlow dataset from directory structure.
    
    Args:
        data_dir: Path to data directory containing 'defect' and 'no_defect' subdirectories
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        
    Returns:
        TensorFlow dataset
    """
    data_dir = Path(data_dir)
    
    # Get all image paths
    defect_dir = data_dir / "defect"
    no_defect_dir = data_dir / "no_defect"
    
    defect_images = list(defect_dir.glob("*.jpg")) + list(defect_dir.glob("*.jpeg"))
    no_defect_images = list(no_defect_dir.glob("*.jpg")) + list(no_defect_dir.glob("*.jpeg"))
    
    all_images = defect_images + no_defect_images
    labels = [1] * len(defect_images) + [0] * len(no_defect_images)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        [str(x) for x in all_images],
        labels
    ))
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(all_images))
    
    # Map preprocessing function
    dataset = dataset.map(
        lambda x, y: (preprocess_image(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset
