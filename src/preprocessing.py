"""Image preprocessing functions."""

from pathlib import Path
from typing import Tuple, Union

import tensorflow as tf
import cv2
import numpy as np

from src.config import IMG_SIZE


def read_yolo_label(label_path: str) -> Tuple[float, float, float, float]:
    """Read YOLO format label file.

    Args:
        label_path: Path to label file

    Returns:
        Tuple of (x_center, y_center, width, height) normalized coordinates
    """
    with open(label_path, "r") as f:
        line = f.readline().strip().split()
        return tuple(map(float, line[1:]))


def preprocess_image(image_path: str) -> tf.Tensor:
    """Preprocess image for model input.

    Args:
        image_path: Path to image file

    Returns:
        Preprocessed image tensor
    """
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def preprocess(image_path):
    """
    Preprocess an image for model input.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to target size
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized


def load_dataset(data_dir, split='train'):
    """
    Load dataset from directory.
    
    Args:
        data_dir (str): Path to data directory
        split (str): Dataset split (train, valid, test)
        
    Returns:
        tuple: (images, labels)
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / split / 'images'
    labels_dir = data_dir / split / 'labels'
    
    # Get all image files
    image_files = sorted(list(images_dir.glob('*.jpg')))
    label_files = sorted(list(labels_dir.glob('*.txt')))
    
    if len(image_files) != len(label_files):
        raise ValueError(f"Number of images ({len(image_files)}) does not match number of labels ({len(label_files)})")
    
    # Load and preprocess images
    images = []
    labels = []
    
    for img_path, label_path in zip(image_files, label_files):
        # Load image
        image = preprocess(str(img_path))
        images.append(image)
        
        # Load label
        with open(label_path, 'r') as f:
            label = int(f.read().strip())
        labels.append(label)
    
    return np.array(images), np.array(labels)


def create_dataset(
    data_dir: Union[str, Path], batch_size: int = 32, shuffle: bool = True
) -> tf.data.Dataset:
    """Create a TensorFlow dataset from directory structure.

    Args:
        data_dir: Root directory containing defect and no_defect subdirs
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset

    Returns:
        TensorFlow dataset
    """
    data_dir = Path(data_dir)

    defect_dir = data_dir / "defect"
    no_defect_dir = data_dir / "no_defect"

    defect_images = (
        list(defect_dir.glob("*.jpg")) + list(defect_dir.glob("*.jpeg"))
    )
    no_defect_images = (
        list(no_defect_dir.glob("*.jpg")) + list(no_defect_dir.glob("*.jpeg"))
    )

    all_images = defect_images + no_defect_images
    labels = [1] * len(defect_images) + [0] * len(no_defect_images)

    dataset = tf.data.Dataset.from_tensor_slices(
        ([str(x) for x in all_images], labels)
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(all_images))

    dataset = dataset.map(
        lambda x, y: (tf.convert_to_tensor(preprocess(x), dtype=tf.float32), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
