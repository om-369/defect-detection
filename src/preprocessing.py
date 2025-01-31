"""Image preprocessing functions."""

from pathlib import Path
from typing import Tuple, Union

import tensorflow as tf
import cv2

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


def preprocess(image) -> None:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


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
        lambda x, y: (preprocess_image(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
