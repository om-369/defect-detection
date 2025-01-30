"""Image preprocessing utilities."""

from pathlib import Path
from typing import Union

import tensorflow as tf

from src.config import IMG_SIZE


def preprocess_image(image_path: Union[str, Path]) -> tf.Tensor:
    """Load and preprocess a single image.

    Args:
        image_path: Path to the image file.

    Returns:
        Preprocessed image tensor.
    """
    # Read and decode image
    image = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(image, channels=3, expand_animations=False)

    # Convert to float32 and normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    # Resize image using size as a list
    size = tf.constant([IMG_SIZE, IMG_SIZE])
    image = tf.image.resize(image, size)

    # Ensure shape is correct
    image = tf.ensure_shape(image, [IMG_SIZE, IMG_SIZE, 3])

    # Add batch dimension
    image = tf.expand_dims(image, 0)

    return image


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
    image_paths = defect_paths + no_defect_paths
    labels = [1] * len(defect_paths) + [0] * len(no_defect_paths)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))

    # Map preprocessing function
    def process_path(path, label):
        image = preprocess_image(path)[0]  # Remove batch dimension
        return image, tf.cast(label, tf.float32)

    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset
