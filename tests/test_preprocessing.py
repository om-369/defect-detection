"""Test suite for data preprocessing functions."""

import sys
from pathlib import Path
import pytest
import tensorflow as tf

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

import pytest

from src.config import IMG_SIZE
from src.preprocessing import create_dataset, preprocess_image


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create test data directory with sample images."""
    test_dir = tmp_path_factory.mktemp("test_images")
    
    # Create subdirectories
    defect_dir = test_dir / "defect"
    no_defect_dir = test_dir / "no_defect"
    defect_dir.mkdir()
    no_defect_dir.mkdir()
    
    # Create test images using TensorFlow
    img = tf.random.uniform((IMG_SIZE, IMG_SIZE, 3), maxval=255, dtype=tf.int32)
    img = tf.cast(img, tf.uint8)
    
    # Save images
    for i in range(2):
        tf.io.write_file(
            str(defect_dir / f"defect_{i}.jpg"),
            tf.io.encode_jpeg(img)
        )
        tf.io.write_file(
            str(no_defect_dir / f"no_defect_{i}.jpg"),
            tf.io.encode_jpeg(img)
        )
    
    return test_dir


@pytest.fixture
def test_image_path(test_data_dir):
    """Get path to a test image."""
    return str(test_data_dir / "defect" / "defect_0.jpg")


def test_image_loading(test_image_path):
    """Test image loading functionality."""
    # Load the image
    loaded_img = preprocess_image(test_image_path)

    # Check if loaded image has correct shape and type
    assert loaded_img.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert loaded_img.dtype == tf.float32
    assert tf.reduce_max(loaded_img) <= 1.0
    assert tf.reduce_min(loaded_img) >= 0.0


def test_image_resizing(test_image_path):
    """Test image resizing functionality."""
    # Load and preprocess image
    img = preprocess_image(test_image_path)

    # Check dimensions
    assert img.shape == (IMG_SIZE, IMG_SIZE, 3)


def test_image_normalization(test_image_path):
    """Test image normalization functionality."""
    # Load and preprocess image
    img = preprocess_image(test_image_path)

    # Check if values are normalized
    assert tf.reduce_max(img) <= 1.0
    assert tf.reduce_min(img) >= 0.0
    assert img.dtype == tf.float32


def test_dataset_creation(test_data_dir):
    """Test dataset creation functionality."""
    # Create dataset
    batch_size = 2
    dataset = create_dataset(test_data_dir, batch_size=batch_size)

    # Check if it's a tf.data.Dataset
    assert isinstance(dataset, tf.data.Dataset)

    # Check batch shape
    for images, labels in dataset.take(1):
        assert images.shape[0] == batch_size
        assert images.shape[1:] == (IMG_SIZE, IMG_SIZE, 3)
        assert labels.shape == (batch_size,)
        assert images.dtype == tf.float32
        assert tf.reduce_max(images) <= 1.0
        assert tf.reduce_min(images) >= 0.0
