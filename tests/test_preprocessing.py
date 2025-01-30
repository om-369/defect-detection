"""Test suite for data preprocessing functions."""

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from src.config import IMG_SIZE
from src.preprocessing import create_dataset, preprocess_image


@pytest.fixture(scope="session")
def test_data_dir():
    """Create and return test data directory."""
    test_dir = Path(__file__).resolve().parent / "data" / "test_images"
    defect_dir = test_dir / "defect"
    no_defect_dir = test_dir / "no_defect"

    # Create test image
    img = Image.new("RGB", (100, 100), color="red")
    img.save(defect_dir / "test_image.jpg")
    img.save(no_defect_dir / "test_image.jpg")

    return test_dir


@pytest.fixture
def test_image_path(test_data_dir):
    """Get path to a test image."""
    return str(test_data_dir / "defect" / "test_image.jpg")


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
