"""Test suite for data preprocessing functionality."""

import pytest
import tensorflow as tf
import numpy as np
from pathlib import Path
import sys
import cv2

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import (
    load_image,
    resize_image,
    normalize_image,
    create_dataset,
    augment_image
)
from config import IMG_SIZE

@pytest.fixture
def test_image_path():
    """Get path to a test image."""
    return str(Path("data/test/defect/defect_0.jpg"))

def test_image_loading(test_image_path):
    """Test image loading functionality."""
    # Load the image
    loaded_img = load_image(test_image_path)
    
    # Check if loaded image has correct shape and type
    assert loaded_img.shape == (224, 224, 3)
    assert loaded_img.dtype == np.uint8

def test_image_resizing(test_image_path):
    """Test image resizing functionality."""
    img = load_image(test_image_path)
    resized_img = resize_image(img, IMG_SIZE)
    
    assert resized_img.shape == (*IMG_SIZE, 3)

def test_image_normalization(test_image_path):
    """Test image normalization functionality."""
    img = load_image(test_image_path)
    normalized_img = normalize_image(img)
    
    assert normalized_img.dtype == np.float32
    assert 0 <= normalized_img.min() <= normalized_img.max() <= 1.0

def test_dataset_creation():
    """Test dataset creation functionality."""
    # Get actual test image paths
    test_dir = Path("data/test")
    defect_paths = list(map(str, (test_dir / "defect").glob("*.jpg")))
    no_defect_paths = list(map(str, (test_dir / "no_defect").glob("*.jpg")))
    
    image_paths = defect_paths + no_defect_paths
    labels = [1] * len(defect_paths) + [0] * len(no_defect_paths)
    
    # Create dataset
    batch_size = 2
    dataset = create_dataset(image_paths, labels, batch_size, augment=False)
    
    # Verify dataset properties
    assert isinstance(dataset, tf.data.Dataset)
    
    # Check batch size and shapes
    for images, batch_labels in dataset:
        assert images.shape[0] <= batch_size  # Last batch might be smaller
        assert images.shape[1:] == (*IMG_SIZE, 3)
        assert batch_labels.shape[0] <= batch_size
        break

def test_augmentation(test_image_path):
    """Test image augmentation functionality."""
    # Load and preprocess image
    img = load_image(test_image_path)
    img = resize_image(img)
    img = normalize_image(img)
    
    # Convert to tensor
    image = tf.convert_to_tensor(img)
    label = tf.constant(1)
    
    # Apply augmentation
    aug_image, aug_label = augment_image(image, label)
    
    # Check if shapes are preserved
    assert aug_image.shape == image.shape
    assert aug_label == label
    
    # Check if values are still in valid range
    assert tf.reduce_min(aug_image) >= 0.0
    assert tf.reduce_max(aug_image) <= 1.0

@pytest.mark.parametrize("batch_size,num_samples", [
    (2, 4),
    (4, 8),
])
def test_dataset_different_batch_sizes(batch_size, num_samples):
    """Test dataset creation with different batch sizes."""
    # Get actual test image paths
    test_dir = Path("data/test")
    image_paths = list(map(str, (test_dir / "defect").glob("*.jpg")))[:num_samples]
    labels = [1] * len(image_paths)
    
    dataset = create_dataset(image_paths, labels, batch_size, augment=False)
    
    for images, batch_labels in dataset:
        assert images.shape[0] <= batch_size  # Last batch might be smaller
        break
