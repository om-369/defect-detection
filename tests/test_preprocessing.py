"""Tests for preprocessing functions."""

import pytest
import numpy as np
import torch
import cv2
import tempfile
from pathlib import Path

from src.preprocessing import preprocess, load_dataset


@pytest.fixture
def test_data_dir():
    """Create temporary test data directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        yield tmp_path


@pytest.fixture
def test_image(test_data_dir):
    """Create a test image."""
    image_path = test_data_dir / "defect" / "defect_0.jpg"
    # Create a random color image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), img)
    return image_path


def test_image_loading(test_image):
    """Test image loading function."""
    image = preprocess(str(test_image))
    assert isinstance(image, np.ndarray)
    assert image.dtype == np.float32


def test_image_resizing(test_image):
    """Test image resizing."""
    image = preprocess(str(test_image))
    assert image.shape == (224, 224)  # Single channel grayscale


def test_image_normalization(test_image):
    """Test image normalization."""
    image = preprocess(str(test_image))
    assert np.all(image >= 0) and np.all(image <= 1)


def test_label_reading(test_data_dir):
    """Test label reading."""
    # Create a test label file
    label_path = test_data_dir / "train" / "labels" / "test.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        f.write("1")
    
    assert int(label_path.read_text().strip()) == 1


def test_dataset_creation(test_data_dir):
    """Test dataset creation."""
    # Create test data
    for split in ["train", "valid", "test"]:
        images_dir = test_data_dir / split / "images"
        labels_dir = test_data_dir / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 2 test images and labels
        for i in range(2):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(images_dir / f"test_{i}.jpg"), img)
            
            with open(labels_dir / f"test_{i}.txt", "w") as f:
                f.write(str(i % 2))  # Binary labels
    
    images, labels = load_dataset(test_data_dir, split="train")
    assert isinstance(images, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert len(images) == len(labels)
    assert images.shape[1:] == (224, 224)  # Single channel grayscale
    assert np.all(images >= 0) and np.all(images <= 1)
