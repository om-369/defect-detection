"""Tests for preprocessing functionality."""

import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path

from src.preprocessing import preprocess, load_dataset


@pytest.fixture
def test_data_dir():
    """Create temporary test data directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_dir = Path(tmp_dir)
        
        # Create directory structure
        (data_dir / 'train' / 'images').mkdir(parents=True)
        (data_dir / 'train' / 'labels').mkdir(parents=True)
        
        # Create test image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(data_dir / 'train' / 'images' / 'test.jpg'), img)
        
        # Create test label
        with open(data_dir / 'train' / 'labels' / 'test.txt', 'w') as f:
            f.write('1')
        
        yield data_dir


def test_image_loading(test_data_dir):
    """Test image loading functionality."""
    img_path = test_data_dir / 'train' / 'images' / 'test.jpg'
    img = preprocess(img_path)
    assert isinstance(img, np.ndarray)
    assert len(img.shape) == 2  # Grayscale


def test_image_resizing(test_data_dir):
    """Test image resizing functionality."""
    img_path = test_data_dir / 'train' / 'images' / 'test.jpg'
    img = preprocess(img_path)
    assert img.shape == (224, 224)


def test_image_normalization(test_data_dir):
    """Test image normalization functionality."""
    img_path = test_data_dir / 'train' / 'images' / 'test.jpg'
    img = preprocess(img_path)
    assert img.dtype == np.float32
    assert np.all(img >= 0) and np.all(img <= 1)


def test_label_reading(test_data_dir):
    """Test label reading functionality."""
    images, labels = load_dataset(test_data_dir, split='train')
    assert len(labels) == 1
    assert labels[0] == 1


def test_dataset_creation(test_data_dir):
    """Test dataset creation functionality."""
    images, labels = load_dataset(test_data_dir, split='train')
    assert isinstance(images, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert len(images) == len(labels)
    assert images.shape[1:] == (224, 224)
