"""Tests for preprocessing functionality."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.preprocessing import load_dataset, preprocess


@pytest.fixture
def test_data_dir():
    """Create temporary test data directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_dir = Path(tmp_dir)
        (data_dir / "train" / "images").mkdir(parents=True)
        (data_dir / "train" / "labels").mkdir(parents=True)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(str(data_dir / "train" / "images" / "test.jpg"), img)
        with open(data_dir / "train" / "labels" / "test.txt", "w") as f:
            f.write("1")
        yield data_dir


def test_image_loading(test_data_dir):
    """Test image loading functionality."""
    img_path = test_data_dir / "train" / "images" / "test.jpg"
    img = preprocess(img_path)
    assert isinstance(img, np.ndarray)
    assert len(img.shape) == 2


def test_image_resizing(test_data_dir):
    """Test image resizing functionality."""
    img_path = test_data_dir / "train" / "images" / "test.jpg"
    img = preprocess(img_path)
    assert img.shape == (224, 224)


def test_image_normalization(test_data_dir):
    """Test image normalization functionality."""
    img_path = test_data_dir / "train" / "images" / "test.jpg"
    img = preprocess(img_path)
    assert img.dtype == np.float32
    assert np.all(img >= 0) and np.all(img <= 1)


def test_dataset_loading(test_data_dir):
    """Test dataset loading functionality."""
    images, labels = load_dataset(test_data_dir, split="train")
    assert isinstance(images, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert len(images) == len(labels) == 1
    assert images.shape[1:] == (224, 224)
    assert labels[0] == 1


def test_invalid_image_path():
    """Test handling of invalid image path."""
    with pytest.raises(ValueError):
        preprocess("nonexistent.jpg")


def test_invalid_dataset_path():
    """Test handling of invalid dataset path."""
    with pytest.raises(ValueError):
        load_dataset("nonexistent_dir", split="train")
