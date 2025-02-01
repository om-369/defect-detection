"""Tests for preprocessing functionality."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from defect_detection.preprocessing import load_dataset, preprocess


@pytest.mark.unit
@pytest.fixture
def test_data_dir():
    """Create temporary test data directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        (data_dir / "train" / "images").mkdir(parents=True)
        (data_dir / "train" / "labels").mkdir(parents=True)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(str(data_dir / "train" / "images" / "test.jpg"), img)
        with open(data_dir / "train" / "labels" / "test.txt", "w") as f:
            f.write("1")
        yield data_dir


def test_load_dataset(test_data_dir):
    """Test dataset loading functionality."""
    # Create test images
    image_size = (224, 224)
    num_images = 5

    # Create some test images
    for i in range(num_images):
        img = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        cv2.imwrite(str(test_data_dir / f"image_{i}.jpg"), img)


def test_invalid_image_path():
    """Test handling of invalid image path."""
    with pytest.raises(ValueError):
        preprocess("nonexistent_image.jpg")


def test_invalid_dataset_path():
    """Test handling of invalid dataset path."""
    with pytest.raises(ValueError):
        load_dataset("nonexistent_dir", split="train")
