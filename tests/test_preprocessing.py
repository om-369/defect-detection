"""Tests for preprocessing functionality."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import defect_detection.preprocessing as preprocessing


@pytest.fixture
def test_data_dir():
    """Create temporary test data directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        (data_dir / "images").mkdir(parents=True)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(str(data_dir / "images" / "good_weld_1.jpg"), img)
        yield data_dir


@pytest.mark.unit
def test_load_dataset(test_data_dir):
    """Test dataset loading functionality."""
    images, labels = preprocessing.load_dataset(str(test_data_dir))
    assert len(images) == 1
    assert len(labels) == 1
    assert isinstance(images[0], np.ndarray)
    assert labels[0] == 0  # good weld


@pytest.mark.unit
def test_preprocess():
    """Test image preprocessing."""
    # Create a test image
    image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)

    # Test preprocessing
    result = preprocessing.preprocess(image)

    # Check output properties
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 224, 224)  # CHW format
    assert result.min() >= 0 and result.max() <= 1  # Normalized values
