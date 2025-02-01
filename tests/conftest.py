"""Test configuration and fixtures."""

import os
import shutil
import cv2
import numpy as np
import pytest

@pytest.fixture
def test_data_dir(tmp_path):
    """Create test data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def test_image_path(test_data_dir):
    """Create test image."""
    image_path = test_data_dir / "test.jpg"
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), img)
    return image_path


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up any files created during tests."""
    yield
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
