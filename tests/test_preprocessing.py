"""Tests for preprocessing functions."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.preprocessing import preprocess
from src.config import IMG_SIZE
from src.preprocessing import create_dataset
from src.preprocessing import read_yolo_label


@pytest.fixture
def test_data_dir():
    """Create temporary test data directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        defect_dir = tmp_path / "defect"
        no_defect_dir = tmp_path / "no_defect"
        label_dir = tmp_path / "label"

        defect_dir.mkdir()
        no_defect_dir.mkdir()
        label_dir.mkdir()

        for i in range(2):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_path = defect_dir / f"defect_{i}.jpg"
            tf.keras.preprocessing.image.save_img(str(img_path), img)

            label_path = label_dir / f"defect_{i}.txt"
            with open(label_path, "w") as f:
                f.write("0 0.5 0.5 0.2 0.2")

            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_path = no_defect_dir / f"no_defect_{i}.jpg"
            tf.keras.preprocessing.image.save_img(str(img_path), img)

        yield tmp_path


def test_image_loading(test_data_dir):
    """Test image loading function."""
    image_path = str(test_data_dir / "defect" / "defect_0.jpg")
    image = preprocess(image_path)

    assert isinstance(image, tf.Tensor)
    assert image.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert tf.reduce_min(image) >= 0.0
    assert tf.reduce_max(image) <= 1.0


def test_image_resizing(test_data_dir):
    """Test image resizing."""
    image_path = str(test_data_dir / "defect" / "defect_0.jpg")
    image = preprocess(image_path)

    assert image.shape == (IMG_SIZE, IMG_SIZE, 3)


def test_image_normalization(test_data_dir):
    """Test image normalization."""
    image_path = str(test_data_dir / "defect" / "defect_0.jpg")
    image = preprocess(image_path)

    assert tf.reduce_min(image) >= 0.0
    assert tf.reduce_max(image) <= 1.0


def test_label_reading(test_data_dir):
    """Test YOLO label reading."""
    label_path = str(test_data_dir / "label" / "defect_0.txt")
    bbox = read_yolo_label(label_path)

    assert len(bbox) == 4
    assert all(0 <= x <= 1 for x in bbox)


def test_dataset_creation(test_data_dir):
    """Test dataset creation."""
    dataset = create_dataset(test_data_dir, batch_size=2)

    assert isinstance(dataset, tf.data.Dataset)

    images, labels = next(iter(dataset))
    assert images.shape == (2, IMG_SIZE, IMG_SIZE, 3)
    assert labels.shape == (2,)
