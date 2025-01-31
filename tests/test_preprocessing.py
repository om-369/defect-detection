"""Test suite for preprocessing functions."""

import sys
from pathlib import Path
import pytest
import tensorflow as tf

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

from src.config import IMG_SIZE
from src.preprocessing import preprocess_image, create_dataset, read_yolo_label


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create test data directory with sample images."""
    test_dir = tmp_path_factory.mktemp("test_data")
    
    # Create subdirectories
    defect_dir = test_dir / "defect"
    no_defect_dir = test_dir / "no_defect"
    label_dir = test_dir / "label"
    
    defect_dir.mkdir()
    no_defect_dir.mkdir()
    label_dir.mkdir()
    
    # Create test images using TensorFlow
    img = tf.random.uniform((IMG_SIZE, IMG_SIZE, 3), maxval=255, dtype=tf.int32)
    img = tf.cast(img, tf.uint8)
    
    # Save images and labels
    for i in range(2):
        # Save defect image and label
        img_path = defect_dir / f"defect_{i}.jpg"
        label_path = label_dir / f"defect_{i}.txt"
        
        tf.io.write_file(str(img_path), tf.io.encode_jpeg(img))
        with open(label_path, 'w') as f:
            f.write("1 0.5 0.5 0.3 0.4\n")  # YOLO format label
        
        # Save no_defect image
        img_path = no_defect_dir / f"no_defect_{i}.jpg"
        tf.io.write_file(str(img_path), tf.io.encode_jpeg(img))
    
    return test_dir


@pytest.fixture
def test_image_path(test_data_dir):
    """Get path to a test image."""
    return str(test_data_dir / "defect" / "defect_0.jpg")


@pytest.fixture
def test_label_path(test_data_dir):
    """Get path to a test label."""
    return str(test_data_dir / "label" / "defect_0.txt")


def test_image_loading(test_image_path):
    """Test image loading functionality."""
    image = preprocess_image(test_image_path)
    assert isinstance(image, tf.Tensor)
    assert image.shape == (IMG_SIZE, IMG_SIZE, 3)


def test_image_resizing(test_image_path):
    """Test image resizing functionality."""
    image = preprocess_image(test_image_path)
    assert image.shape == (IMG_SIZE, IMG_SIZE, 3)


def test_image_normalization(test_image_path):
    """Test image normalization functionality."""
    image = preprocess_image(test_image_path)
    assert tf.reduce_min(image) >= 0.0
    assert tf.reduce_max(image) <= 1.0


def test_label_reading(test_label_path):
    """Test YOLO label reading functionality."""
    bbox = read_yolo_label(test_label_path)
    assert len(bbox) == 4
    assert all(0 <= x <= 1 for x in bbox)


def test_dataset_creation(test_data_dir):
    """Test dataset creation functionality."""
    # Create dataset
    batch_size = 2
    dataset = create_dataset(test_data_dir, batch_size=batch_size)
    
    # Check if it's a tf.data.Dataset
    assert isinstance(dataset, tf.data.Dataset)
    
    # Check batch shape
    for images, labels in dataset.take(1):
        assert images.shape[0] == batch_size  # Batch size
        assert images.shape[1:] == (IMG_SIZE, IMG_SIZE, 3)  # Image dimensions
        assert labels.shape == (batch_size,)  # Label shape
