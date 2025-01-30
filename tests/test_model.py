"""Test suite for model functionality."""

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from src.config import IMG_SIZE
from src.models.model import create_model
from src.preprocessing import create_dataset


@pytest.fixture
def test_model():
    """Create a test model."""
    return create_model(num_classes=1)


@pytest.fixture
def test_data_dir():
    """Get path to test data directory."""
    return Path(__file__).resolve().parent / "data" / "test_images"


def test_model_creation(test_model):
    """Test model creation."""
    # Check model architecture
    assert isinstance(test_model, tf.keras.Model)
    
    # Check input shape
    assert test_model.input_shape == (None, IMG_SIZE, IMG_SIZE, 3)
    
    # Check output shape (binary classification)
    assert test_model.output_shape == (None, 1)


def test_model_compilation(test_model):
    """Test model compilation."""
    # Check if model is compiled
    assert test_model.optimizer is not None
    assert test_model.loss is not None
    assert "accuracy" in test_model.metrics_names


def test_model_prediction(test_model):
    """Test model prediction."""
    # Create random input
    batch_size = 4
    input_shape = (batch_size, IMG_SIZE, IMG_SIZE, 3)
    test_input = tf.random.uniform(input_shape)
    
    # Get predictions
    predictions = test_model.predict(test_input)
    
    # Check prediction shape and range
    assert predictions.shape == (batch_size, 1)
    assert np.all(predictions >= 0) and np.all(predictions <= 1)


def test_model_training(test_model, test_data_dir):
    """Test model training."""
    # Create small test dataset
    batch_size = 4
    test_dataset = create_dataset(
        str(test_data_dir),
        batch_size=batch_size,
        shuffle=True
    )
    
    # Train for one epoch
    history = test_model.fit(
        test_dataset,
        epochs=1,
        steps_per_epoch=2
    )
    
    # Check if training metrics are present
    assert "loss" in history.history
    assert "accuracy" in history.history
