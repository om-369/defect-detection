"""Test suite for the defect detection model."""

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from src.config import IMG_SIZE
from src.models.model import compile_model, create_model
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
    assert test_model.input_shape == (None, *IMG_SIZE, 3)

    # Check output shape (binary classification)
    assert test_model.output_shape == (None, 1)


def test_model_compilation(test_model):
    """Test model compilation."""
    compile_model(test_model)

    # Check if model is compiled
    assert test_model.optimizer is not None
    assert test_model.loss is not None


def test_model_prediction(test_model):
    """Test model prediction."""
    # Create random input
    test_input = tf.random.uniform((1, *IMG_SIZE, 3))

    # Get prediction
    prediction = test_model(test_input)

    # Check prediction shape and range
    assert prediction.shape == (1, 1)
    assert 0 <= prediction[0, 0] <= 1


def test_model_training(test_model, test_data_dir):
    """Test model training."""
    # Create test dataset
    dataset = create_dataset(test_data_dir, batch_size=1)

    # Compile model
    compile_model(test_model)

    # Train for one epoch
    history = test_model.fit(dataset, epochs=1)

    # Check if training happened
    assert "loss" in history.history
    assert len(history.history["loss"]) == 1
