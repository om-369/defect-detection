"""Test suite for model creation and training."""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

import numpy as np
import tensorflow as tf
import pytest

from src.config import IMG_SIZE, MODEL_CONFIG
from src.models.model import create_model
from src.preprocessing import create_dataset


@pytest.fixture
def test_model():
    """Create a test model."""
    return create_model()


@pytest.fixture
def test_data_dir():
    """Get path to test data directory."""
    test_data_dir = Path(__file__).resolve().parent / "data" / "test_images"
    test_data_dir = os.path.abspath(test_data_dir).replace("\\", "/")
    return test_data_dir


def test_model_creation(test_model):
    """Test model creation."""
    # Check model architecture
    assert isinstance(test_model, tf.keras.Model)

    # Check input shape
    assert test_model.input_shape == (None, IMG_SIZE, IMG_SIZE, 3)

    # Check output shape 
    assert test_model.output_shape == (None, MODEL_CONFIG["num_classes"])


def test_model_compilation(test_model):
    """Test model compilation."""
    # Check if model is compiled
    assert test_model.optimizer is not None
    assert test_model.loss is not None
    assert any(isinstance(metric, tf.keras.metrics.BinaryAccuracy) for metric in test_model.metrics)


def test_model_prediction(test_model):
    """Test model prediction."""
    # Create random input
    batch_size = 4
    input_shape = (batch_size, IMG_SIZE, IMG_SIZE, 3)
    test_input = tf.random.uniform(input_shape)
    
    # Get predictions
    predictions = test_model.predict(test_input)
    
    # Check prediction shape and range
    assert predictions.shape == (batch_size, MODEL_CONFIG["num_classes"])
    assert np.all(predictions >= 0) and np.all(predictions <= 1)


@pytest.fixture
def test_dataset(test_model, test_data_dir):
    """Create small test dataset."""
    batch_size = 2
    return create_dataset(test_data_dir, batch_size=batch_size, shuffle=True)


def test_model_training(test_model, test_dataset):
    """Test model training."""
    # Train for one epoch
    history = test_model.fit(test_dataset, epochs=1, steps_per_epoch=1)
    
    # Check if training completed
    assert len(history.history["loss"]) == 1
