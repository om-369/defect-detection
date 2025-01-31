"""Tests for model creation and training."""

import numpy as np
import pytest
import tensorflow as tf
from src.config import IMG_SIZE, MODEL_CONFIG
from src.models.model import create_model


@pytest.fixture
def test_model():
    """Create a test model."""
    return create_model()


@pytest.fixture
def test_data():
    """Create test data for model training."""
    num_samples = 4
    images = np.random.random((num_samples, IMG_SIZE, IMG_SIZE, 3))
    labels = np.random.randint(0, MODEL_CONFIG["num_classes"], (num_samples,))
    return images, labels


def test_model_creation(test_model):
    """Test model creation."""
    # Check model structure
    assert isinstance(test_model, tf.keras.Model)
    assert test_model.input_shape == (None, IMG_SIZE, IMG_SIZE, 3)
    assert test_model.output_shape == (None, MODEL_CONFIG["num_classes"])


def test_model_compilation(test_model):
    """Test model compilation."""
    # Check if model is compiled
    assert test_model.optimizer is not None, "Model optimizer not found"
    assert test_model.loss is not None, "Model loss not found"
    assert len(test_model.metrics) > 0, "Model metrics not found"


def test_model_prediction(test_model):
    """Test model prediction."""
    # Create sample input
    sample_input = np.random.random((1, IMG_SIZE, IMG_SIZE, 3))

    # Get prediction
    prediction = test_model.predict(sample_input)

    # Check prediction shape and values
    assert prediction.shape == (1, MODEL_CONFIG["num_classes"])
    assert np.all(prediction >= 0) and np.all(prediction <= 1)


def test_model_training(test_model, test_data):
    """Test model training."""
    images, labels = test_data

    # Train for one epoch
    history = test_model.fit(images, labels, epochs=1, verbose=0)

    # Check if training happened
    assert "loss" in history.history
    assert len(history.history["loss"]) == 1
