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
from src.data.preprocessing import create_dataset
from src.models.model import compile_model, create_model


def test_model_creation():
    """Test if model is created with correct architecture."""
    model = create_model()
    assert isinstance(model, tf.keras.Model)

    # Test input shape
    assert model.input_shape == (None, *IMG_SIZE, 3)

    # Test output shape (binary classification)
    assert model.output_shape == (None, 1)


def test_model_compilation():
    """Test if model compiles correctly."""
    model = create_model()
    compiled_model = compile_model(model)

    # Check if model is compiled
    assert isinstance(compiled_model.optimizer, tf.keras.optimizers.Optimizer)
    assert isinstance(compiled_model.loss, tf.keras.losses.Loss)

    # Check metrics
    metrics = [metric.__class__.__name__ for metric in compiled_model.metrics]
    assert any("Accuracy" in metric for metric in metrics)
    assert any("AUC" in metric for metric in metrics)


def test_model_prediction():
    """Test if model can make predictions on sample data."""
    model = create_model()
    model = compile_model(model)

    # Create sample input
    sample_input = np.random.random((1, *IMG_SIZE, 3)).astype(np.float32)

    # Make prediction
    prediction = model.predict(sample_input)

    # Check prediction shape and range
    assert prediction.shape == (1, 1)
    assert 0 <= prediction[0, 0] <= 1


def test_model_training():
    """Test if model can be trained on a small dataset."""
    # Get test data
    test_dir = Path("data/test")
    defect_paths = list(map(str, (test_dir / "defect").glob("*.jpg")))
    no_defect_paths = list(map(str, (test_dir / "no_defect").glob("*.jpg")))

    image_paths = defect_paths + no_defect_paths
    labels = [1] * len(defect_paths) + [0] * len(no_defect_paths)

    # Create dataset
    dataset = create_dataset(image_paths, labels, batch_size=2, augment=False)

    # Create and compile model
    model = create_model()
    model = compile_model(model)

    # Train for one epoch
    history = model.fit(dataset, epochs=1, verbose=0)

    # Check if training happened
    assert "loss" in history.history
    assert len(history.history["loss"]) == 1
