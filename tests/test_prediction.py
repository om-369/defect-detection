"""Tests for defect detection prediction."""

import cv2
import numpy as np
import pytest
import torch

from defect_detection.models.model import DefectDetectionModel
from defect_detection.preprocessing import preprocess


@pytest.fixture
def test_model():
    """Create a test model."""
    model = DefectDetectionModel()
    model.eval()
    return model


@pytest.fixture
def test_image_path(tmp_path):
    """Create a test image."""
    image_path = tmp_path / "test_image.jpg"
    # Create a random image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), img)
    return image_path


@pytest.mark.unit
def test_preprocess_image(test_image_path):
    """Test image preprocessing function."""
    # Load and preprocess image
    img = cv2.imread(str(test_image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed = preprocess(img)
    # Check output properties
    assert isinstance(processed, torch.Tensor)
    assert processed.shape == (3, 224, 224)
    assert processed.min() >= 0 and processed.max() <= 1


@pytest.mark.unit
def test_predict(test_model, test_image_path):
    """Test prediction function."""
    # Make prediction
    result = test_model.predict(str(test_image_path))
    # Check output properties
    assert isinstance(result, dict)
    assert "class" in result
    assert "confidence" in result
    assert "defect_probability" in result
    assert isinstance(result["class"], int)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["defect_probability"], float)
    assert 0 <= result["confidence"] <= 1
    assert 0 <= result["defect_probability"] <= 1


@pytest.mark.unit
def test_model_prediction_range(test_model, test_image_path):
    """Test that model predictions are in valid range."""
    # Make prediction
    result = test_model.predict(str(test_image_path))
    # Check prediction range
    assert result["defect_probability"] >= 0
    assert result["defect_probability"] <= 1
    assert result["confidence"] >= 0
    assert result["confidence"] <= 1
