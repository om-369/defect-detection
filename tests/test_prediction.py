"""Tests for prediction functionality."""

import pytest
import numpy as np
import torch
import cv2
from src.models.model import (
    DefectDetectionModel,
    preprocess_image,
    predict
)


@pytest.fixture
def test_model():
    """Create a test model."""
    model = DefectDetectionModel(num_classes=3)
    model.eval()
    return model


@pytest.fixture
def test_image_path(tmp_path):
    """Create a test image."""
    image_path = tmp_path / "test_image.jpg"
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), img)
    return image_path


def test_preprocess_image(test_image_path):
    """Test image preprocessing function."""
    img = preprocess_image(test_image_path)
    assert isinstance(img, torch.Tensor)
    assert img.shape == (1, 3, 224, 224)
    assert img.min() >= 0 and img.max() <= 1


def test_predict(test_model, test_image_path):
    """Test prediction function."""
    result = predict(test_model, test_image_path)
    assert isinstance(result, dict)
    assert 'class' in result
    assert 'confidence' in result
    assert 'all_probabilities' in result
    assert isinstance(result['class'], int)
    assert isinstance(result['confidence'], float)
    assert isinstance(result['all_probabilities'], dict)


def test_model_load(test_model):
    """Test model loading."""
    assert isinstance(test_model, DefectDetectionModel)
    test_input = torch.randn(1, 3, 224, 224)
    output = test_model(test_input)
    assert output.shape == (1, 3)
    assert not torch.isnan(output).any()


def test_model_prediction_range(test_model, test_image_path):
    """Test model prediction range."""
    result = predict(test_model, test_image_path)
    assert 0 <= result['class'] <= 2
    assert 0 <= result['confidence'] <= 100
    for prob in result['all_probabilities'].values():
        assert 0 <= prob <= 100


def test_model_output_format(test_model, test_image_path):
    """Test model output format."""
    result = predict(test_model, test_image_path)
    expected = {'class', 'confidence', 'all_probabilities'}
    assert set(result.keys()) == expected
    assert len(result['all_probabilities']) == 3
    assert all(isinstance(k, int) for k in result['all_probabilities'])
    probs = result['all_probabilities'].values()
    assert all(isinstance(v, float) for v in probs)
