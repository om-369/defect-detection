"""Tests for model functionality."""

import torch
from src.models.model import DefectDetectionModel


def test_model_creation():
    """Test model creation."""
    model = DefectDetectionModel(num_classes=3)
    assert isinstance(model, DefectDetectionModel)
    assert not model.training


def test_model_architecture():
    """Test model architecture."""
    model = DefectDetectionModel(num_classes=3)
    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    assert output.shape == (1, 3)


def test_model_output_range():
    """Test model output range."""
    model = DefectDetectionModel(num_classes=3)
    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    probs = torch.nn.functional.softmax(output, dim=1)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)
    assert torch.allclose(torch.sum(probs, dim=1), torch.tensor([1.0]))
