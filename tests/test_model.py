"""Tests for model module."""

import torch
from src.models.model import DefectDetectionModel


def test_model_creation():
    """Test model creation."""
    model = DefectDetectionModel(num_classes=3)
    model.eval()  # Set model to evaluation mode
    assert isinstance(model, DefectDetectionModel)
    assert not model.training


def test_model_output_shape():
    """Test model output shape."""
    model = DefectDetectionModel(num_classes=3)
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    output = model(input_tensor)
    assert output.shape == (batch_size, 3)


def test_model_output_range():
    """Test model output range."""
    model = DefectDetectionModel(num_classes=3)
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)
    assert torch.allclose(torch.sum(probabilities, dim=1), torch.ones(batch_size))
