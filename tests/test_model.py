"""Tests for model creation and training."""

import pytest
import torch
from src.models.model import DefectDetectionModel, load_model, predict


def test_model_creation():
    """Test model creation."""
    model = DefectDetectionModel(num_classes=3)
    assert isinstance(model, DefectDetectionModel)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (1, 3)  # Batch size 1, 3 classes


def test_model_architecture():
    """Test model architecture."""
    model = DefectDetectionModel()
    
    # Test feature extractor
    assert len(model.features) == 9  # 3 conv blocks with 3 layers each
    assert isinstance(model.features[0], torch.nn.Conv2d)
    
    # Test classifier
    assert len(model.classifier) == 5  # Dropout, Linear, ReLU, Dropout, Linear
    assert isinstance(model.classifier[-1], torch.nn.Linear)


def test_model_output_range():
    """Test model output range."""
    model = DefectDetectionModel()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Test probability properties
        assert torch.all(probabilities >= 0)
        assert torch.all(probabilities <= 1)
        assert torch.allclose(torch.sum(probabilities, dim=1), torch.tensor([1.0]))
