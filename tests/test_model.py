"""Tests for defect detection model."""

import torch
from src.models.model import DefectDetectionModel


def test_model_initialization():
    """Test model initialization."""
    model = DefectDetectionModel()
    assert isinstance(model, DefectDetectionModel)


def test_model_forward():
    """Test model forward pass."""
    model = DefectDetectionModel()
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    output = model(input_tensor)
    assert output.shape == (batch_size, 1)


def test_model_checkpoint(tmp_path):
    """Test model checkpoint save and load."""
    model = DefectDetectionModel()
    checkpoint_path = tmp_path / "model.pth"
    # Save model
    torch.save(model.state_dict(), checkpoint_path)
    # Load model
    loaded_model = DefectDetectionModel.load_from_checkpoint(checkpoint_path)
    assert isinstance(loaded_model, DefectDetectionModel)
    # Compare parameters
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(p1, p2)
