import pytest
import torch
import os
import numpy as np
import cv2
from pathlib import Path
from src.models.model import DefectDetectionModel, predict, preprocess_image, load_model

@pytest.fixture
def model():
    """Create a test model."""
    return DefectDetectionModel(num_classes=3)

@pytest.fixture
def test_image(tmp_path):
    """Create a test image."""
    image_path = tmp_path / "test_image.jpg"
    # Create a random color image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), img)
    return image_path

def test_preprocess_image(test_image):
    """Test image preprocessing."""
    processed = preprocess_image(test_image)
    
    # Check output shape and type
    assert isinstance(processed, torch.Tensor)
    assert processed.shape == (1, 3, 224, 224)
    assert processed.dtype == torch.float32
    
    # Check normalization
    assert torch.all(processed >= 0)
    assert torch.all(processed <= 1)

def test_predict(model, test_image):
    """Test prediction functionality."""
    result = predict(model, test_image)
    
    # Check result structure
    assert isinstance(result, dict)
    assert 'class' in result
    assert 'confidence' in result
    assert 'all_probabilities' in result
    
    # Check confidence value
    assert isinstance(result['confidence'], float)
    assert 0 <= result['confidence'] <= 100
    
    # Check probabilities
    assert isinstance(result['all_probabilities'], dict)
    assert len(result['all_probabilities']) == 3  # Number of classes
    
    # Check that probabilities sum to approximately 100
    total_prob = sum(result['all_probabilities'].values())
    assert 99.9 <= total_prob <= 100.1

def test_model_load(tmp_path):
    """Test model loading."""
    # Create a dummy model file
    model = DefectDetectionModel(num_classes=3)
    model.eval()  # Set to eval mode before saving
    model_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_path)
    
    # Create a dummy data.yaml file
    yaml_content = """
    nc: 3
    names: ['Bad Weld', 'Good Weld', 'Defect']
    """
    yaml_path = tmp_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    # Test loading
    loaded_model = load_model(model_path)
    assert isinstance(loaded_model, DefectDetectionModel)
    assert loaded_model.training == False  # Should be in eval mode
