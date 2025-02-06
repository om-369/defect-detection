"""Configuration settings for the defect detection application."""

import os
from pathlib import Path


class Config:
    """Configuration class for the application."""

    def __init__(self):
        """Initialize configuration with default values."""
        self.base_dir = Path(__file__).parent.parent.parent
        self.model_dir = self.base_dir / "models"
        self.upload_dir = self.base_dir / "uploads"
        
        # Create necessary directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Model settings
        self.model_path = self.model_dir / "best_model.h5"
        
        # Image settings for ResNet50
        self.image_size = (224, 224)  # ResNet50 standard input size
        self.channels = 3  # RGB images
        
        # Class mapping
        self.class_names = {
            0: "Good Weld",
            1: "Defective Weld"
        }
