"""Preprocessing utilities for defect detection."""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Define standard image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess(image: np.ndarray) -> torch.Tensor:
    """Preprocess an image for model input.

    Args:
        image: Input image as numpy array in RGB format

    Returns:
        Preprocessed image tensor
    """
    # Convert numpy array to PIL Image
    image = Image.fromarray(image)
    
    # Apply transformations
    return transform(image)


def load_dataset(data_dir: str) -> Tuple[List[np.ndarray], List[int]]:
    """Load dataset from directory.

    Args:
        data_dir: Path to dataset directory

    Returns:
        Tuple of (images, labels)
    """
    images = []
    labels = []
    
    image_dir = Path(data_dir) / "images"
    
    for img_path in image_dir.glob("*.jpg"):
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        images.append(image)
        
        # Determine label from filename
        # Assuming filenames start with 'good_' or 'defect_'
        label = 0 if img_path.stem.startswith('good_') else 1
        labels.append(label)
    
    return images, labels
