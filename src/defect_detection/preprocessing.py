"""Preprocessing utilities for defect detection."""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

__all__ = ["load_dataset", "preprocess", "preprocess_batch"]


def load_dataset(data_dir: str) -> Tuple[List[np.ndarray], List[int]]:
    """Load dataset from directory.
    Args:
        data_dir: Path to data directory containing images
    Returns:
        Tuple of (images, labels) where:
            images: List of image arrays
            labels: List of integer class labels
    """
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    images = []
    labels = []
    # Load all images from the images directory
    for img_path in images_dir.glob("*.jpg"):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        # Determine label from filename
        # 0 = good weld
        # 1 = bad weld/defect
        filename = img_path.name.lower()
        if "good" in filename:
            labels.append(0)
        else:
            labels.append(1)
    return images, labels


def preprocess_batch(images: List[np.ndarray]) -> torch.Tensor:
    """Preprocess a batch of images.
    Args:
        images: List of image arrays
    Returns:
        Preprocessed images as a torch tensor
    """
    # Convert to torch tensor
    batch = torch.stack([preprocess(img) for img in images])
    return batch


def preprocess(image: np.ndarray) -> torch.Tensor:
    """Preprocess a single image.
    Args:
        image: Input image array
    Returns:
        Preprocessed image as torch tensor
    """
    # Resize to standard size
    image = cv2.resize(image, (224, 224))
    # Scale pixel values to [0,1]
    image = image.astype(np.float32) / 255.0
    # Convert to torch tensor and add batch dimension
    tensor = torch.from_numpy(image)
    # Rearrange dimensions to [C,H,W]
    tensor = tensor.permute(2, 0, 1)
    return tensor
