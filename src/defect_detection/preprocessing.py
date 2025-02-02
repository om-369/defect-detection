"""Preprocessing utilities for defect detection."""

# Standard library imports
from pathlib import Path
from typing import Callable, List, Tuple, Union

# Third-party imports
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Define standard image transformations
transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess(image: np.ndarray) -> torch.Tensor:
    """Preprocess an image for model input.

    Args:
        image: Input image as numpy array in RGB format

    Returns:
        Preprocessed image tensor
    """
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image.astype("uint8"))

    # Apply transformations
    return transform(pil_image)


def load_dataset(data_dir: Union[str, Path]) -> Tuple[List[np.ndarray], List[int]]:
    """Load dataset from directory.

    Args:
        data_dir: Path to dataset directory

    Returns:
        Tuple of (images, labels) where images is a list of numpy arrays
        and labels is a list of integers (0 for good, 1 for defect)
    """
    images: List[np.ndarray] = []
    labels: List[int] = []

    image_dir = Path(data_dir) / "images"

    for img_path in image_dir.glob("*.jpg"):
        # Load image
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            image = np.array(img, dtype=np.uint8)
            images.append(image)

        # Determine label from filename
        # Assuming filenames start with 'good_' or 'defect_'
        label = 0 if img_path.stem.startswith("good_") else 1
        labels.append(label)

    return images, labels
