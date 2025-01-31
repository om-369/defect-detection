"""Image preprocessing utilities."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms


def load_dataset(data_dir, image_size=(224, 224)):
    """Load dataset from directory.

    Args:
        data_dir (str): Directory containing the dataset
        image_size (tuple): Target image size (height, width)

    Returns:
        tuple: Arrays of images and labels
    """
    data_path = Path(data_dir)
    images_dir = data_path / 'images'
    
    images = []
    labels = []
    
    # Load all images from the images directory
    for img_path in images_dir.glob('*.jpg'):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, image_size)
        
        # Normalize pixel values to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        images.append(img)
        
        # Determine label from filename
        # 0 = good weld
        # 1 = bad weld/defect
        filename = img_path.name.lower()
        if 'good' in filename:
            labels.append(0)
        else:
            labels.append(1)
            
    if not images:
        raise ValueError(f"No valid images found in {data_dir}")

    # Convert to numpy arrays
    images = np.stack(images)
    labels = np.array(labels)

    return images, labels


def load_image(image_path, image_size=(224, 224)):
    """Load and preprocess a single image.

    Args:
        image_path (str): Path to image file
        image_size (tuple): Target image size (height, width)

    Returns:
        numpy.ndarray: Preprocessed image array
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return None

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image
    image = cv2.resize(image, image_size)

    # Normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0

    return image


def preprocess_batch(images, device=None):
    """Preprocess a batch of images for model input.

    Args:
        images (numpy.ndarray): Batch of images
        device (torch.device): Device to move tensors to

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to torch tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Apply transformations
    batch = torch.stack([transform(img) for img in images])

    return batch.to(device)
