"""Configuration settings for the defect detection project."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "raw").mkdir(exist_ok=True)
(DATA_DIR / "processed").mkdir(exist_ok=True)
(DATA_DIR / "labeled").mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Data configuration
IMG_SIZE = 224  # Input image size (single integer)
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Model configuration
MODEL_CONFIG = {
    "backbone": "resnet50",
    "weights": "imagenet",
    "num_classes": 2,  # Binary classification: defect/no-defect
    "learning_rate": 1e-5,  # Reduced learning rate
    "epochs": 20,  # Fewer epochs for small dataset
}

# Training configuration
TRAIN_CONFIG = {
    "early_stopping_patience": 3,  # Reduced patience
    "reduce_lr_patience": 2,
    "min_lr": 1e-7,
}

# Augmentation configuration
AUG_CONFIG = {
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "horizontal_flip": True,
    "vertical_flip": True,
    "fill_mode": "nearest",
}
