"""Test script to verify the setup of our defect detection project."""

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def test_imports():
    """Test if all required packages are properly imported."""
    print("Testing imports:")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"OpenCV version: {cv2.__version__}")

    # Test TensorFlow GPU availability
    print("\nTensorFlow GPU available:", tf.config.list_physical_devices("GPU"))

    # Test basic TensorFlow operations
    print("\nTesting TensorFlow operations:")
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])
    print("Matrix multiplication result:")
    print(tf.matmul(a, b))


def test_directory_structure():
    """Test if the project directory structure is properly set up."""
    print("\nChecking directory structure:")
    required_dirs = [
        "data/labeled/defect",
        "data/labeled/no_defect",
        "src/data",
        "src/models",
        "src/utils",
    ]

    for dir_path in required_dirs:
        path = Path(dir_path)
        exists = path.exists()
        print(f"{dir_path}: {'✓' if exists else '✗'}")


if __name__ == "__main__":
    print("Running setup tests...\n")
    test_imports()
    test_directory_structure()
