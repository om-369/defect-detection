"""Script for making predictions using the trained model."""

import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))
from config import IMG_SIZE
from src.data.preprocessing import load_image
from src.data.preprocessing import normalize_image
from src.data.preprocessing import resize_image
from src.utils.visualization import visualize_predictions


def load_model(model_path: str = "models/best_model.h5") -> tf.keras.Model:
    """Load the trained model."""
    return tf.keras.models.load_model(model_path)


def predict_single_image(model: tf.keras.Model, image_path: str) -> float:
    """Make prediction on a single image."""
    # Load and preprocess image
    img = load_image(image_path)
    img = resize_image(img)
    img = normalize_image(img)

    # Make prediction
    prediction = model.predict(np.expand_dims(img, axis=0))[0]

    return float(prediction)


def predict_batch(
    model: tf.keras.Model, image_paths: list, batch_size: int = 32
) -> np.ndarray:
    """Make predictions on a batch of images."""
    # Prepare images
    images = []
    for path in image_paths:
        img = load_image(path)
        img = resize_image(img)
        img = normalize_image(img)
        images.append(img)

    # Convert to numpy array
    images = np.array(images)

    # Make predictions
    predictions = model.predict(images, batch_size=batch_size)

    return predictions


def main():
    """Main prediction function."""
    # Load model
    model = load_model()

    # Directory containing images to predict
    predict_dir = Path("data/predict")

    if not predict_dir.exists():
        print(f"Directory {predict_dir} does not exist.")
        return

    # Get all image paths
    image_paths = list(predict_dir.glob("*.jpg"))

    if not image_paths:
        print(f"No images found in {predict_dir}")
        return

    # Make predictions
    predictions = predict_batch(model, [str(p) for p in image_paths])

    # Process and display results
    for path, pred in zip(image_paths, predictions):
        pred_class = "Defect" if pred > 0.5 else "No Defect"
        confidence = pred if pred > 0.5 else 1 - pred
        print(f"Image: {path.name}")
        print(f"Prediction: {pred_class}")
        print(f"Confidence: {confidence:.2%}\n")


if __name__ == "__main__":
    main()
