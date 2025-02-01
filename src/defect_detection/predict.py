"""Module for making predictions using the trained model."""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .models import DefectDetectionModel

logger = logging.getLogger(__name__)

# Define image transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class PredictionResult:
    """Container for prediction results."""

    def __init__(self, filename: str, prediction: dict):
        """Initialize prediction result.

        Args:
            filename: Name of the image file
            prediction: Dictionary containing prediction details
        """
        self.filename = filename
        self.class_id = prediction["class"]
        self.confidence = prediction["confidence"]
        self.probabilities = prediction["all_probabilities"]


def load_image(image_path):
    """Load and preprocess an image.

    Args:
        image_path: Path to image file or file-like object.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    try:
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")

        return transform(image).unsqueeze(0)
    except Exception as e:
        logger.error(f"Failed to load image: {str(e)}")
        raise


def predict_image(image_path, model_path="models/model.pth"):
    """Make prediction for a single image.

    Args:
        image_path: Path to image file or file-like object.
        model_path (str): Path to the model file.

    Returns:
        dict: Prediction results including defect probability.
    """
    try:
        # Load and preprocess image
        image_tensor = load_image(image_path)

        # Load model
        model = DefectDetectionModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probability = torch.sigmoid(output).item()

        result = {
            "defect_probability": float(probability),
            "has_defect": probability >= 0.5,
        }

        logger.info(f"Prediction result: {result}")
        return result

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


def process_directory(model: DefectDetectionModel, directory: str) -> list:
    """Process all images in a directory.

    Args:
        model: Trained model
        directory: Directory containing images

    Returns:
        List of PredictionResult objects
    """

    results = []
    dir_path = Path(directory)
    image_files = list(dir_path.glob("*.jpg"))

    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            prediction = predict_image(str(img_path))
            result = PredictionResult(img_path.name, prediction)
            results.append(result)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return results


def save_results(results: list, output_path: str) -> None:
    """Save prediction results to JSON file.

    Args:
        results: List of PredictionResult objects
        output_path: Path to save results
    """

    output_data = []
    for result in results:
        output_data.append(
            {
                "filename": result.filename,
                "class": result.class_id,
                "confidence": result.confidence,
                "probabilities": result.probabilities,
            }
        )

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def main():
    """Run predictions on images."""
    parser = argparse.ArgumentParser(description="Run defect detection predictions")
    parser.add_argument(
        "--model",
        type=str,
        default="models/model.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input directory containing images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.json",
        help="Path to save prediction results",
    )
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model = DefectDetectionModel()
    model.load_state_dict(torch.load(args.model))
    model.eval()

    # Process images
    print("Processing images...")
    results = process_directory(model, args.input)

    # Save results
    print("Saving results...")
    save_results(results, args.output)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
