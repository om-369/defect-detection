"""Script for making predictions with trained model."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from defect_detection.models.model import DefectDetectionModel, predict


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
            prediction = predict(model, str(img_path))
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
    output = []
    for result in results:
        output.append({
            "filename": result.filename,
            "class": result.class_id,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def main():
    """Run predictions on images."""
    parser = argparse.ArgumentParser(description="Run defect detection predictions")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--input", required=True, help="Path to input directory or image")
    parser.add_argument("--output", help="Path to save results JSON")
    args = parser.parse_args()

    # Load model
    model = DefectDetectionModel()
    model.load_state_dict(torch.load(args.model))
    model.eval()

    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        # Single image
        try:
            result = predict(model, str(input_path))
            print(f"\nPrediction for {input_path.name}:")
            print(f"Class: {'Defect' if result['class'] == 1 else 'No defect'}")
            print(f"Confidence: {result['confidence']:.1f}%")
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
    else:
        # Directory of images
        results = process_directory(model, str(input_path))
        
        # Print summary
        print("\nPrediction Summary:")
        defect_count = sum(1 for r in results if r.class_id == 1)
        total = len(results)
        print(f"Total images processed: {total}")
        print(f"Defects detected: {defect_count}")
        print(f"No defects detected: {total - defect_count}")

        # Save results if output path provided
        if args.output:
            save_results(results, args.output)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
