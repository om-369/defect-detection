"""Module for making predictions using the trained model."""

# Standard library imports
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Union, cast

# Third-party imports
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm

# Local imports
from .models import DefectDetectionModel

logger = logging.getLogger(__name__)

# Define standard image transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class PredictionDict(TypedDict):
    """Type definition for prediction dictionary."""

    class_id: int
    confidence: float
    all_probabilities: List[float]


class ResultDict(TypedDict):
    """Type definition for result dictionary."""

    filename: str
    class_id: int
    confidence: float
    probabilities: List[float]


class PredictionResult:
    """Container for prediction results."""

    def __init__(self, filename: str, prediction: PredictionDict) -> None:
        """Initialize prediction result.

        Args:
            filename: Name of the image file
            prediction: Dictionary containing prediction details
        """
        self.filename = filename
        self.class_id = prediction["class_id"]
        self.confidence = prediction["confidence"]
        self.probabilities = prediction["all_probabilities"]


def load_image(image_path: Union[str, Path]) -> Tensor:
    """Load and preprocess an image.

    Args:
        image_path: Path to image file or file-like object.

    Returns:
        Preprocessed image tensor
    """
    image = Image.open(image_path).convert("RGB")
    tensor = cast(Tensor, transform(image))
    return tensor.unsqueeze(0)


def predict_image(
    image_path: Union[str, Path], model_path: str = "models/model.pth"
) -> PredictionDict:
    """Make prediction for a single image.

    Args:
        image_path: Path to image file or file-like object
        model_path: Path to the model file

    Returns:
        Dictionary containing prediction results
    """
    # Load model
    model = DefectDetectionModel.load_from_checkpoint(model_path)
    model.eval()

    # Load and preprocess image
    image_tensor = load_image(image_path)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_class = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class].item())

    return {
        "class_id": predicted_class,
        "confidence": confidence,
        "all_probabilities": cast(List[float], probabilities.tolist()),
    }


def process_directory(
    model: DefectDetectionModel, directory: Union[str, Path]
) -> List[PredictionResult]:
    """Process all images in a directory.

    Args:
        model: Trained model
        directory: Directory containing images

    Returns:
        List of PredictionResult objects
    """
    directory = Path(directory)
    results = []

    for image_path in tqdm(list(directory.glob("*.jpg"))):
        image_tensor = load_image(image_path)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            predicted_class = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[predicted_class].item())

        prediction: PredictionDict = {
            "class_id": predicted_class,
            "confidence": confidence,
            "all_probabilities": cast(List[float], probabilities.tolist()),
        }
        results.append(PredictionResult(image_path.name, prediction))

    return results


def save_results(
    results: List[PredictionResult], output_path: Union[str, Path]
) -> None:
    """Save prediction results to JSON file.

    Args:
        results: List of PredictionResult objects
        output_path: Path to save results
    """
    output_data: List[ResultDict] = []
    for result in results:
        output_data.append(
            {
                "filename": result.filename,
                "class_id": result.class_id,
                "confidence": result.confidence,
                "probabilities": result.probabilities,
            }
        )

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)


def main() -> None:
    """Run predictions on images."""
    parser = argparse.ArgumentParser(description="Run predictions on images")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input image or directory"
    )
    parser.add_argument("--output", type=str, help="Path to save results")
    args = parser.parse_args()

    # Load model
    model = DefectDetectionModel.load_from_checkpoint(args.model)
    model.eval()

    input_path = Path(args.input)
    if input_path.is_file():
        # Single image prediction
        result = predict_image(input_path, args.model)
        print(json.dumps(result, indent=4))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=4)
    else:
        # Directory processing
        results = process_directory(model, input_path)
        if args.output:
            save_results(results, args.output)
        else:
            for result in results:
                print(
                    f"{result.filename}: Class {result.class_id} "
                    f"(Confidence: {result.confidence*100:.2f}%)"
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
