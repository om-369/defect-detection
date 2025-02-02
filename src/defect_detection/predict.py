"""Module for making predictions using trained model."""

# Standard library imports
import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Union, cast

# Third-party imports
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

# Local imports
from .models import DefectDetectionModel

logger = logging.getLogger(__name__)


class PredictionDict(TypedDict, total=True):
    """Type definition for prediction results."""

    filename: str
    class_id: int
    confidence: float
    error: Optional[str]


@dataclass
class PredictionResult:
    """Class to hold prediction results."""

    filename: str
    class_id: int
    confidence: float

    def to_dict(self) -> Dict[str, Union[str, int, float]]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return asdict(self)

    def to_prediction_dict(self) -> PredictionDict:
        """Convert to PredictionDict.

        Returns:
            PredictionDict representation
        """
        d: Dict[str, Union[str, int, float, None]] = {
            "filename": self.filename,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "error": None,
        }
        return cast(PredictionDict, d)


def load_image(image_path: Union[str, Path]) -> Optional[Image.Image]:
    """Load an image from file.

    Args:
        image_path: Path to image file

    Returns:
        Loaded PIL Image or None if loading fails
    """
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        return None


def preprocess_image(image: Image.Image) -> Tensor:
    """Preprocess image for model input.

    Args:
        image: PIL Image to preprocess

    Returns:
        Preprocessed image tensor
    """
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor: Tensor = transform(image).unsqueeze(0)
    return tensor


def create_error_prediction(
    filename: str, error_message: str, class_id: int = -1, confidence: float = 0.0
) -> PredictionDict:
    """Create a PredictionDict for error cases.

    Args:
        filename: Name of the image file
        error_message: Error message to include
        class_id: Class ID to use (default: -1)
        confidence: Confidence score to use (default: 0.0)

    Returns:
        PredictionDict with error information
    """
    d: Dict[str, Union[str, int, float, None]] = {
        "filename": filename,
        "class_id": class_id,
        "confidence": confidence,
        "error": error_message,
    }
    return cast(PredictionDict, d)


def create_success_prediction(
    filename: str, class_id: int, confidence: float
) -> PredictionDict:
    """Create a PredictionDict for successful predictions.

    Args:
        filename: Name of the image file
        class_id: Predicted class ID
        confidence: Confidence score

    Returns:
        PredictionDict with prediction results
    """
    d: Dict[str, Union[str, int, float, None]] = {
        "filename": filename,
        "class_id": class_id,
        "confidence": confidence,
        "error": None,
    }
    return cast(PredictionDict, d)


def predict_single_image(
    image_path: Union[str, Path], model_path: str = "checkpoints/model.pth"
) -> PredictionDict:
    """Make prediction on a single image.

    Args:
        image_path: Path to image file
        model_path: Path to model checkpoint

    Returns:
        Dictionary containing prediction results
    """
    image_path = Path(image_path)
    image = load_image(image_path)
    if image is None:
        return create_error_prediction(
            image_path.name, f"Failed to load image: {image_path}"
        )

    # Load model
    try:
        model = DefectDetectionModel.load_from_checkpoint(model_path)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return create_error_prediction(image_path.name, "Failed to load model")

    # Preprocess and predict
    try:
        with torch.no_grad():
            inputs = preprocess_image(image)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

            return create_success_prediction(
                image_path.name,
                int(predicted.item()),
                float(confidence.item()),
            )

    except Exception as e:
        logger.error(f"Failed to make prediction: {e}")
        return create_error_prediction(image_path.name, "Failed to make prediction")


def process_directory(
    directory: Union[str, Path], model: DefectDetectionModel
) -> List[PredictionResult]:
    """Process all images in a directory.

    Args:
        directory: Directory containing images
        model: Loaded model to use for predictions

    Returns:
        List of prediction results
    """
    directory = Path(directory)
    results: List[PredictionResult] = []

    for image_path in directory.glob("*.jpg"):
        image = load_image(image_path)
        if image is None:
            continue

        # Preprocess and predict
        try:
            with torch.no_grad():
                inputs = preprocess_image(image)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, dim=1)

                result = PredictionResult(
                    filename=image_path.name,
                    class_id=int(predicted.item()),
                    confidence=float(confidence.item()),
                )
                results.append(result)

        except Exception as e:
            logger.error(f"Failed to make prediction: {e}")

    return results


def save_results(
    results: List[PredictionResult], output_path: Union[str, Path]
) -> None:
    """Save prediction results to a file.

    Args:
        results: List of PredictionResult objects
        output_path: Path to save results
    """
    output_data = [result.to_prediction_dict() for result in results]
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)


def main() -> None:
    """Run predictions on images."""
    parser = argparse.ArgumentParser(description="Run predictions on images")
    parser.add_argument("input", type=str, help="Path to image or directory")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--output", type=str, help="Path to save results (optional)")
    args = parser.parse_args()

    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        # Single image prediction
        result = predict_single_image(input_path, args.model)
        print(json.dumps(result, indent=4))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=4)
    else:
        # Directory processing
        try:
            model = DefectDetectionModel.load_from_checkpoint(args.model)
            model.eval()
            results = process_directory(input_path, model)
            if args.output:
                save_results(results, args.output)
            else:
                for result in results:
                    print(json.dumps(result.to_prediction_dict(), indent=4))
        except Exception as e:
            logger.error(f"Failed to process directory: {e}")


if __name__ == "__main__":
    main()
