"""Module for making predictions using trained model."""

# Standard library imports
import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Union

# Third-party imports
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# Local imports
from .models import DefectDetectionModel
from .config import Config

logger = logging.getLogger(__name__)
config = Config()


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
        return d


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


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model input.

    Args:
        image: PIL Image to preprocess

    Returns:
        Preprocessed image array
    """
    # Resize image to model's expected size
    image = image.resize(config.image_size)
    
    # Convert to array and add batch dimension
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Keep as uint8 - model's Rescaling layer will handle normalization
    return img_array.astype(np.uint8)


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
    return d


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
    return d


def predict_single_image(
    image_path: Union[str, Path]
) -> PredictionDict:
    """Make prediction on a single image.

    Args:
        image_path: Path to image file

    Returns:
        Dictionary containing prediction results
    """
    image_path = Path(image_path)
    try:
        # Load and preprocess image
        image = load_image(image_path)
        if image is None:
            return create_error_prediction(
                image_path.name, f"Failed to load image: {image_path}"
            )

        # Load model
        try:
            model = DefectDetectionModel.load_from_checkpoint(config.model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return create_error_prediction(image_path.name, f"Failed to load model: {str(e)}")

        # Preprocess and predict
        inputs = preprocess_image(image)
        outputs = model(inputs)
        
        # Get prediction and confidence
        class_id = int(np.argmax(outputs[0]))
        confidence = float(outputs[0][class_id])

        return create_success_prediction(
            image_path.name,
            class_id,
            confidence
        )

    except Exception as e:
        logger.error(f"Failed to make prediction: {e}")
        return create_error_prediction(image_path.name, f"Failed to make prediction: {str(e)}")


def process_directory(
    directory: Union[str, Path], model
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
            inputs = preprocess_image(image)
            outputs = model(inputs)
            probabilities = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]))
            confidence = float(np.max(probabilities))
            predicted = int(np.argmax(probabilities))

            result = PredictionResult(
                filename=image_path.name,
                class_id=predicted,
                confidence=confidence,
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
        result = predict_single_image(input_path)
        print(json.dumps(result, indent=4))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=4)
    else:
        # Directory processing
        try:
            model = DefectDetectionModel.load_from_checkpoint(config.model_path)
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
