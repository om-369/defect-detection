"""Module for evaluating the defect detection model."""

# Standard library imports
import logging
from pathlib import Path
from typing import Dict, List

# Third-party imports
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from .data import DefectDataset
from .models import DefectDetectionModel

logger = logging.getLogger(__name__)


def load_model(model_path: str) -> DefectDetectionModel:
    """Load the trained model from disk.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        DefectDetectionModel: Loaded model.
    """
    try:
        model = DefectDetectionModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def evaluate_model(
    test_data_dir: str = "data/test", model_path: str = "models/model.pth"
) -> Dict[str, float]:
    """Evaluate model performance on test dataset.

    Args:
        test_data_dir (str): Directory containing test data.
        model_path (str): Path to the saved model file.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    try:
        # Load model
        model = load_model(model_path)

        # Prepare test data
        test_dataset = DefectDataset(Path(test_data_dir))
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=4
        )

        # Evaluate
        true_labels: List[int] = []
        pred_labels: List[int] = []
        pred_probs: List[float] = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()

                true_labels.extend(labels.numpy())
                pred_labels.extend(preds.numpy())
                pred_probs.extend(probs.numpy())

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="binary"
        )
        accuracy = accuracy_score(true_labels, pred_labels)

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


def save_metrics(metrics: Dict[str, float], output_path: str | Path) -> None:
    """Save evaluation metrics to file.

    Args:
        metrics: Dictionary containing evaluation metrics
        output_path: Path to save metrics
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w") as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate_model()
