"""Model evaluation script."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.data.preprocessing import create_dataset


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_model(model_path: str) -> torch.nn.Module:
    """Load trained model.

    Args:
        model_path: Path to model file

    Returns:
        Loaded model
    """
    try:
        model = torch.load(model_path)
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
) -> dict:
    """Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: DataLoader for test set

    Returns:
        Dictionary containing evaluation metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    true_labels = []
    predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.numpy())

    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Calculate binary predictions
    binary_predictions = (predictions > 0.5).astype(int)

    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(true_labels, binary_predictions)),
        "precision": float(precision_score(true_labels, binary_predictions)),
        "recall": float(recall_score(true_labels, binary_predictions)),
        "f1": float(f1_score(true_labels, binary_predictions)),
        "auc_roc": float(roc_auc_score(true_labels, predictions)),
    }

    return metrics


def save_metrics(metrics: dict, output_path: str) -> None:
    """Save evaluation metrics to file.

    Args:
        metrics: Dictionary containing evaluation metrics
        output_path: Path to save metrics
    """
    try:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Metrics saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving metrics: {str(e)}")
        raise


def main():
    """Evaluate model performance."""
    parser = argparse.ArgumentParser(
        description="Evaluate model performance on test set"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--test-dir",
        required=True,
        help="Directory containing test data",
    )
    parser.add_argument(
        "--output",
        default="evaluation_metrics.json",
        help="Path to save metrics",
    )
    args = parser.parse_args()

    setup_logging()

    try:
        # Load model and data
        model = load_model(args.model)
        test_dataset = create_dataset(
            args.test_dir,
            batch_size=32,
            augment=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
        )

        # Evaluate model
        metrics = evaluate_model(model, test_loader)
        save_metrics(metrics, args.output)
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
