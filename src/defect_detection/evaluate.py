"""Module for evaluating the defect detection model."""
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from .models import DefectDetectionModel
from .data import DefectDataset

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


def evaluate_model(test_data_dir: str = 'data/test', model_path: str = 'models/model.pth') -> dict:
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
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )

        # Evaluate
        true_labels = []
        pred_labels = []
        pred_probs = []

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()

                true_labels.extend(labels.numpy())
                pred_labels.extend(preds.numpy())
                pred_probs.extend(probs.numpy())

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            pred_labels,
            average='binary'
        )
        accuracy = accuracy_score(true_labels, pred_labels)

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    evaluate_model()
