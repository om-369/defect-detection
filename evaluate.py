"""Model evaluation script."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import MODEL_CONFIG
from src.data.preprocessing import create_dataset


def evaluate_model(
    model_path: str = "models/latest", test_data_dir: str = "data/labeled"
) -> dict:
    """Evaluate the model on test data.

    Args:
        model_path: Path to the saved model
        test_data_dir: Directory containing test data

    Returns:
        Dictionary containing evaluation metrics
    """
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)

    print("Loading test data...")
    test_dir = Path(test_data_dir)

    # Load defect images
    defect_images = list(map(str, (test_dir / "defect").glob("*.jpg")))
    defect_labels = [1] * len(defect_images)

    # Load no-defect images
    no_defect_images = list(map(str, (test_dir / "no_defect").glob("*.jpg")))
    no_defect_labels = [0] * len(no_defect_images)

    # Combine data
    all_images = defect_images + no_defect_images
    all_labels = defect_labels + no_defect_labels

    # Create dataset
    test_dataset = create_dataset(all_images, all_labels, batch_size=32, augment=False)

    print("Making predictions...")
    predictions = []
    true_labels = []

    for images, labels in test_dataset:
        batch_preds = model.predict(images, verbose=0)
        predictions.extend(batch_preds.flatten())
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

    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save metrics
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    evaluate_model()
