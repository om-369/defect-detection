"""Module for evaluating model performance."""

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.defect_detection.models import DefectDetectionModel
from src.defect_detection.preprocessing import preprocess_image


def load_model(model_path: str) -> DefectDetectionModel:
    """Load the trained model from disk."""
    try:
        model = torch.load(model_path)
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise


def evaluate_model(
    model: DefectDetectionModel,
    test_dir: str,
    batch_size: int = 32
) -> Dict[str, float]:
    """Evaluate model performance on test set."""
    try:
        test_dir = Path(test_dir)
        image_paths = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        
        y_true = []
        y_pred = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            batch_labels = []
            
            for path in batch_paths:
                try:
                    image = preprocess_image(str(path))
                    label = int(path.parent.name)  # Assuming directory name is label
                    batch_images.append(image)
                    batch_labels.append(label)
                except Exception as e:
                    logging.error(f"Error processing {path}: {str(e)}")
                    continue
            
            if not batch_images:
                continue
                
            batch_tensor = torch.stack(batch_images)
            with torch.no_grad():
                outputs = model(batch_tensor)
                predictions = torch.argmax(outputs, dim=1)
            
            y_true.extend(batch_labels)
            y_pred.extend(predictions.tolist())
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1": f1_score(y_true, y_pred, average="weighted"),
        }
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error evaluating model: {str(e)}")
        raise


def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    """Save evaluation metrics to a file."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
            
        logging.info(f"Saved metrics to {output_path}")
        
    except Exception as e:
        logging.error(f"Error saving metrics: {str(e)}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_path = "models/latest/model.pth"
    test_dir = "data/test"
    output_path = "metrics/evaluation_results.json"
    
    try:
        model = load_model(model_path)
        metrics = evaluate_model(model, test_dir)
        save_metrics(metrics, output_path)
    except Exception as e:
        logging.error(f"Evaluation pipeline failed: {str(e)}")
