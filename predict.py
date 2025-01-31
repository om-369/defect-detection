"""Script for making predictions with trained model."""

import torch
from pathlib import Path
import yaml
from prometheus_client import Counter, start_http_server

from src.models.model import DefectDetectionModel
from src.preprocessing import preprocess
from src.utils.notifications import setup_logging, log_prediction

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'defect_predictions_total',
    'Total number of defect predictions made',
    ['predicted_class']
)

def load_config():
    """Load configuration from yaml file."""
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_model(checkpoint_path, num_classes):
    """Load trained model from checkpoint.

    Args:
        checkpoint_path (str): Path to model checkpoint
        num_classes (int): Number of classes

    Returns:
        DefectDetectionModel: Loaded model
    """
    model = DefectDetectionModel(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def predict_image(model, image_path, logger):
    """Make prediction for a single image.

    Args:
        model (DefectDetectionModel): Trained model
        image_path (str): Path to input image
        logger (logging.Logger): Logger instance

    Returns:
        dict: Prediction results
    """
    # Preprocess image
    image = preprocess(image_path)
    image_tensor = torch.tensor(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = int(torch.argmax(probabilities))
        confidence = float(probabilities[predicted_class]) * 100

    # Create result dictionary
    result = {
        'class': predicted_class,
        'confidence': confidence,
        'all_probabilities': {
            i: float(prob) * 100
            for i, prob in enumerate(probabilities)
        }
    }

    # Log prediction
    log_prediction(logger, image_path, result)

    # Update Prometheus metrics
    PREDICTION_COUNTER.labels(predicted_class=str(predicted_class)).inc()

    return result

def main():
    """Main prediction function."""
    # Load configuration
    config = load_config()

    # Setup logging
    logger = setup_logging(config['logging']['log_dir'])
    logger.info("Starting prediction service...")

    # Start Prometheus metrics server
    start_http_server(config['monitoring']['prometheus_port'])
    logger.info(
        f"Started Prometheus metrics server on port "
        f"{config['monitoring']['prometheus_port']}"
    )

    # Load model
    checkpoint_path = (
        Path(config['model']['checkpoint_dir']) / 'best_model.pth'
    )
    try:
        model = load_model(
            checkpoint_path,
            config['model']['num_classes']
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

    # Example prediction
    try:
        image_path = "data/test/test_image.jpg"
        result = predict_image(model, image_path, logger)
        print(f"Prediction result: {result}")
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
