"""Create a test model for deployment verification."""

import logging
import os
from pathlib import Path

import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_model():
    """Create and save a test model."""
    try:
        # Import here to ensure all dependencies are available
        from src.models.model import compile_model, create_model

        # Create model directory if it doesn't exist
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)

        # Create and compile model
        logger.info("Creating model...")
        model = create_model()

        logger.info("Compiling model...")
        model = compile_model(model)

        # Save model
        model_path = model_dir / "latest.h5"
        logger.info(f"Saving model to {model_path}...")
        model.save(str(model_path), save_format="h5")

        # Verify the saved model
        logger.info("Verifying saved model...")
        test_input = tf.random.normal([1, 224, 224, 3])
        prediction = model.predict(test_input)
        logger.info(f"Test prediction shape: {prediction.shape}")

        logger.info("Model creation successful!")
        return True

    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        return False


if __name__ == "__main__":
    success = create_test_model()
    exit(0 if success else 1)
