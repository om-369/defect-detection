"""Notifications and logging utilities."""

import logging
from pathlib import Path


def setup_logging(log_dir, level=logging.INFO):
    """Set up logging configuration.

    Args:
        log_dir (str): Directory to store log files
        level (int): Logging level

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("defect_detection")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Create formatters
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler for human-readable logs
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def log_training_metrics(logger, epoch, metrics):
    """Log training metrics.

    Args:
        logger (logging.Logger): Logger instance
        epoch (int): Current epoch number
        metrics (dict): Dictionary containing metrics
    """
    logger.info(
        f"Epoch {epoch}: "
        f"loss={metrics['loss']:.4f}, "
        f"accuracy={metrics['accuracy']:.2f}%, "
        f"valid_loss={metrics['valid_loss']:.4f}, "
        f"valid_accuracy={metrics['valid_accuracy']:.2f}%"
    )


def log_prediction(logger, image_path, result):
    """Log prediction results.

    Args:
        logger (logging.Logger): Logger instance
        image_path (str): Path to input image
        result (dict): Dictionary containing prediction results
    """
    logger.info(
        f"Prediction for {image_path}: "
        f"class={result['class']}, "
        f"confidence={result['confidence']:.2f}%"
    )
