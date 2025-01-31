"""Training script for defect detection model."""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import yaml

from src.models.model import DefectDetectionModel
from src.preprocessing import load_dataset, preprocess_batch
from src.utils.notifications import setup_logging, log_training_metrics
from src.utils.backup import create_backup


def load_config():
    """Load configuration from yaml file."""
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_model(config, logger):
    """Train the defect detection model.
    Args:
        config (dict): Configuration dictionary
        logger (logging.Logger): Logger instance
    """
    # Load datasets
    train_images, train_labels = load_dataset(config['data']['train_dir'])
    valid_images, valid_labels = load_dataset(config['data']['valid_dir'])

    # Create model
    model = DefectDetectionModel()
    criterion = nn.BCELoss()  # Binary cross entropy loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['model']['learning_rate']
    )

    # Training loop
    best_accuracy = 0.0
    for epoch in range(config['model']['epochs']):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Training
        for i in range(0, len(train_images), config['model']['batch_size']):
            batch_images = preprocess_batch(
                train_images[i:i + config['model']['batch_size']]
            )
            batch_labels = torch.tensor(
                train_labels[i:i + config['model']['batch_size']],
                dtype=torch.float32  # Binary labels as floats
            ).unsqueeze(1)  # Add dimension to match model output

            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()  # Binary threshold at 0.5
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        train_accuracy = 100.0 * correct / total

        # Validation
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            batch_size = config['model']['batch_size']
            for i in range(0, len(valid_images), batch_size):
                batch_images = preprocess_batch(
                    valid_images[i:i + batch_size]
                )
                batch_labels = torch.tensor(
                    valid_labels[i:i + batch_size],
                    dtype=torch.float32
                ).unsqueeze(1)

                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                valid_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        valid_accuracy = 100.0 * correct / total

        # Log metrics
        metrics = {
            'loss': train_loss,
            'accuracy': train_accuracy,
            'valid_loss': valid_loss,
            'valid_accuracy': valid_accuracy
        }
        log_training_metrics(logger, epoch, metrics)

        # Save best model
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            checkpoint_path = Path(config['model']['checkpoint_dir'])
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                model.state_dict(),
                checkpoint_path / 'best_model.pth'
            )
            logger.info(
                f"Saved best model with accuracy: {best_accuracy:.2f}%"
            )


def main():
    """Main training function."""
    # Load configuration
    config = load_config()

    # Setup logging
    logger = setup_logging(config['logging']['log_dir'])
    logger.info("Starting training...")

    # Create backup before training
    backup_dir = config['backup']['backup_dir']
    try:
        create_backup('models', backup_dir)
        logger.info("Created backup of existing models")
    except Exception as e:
        logger.error(f"Backup failed: {str(e)}")

    # Train model
    try:
        train_model(config, logger)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
