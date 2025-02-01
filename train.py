"""Script for training the defect detection model."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from defect_detection.data import DefectDataset
from defect_detection.models.model import DefectDetectionModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
) -> dict:
    """Train model and return training history.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on

    Returns:
        Dictionary containing training history
    """
    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    best_model_path = "models/best_model.pth"

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Calculate training metrics
        epoch_loss = train_loss / len(train_loader)
        epoch_acc = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # Calculate validation metrics
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100.0 * val_correct / val_total

        # Update history
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)
        history["val_loss"].append(val_epoch_loss)
        history["val_accuracy"].append(val_epoch_acc)

        # Save best model
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), best_model_path)

        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {epoch_loss:.4f} - "
            f"Acc: {epoch_acc:.2f}% - "
            f"Val Loss: {val_epoch_loss:.4f} - "
            f"Val Acc: {val_epoch_acc:.2f}%"
        )

    return history


def main():
    """Run model training."""
    parser = argparse.ArgumentParser(description="Train defect detection model")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing training data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for optimizer",
    )
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create data loaders
    train_dataset = DefectDataset(
        Path(args.data_dir) / "train",
        transform=True,
    )
    val_dataset = DefectDataset(
        Path(args.data_dir) / "val",
        transform=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Initialize model and training components
    model = DefectDetectionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train model
    history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        args.epochs,
        device,
    )

    # Save training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = f"models/history_{timestamp}.json"
    with open(history_path, "w") as f:
        json.dump(history, f)
    logger.info(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
