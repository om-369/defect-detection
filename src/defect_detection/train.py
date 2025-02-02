"""Script for training the defect detection model."""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
    save_dir: Path,
) -> Dict[str, List[float]]:
    """Train model and return training history.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on
        save_dir: Directory to save model checkpoints

    Returns:
        Dictionary containing training history with metrics
    """
    history: Dict[str, List[float]] = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    best_model_path = save_dir / "best_model.pth"

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {epoch_loss:.4f} - "
            f"Acc: {epoch_acc:.2f}% - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val Acc: {val_acc:.2f}%"
        )

    return history


def main() -> None:
    """Run model training."""
    parser = argparse.ArgumentParser(description="Train defect detection model")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--save-dir", type=str, default="models", help="Directory to save models"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    train_dataset = DefectDataset(args.data_dir, train=True)
    val_dataset = DefectDataset(args.data_dir, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Create model
    model = DefectDetectionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=save_dir,
    )

    # Save training history
    history_path = save_dir / f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_path, "w") as f:
        import json
        json.dump(history, f, indent=4)
    logger.info(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
