"""Training script for defect detection model."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .preprocessing import preprocess_image


class DefectDataset(Dataset):
    """Dataset class for defect detection."""

    def __init__(self, data_dir: str, transform=None):
        """Initialize dataset."""
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths = list(self.data_dir.glob("**/*.jpg"))
        self.labels = [
            1 if "defect" in str(path).lower() else 0 for path in self.image_paths
        ]

    def __len__(self):
        """Get dataset length."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get item from dataset."""
        image_path = self.image_paths[idx]
        image = preprocess_image(str(image_path))
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: str,
) -> nn.Module:
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float("inf")
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

        logging.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Accuracy: {accuracy:.2f}%"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()

    model.load_state_dict(best_model)
    return model


def save_model(model: nn.Module, save_path: str):
    """Save model to file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    train_dataset = DefectDataset("data/train")
    val_dataset = DefectDataset("data/val")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize and train model
    model = DefectDataset().to(device)
    model = train_model(model, train_loader, val_loader, num_epochs=10, device=device)

    # Save model
    save_model(model, "models/defect_detection_model.pth")
