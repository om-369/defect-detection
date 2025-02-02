"""Model definition for defect detection."""

from pathlib import Path
from typing import Dict, Union

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

# Define standard transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class DefectDetectionModel(nn.Module):
    """PyTorch model for defect detection."""

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),  # Adjust based on input size
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.conv_layers(x)
        return self.classifier(x)

    @staticmethod
    def load_from_checkpoint(checkpoint_path: str) -> "DefectDetectionModel":
        """Load model from saved checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Loaded model

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint file is invalid
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            # Load on CPU explicitly for better security
            device = torch.device("cpu")
            model = DefectDetectionModel()
            state_dict = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,  # Only load model weights, not arbitrary objects
            )
            model.load_state_dict(state_dict)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}")

    def save_checkpoint(self, save_path: str) -> None:
        """Save model checkpoint.

        Args:
            save_path: Path where to save the checkpoint

        Raises:
            RuntimeError: If saving fails
        """
        try:
            # Ensure the directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            # Save only the state dict for security
            torch.save(self.state_dict(), save_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save checkpoint: {str(e)}")

    def predict(self, image_path: Union[str, Path]) -> Dict[str, float]:
        """Make prediction on an image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing prediction results

        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If prediction fails
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)

            # Make prediction
            self.eval()
            with torch.no_grad():
                output = self(image_tensor)
                probability = torch.sigmoid(output).item()

            # Format results
            return {
                "class": 1 if probability > 0.5 else 0,
                "confidence": probability if probability > 0.5 else 1 - probability,
                "defect_probability": probability,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to make prediction: {str(e)}")
