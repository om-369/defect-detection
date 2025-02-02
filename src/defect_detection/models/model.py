"""Model definition for defect detection."""

# Standard library imports
from pathlib import Path
from typing import Union

# Third-party imports
import torch
import torch.nn as nn
from torchvision.models import resnet50


class DefectDetectionModel(nn.Module):
    """Neural network model for defect detection."""

    def __init__(self, num_classes: int = 2) -> None:
        """Initialize the model.

        Args:
            num_classes: Number of output classes
        """
        super().__init__()
        self.model = resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        output: torch.Tensor = self.model(x)
        return output

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: Union[str, Path]
    ) -> "DefectDetectionModel":
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded model
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path)
        model = cls()
        model.load_state_dict(state_dict)
        return model

    def save_checkpoint(self, save_path: Union[str, Path]) -> None:
        """Save model checkpoint.

        Args:
            save_path: Path to save checkpoint
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path)
