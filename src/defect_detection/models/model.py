import torch
from torch import nn

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
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),  # Adjust based on input size
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.conv_layers(x)
        return self.classifier(x)

    @staticmethod
    def load_from_checkpoint(checkpoint_path: str) -> 'DefectDetectionModel':
        """Load model from saved checkpoint."""
        model = DefectDetectionModel()
        model.load_state_dict(torch.load(checkpoint_path))
        return model
