"""Deep learning model for defect detection."""

from typing import TypeVar

import cv2
import torch
import torch.nn as nn
import torchvision.models as models

Model = TypeVar("Model", bound=nn.Module)


class DefectDetectionModel(nn.Module):
    """CNN model for weld defect detection."""

    def __init__(self):
        """Initialize model architecture."""
        super().__init__()
        # Use ResNet18 backbone pretrained on ImageNet
        self.backbone = models.resnet18(pretrained=True)
        # Replace final layer for binary classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),  # Binary classification
            nn.Sigmoid(),  # Output probability between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Output probabilities
        """
        return self.backbone(x)

    @staticmethod
    def load_from_checkpoint(path: str) -> nn.Module:
        """Load model from checkpoint file.

        Args:
            path: Path to checkpoint file

        Returns:
            Model loaded from checkpoint
        """
        model = DefectDetectionModel()
        model.load_state_dict(torch.load(path))
        return model


def load_model(model_path):
    """Load model from path."""
    model = DefectDetectionModel()
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model input."""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img


def predict(model: nn.Module, image_path):
    """Make prediction on an image."""
    img = preprocess_image(image_path)
    with torch.no_grad():
        output = model(img)
        prob = output[0][0].item() * 100
        pred_class = 1 if prob > 50 else 0
        all_probs = {0: 100 - prob, 1: prob}

    return {
        "class": pred_class,
        "confidence": prob,
        "all_probabilities": all_probs,
    }
