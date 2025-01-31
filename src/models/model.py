"""Model creation and training functions."""

import torch
import torch.nn as nn
import yaml
import cv2
import numpy as np
from pathlib import Path

class DefectDetectionModel(nn.Module):
    """DefectDetectionModel class for welding defect detection."""

    def __init__(self, num_classes=3):
        """Initialize the model."""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Forward pass of the model."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_model(model_path):
    """Load the trained model."""
    with open('data/data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)

    model = DefectDetectionModel(num_classes=data_config['nc'])
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model input."""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img


def predict(model, image_path):
    """Make prediction on an image."""
    img = preprocess_image(image_path)
    with torch.no_grad():
        output = model(img)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item() * 100
        all_probs = {i: p.item() * 100 for i, p in enumerate(probs)}
    return {
        'class': pred_class,
        'confidence': confidence,
        'all_probabilities': all_probs
    }
