"""Model creation and training functions."""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
import cv2
import numpy as np

class DefectDetectionModel(nn.Module):
    def __init__(self, num_classes=3):
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
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def load_model(model_path):
    """Load the trained model."""
    with open('data/data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    model = DefectDetectionModel(num_classes=data_config['nc'])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model input."""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.transpose((2, 0, 1))
    img = torch.FloatTensor(img).unsqueeze(0) / 255.0
    return img

def predict(model, image_path):
    """Make prediction on an image."""
    img = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    with open('data/data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config['names']
    return {
        'class': class_names[predicted_class],
        'confidence': round(confidence * 100, 2),
        'all_probabilities': {
            name: round(prob.item() * 100, 2)
            for name, prob in zip(class_names, probabilities[0])
        }
    }
