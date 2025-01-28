"""Visualization utilities for the defect detection project."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from typing import List, Tuple
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_history(history):
    """Plot training history metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot accuracy
    ax1.plot(history.history["accuracy"], label="Training Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    # Plot loss
    ax2.plot(history.history["loss"], label="Training Loss")
    ax2.plot(history.history["val_loss"], label="Validation Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("confusion_matrix.png")
    plt.close()


def visualize_predictions(
    images: List[np.ndarray],
    true_labels: List[int],
    pred_labels: List[int],
    num_samples: int = 10,
):
    """Visualize model predictions on sample images."""
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for idx in range(num_samples):
        axes[idx].imshow(images[idx])
        color = "green" if true_labels[idx] == pred_labels[idx] else "red"
        title = f"True: {true_labels[idx]}\nPred: {pred_labels[idx]}"
        axes[idx].set_title(title, color=color)
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig("prediction_samples.png")
    plt.close()


def plot_feature_maps(
    model: tf.keras.Model, image: np.ndarray, layer_name: str, num_features: int = 16
):
    """Visualize feature maps from a specific layer for an input image."""
    # Create a model that outputs the feature maps
    layer_outputs = [layer.output for layer in model.layers if layer.name == layer_name]
    visualization_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

    # Get feature maps
    feature_maps = visualization_model.predict(np.expand_dims(image, axis=0))

    # Plot feature maps
    if len(feature_maps[0].shape) == 4:
        feature_maps = np.squeeze(feature_maps[0])
        num_features = min(num_features, feature_maps.shape[-1])

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()

        for idx in range(num_features):
            axes[idx].imshow(feature_maps[:, :, idx], cmap="viridis")
            axes[idx].axis("off")

        plt.suptitle(f"Feature Maps from {layer_name}")
        plt.tight_layout()
        plt.savefig(f"feature_maps_{layer_name}.png")
        plt.close()
