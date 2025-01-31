"""Visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Path = None,
    show: bool = True
) -> None:
    """Plot training history metrics.

    Args:
        history: Training history dictionary
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Training")
    if "val_accuracy" in history:
        plt.plot(history["val_accuracy"], label="Validation")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Training")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Validation")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_predictions(
    model: tf.keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    num_samples: int = 5,
    save_path: Path = None,
    show: bool = True
) -> None:
    """Plot model predictions on sample images.

    Args:
        model: Trained model
        images: Input images
        labels: True labels
        num_samples: Number of samples to plot
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    preds = model.predict(images[:num_samples])
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    for i, (img, pred, label) in enumerate(
        zip(images[:num_samples], preds, labels[:num_samples])
    ):
        axes[i].imshow(img)
        title = f"Pred: {pred[0]:.2f}\nTrue: {label}"
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray
) -> plt.Figure:
    """Plot ROC curve using matplotlib.

    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities

    Returns:
        Matplotlib figure object
    """
    fpr, tpr, _ = tf.metrics.roc_curve(y_true, y_pred_prob)
    roc_auc = tf.metrics.auc(fpr, tpr)

    fig, ax = plt.subplots()
    label = f"ROC curve (AUC = {roc_auc:.2f})"
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=label)
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    return fig


def plot_defect_trend(
    history_data: List[Dict[str, str]]
) -> plt.Figure:
    """Plot defect detection trend over time.

    Args:
        history_data: List of dictionaries containing historical data

    Returns:
        Matplotlib figure object
    """
    dates = [record["date"] for record in history_data]
    defects = [record["defects"] for record in history_data]

    fig, ax = plt.subplots()
    ax.plot(dates, defects)
    ax.set_title("Daily Defect Detection Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Defects")

    return fig


def plot_results(results):
    return plt.plot(results["loss"], label="loss")
