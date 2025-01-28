"""Visualization utilities for the defect detection project."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import tensorflow as tf
from plotly.subplots import make_subplots
from sklearn.metrics import auc, confusion_matrix, roc_curve


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


def plot_roc_curve(y_true: np.ndarray, y_pred_prob: np.ndarray) -> go.Figure:
    """Plot ROC curve using plotly."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, name=f"ROC curve (AUC = {roc_auc:.2f})", mode="lines")
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], name="Random", mode="lines", line=dict(dash="dash")
        )
    )

    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=True,
    )

    return fig


def plot_defect_trend(history_data: List[Dict[str, Any]]) -> go.Figure:
    """Plot defect detection trend over time."""
    df = pd.DataFrame(history_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    # Calculate daily statistics
    daily_stats = (
        df.groupby("date").agg({"defect_detected": ["count", "sum"]}).reset_index()
    )

    daily_stats.columns = ["date", "total_predictions", "defects_found"]
    daily_stats["defect_rate"] = (
        daily_stats["defects_found"] / daily_stats["total_predictions"]
    ) * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=daily_stats["date"],
            y=daily_stats["total_predictions"],
            name="Total Predictions",
            marker_color="rgb(158,202,225)",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=daily_stats["date"],
            y=daily_stats["defect_rate"],
            name="Defect Rate (%)",
            line=dict(color="rgb(255,127,14)"),
            mode="lines+markers",
        ),
        secondary_y=True,
    )

    fig.update_layout(title="Daily Defect Detection Trend", showlegend=True)

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Total Predictions", secondary_y=False)
    fig.update_yaxes(title_text="Defect Rate (%)", secondary_y=True)

    return fig


def plot_performance_metrics(history_data: List[Dict[str, Any]]) -> go.Figure:
    """Plot system performance metrics."""
    df = pd.DataFrame(history_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.floor("H")

    # Calculate hourly statistics
    hourly_stats = (
        df.groupby("hour")
        .agg({"processing_time": ["mean", "max", "count"]})
        .reset_index()
    )

    hourly_stats.columns = ["hour", "avg_time", "max_time", "requests"]

    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("Processing Time", "Request Volume")
    )

    # Processing time plot
    fig.add_trace(
        go.Scatter(
            x=hourly_stats["hour"],
            y=hourly_stats["avg_time"],
            name="Avg Processing Time",
            line=dict(color="rgb(31,119,180)"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=hourly_stats["hour"],
            y=hourly_stats["max_time"],
            name="Max Processing Time",
            line=dict(color="rgb(255,127,14)", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Request volume plot
    fig.add_trace(
        go.Bar(
            x=hourly_stats["hour"],
            y=hourly_stats["requests"],
            name="Request Volume",
            marker_color="rgb(158,202,225)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(height=800, title="System Performance Metrics", showlegend=True)

    return fig


def plot_confidence_distribution(predictions: List[float]) -> go.Figure:
    """Plot distribution of prediction confidence scores."""
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=predictions,
            nbinsx=50,
            name="Confidence Distribution",
            marker_color="rgb(158,202,225)",
        )
    )

    fig.update_layout(
        title="Distribution of Prediction Confidence Scores",
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        showlegend=True,
        bargap=0.1,
    )

    return fig


def create_monitoring_dashboard(
    history_data: List[Dict[str, Any]]
) -> Dict[str, go.Figure]:
    """Create a complete monitoring dashboard."""
    df = pd.DataFrame(history_data)

    return {
        "defect_trend": plot_defect_trend(history_data),
        "performance": plot_performance_metrics(history_data),
        "confidence": plot_confidence_distribution(df["defect_probability"].tolist()),
        "roc": plot_roc_curve(
            df["defect_detected"].astype(int).values, df["defect_probability"].values
        ),
    }
