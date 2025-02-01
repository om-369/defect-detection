"""Script to visualize training history."""

import argparse
import json
import matplotlib.pyplot as plt

def plot_training_history(history_file: str, output_file: str) -> None:
    """Plot training history from JSON file.

    Args:
        history_file: Path to training history JSON file
        output_file: Path to save plot
    """
    # Load history data
    with open(history_file) as f:
        history = json.load(f)

    # Create figure
    plt.figure(figsize=(12, 4))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def main():
    """Visualize training history."""
    parser = argparse.ArgumentParser(description="Visualize model training history")
    parser.add_argument("--history", required=True, help="Path to training history JSON file")
    parser.add_argument("--output", default="training_history.png", help="Path to save plot")
    args = parser.parse_args()

    plot_training_history(args.history, args.output)


if __name__ == "__main__":
    main()
