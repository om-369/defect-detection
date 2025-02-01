"""Script to visualize model training history."""

import logging
import matplotlib.pyplot as plt
import pandas as pd


def load_history(history_path: str) -> pd.DataFrame:
    """Load training history from CSV file."""
    try:
        history_path = pd.read_csv(history_path)
        logging.info(f"Loaded history from {history_path}")
        return history_path

    except Exception as e:
        logging.error(f"Error loading history: {str(e)}")
        raise


def plot_metrics(df: pd.DataFrame, output_path: str) -> None:
    """Plot training metrics."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss
        df.plot(
            x="epoch",
            y=["train_loss", "val_loss"],
            ax=ax1,
            title="Loss vs. Epoch",
        )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True)

        # Plot accuracy
        df.plot(
            x="epoch",
            y=["train_acc", "val_acc"],
            ax=ax2,
            title="Accuracy vs. Epoch",
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(output_path)
        logging.info(f"Saved plot to {output_path}")

    except Exception as e:
        logging.error(f"Error plotting metrics: {str(e)}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    history_path = "training_history.csv"
    output_path = "training_history.png"

    try:
        df = load_history(history_path)
        plot_metrics(df, output_path)
    except Exception as e:
        logging.error(f"Failed to visualize history: {str(e)}")
        raise
