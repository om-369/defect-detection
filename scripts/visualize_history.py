import matplotlib.pyplot as plt
import json
import os
import seaborn as sns
import numpy as np

def plot_training_history():
    # Set style
    plt.style.use('seaborn-v0_8')  # Updated style name
    sns.set_theme()  # Use seaborn's default theme
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Sample training history (replace with actual data loading)
    history = {
        'accuracy': [0.65, 0.75, 0.82, 0.86, 0.89, 0.91, 0.92, 0.93],
        'val_accuracy': [0.63, 0.72, 0.79, 0.83, 0.85, 0.86, 0.87, 0.87],
        'loss': [0.98, 0.72, 0.54, 0.42, 0.35, 0.31, 0.28, 0.26],
        'val_loss': [1.02, 0.78, 0.62, 0.51, 0.45, 0.42, 0.40, 0.39]
    }
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    # Plot accuracy
    ax1.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, history['val_accuracy'], 'r--', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Epochs', fontsize=14, pad=15)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    
    # Plot loss
    ax2.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Over Epochs', fontsize=14, pad=15)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_training_history()
