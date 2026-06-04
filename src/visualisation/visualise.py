# Script to visualise the made dataset
from pathlib import Path
import sys

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def visualise_samples(X, y, num_samples=1):
    """
    Visualises a few samples from the dataset.
    
    Parameters:
    - X: numpy array of images
    - y: numpy array of labels
    - num_samples: number of samples to visualise
    """
    plt.figure(figsize=(4, 4 * num_samples))
    
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(X[i].squeeze(), cmap='gray')
        plt.title(f'Label: {y[i]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from src.data.make_dataset import load_pneumoniamnist
    
    # Load the dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_pneumoniamnist()
    
    # Visualise the first sample from the training set
    visualise_samples(X_train, y_train, num_samples=1)