# Visualisation utilities for MedMNIST datasets
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
# For importing the dataset
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.data.make_dataset import load_pneumonia_mnist_arrays
# Show a grid of sample images from a MedMNIST dataset object
def show_samples(images, labels=None, num_samples=16):
    #Display a grid of sample images from NumPy arrays.
    if labels is None:
        # If no labels provided, just show images without titles
        for index in range(num_samples):
            plt.subplot(4, 4, index + 1)
            plt.imshow(images[index], cmap="gray")
            plt.axis("off")
    else:
        # Show images with labels
        for index in range(num_samples):
            plt.subplot(4, 4, index + 1)
            plt.imshow(images[index], cmap="gray")
            plt.title(f"Label: {labels[index][0] if labels[index].ndim > 0 else labels[index]}")
            plt.axis("off")
    plt.tight_layout()
    plt.show()

# Save a class distribution plot for the training labels
def save_training_class_distribution_plot(labels, split_name, output_path=None):
    """Save a class distribution plot for the labels of a given split."""
    labels = np.asarray(labels).squeeze()
    
    output_dir = PROJECT_ROOT / "src" / "visualisation"
    output_path = output_dir / f"class_distribution_{split_name}.png"

    plt.figure(figsize=(8, 5))
    plt.title(f"Class Distribution in the {split_name.capitalize()} Set")
    plt.xlabel("Class (0 = no pneumonia, 1 = pneumonia)")
    plt.ylabel("Frequency")

    n, bins, patches = plt.hist(labels, bins=2, edgecolor="black")

    for count, left_edge, right_edge in zip(n, bins[:-1], bins[1:]):
        center = (left_edge + right_edge) / 2.0
        plt.text(center, count, f"{int(count)}", ha="center", va="bottom")

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    #Load the PneumoniaMNIST splits
    print("---------------------- Import, preprocess and load data ----------------------")
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_pneumonia_mnist_arrays()
    #Visualise the PneumoniaMNIST splits
    print("Visualising the PneumoniaMNIST splits")
    show_samples(train_images, train_labels)
    show_samples(val_images, val_labels)
    show_samples(test_images, test_labels)
    #Save class distribution plot for the training labels
    print("Saving class distribution plots")
    save_training_class_distribution_plot(train_labels, split_name="train")
    save_training_class_distribution_plot(val_labels, split_name="val")
    save_training_class_distribution_plot(test_labels, split_name="test")

if __name__ == "__main__":
    main()
