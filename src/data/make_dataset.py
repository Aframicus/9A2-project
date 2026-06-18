# Packages for downloading and loading the PneumoniaMNIST dataset
import matplotlib
import matplotlib.pyplot as plt
import medmnist
import numpy as np
import torch.utils.data as data
from medmnist import INFO
from torchvision import transforms

# Constants
DATA_FLAG = "pneumoniamnist"
DEFAULT_BATCH_SIZE = 128
SEED = 42

# Helper functions to load the PneumoniaMNIST dataset as NumPy arrays or PyTorch DataLoaders
def _get_data_class():
    info = INFO[DATA_FLAG]
    data_class = getattr(medmnist, info["python_class"])
    return info, data_class

# Load the raw PneumoniaMNIST splits as NumPy arrays
def load_pneumonia_mnist_arrays(download=True):
    """Load the raw PneumoniaMNIST splits as NumPy arrays."""
    _, data_class = _get_data_class()

    train_dataset_raw = data_class(split="train", download=download)
    val_dataset_raw = data_class(split="val", download=download)
    test_dataset_raw = data_class(split="test", download=download)

    train_images = train_dataset_raw.imgs 
    train_labels = train_dataset_raw.labels

    val_images = val_dataset_raw.imgs 
    val_labels = val_dataset_raw.labels

    test_images = test_dataset_raw.imgs
    test_labels = test_dataset_raw.labels

    return train_images, train_labels, val_images, val_labels, test_images, test_labels

# Load the transformed PneumoniaMNIST splits as PyTorch DataLoaders
def load_pneumonia_mnist_loaders(batch_size=DEFAULT_BATCH_SIZE, download=True):
    """Load the transformed PneumoniaMNIST splits as PyTorch DataLoaders."""
    _, data_class = _get_data_class()

    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_dataset_transformed = data_class(split="train", download=download, transform=data_transform)
    val_dataset_transformed = data_class(split="val", download=download, transform=data_transform)
    test_dataset_transformed = data_class(split="test", download=download, transform=data_transform)

    train_loader = data.DataLoader(dataset=train_dataset_transformed, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset_transformed, batch_size=2 * batch_size, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset_transformed, batch_size=2 * batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def save_training_class_distribution_plot(train_labels, output_path="class_distribution_train.png"):
    """Save a class distribution plot for the training labels (0 = no pneumonia, 1 = pneumonia)."""
    labels = np.asarray(train_labels).squeeze()

    plt.figure(figsize=(8, 5))
    plt.title("Class Distribution (0 = no pneumonia, 1 = pneumonia)")
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.hist(
        labels, bins=np.arange(labels.min() - 0.5, labels.max() + 1.5, 1), edgecolor="black")

    classes, counts = np.unique(labels, return_counts=True)
    total = counts.sum()

    for c, count in zip(classes, counts):
        pct = 100 * count / total
        plt.text(c, count, f"{int(count)}\n", ha="center", va="bottom")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():

    print("---------------------- Import, preprocess and load data ----------------------")
    print("Loading the PneumoniaMNIST splits")

    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_pneumonia_mnist_arrays()
    print("Raw data loaded successfully!")

    train_loader, val_loader, test_loader = load_pneumonia_mnist_loaders(batch_size=DEFAULT_BATCH_SIZE)
    print("Transformed data loaded successfully!")

    save_training_class_distribution_plot(train_labels)

    return (train_images, train_labels, val_images, val_labels, test_images, test_labels, train_loader, val_loader,test_loader,)


if __name__ == "__main__":
    main()