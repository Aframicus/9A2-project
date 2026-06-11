"""Utilities for downloading and loading the PneumoniaMNIST dataset."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import medmnist
import numpy as np
import torch.utils.data as data
from medmnist import INFO
from torchvision import transforms


DATA_FLAG = "pneumoniamnist"
DEFAULT_BATCH_SIZE = 128


def _get_data_class():
    info = INFO[DATA_FLAG]
    data_class = getattr(medmnist, info["python_class"])
    return info, data_class


def load_pneumonia_mnist_arrays(download=True):
    """Load the raw PneumoniaMNIST splits as NumPy arrays."""
    _, data_class = _get_data_class()

    train_dataset_raw = data_class(split="train", download=download)
    val_dataset_raw = data_class(split="val", download=download)
    test_dataset_raw = data_class(split="test", download=download)

    train_images = train_dataset_raw.imgs / 255.0
    train_labels = train_dataset_raw.labels

    val_images = val_dataset_raw.imgs / 255.0
    val_labels = val_dataset_raw.labels

    test_images = test_dataset_raw.imgs / 255.0
    test_labels = test_dataset_raw.labels

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


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


def show_samples(dataset, num_samples=16):
    """Display a grid of sample images from a MedMNIST dataset object."""
    images = dataset.imgs
    labels = dataset.labels

    plt.figure(figsize=(8, 8))
    for index in range(num_samples):
        plt.subplot(4, 4, index + 1)
        plt.imshow(images[index], cmap="gray")
        plt.title(f"Label: {labels[index][0]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def save_training_class_distribution_plot(train_labels, output_path="class_distribution_train.png"):
    """Save a class distribution plot for the training labels."""
    labels = np.asarray(train_labels).squeeze()

    plt.figure(figsize=(8, 5))
    plt.title("Class Distribution in the Training Set")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    print("---------------------- Import, preprocess and load data ----------------------")
    print("Loading the PneumoniaMNIST splits")

    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_pneumonia_mnist_arrays()
    print("Raw data loaded successfully!")

    train_loader, val_loader, test_loader = load_pneumonia_mnist_loaders()
    print("Transformed data loaded successfully!")

    save_training_class_distribution_plot(train_labels)

    return (
        train_images,
        train_labels,
        val_images,
        val_labels,
        test_images,
        test_labels,
        train_loader,
        val_loader,
        test_loader,
    )


if __name__ == "__main__":
    main()