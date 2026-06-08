# Scripts to download or generate data from PneumoniaMNIST
import pickle
import random
import sys
import matplotlib.pyplot as plt
import medmnist
import numpy as np
import torch
import torch.utils.data as data
from medmnist import INFO
from torchvision import transforms



print("---------------------- Import, preprocess and load data ----------------------")
# Import data
data_flag = 'pneumoniamnist'
download = True

BATCH_SIZE = 128

info = INFO[data_flag]
task = info['task']
DataClass = getattr(medmnist, info['python_class'])

n_channels = info['n_channels']
n_classes = len(info['label'])

print("Loading the PneumoniaMNIST splits")

# Load the raw data
train_dataset_raw = DataClass(split='train', download=download)
val_dataset_raw = DataClass(split='val', download=download)
test_dataset_raw = DataClass(split='test', download=download)

print("Raw data loaded successfully!")

#Labels and images are stored in the .imgs and .labels attributes of the dataset objects. We normalise the images to [0.0, 1.0] by dividing by 255.0, since the original pixel values are in the range [0, 255].
train_images = train_dataset_raw.imgs / 255.0    
train_labels = train_dataset_raw.labels
 
val_images   = val_dataset_raw.imgs   / 255.0
val_labels   = val_dataset_raw.labels
 
test_images  = test_dataset_raw.imgs  / 255.0
test_labels  = test_dataset_raw.labels

#Preprocessing, here we transform the data into a tensor. Then we normalize it using a mean of 0.5 and standard deviation of 0.5 This is more usefull
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Transform the raw data into tensors and normalise them. We will use these transformed datasets for training and evaluation of our deep learning models.
# Load transformed data
train_dataset_transformed = DataClass(split='train', download=download, transform=data_transform)
val_dataset_transformed = DataClass(split='val', download=download, transform=data_transform)
test_dataset_transformed = DataClass(split='test', download=download, transform=data_transform)

# Create data loaders
train_loader = data.DataLoader(dataset=train_dataset_transformed, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset_transformed, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset_transformed, batch_size=2*BATCH_SIZE, shuffle=False)

print("Transformed data loaded successfully!")


#show some samples from the raw and transformed datasets to verify that the transformations are correct. We will display a grid of images with their corresponding labels for both the raw and transformed datasets.
def show_samples(dataset, num_samples=16):
    images = dataset.imgs
    labels = dataset.labels
    plt.figure(figsize=(8, 8))
    for i in range(num_samples):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i][0]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
# show_samples(train_dataset_transformed)
# show_samples(train_dataset_raw)

# #Class distribution in the training set
# labels = (labels for labels in train_labels)
# sns.countplot(x=list(labels))
# plt.title("Class Distribution in the Training Set")
# plt.show()

# # Make a montage from raw images (before normalization)
# images = train_dataset_raw.imgs          # shape: (N, 28, 28)
# num = 64                           # how many images to show
# rows, cols = 8, 8

# fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))

# for i, ax in enumerate(axes.flat):
#     if i < num:
#         ax.imshow(images[i], cmap="gray")
#     ax.axis("off")

# plt.tight_layout()
# plt.savefig("pneumonia_montage.png", dpi=200, bbox_inches="tight")
# plt.close()
# print("Saved pneumonia_montage.png")