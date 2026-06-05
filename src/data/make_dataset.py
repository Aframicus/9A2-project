# Scripts to download or generate data
# download the packages
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist

from tqdm import tqdm
from medmnist import PneumoniaMNIST
from medmnist import INFO, Evaluator

# flag the dataset we want to use
data_flag='pneumoniamnist'

# define certain parameters, epochs are the amount of iterations it does per time-unit, barch size is the amount image it takes per batch
# and lr is the learning rate, so the size of steps it uses during learning.
Num_Epochs = 3
Batch_Size = 128
lr = 0.001


info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# preprocessing, here we transform the data into a tensor. Then we normalize it using a mean of 0.5 and standard deviation of 0.5 This is more usefull
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=True)
test_dataset = DataClass(split='test', transform=data_transform, download=True)
val_dataset = DataClass(split='val', transform=data_transform, download=True)
pil_dataset = DataClass(split='train', download=True)

# Here we pick an image for the PIL dataset, which has not been transformed and normalized.
img, label = pil_dataset[0]   # PIL image
npimg = np.array(img)
plt.imshow(npimg, cmap="gray")
plt.axis("off")
plt.savefig("pneumonia_example.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved pneumonia_example.png")

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=Batch_Size, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*Batch_Size, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*Batch_Size, shuffle=False)

print(train_dataset)
print("===================")
print(test_dataset)


# Make a montage from raw images (before normalization)
images = pil_dataset.imgs          # shape: (N, 28, 28)
num = 64                           # how many images to show
rows, cols = 8, 8

fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))

for i, ax in enumerate(axes.flat):
    if i < num:
        ax.imshow(images[i], cmap="gray")
    ax.axis("off")

plt.tight_layout()
plt.savefig("pneumonia_montage.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved pneumonia_montage.png")