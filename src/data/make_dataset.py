# Scripts to download or generate data from PneumoniaMNIST
import pickle
import medmnist
import numpy as np
import torch
import torch.utils.data as data
from medmnist import INFO

print("---------------------- Import, preprocess and load data ----------------------")
# Import data
data_flag = 'pneumoniamnist'
download = True

if data_flag not in ['pneumoniamnist']:
    raise ValueError("The data_flag should be 'pneumoniamnist'.")

NUM_EPOCHS = 3
BATCH_SIZE = 128

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])

# Load data
train_dataset = DataClass(split='train', download=download)
val_dataset = DataClass(split='val', download=download)
test_dataset = DataClass(split='test', download=download)

# Set seed for reproducibility
torch.manual_seed(123)

# Create data loaders
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=2 * BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False)

# Get out images and information
X_train = train_loader.dataset.imgs
y_train = np.reshape(train_loader.dataset.labels, (len(train_loader.dataset.labels)))
X_val = val_loader.dataset.imgs
y_val = np.reshape(val_loader.dataset.labels, (len(val_loader.dataset.labels)))
X_test = test_loader.dataset.imgs
y_test = np.reshape(test_loader.dataset.labels, (len(test_loader.dataset.labels)))

print("---------- Store data ---------- ")

file_name1 = 'pneumoniamnist_X_train.pickle'
file_name2 = 'pneumoniamnist_y_train.pickle'
file_name3 = 'pneumoniamnist_X_val.pickle'
file_name4 = 'pneumoniamnist_y_val.pickle'
file_name5 = 'pneumoniamnist_X_test.pickle'
file_name6 = 'pneumoniamnist_y_test.pickle'


with open(file_name1, 'wb') as f:
    pickle.dump(np.array(X_train), f, pickle.HIGHEST_PROTOCOL)

with open(file_name2, 'wb') as f:
    pickle.dump(y_train, f, pickle.HIGHEST_PROTOCOL)

with open(file_name3, 'wb') as f:
    pickle.dump(X_val, f, pickle.HIGHEST_PROTOCOL)

with open(file_name4, 'wb') as f:
    pickle.dump(y_val, f, pickle.HIGHEST_PROTOCOL)

with open(file_name5, 'wb') as f:
    pickle.dump(X_test, f, pickle.HIGHEST_PROTOCOL)

with open(file_name6, 'wb') as f:
    pickle.dump(y_test, f, pickle.HIGHEST_PROTOCOL)


print("---------- End ----------")