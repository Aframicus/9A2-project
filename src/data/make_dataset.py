# Scripts to download or generate data from PneumoniaMNIST
import numpy as np
import medmnist
from medmnist import INFO
def load_pneumoniamnist(download=True):
    """
    Loads the PneumoniaMNIST dataset and returns train, val, test splits.
    """
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    
    # Get the Python class for this specific dataset
    DataClass = getattr(medmnist, info['python_class'])
    
    # Initialize datasets for all splits
    train_dataset = DataClass(split='train', download=download)
    val_dataset = DataClass(split='val', download=download)
    test_dataset = DataClass(split='test', download=download)
    
    # Extract images and labels (squeeze removes unnecessary 1D dimensions from labels)
    X_train, y_train = train_dataset.imgs, train_dataset.labels.squeeze()
    X_val, y_val = val_dataset.imgs, val_dataset.labels.squeeze()
    X_test, y_test = test_dataset.imgs, test_dataset.labels.squeeze()
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

