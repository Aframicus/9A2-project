#Script to train deep learning models for pneumonia classification. We will use a simple CNN architecture and train it on the PneumoniaMNIST dataset.
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.make_dataset import train_loader, val_loader, test_loader

#Setting seed for reproducibility
torch.manual_seed(42)

# Define the CNN architecture
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1) # Input channels = 1 (grayscale), output channels = 16, kernel size = 3, padding ensures output size is the same as input size
        self.pool = nn.MaxPool2d(2, 2) # Max pooling with kernel size 2 and stride 2, reduces spatial dimensions by half
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # Input channels = 16 (from previous layer), output channels = 32, kernel size = 3, padding ensures output size is the same as input size
        self.fc1 = nn.Linear(32 * 7 * 7, 512) # Fully connected layer, input features = 32*7*7 (after two pooling layers), output features = 512
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Instantiate the model
model = CNNModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item() * images.size(0)  # Accumulate loss
    epoch_loss = running_loss / len(train_loader.dataset)  # Average loss for the epoch
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Plot training loss. The loss should decrease as we train the model, indicating that the model is learning to classify the images correctly. If the loss does not decrease, it may indicate that the model is not learning effectively, and we may need to adjust the architecture, learning rate, or other hyperparameters.
plt.figure(figsize=(10, 5))
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.show()

#Evaluate the model on the test set
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

print("Test Accuracy:", accuracy_score(all_labels, all_preds))
print("Classification Report:\n", classification_report(all_labels, all_preds))
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))

# Save the model
torch.save(model.state_dict(), 'pneumonia_model.pth')

# Load the model
model = CNNModel()
model.load_state_dict(torch.load('pneumonia_model.pth'))
model.eval()