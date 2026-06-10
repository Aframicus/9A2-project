from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Project imports: make sure this matches your structure
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.make_dataset import train_loader, val_loader, test_loader

# -------------------------------------------------------------------
# Reproducibility & device
# -------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------------------------
# Deeper CNN architecture
# -------------------------------------------------------------------
class DeeperCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DeeperCNN, self).__init__()

        # Block 1: 1x28x28 -> 32x14x14
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: 32x14x14 -> 64x7x7
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: 64x7x7 -> 128x3x3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 7 -> 3
        )

        # Fully connected classifier
        self.fc = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)         # (B, 32, 14, 14)
        x = self.block2(x)         # (B, 64, 7, 7)
        x = self.block3(x)         # (B, 128, 3, 3)
        x = x.view(x.size(0), -1)  # Flatten: (B, 128*3*3)
        x = self.fc(x)             # (B, num_classes)
        return x

# -------------------------------------------------------------------
# Instantiate model, loss, optimizer
# -------------------------------------------------------------------
model = DeeperCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -------------------------------------------------------------------
# Training loop with validation
# -------------------------------------------------------------------
num_epochs = 10
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    # ----------------- Train -----------------
    model.train()
    running_loss = 0.0
    all_train_preds = []
    all_train_labels = []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.squeeze(1).to(device).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())

    epoch_train_loss = running_loss / len(train_loader.dataset)
    epoch_train_acc = accuracy_score(all_train_labels, all_train_preds)
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    # ----------------- Validate -----------------
    model.eval()
    running_val_loss = 0.0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.squeeze(1).to(device).long()

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_val_acc = accuracy_score(all_val_labels, all_val_preds)
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

# -------------------------------------------------------------------
# Plot training & validation loss (optional)
# -------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train")
plt.plot(val_accuracies, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# Evaluate on test set
# -------------------------------------------------------------------
model.eval()
all_test_preds = []
all_test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.squeeze(1).to(device).long()

        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_test_preds.extend(preds.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

print("Test Accuracy:", accuracy_score(all_test_labels, all_test_preds))
print("Classification Report:\n", classification_report(all_test_labels, all_test_preds, digits=4))
print("Confusion Matrix:\n", confusion_matrix(all_test_labels, all_test_preds))

# -------------------------------------------------------------------
# Save model
# -------------------------------------------------------------------
torch.save(model.state_dict(), "pneumonia_deeper_cnn.pth")
print("Saved model to pneumonia_deeper_cnn.pth")