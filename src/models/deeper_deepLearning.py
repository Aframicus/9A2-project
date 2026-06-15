from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from time import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------------------------------------------------
# Project imports
# -------------------------------------------------------------------
from src.data.make_dataset import load_pneumonia_mnist_loaders
from src.utils import set_seed, get_experiment_dir, save_checkpoint, save_best_model, load_checkpoint

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 128
NUM_EPOCHS = 10
NUM_EPOCHS_SEARCH = 5
CHECKPOINT_EVERY = 1
EXPERIMENTS_DIR = "experiments/"
LR_VALUES = [0.0001, 0.0005, 0.001]
WEIGHT_DECAY_VALUES = [0.0, 0.0001, 0.001]
EARLY_STOPPING_PATIENCE = 3

# -------------------------------------------------------------------
# Set random seed for reproducibility
# -------------------------------------------------------------------
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# -------------------------------------------------------------------
# Experiment directory
# -------------------------------------------------------------------
# change run_name per experiment
run_dir = get_experiment_dir(EXPERIMENTS_DIR, run_name="run_001")
print(f"Experiment directory: {run_dir}")
# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
train_loader, val_loader, test_loader = load_pneumonia_mnist_loaders(batch_size=BATCH_SIZE)

# -------------------------------------------------------------------
# Deeper CNN architecture
# -------------------------------------------------------------------
class DeeperCNN(nn.Module): #creating a new neural network type from the nn module
    def __init__(self, num_classes=2): #there are two classes, namely present and absent of pneumonia
        super(DeeperCNN, self).__init__() #requirement to call the class of an neural network

        # Block 1: 1x28x28 -> 32x14x14
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), #defines the kernel_size, with padding of 1 which spatiality. The 1 is the amount of channels in and 32 for channels out.
            nn.BatchNorm2d(32), #batchnormalisation of 32 for stable training
            nn.ReLU(inplace=True), #uses ReLu
            nn.MaxPool2d(kernel_size=2, stride=2) #halves the hight and width
        )

        # Block 2: 32x14x14 -> 64x7x7
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) #Almost the same explanation as above in block 1, but in the second block, it converges 32 to 64 channels. The batch is also 64.

        # Block 3: 64x7x7 -> 128x3x3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        ) #Also the same explanation as above, but with 64 in channels and 128 outchannels. Here, the batch is also 128.

        # Fully connected classifier
        self.fc = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256), #the in channels 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5), #During training, aroung half of all the neurons are deactivates
            nn.Linear(256, num_classes) #last layer will be placed from 256 to 2 since we have binairy outcomes
        )  #docs.pytorch.org/docs/2.12/generated/torch.nn.Conv2d.html

    def forward(self, x):
        x = self.block1(x)         # (B, 32, 14, 14)
        x = self.block2(x)         # (B, 64, 7, 7)
        x = self.block3(x)         # (B, 128, 3, 3)
        x = x.view(x.size(0), -1)  # Flatten: (B, 128*3*3)
        x = self.fc(x)             # (B, num_classes)
        return x
    #pushes the image through multiple layers and at the end it is flattend to a tensor. At the self.fc(x) it is passed through a fully connected layer. Then it is returned as a logits.

# -------------------------------------------------------------------
# Hyperparameter search for learning rate and weight decay
# -------------------------------------------------------------------
def train_for_search(lr, weight_decay, num_epochs_search=NUM_EPOCHS_SEARCH):
    set_seed(SEED)  # Ensure reproducibility for each hyperparameter combination
    model = DeeperCNN(num_classes=2).to(device) #Creates a new model
    criterion = nn.CrossEntropyLoss() #the criteria is set on CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) #These are the optimizer values put in the new model.
    best_val_acc = 0.0

    for epoch in range(num_epochs_search):
        # ----------------- Train -----------------
        model.train() #puts the model in training mode
        for images, labels in train_loader: #loops over batches of training data
            images = images.to(device)
            labels = labels.squeeze(1).to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # ----------------- Validate -----------------
        model.eval()
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.squeeze(1).to(device).long()
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_val_labels, all_val_preds)
        best_val_acc = max(best_val_acc, val_acc)

    return best_val_acc

best_val_acc_overall = 0.0
best_params = None

print("---------------------- Hyperparameter search (DeeperCNN) ----------------------")
for lr in LR_VALUES:
    for wd in WEIGHT_DECAY_VALUES:
        print(f"Try lr={lr}, weight_decay={wd}")
        val_acc = train_for_search(lr, wd, num_epochs_search=NUM_EPOCHS_SEARCH)
        print(f"   -> best val accuracy during search: {val_acc:.4f}")
        if val_acc > best_val_acc_overall:
            best_val_acc_overall = val_acc
            best_params = {"lr": lr, "weight_decay": wd}

print("Best hyperparameters:", best_params)
print("Best validation accuracy (search):", best_val_acc_overall)

# -------------------------------------------------------------------
# Instantiate final model, loss, optimizer with best hyperparameters
# -------------------------------------------------------------------
set_seed(SEED)  # Ensure reproducibility for final training
model = DeeperCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=best_params["lr"],
    weight_decay=best_params["weight_decay"]
)

print(f"Train final model with lr={best_params['lr']}, weight_decay={best_params['weight_decay']}")

# -------------------------------------------------------------------
# Complete training with chosen hyperparameters
# -------------------------------------------------------------------

train_losses, val_losses= [], []
train_accuracies, val_accuracies = [], []
best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    # ── Train ──
    model.train()
    running_loss = 0.0
    all_train_preds, all_train_labels = [], []

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
    epoch_train_acc  = accuracy_score(all_train_labels, all_train_preds)
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

            running_val_loss += criterion(outputs, labels).item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_val_acc = accuracy_score(all_val_labels, all_val_preds)
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

# if validation accuracy improved, save best model checkpoint
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        save_best_model(run_dir, epoch, model, best_val_acc)
# periodic checkpointing
    if epoch % CHECKPOINT_EVERY == 0:
        save_checkpoint(
            run_dir, epoch, model, optimizer,
            train_losses, val_losses, train_accuracies, val_accuracies,
        )
# -------------------------------------------------------------------
# LOSS-plot (training & validation)
# -------------------------------------------------------------------
epochs_ran = len(train_losses)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
axes[0].plot(range(1, epochs_ran + 1), train_losses,  marker='o', label="Train loss")
axes[0].plot(range(1, epochs_ran + 1), val_losses,    marker='o', label="Val loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Deeper CNN — Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(range(1, epochs_ran + 1), train_accuracies, marker='o', label="Train acc")
axes[1].plot(range(1, epochs_ran + 1), val_accuracies,   marker='o', label="Val acc")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Deeper CNN — Accuracy")
axes[1].legend()
axes[1].grid(True)
 
plt.tight_layout()
plt.savefig(Path(run_dir) / "deeper_cnn_curves.png", dpi=200, bbox_inches="tight")
plt.show()

# -------------------------------------------------------------------
# Load best model and Evaluate on test set
# -------------------------------------------------------------------
best_ckpt_path = Path(run_dir) / "checkpoints" / "best_model.pt"
load_checkpoint(best_ckpt_path, model)
model.to(device)
model.eval()

all_test_preds, all_test_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.squeeze(1).to(device).long()
        _, preds = torch.max(model(images), 1)
        all_test_preds.extend(preds.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

print("Test Accuracy:", accuracy_score(all_test_labels, all_test_preds))
print("Classification Report:\n", classification_report(all_test_labels, all_test_preds, digits=4))
print("Confusion Matrix:\n", confusion_matrix(all_test_labels, all_test_preds))

# -------------------------------------------------------------------
# Save model for later usages such as predictions with new datasets
# -------------------------------------------------------------------
model_save_path = Path(run_dir) / "pneumonia_deeper_cnn.pth"
torch.save(model.state_dict(), model_save_path)  
print("Saved model to pneumonia_deeper_cnn.pth")