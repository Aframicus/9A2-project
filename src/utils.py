import os
import random
from pathlib import Path
import numpy as np
import torch

# Project root is two levels up from this file (src/utils.py -> src/ -> project/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Utility functions for reproducibility, configuration loading, and experiment directory management.
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Utility function to create a directory for an experiment
def get_experiment_dir(base: str, run_name: str) -> str:
    run_dir = os.path.join(base, run_name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    return run_dir

#Save a periodic checkpoint of the model and optimizer state, along with training/validation metrics
def save_checkpoint(run_dir, epoch, model, optimizer,
                    train_losses, val_losses, train_accuracies, val_accuracies):
    path = os.path.join(run_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
        },
        path,
    )
    print(f"  [Checkpoint] Saved periodic checkpoint -> {path}")

# Save the best model based on validation accuracy
def save_best_model(run_dir, epoch, model, optimizer, train_losses, val_losses, train_accuracies, val_accuracies):
    path = os.path.join(run_dir, "checkpoints", "best_model.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
        },
        path,
    )
    print(f"  [Checkpoint] New best model (val_acc={val_accuracies[-1]:.4f}) -> {path}")


# Load a checkpoint and restore the model and optimizer state, along with training/validation metrics
def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"  [Checkpoint] Loaded from {path} (epoch {checkpoint.get('epoch')})")
    return checkpoint