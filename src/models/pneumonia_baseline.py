from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import random
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.make_dataset import load_pneumonia_mnist_arrays

# --- Constants ---
SEED = 42
VISUALIZE_DIR = PROJECT_ROOT / "src" / "visualisation"
KNN_PCA_COMPONENTS = 50
KNN_K_RANGE = (1, 30)
KNN_WEIGHTS = ["uniform", "distance"]
KNN_METRICS = ["euclidean", "manhattan", "minkowski"]
KNN_CV_FOLDS = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
VISUALIZE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
train_images, train_labels, val_images, val_labels, test_images, test_labels = load_pneumonia_mnist_arrays()
print("---------------------- Load data ----------------------")

X_train = train_images.reshape(train_images.shape[0], -1)
X_val = val_images.reshape(val_images.shape[0], -1)
X_test = test_images.reshape(test_images.shape[0], -1)

y_train = np.asarray(train_labels).reshape(-1)
y_val = np.asarray(val_labels).reshape(-1)
y_test = np.asarray(test_labels).reshape(-1)
# -------------------------------------------------------------------
# Hyperparameter search
# -------------------------------------------------------------------
print("---------------------- Determining hyperparameters for KNN: -----------------------")
k_min, k_max = KNN_K_RANGE
k_values = list(range(k_min, k_max + 1))  # From k_min to k_max 
hyperparameter_space = {
    'knn__n_neighbors': k_values,
    'knn__weights': KNN_WEIGHTS,
    'knn__metric': KNN_METRICS
}

pipe = Pipeline([
    ('pca', PCA(n_components=KNN_PCA_COMPONENTS, random_state=SEED)), # Reduce dimensionality to 50 components instead of 784
    ('knn', KNeighborsClassifier())
])

gs = GridSearchCV(pipe, param_grid=hyperparameter_space,
                  scoring='accuracy', cv=KNN_CV_FOLDS, n_jobs=-1, refit=True)

gs.fit(X_train, y_train)

print("Best hyperparameters: ", gs.best_params_)
print("Mean CV accuracy of best hyperparameters: ", gs.best_score_)

# -------------------------------------------------------------------
# FIGURE 1 — Validation Loss (1 - accuracy) vs k
# -------------------------------------------------------------------
print(" ---------------------- LOSS-curve (1 - validation accuracy) vs k ----------------------")
results_df = pd.DataFrame(gs.cv_results_)

# Filter results for the best weights and metric
best_weights = gs.best_params_['knn__weights']
best_metric  = gs.best_params_['knn__metric']

mask = (
    (results_df['param_knn__weights'] == best_weights) &
    (results_df['param_knn__metric']  == best_metric)
)
filtered = results_df[mask].copy()
filtered['k'] = filtered['param_knn__n_neighbors'].astype(int)
filtered = filtered.sort_values('k')

val_losses = 1.0 - filtered['mean_test_score'].values
k_plot     = filtered['k'].values
plt.figure(figsize=(6, 4))
plt.plot(k_plot, val_losses, marker='o')
plt.xlabel("k (n_neighbors)")
plt.ylabel("Loss (1 - validation accuracy)")
plt.title("KNN validation loss vs k")
plt.grid(True)
plt.tight_layout()
plot_path = VISUALIZE_DIR / "knn_validation_loss.png"
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Validation loss vs k plot saved to: {plot_path}")
# -------------------------------------------------------------------
# FIGURE 2 — Validation Accuracy vs k
# -------------------------------------------------------------------
val_accs_k = filtered['mean_test_score'].values

plt.figure(figsize=(6, 4))
plt.plot(k_plot, val_accs_k, marker='o', color='green')
plt.xlabel("k (n_neighbors)")
plt.ylabel("Validation Accuracy")
plt.title("KNN — Validation Accuracy vs k")
plt.grid(True)
plt.tight_layout()
plt.savefig(VISUALIZE_DIR / "knn_validation_accuracy.png", dpi=200, bbox_inches="tight")
plt.show()

# -------------------------------------------------------------------
# Train / validate / test with best model
# -------------------------------------------------------------------
print("---------------------- Train model ----------------------")
model = gs.best_estimator_
print("Model trained successfully!")

print("---------------------- Validate model ----------------------")
y_val_pred = model.predict(X_val)
print(f"Validation accuracy: {accuracy_score(y_val, y_val_pred):.4f}")

print("---------------------- Evaluate model ----------------------")
y_test_pred = model.predict(X_test)
y_test_probs = model.predict_proba(X_test)[:, 1]
print(f"Test accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(classification_report(y_test, y_test_pred, digits=4))
# -------------------------------------------------------------------
# FIGURE 3 — Confusion Matrix Heatmap (matplotlib only)
# -------------------------------------------------------------------
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12)

plt.xticks([0, 1], ["Normal", "Pneumonia"])
plt.yticks([0, 1], ["Normal", "Pneumonia"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("KNN — Confusion Matrix")
plt.tight_layout()
plt.savefig(VISUALIZE_DIR / "knn_confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()

# -------------------------------------------------------------------
# FIGURE 4 — Precision-Recall Curve
# -------------------------------------------------------------------
precision, recall, _ = precision_recall_curve(y_test, y_test_probs)
avg_precision = average_precision_score(y_test, y_test_probs)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='.', label=f"AP = {avg_precision:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("KNN — Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(VISUALIZE_DIR / "knn_precision_recall.png", dpi=200, bbox_inches="tight")
plt.show()