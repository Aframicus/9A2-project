from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.make_dataset import load_pneumonia_mnist_arrays
from src.utils import load_config, set_seed

cfg = load_config()
set_seed(cfg["seed"])
outputs_dir = PROJECT_ROOT / cfg["paths"]["outputs"]
os.makedirs(outputs_dir, exist_ok=True)
knn_cfg = cfg["knn"]

train_images, train_labels, val_images, val_labels, test_images, test_labels = load_pneumonia_mnist_arrays()
print("---------------------- Load data ----------------------")

X_train = train_images.reshape(train_images.shape[0], -1)
X_val = val_images.reshape(val_images.shape[0], -1)
X_test = test_images.reshape(test_images.shape[0], -1)

y_train = np.asarray(train_labels).reshape(-1)
y_val = np.asarray(val_labels).reshape(-1)
y_test = np.asarray(test_labels).reshape(-1)

print("---------------------- Determining hyperparameters for KNN: -----------------------")
k_min, k_max = knn_cfg["k_range"]
k_values = list(range(k_min, k_max + 1))  # From k_min to k_max (inclusive)
hyperparameter_space = {
    'knn__n_neighbors': k_values,
    'knn__weights': knn_cfg["weights"],
    'knn__metric': knn_cfg["metrics"]
}

pipe = Pipeline([
    ('pca', PCA(n_components=knn_cfg["pca_components"],
        random_state=cfg["seed"])), # Reduce dimensionality to 50 components instead of 784
    ('knn', KNeighborsClassifier())
])

gs = GridSearchCV(pipe, param_grid=hyperparameter_space,
                  scoring='accuracy', cv=knn_cfg["cv_folds"], n_jobs=-1, refit=True)

gs.fit(X_train, y_train)

print("Best hyperparameters: ", gs.best_params_)
print("Mean CV accuracy of best hyperparameters: ", gs.best_score_)

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
plot_path = os.path.join(outputs_dir, "knn_validation_loss.png")
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Validation loss vs k plot saved to: {plot_path}")

print("---------------------- Train model ----------------------")
model = gs.best_estimator_
print("Model trained successfully!")

print("---------------------- Validate model ----------------------")
y_val_pred = model.predict(X_val)
print(f"Validation accuracy: {accuracy_score(y_val, y_val_pred):.4f}")

print("---------------------- Evaluate model ----------------------")
y_test_pred = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(classification_report(y_test, y_test_pred, digits=4))
print(confusion_matrix(y_test, y_test_pred))