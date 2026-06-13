from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.make_dataset import load_pneumonia_mnist_arrays

train_images, train_labels, val_images, val_labels, test_images, test_labels = load_pneumonia_mnist_arrays()
print("---------------------- Load data ----------------------")

X_train = train_images.reshape(train_images.shape[0], -1)
X_val = val_images.reshape(val_images.shape[0], -1)
X_test = test_images.reshape(test_images.shape[0], -1)

y_train = np.asarray(train_labels).reshape(-1)
y_val = np.asarray(val_labels).reshape(-1)
y_test = np.asarray(test_labels).reshape(-1)

print("---------------------- Determining hyperparameters for KNN: -----------------------")
model = KNeighborsClassifier()
k_values = list(range(1, 31))  # From 1 to 30
hyperparameter_space = {
	'n_neighbors': k_values,
	'weights': ['uniform', 'distance'],
	'metric': ['euclidean', 'manhattan', 'minkowski']
}

gs = GridSearchCV(model, param_grid=hyperparameter_space,
                  scoring='accuracy', cv=5)

gs.fit(X_train, y_train)

print("Best hyperparameters: ", gs.best_params_)
print("Mean CV accuracy of best hyperparameters: ", gs.best_score_)

# ---------------------- LOSS-curve (1 - validation accuracy) vs k ----------------------
val_losses = []

# We use the best weights/metrics from GridSearch, we only vary k
for k in k_values:
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights=gs.best_params_['weights'],
        metric=gs.best_params_['metric']
    )
    knn.fit(X_train, y_train)
    y_val_pred_k = knn.predict(X_val)
    val_acc_k = accuracy_score(y_val, y_val_pred_k)
    val_loss_k = 1.0 - val_acc_k
    val_losses.append(val_loss_k)

plt.figure(figsize=(6, 4))
plt.plot(k_values, val_losses, marker='o')
plt.xlabel("k (n_neighbors)")
plt.ylabel("Loss (1 - validation accuracy)")
plt.title("KNN validation loss vs k")
plt.grid(True)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "src" / "visualisation" / "knn_validation_loss.png", dpi=200, bbox_inches="tight")
plt.show()

print("---------------------- Train model ----------------------")
model = KNeighborsClassifier(
	n_neighbors=gs.best_params_['n_neighbors'],
	weights=gs.best_params_['weights'],
	metric=gs.best_params_['metric']
)
model.fit(X_train, y_train)
print("Model trained successfully!")

print("---------------------- Validate model ----------------------")
y_val_pred = model.predict(X_val)
print(f"Validation accuracy: {accuracy_score(y_val, y_val_pred):.4f}")

print("---------------------- Evaluate model ----------------------")
y_test_pred = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(classification_report(y_test, y_test_pred, digits=4))
print(confusion_matrix(y_test, y_test_pred))