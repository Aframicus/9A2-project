from pathlib import Path
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.make_dataset import train_images, train_labels, val_images, val_labels, test_images, test_labels
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