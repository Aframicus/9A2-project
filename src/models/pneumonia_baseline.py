#Script to generate baseline results for PneumoniaMNIST using random forest classifier
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utils import data_feed

print("-------------- Import data --------------")
X_train_base, y_train_base, X_val, y_val, X_test, y_test = data_feed(data_flag='pneumoniamnist')
print("Training shape", X_train_base.shape)
print("Validation shape", X_val.shape)
print("Testing shape", X_test.shape)

print("-------------- Train model Random Forest --------------")
X_train = np.vstack((X_train_base, X_val))
y_train = np.hstack((y_train_base, y_val))

print("Training predictors shape", X_train.shape)
print("Training target shape", y_train.shape)
print("Testing predictors shape", X_test.shape)
print("Testing target shape", y_test.shape)



