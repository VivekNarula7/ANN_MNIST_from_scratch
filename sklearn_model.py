import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from data_loader import X_train, Y_train, X_test, Y_test  # Load the data from data_loader.py

# Flatten input images if necessary (ensure X_train and X_test are 2D arrays)
if X_train.ndim > 2:
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

# One-hot encode labels for multi-class classification
encoder = OneHotEncoder()
Y_train_onehot = encoder.fit_transform(Y_train.reshape(-1, 1))

# Define the neural network architecture
# Using two different architectures: 
# 1. One hidden layer with 100 units
# 2. Three hidden layers with 100, 50, and 50 units

# Architecture 1: One hidden layer
mlp1 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, alpha=0.01, 
                     activation='relu', solver='adam', random_state=42)

# Train the model
mlp1.fit(X_train, Y_train)

# Predict on test set
predictions_1 = mlp1.predict(X_test)
accuracy_1 = accuracy_score(Y_test, predictions_1) * 100
print(f"Accuracy with 1 hidden layer: {accuracy_1:.2f}%")

# Architecture 2: Three hidden layers
mlp2 = MLPClassifier(hidden_layer_sizes=(100, 50, 50), max_iter=500, alpha=0.01, 
                     activation='relu', solver='adam', random_state=42)

# Train the model
mlp2.fit(X_train, Y_train)

# Predict on test set
predictions_2 = mlp2.predict(X_test)
accuracy_2 = accuracy_score(Y_test, predictions_2) * 100
print(f"Accuracy with 3 hidden layers: {accuracy_2:.2f}%")
