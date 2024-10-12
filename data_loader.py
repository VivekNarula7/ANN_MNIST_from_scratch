import numpy as np
import h5py

# Load dataset from the file path
file_path = 'data/data.h5'
with h5py.File(file_path, 'r') as file:
    X = file['X'][:]  # Load the input data (images)
    Y = file['Y'][:]  # Load the labels

# Step 1: Normalize the data (assuming X contains image data with pixel values)
X = X / 255.0  # Normalize pixel values between 0 and 1

# Step 2: Reshape X to be 2D (num_samples, 784)
X = X.reshape(X.shape[0], -1)

# Step 3: Map the labels Y to binary (7 -> 0, 9 -> 1)
Y_binary = np.where(Y == 7, 0, 1)

# Step 4: Split the dataset manually (80% training, 20% testing)
split_index = int(0.8 * X.shape[0])
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y_binary[:split_index], Y_binary[split_index:]

# Now X_train, Y_train can be fed into your binary neural network for training
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_test shape: {Y_test.shape}")
print(X.shape, Y.shape)