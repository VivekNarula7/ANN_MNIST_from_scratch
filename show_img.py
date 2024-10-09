import numpy as np
import h5py
import matplotlib.pyplot as plt

# Load dataset from the file path
file_path = '/home/vivek/Code/ANN/ANN_MNIST_from_scratch/data/data.h5'


with h5py.File(file_path, 'r') as file:
    X = file['X'][:]  # Load the input data (images)
    Y = file['Y'][:]  # Load the labels

# Normalize the images
X = X / 255.0  # Normalize pixel values between 0 and 1

# Display a few images using Matplotlib
for i in range(5):  # Display first 5 images as an example
    img = X[i]  # Select image i
    label = Y[i]  # Corresponding label for image i

    # Plot the image with the label
    plt.imshow(img, cmap='gray')  # Display the image in grayscale
    plt.title(f"Label: {label}")  # Display the label as the title
    plt.axis('off')  # Hide axis for better visualization
    plt.show()  # Show the image