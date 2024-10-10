# ANN MNIST from Scratch

This repository contains implementations of Artificial Neural Networks (ANNs) to classify the MNIST dataset using different approaches: a custom implementation using NumPy and a model using Scikit-learn. It includes various architectures with different activation functions and evaluates their performance metrics.

## Table of Contents
- [Features](#features)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Improvements](#future-improvements)
- [License](#license)

## Features
- Custom ANN implementations using NumPy with:
  - ReLU activation function
  - Sigmoid activation function
- Use of Scikit-learn's `MLPClassifier` for comparison.
- Visualization of sample images from the MNIST dataset.
- Evaluation of model accuracy and loss metrics.

## Installation Instructions

To get started with this project, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/ANN_MNIST_from_scratch.git
   cd ANN_MNIST_from_scratch
   ```

2. **Create a Virtual Environment (Optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Packages:**
   You can install the necessary libraries using pip. Make sure to have `numpy`, `matplotlib`, `scikit-learn` and `h5py` installed:
   ```bash
   pip install numpy matplotlib scikit-learn h5py 
   ```

4. **Download the MNIST Dataset:**
   Ensure that the MNIST dataset is saved as an HDF5 file at the path specified in `show_img.py`.

## Usage

You can run the scripts to train the neural networks and visualize the results:

```bash
python show_img.py          # Visualize the first 5 images with labels
python model_RELU.py       # Train custom ANN with ReLU activation
python model_Sigmoid.py    # Train custom ANN with Sigmoid activation
python sklearn_model.py     # Train ANN using Scikit-learn's MLPClassifier
```

## File Descriptions

| File               | Description                                                                                       |
|--------------------|---------------------------------------------------------------------------------------------------|
| `data_loader.py`   | Contains code to load the MNIST dataset and preprocess it for training and testing.              |
| `model_RELU.py`    | Implements a custom neural network using ReLU activation with two architectures (1 and 3 hidden layers). |
| `model_Sigmoid.py` | Implements a custom neural network using Sigmoid activation with two architectures (1 and 3 hidden layers). |
| `show_img.py`      | Loads and visualizes the first 5 images from the dataset along with their labels.                     |
| `sklearn_model.py` | Uses Scikit-learn's `MLPClassifier` to implement two neural network architectures and evaluates their performance. |

## Results

| **Model**           | **Activation Function** | **Architecture** | **Test Accuracy (%)** |
|---------------------|-------------------------|-------------------|------------------------|
| **Custom Model**    | ReLU                    | 1 Hidden Layer    | 94.46                  |
| **Custom Model**    | ReLU                    | 3 Hidden Layers   | 95.69                  |
| **Custom Model**    | Sigmoid                 | 1 Hidden Layer    | 94.91                  |
| **Custom Model**    | Sigmoid                 | 3 Hidden Layers   | 84.32                  |
| **Scikit-learn MLP**| ReLU                    | 1 Hidden Layer    | 99.30                  |
| **Scikit-learn MLP**| ReLU                    | 3 Hidden Layers   | 99.33                  |

## Conclusion

This project demonstrates various implementations of artificial neural networks, showcasing the performance of different activation functions and architectures on the MNIST dataset. The comparison with Scikit-learn's MLPClassifier illustrates the effectiveness of optimized libraries for neural network training.

## Future Improvements

- Experiment with additional activation functions (e.g., Leaky ReLU, ELU) and optimizers (e.g., Adam, RMSprop).
- Implement regularization techniques (e.g., dropout, L2 regularization) to improve generalization.
- Extend the project to other datasets (e.g., CIFAR-10, Fashion MNIST) for further experimentation.
- Visualize the learning curves for training and validation to understand model performance better.
