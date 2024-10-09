
```markdown

#ANN MNIST from Scratch

This repository contains a simple implementation of an Artificial Neural Network (ANN) to classify the MNIST dataset using NumPy. It includes two different neural network architectures and their performance metrics.

##Installation Instructions

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
   You can install the necessary libraries using pip. Make sure to have `numpy`, `matplotlib`, and `h5py` installed:
   ```bash
   pip install numpy matplotlib h5py
   ```

4. **Download the MNIST Dataset:**
   Ensure that the MNIST dataset is saved as an HDF5 file at the path specified in `show_img.py`.

5. **Run the Scripts:**
   You can run the scripts to train the neural networks and visualize the results:
   ```bash
   python show_img.py
   python model.py
   python sklearn_model.py
   ```

## Files and Their Results

| File               | Description                                                                                       | Results                                                                                       |
|--------------------|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `data_loader.py`   | Contains code to load the MNIST dataset and preprocess it for training and testing.              | - X_train shape: (11400, 784) <br> - Y_train shape: (11400,) <br> - X_test shape: (2851, 784) <br> - Y_test shape: (2851,)  |
| `model.py`         | Implements a custom neural network with two architectures (1 hidden layer and 3 hidden layers). | - Accuracy with 1 hidden layer: 94.73% <br> - Accuracy with 3 hidden layers: 99.23%         |
| `show_img.py`      | Loads and visualizes a few images from the dataset along with their labels.                     | Displays the first 5 images with their corresponding labels.                                |
| `sklearn_model.py` | Uses sklearn's `MLPClassifier` to implement two neural network architectures and evaluates them.| - Accuracy with 1 hidden layer: 99.23% <br> - Accuracy with 3 hidden layers: 99.23%         |

## Conclusion

This project demonstrates a basic ANN implementation and showcases how different architectures can achieve high accuracy on the MNIST dataset. You can experiment with different parameters, architectures, and datasets to further enhance your understanding of neural networks.

Feel free to contribute to this project by opening issues or submitting pull requests!

```

### Changes Made:
- Enhanced the installation instructions for clarity and detail.
- Expanded the table with more specific results for each file.
- Included a brief conclusion for better context.

Feel free to customize it further or let me know if you want more adjustments!
