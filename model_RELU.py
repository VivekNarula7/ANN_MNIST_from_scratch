import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from data_loader import X_train, Y_train, X_test, Y_test  # Load the data from data_loader.py

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases using He initialization
        for i in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[i-1], layers[i]) * np.sqrt(2. / layers[i-1]))
            self.biases.append(np.zeros((1, layers[i])))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def feedforward(self, X):
        self.activations = [X]
        z = X
        for i in range(len(self.weights) - 1):
            z = self.relu(np.dot(z, self.weights[i]) + self.biases[i])
            self.activations.append(z)
        
        z = np.dot(z, self.weights[-1]) + self.biases[-1]
        z = self.softmax(z)
        self.activations.append(z)
        return z

    def backprop(self, X, y, learning_rate):
        m = X.shape[0]
        self.feedforward(X)
        
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        error = self.activations[-1] - y
        weight_gradients[-1] = np.dot(self.activations[-2].T, error) / m
        bias_gradients[-1] = np.sum(error, axis=0, keepdims=True) / m
        
        error = np.dot(error, self.weights[-1].T)
        for i in reversed(range(len(self.weights) - 1)):
            delta = error * self.relu_derivative(self.activations[i + 1])
            weight_gradients[i] = np.dot(self.activations[i].T, delta) / m
            bias_gradients[i] = np.sum(delta, axis=0, keepdims=True) / m
            error = np.dot(delta, self.weights[i].T)

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * weight_gradients[i]
            self.biases[i] -= learning_rate * bias_gradients[i]

    def one_hot_encode(self, y, num_classes):
        one_hot = np.zeros((y.size, num_classes))
        one_hot[np.arange(y.size), y.astype(int)] = 1
        return one_hot

    def compute_loss(self, y_true, y_pred):
        """Cross-entropy loss."""
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

    def compute_accuracy(self, y_true, y_pred):
        """Compute accuracy by comparing predicted and true labels."""
        return np.mean(np.argmax(y_pred, axis=1) == y_true)

    def save_weights(self, save_path):
        """Save the current weights and biases to a file."""
        with open(save_path, 'wb') as f:
            pickle.dump((self.weights, self.biases), f)
        # print(f"Model weights saved to {save_path}")

    def load_weights(self, load_path):
        """Load weights and biases from a file."""
        with open(load_path, 'rb') as f:
            self.weights, self.biases = pickle.load(f)
        print(f"Model weights loaded from {load_path}")

    def train(self, X_train, y_train, X_test, y_test, epochs, learning_rate, save_path=None):
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        best_accuracy = 0

        for epoch in range(epochs):
            # Training step
            self.backprop(X_train, y_train, learning_rate)
            
            # Calculate training loss and accuracy
            train_loss = self.compute_loss(y_train, self.activations[-1])
            train_acc = self.compute_accuracy(np.argmax(y_train, axis=1), self.activations[-1])
            
            # Test loss and accuracy
            test_preds = self.feedforward(X_test)
            test_loss = self.compute_loss(self.one_hot_encode(y_test, y_train.shape[1]), test_preds)
            test_acc = self.compute_accuracy(y_test, test_preds)
            
            # Store the metrics
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
            
            # Save the best model
            if save_path and test_acc > best_accuracy:
                best_accuracy = test_acc
                self.save_weights(save_path)

        return train_losses, test_losses, train_accuracies, test_accuracies

    def predict(self, X):
        output = self.feedforward(X)
        return np.argmax(output, axis=1)

def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, save_path='images/', filename='training_metrics'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Plot Losses
    plt.figure(figsize=(12, 10))
    
    # Training and Test Loss
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Time')
    plt.legend()
    
    # Training and Test Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Time')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}/{filename}.png")
    plt.show()

# Example Usage
if __name__ == '__main__':
    # Train and evaluate the first neural network (1 hidden layer)
    nn1 = NeuralNetwork([X_train.shape[1], 100, 2])
    Y_train_onehot = nn1.one_hot_encode(Y_train, num_classes=2)

    # Train the model, save the best model
    train_losses_1, test_losses_1, train_accuracies_1, test_accuracies_1 = nn1.train(
        X_train, Y_train_onehot, X_test, Y_test, epochs=201, learning_rate=0.01, save_path='models/best_model_1_hidden.pkl'
    )

    # Load the best weights before prediction
    nn1.load_weights('models/best_model_1_hidden.pkl')

    # Predict with the best model (1 hidden layer)
    predictions_1 = nn1.predict(X_test)

    # Compute accuracy for the 1-hidden-layer model
    accuracy_1 = nn1.compute_accuracy(Y_test, nn1.feedforward(X_test))
    print(f"Accuracy of the best model with 1 hidden layer: {accuracy_1 * 100:.2f}%")

    # Plot and save the metrics for the first network
    plot_metrics(train_losses_1, test_losses_1, train_accuracies_1, test_accuracies_1, filename='nn_1_hidden_layer_metrics_RELU')

    # Train and evaluate the second neural network (3 hidden layers)
    nn2 = NeuralNetwork([X_train.shape[1], 100, 50, 50, 2])
    train_losses_2, test_losses_2, train_accuracies_2, test_accuracies_2 = nn2.train(
        X_train, Y_train_onehot, X_test, Y_test, epochs=201, learning_rate=0.01, save_path='models/best_model_3_hidden.pkl'
    )

    # Load the best weights before prediction
    nn2.load_weights('models/best_model_3_hidden.pkl')

    # Predict with the best model (3 hidden layers)
    predictions_2 = nn2.predict(X_test)

    # Compute accuracy for the 3-hidden-layer model
    accuracy_2 = nn2.compute_accuracy(Y_test, nn2.feedforward(X_test))
    print(f"Accuracy of the best model with 3 hidden layers: {accuracy_2 * 100:.2f}%")

    # Plot and save the metrics for the second network
    plot_metrics(train_losses_2, test_losses_2, train_accuracies_2, test_accuracies_2, filename='nn_3_hidden_layers_metrics_RELU')
