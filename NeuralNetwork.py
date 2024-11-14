import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the dataset
from ucimlrepo import fetch_ucirepo

# fetch dataset
banknote_authentication = fetch_ucirepo(id=267)
X = banknote_authentication.data.features
y = banknote_authentication.data.targets


# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# Cost Functions
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def binary_crossentropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # To prevent log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Neuron Class
class Neuron:
    def __init__(self, input_size, output_size, activation_fn, activation_deriv_fn):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(output_size)
        self.activation_fn = activation_fn
        self.activation_deriv_fn = activation_deriv_fn

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.activation_fn(np.dot(self.inputs, self.weights) + self.biases)
        return self.output

    def backward(self, d_output):
        d_activation = d_output * self.activation_deriv_fn(self.output)
        self.d_weights = np.dot(self.inputs.T, d_activation)
        self.d_biases = np.sum(d_activation, axis=0)
        d_input = np.dot(d_activation, self.weights.T)
        return d_input

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases


# Network Class
class Network:
    def __init__(self, input_size, hidden_layer_width, num_outputs, activation_fn, activation_deriv_fn, cost_fn):
        self.hidden_layer = Neuron(input_size, hidden_layer_width, activation_fn, activation_deriv_fn)
        self.output_layer = Neuron(hidden_layer_width, num_outputs, sigmoid, sigmoid_derivative)
        self.cost_fn = cost_fn

    def forward(self, X):
        hidden_output = self.hidden_layer.forward(X)
        output = self.output_layer.forward(hidden_output)
        return output

    def backward(self, X, y, output):
        cost_error = output - y
        d_output = cost_error
        d_hidden = self.output_layer.backward(d_output)
        self.hidden_layer.backward(d_hidden)

    def update(self, learning_rate):
        self.hidden_layer.update_weights(learning_rate)
        self.output_layer.update_weights(learning_rate)

    def train(self, X, y, num_epochs, learning_rate, progress_callback):
        for epoch in range(num_epochs):
            for i in range(len(X)):
                output = self.forward(X[i].reshape(1, -1))
                self.backward(X[i].reshape(1, -1), y[i], output)
                self.update(learning_rate)

            if epoch % 100 == 0:
                predictions = self.test_network(X)
                cost = self.cost_fn(y, predictions)
                progress_callback(epoch, cost)

    def test_network(self, X):
        predictions = []
        for i in range(len(X)):
            output = self.forward(X[i].reshape(1, -1))
            predictions.append(output)
        return np.array(predictions)


# Function to select activation function and cost function
def get_selected_functions(activation_var, cost_var):
    activation_choice = activation_var.get()
    if activation_choice == 'Sigmoid':
        activation_fn = sigmoid
        activation_deriv_fn = sigmoid_derivative
    elif activation_choice == 'ReLU':
        activation_fn = relu
        activation_deriv_fn = relu_derivative
    elif activation_choice == 'Tanh':
        activation_fn = tanh
        activation_deriv_fn = tanh_derivative

    cost_choice = cost_var.get()
    if cost_choice == 'Mean Squared Error':
        cost_fn = mean_squared_error
    elif cost_choice == 'Binary Cross-Entropy':
        cost_fn = binary_crossentropy

    return activation_fn, activation_deriv_fn, cost_fn



# Function to update the progress label, plot loss, and visualize the neural network

def update_progress(epoch, cost, ax, canvas, costs, progress_label, network, ax_nn, canvas_nn):
    progress_label.config(text=f"Epoch {epoch} - Cost: {cost:.4f}")
    progress_label.update()

    # Plot the neural network loss (cost) over time
    costs.append(cost)
    ax.clear()
    ax.set_title("Neural Network Loss")
    ax.plot(costs, label="Cost", color="blue")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Cost")
    ax.legend()

    # Visualize the neural network architecture (can be updated each epoch if needed)
    ax_nn.clear()
    ax_nn.set_title("Neural Network Architecture")

    # Visualizing input and output layer
    ax_nn.scatter([0] * len(network.hidden_layer.weights), range(len(network.hidden_layer.weights)), s=500,
                  color='lightblue', label="Input Layer")
    ax_nn.scatter([1] * len(network.output_layer.weights), range(len(network.output_layer.weights)), s=500,
                  color='orange', label="Output Layer")

    # Drawing connections between layers
    for i in range(len(network.hidden_layer.weights)):
        for j in range(len(network.output_layer.weights)):
            ax_nn.plot([0, 1], [i, j], color='gray')

    ax_nn.legend()
    canvas_nn.draw()

    # Redraw the loss graph
    canvas.draw()


# Updated start_training function with the corrected update_progress call
def start_training(activation_var, cost_var, epochs_var, learning_rate_var, progress_label, training_label, canvas, ax,
                   canvas_nn, ax_nn):
    activation_fn, activation_deriv_fn, cost_fn = get_selected_functions(activation_var, cost_var)

    # Initialize the network
    input_size = X.shape[1]
    hidden_layer_width = 64
    num_outputs = 1
    learning_rate = float(learning_rate_var.get())
    num_epochs = int(epochs_var.get())

    network = Network(input_size, hidden_layer_width, num_outputs, activation_fn, activation_deriv_fn, cost_fn)

    # List to store cost history for plotting
    costs = []

    # Function to update the progress label during training
    def progress_callback(epoch, cost):
        update_progress(epoch, cost, ax, canvas, costs, progress_label, network, ax_nn, canvas_nn)

    # Update the training label
    training_label.config(text="Training...")
    training_label.update()

    # Train the network
    network.train(X.values, y.values, num_epochs, learning_rate, progress_callback)

    # After training, reset training label
    training_label.config(text="Training Completed!")
    training_label.update()


# Create the GUI

def create_ui():
    root = tk.Tk()
    root.title("Neural Network Trainer")

    # Activation Function Selection
    activation_label = tk.Label(root, text="Select Activation Function:")
    activation_label.grid(row=0, column=0)
    activation_var = ttk.Combobox(root, values=["Sigmoid", "ReLU", "Tanh"])
    activation_var.set("Sigmoid")
    activation_var.grid(row=0, column=1)

    # Cost Function Selection
    cost_label = tk.Label(root, text="Select Cost Function:")
    cost_label.grid(row=1, column=0)
    cost_var = ttk.Combobox(root, values=["Mean Squared Error", "Binary Cross-Entropy"])
    cost_var.set("Mean Squared Error")
    cost_var.grid(row=1, column=1)

    # Number of Epochs Input
    epochs_label = tk.Label(root, text="Number of Epochs:")
    epochs_label.grid(row=2, column=0)
    epochs_var = tk.Entry(root)
    epochs_var.insert(0, "1000")
    epochs_var.grid(row=2, column=1)

    # Learning Rate Input
    learning_rate_label = tk.Label(root, text="Learning Rate:")
    learning_rate_label.grid(row=3, column=0)
    learning_rate_var = tk.Entry(root)
    learning_rate_var.insert(0, "0.01")
    learning_rate_var.grid(row=3, column=1)

    # Progress Label
    progress_label = tk.Label(root, text="Training Progress")
    progress_label.grid(row=4, column=0, columnspan=2)

    # Training Label
    training_label = tk.Label(root, text="Ready to Train")
    training_label.grid(row=5, column=0, columnspan=2)

    # Create the plot for loss visualization
    fig, ax = plt.subplots(figsize=(6, 4))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=7, column=0, columnspan=2)
    ax.axis('off')  # Hide axes

    # Create the plot for neural network architecture visualization
    fig_nn, ax_nn = plt.subplots(figsize=(6, 4))
    canvas_nn = FigureCanvasTkAgg(fig_nn, master=root)
    canvas_nn.get_tk_widget().grid(row=8, column=0, columnspan=2)
    ax_nn.axis('off')  # Hide axes

    # Train Button
    train_button = tk.Button(root, text="Start Training",
                             command=lambda: start_training(
                                 activation_var, cost_var, epochs_var, learning_rate_var,
                                 progress_label, training_label, canvas, ax, canvas_nn, ax_nn))
    train_button.grid(row=6, column=0, columnspan=2)

    # Start the GUI loop
    root.mainloop()


# Main function to run the UI
def main():
    create_ui()


if __name__ == "__main__":
    main()