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
    def __init__(self, input_size, hidden_layer_sizes, num_outputs, activation_fn, activation_deriv_fn, cost_fn):
        self.layers = []
        self.activation_fn = activation_fn
        self.activation_deriv_fn = activation_deriv_fn
        self.cost_fn = cost_fn

        # Initialize hidden layers dynamically
        layer_sizes = [input_size] + hidden_layer_sizes
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Neuron(layer_sizes[i], layer_sizes[i + 1], activation_fn, activation_deriv_fn))

        # Add output layer
        self.layers.append(Neuron(hidden_layer_sizes[-1], num_outputs, sigmoid, sigmoid_derivative))

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, X, y, output):
        error = output - y
        d_output = error
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update_weights(learning_rate)

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
        return np.array([self.forward(x.reshape(1, -1)) for x in X])



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

def calculate_accuracy(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5).astype(int)  # Convert predictions to binary (0 or 1)
    return np.mean(y_true == y_pred_binary) * 100  # Accuracy in percentage


# Function to update the progress label, plot loss, and visualize the neural network

def update_progress(epoch, cost, accuracy, ax, canvas, costs, progress_label, network, ax_nn, canvas_nn):
    progress_label.config(text=f"Epoch {epoch} - Cost: {cost:.4f} - Accuracy: {accuracy:.2f}%")
    progress_label.update()

    # Update loss graph
    costs.append(cost)
    ax.clear()
    ax.set_title("Neural Network Loss")
    ax.plot(costs, label="Loss", color="blue")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Cost")
    ax.legend()
    canvas.draw()



    # Update neural network visualization
    # ax_nn.clear()
    ax_nn.set_title("Neural Network Architecture")


    num_layers = len(network.layers)
    max_nodes = max(layer.weights.shape[0] for layer in network.layers)  # Max nodes for scaling

    sample_input = np.random.rand(1, network.layers[0].weights.shape[0])  # Random sample input
    activations = [sample_input]

    # Perform a forward pass to get activations for all layers
    for layer in network.layers:
        activations.append(layer.forward(activations[-1]))

    layer_labels = ["Input Layer"] + [f"Hidden Layer {i + 1}" for i in range(len(network.layers) - 1)] + ["Output Layer"]

    # Visualize layers and activations
    for layer_idx, (layer, activation) in enumerate(zip(network.layers, activations)):
        num_nodes = layer.weights.shape[0]
        x = [layer_idx] * num_nodes
        y = np.linspace(0, max_nodes, num_nodes)
        activation_values = activation.flatten()

        # Determine colormap based on layer type
        cmap = "Blues" if layer_idx == 0 else "Greens" if layer_idx < num_layers - 1 else "Reds"
        scatter = ax_nn.scatter(x, y, s=500, c=activation_values, cmap=cmap, vmin=0, vmax=1)

        # Annotate nodes with activation values
        for i, act_val in enumerate(activation_values):
            ax_nn.text(x[i], y[i], f"{act_val:.2f}", ha='center', va='center', fontsize=8, color="white")

        # Draw connections to next layer
        if layer_idx < num_layers - 1:
            next_num_nodes = network.layers[layer_idx + 1].weights.shape[0]
            next_y = np.linspace(0, max_nodes, next_num_nodes)
            for i in range(num_nodes):
                for j in range(next_num_nodes):
                    ax_nn.plot([layer_idx, layer_idx + 1], [y[i], next_y[j]], color="gray", alpha=0.5)

    ax_nn.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label="Input Layer"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label="Hidden Layers"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label="Output Layer"),
    ], loc="lower center", fontsize=10, ncol=3, bbox_to_anchor=(0.5, -0.15))
    canvas_nn.draw()





# Updated start_training function with the corrected update_progress call
def start_training(activation_var, cost_var, epochs_var, learning_rate_var, hidden_layers_var,
                   progress_label, training_label, canvas, ax, canvas_nn, ax_nn):
    activation_fn, activation_deriv_fn, cost_fn = get_selected_functions(activation_var, cost_var)

    # Parse hidden layers from input
    hidden_layers = list(map(int, hidden_layers_var.get().split(',')))

    # Initialize the network
    input_size = X.shape[1]
    num_outputs = 1
    learning_rate = float(learning_rate_var.get())
    num_epochs = int(epochs_var.get())

    network = Network(input_size, hidden_layers, num_outputs, activation_fn, activation_deriv_fn, cost_fn)

    costs = []

    def progress_callback(epoch, cost):
        predictions = network.test_network(X.values).flatten()
        accuracy = calculate_accuracy(y.values, predictions)
        update_progress(epoch, cost, accuracy, ax, canvas, costs, progress_label, network, ax_nn, canvas_nn)


    training_label.config(text="Training...")
    network.train(X.values, y.values, num_epochs, learning_rate, progress_callback)
    training_label.config(text="Training Completed!")


# Create the GUI
def create_ui():
    root = tk.Tk()
    root.title("Neural Network Trainer")
    root.state('zoomed')  # Fullscreen window

    # Activation Function Selection
    activation_label = tk.Label(root, text="Select Activation Function:")
    activation_label.grid(row=0, column=0, padx=10, pady=5)
    activation_var = ttk.Combobox(root, values=["Sigmoid", "ReLU", "Tanh"])
    activation_var.set("Sigmoid")
    activation_var.grid(row=0, column=1, padx=10, pady=5)

    # Cost Function Selection
    cost_label = tk.Label(root, text="Select Cost Function:")
    cost_label.grid(row=1, column=0, padx=10, pady=5)
    cost_var = ttk.Combobox(root, values=["Mean Squared Error", "Binary Cross-Entropy"])
    cost_var.set("Mean Squared Error")
    cost_var.grid(row=1, column=1, padx=10, pady=5)

    # Number of Hidden Layers Input
    hidden_layers_label = tk.Label(root, text="Hidden Layers (comma-separated), Output Layer:")
    hidden_layers_label.grid(row=2, column=0, padx=10, pady=5)
    hidden_layers_var = tk.Entry(root)
    hidden_layers_var.insert(0, "5,7,8")  # Default configuration
    hidden_layers_var.grid(row=2, column=1, padx=10, pady=5)

    # Number of Epochs Input
    epochs_label = tk.Label(root, text="Number of Epochs:")
    epochs_label.grid(row=3, column=0, padx=10, pady=5)
    epochs_var = tk.Entry(root)
    epochs_var.insert(0, "1000")
    epochs_var.grid(row=3, column=1, padx=10, pady=5)

    # Learning Rate Input
    learning_rate_label = tk.Label(root, text="Learning Rate:")
    learning_rate_label.grid(row=4, column=0, padx=10, pady=5)
    learning_rate_var = tk.Entry(root)
    learning_rate_var.insert(0, "0.01")
    learning_rate_var.grid(row=4, column=1, padx=10, pady=5)

    # Progress Label
    progress_label = tk.Label(root, text="Training Progress")
    progress_label.grid(row=5, column=0, columnspan=2)

    # Training Label
    training_label = tk.Label(root, text="Ready to Train")
    training_label.grid(row=6, column=0, columnspan=2)

    # Loss Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=7, column=0, columnspan=2)
    ax.axis('off')

    # NN Architecture Plot

    fig_nn, ax_nn = plt.subplots(figsize=(12, 6))
    canvas_nn = FigureCanvasTkAgg(fig_nn, master=root)
    canvas_nn.get_tk_widget().grid(row=8, column=0, columnspan=2)
    ax_nn.axis('off')

    # Train Button
    train_button = tk.Button(
        root, text="Start Training",
        command=lambda: start_training(
            activation_var, cost_var, epochs_var, learning_rate_var, hidden_layers_var,
            progress_label, training_label, canvas, ax, canvas_nn, ax_nn
        )
    )
    train_button.grid(row=4, column=0, columnspan=2)

    root.mainloop()



# Main function to run the UI
def main():
    create_ui()


if __name__ == "__main__":
    main()