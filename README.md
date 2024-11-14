# Neural Network Trainer

This project implements a basic neural network from scratch using Python and Tkinter for the graphical user interface (GUI). The network supports binary and real-valued inputs and offers a variety of activation functions (Sigmoid, ReLU, Tanh) and cost functions (Mean Squared Error, Binary Cross-Entropy). The model can be trained on a dataset (e.g., the Banknote Authentication dataset from UCI) and visualizes the training progress in real-time.

## Features
- **Custom Neural Network**: Built from scratch using Python (without any specialized libraries like TensorFlow or PyTorch).
- **Activation Functions**: Supports Sigmoid, ReLU, and Tanh activation functions.
- **Cost Functions**: Includes Mean Squared Error and Binary Cross-Entropy cost functions.
- **Real-Time Training Visualization**: Real-time visualization of the neural network's loss progress during training.
- **GUI Interface**: Built using Tkinter, allowing users to easily interact with the neural network by configuring training parameters and visualizing the network architecture and loss graph.
- **Interactive Plots**: Graphical representations of the neural network architecture and the training loss graph, providing intuitive feedback during the training process.

## Requirements

Make sure you have Python 3.x installed on your system. You'll also need the following Python packages:

- `numpy`
- `pandas`
- `tkinter`
- `matplotlib`
- `ucimlrepo` (for dataset fetching)

You can install the required packages by running: 
pip install numpy pandas matplotlib ucimlrepo



### Key Sections:
1. **Features**: Overview of the core functionality of the project.
2. **Requirements**: Information about necessary libraries and installation steps.
3. **Installation**: Instructions on how to set up the project locally.
4. **Usage**: A guide on how to use the GUI and train the neural network.
5. **Neural Network Architecture**: Describes the model's structure and functionality.
6. **Example Dataset**: Description of the dataset used in the project.
7. **Future Enhancements**: Ideas for improving the project.
8. **License**: Licensing information (you can change this based on the license you choose).
9. **Acknowledgements**: Credit to third-party resources used in the project.

# Neural Network Trainer

## Usage

### Step 1: Choose Activation and Cost Functions

In the GUI, you can choose:

- **Activation Function**: Sigmoid, ReLU, or Tanh.
- **Cost Function**: Mean Squared Error or Binary Cross-Entropy.

### Step 2: Set Hyperparameters

Set the following hyperparameters:

- **Number of Epochs**: The number of training iterations.
- **Learning Rate**: The rate at which the model adjusts its weights during training.

### Step 3: Start Training

Click the **Start Training** button to begin the training process. The neural network will train and update in real-time, showing the current training progress (epoch and cost) and a graphical representation of the loss over epochs.

### Step 4: Monitor Training Progress

The following will be displayed during training:

- **Epoch**: The current epoch number.
- **Cost**: The current value of the loss function.
- **Loss Graph**: A plot of the loss during training, showing how the model is improving over time.

## UI Components

The GUI is built using Tkinter and provides an easy-to-use interface for interacting with the neural network model. The key components include:

### Activation and Cost Function Selection:

- Drop-down menus allow you to select the activation and cost functions that best suit your needs.

### Hyperparameter Inputs:

- **Number of Epochs**: Input box to specify the number of training epochs.
- **Learning Rate**: Input box to set the learning rate for model training.

### Training Controls:

- **Start Training Button**: Starts the training process, which will run in the background and update the UI in real-time.
- **Reset Button**: Resets the training parameters and clears the progress graphs for a fresh training run.

### Training Progress Display:

- **Epoch/Cost Label**: Displays the current epoch and the cost value (loss) during training.
- **Training Status**: A label to show the current status of the training (e.g., "Training in Progress").

### Loss Graph:

- The loss graph updates in real-time to show how the cost value changes over the epochs. This helps in visualizing the model's performance and diagnosing issues like underfitting or overfitting.

### Neural Network Architecture Graph:

- The architecture of the neural network is displayed visually. Layers are shown as nodes (circles), and the connections between them are represented as lines. This visualization helps users understand the structure of the network, including the number of neurons in each layer and how they are interconnected.

## Neural Network Architecture

The neural network consists of:

- **Input Layer**: The input features of the dataset.
- **Hidden Layer**: A layer with 64 neurons by default (can be adjusted).
- **Output Layer**: The output, which predicts a binary or real-valued output.

The architecture is visualized using a graph:

- **Input Layer** and **Output Layer** are shown as circles.
- Connections between layers are drawn as lines.

### Neural Network Visualization

The neural network architecture is displayed in a static plot that shows the structure of the network. Each layer's neurons are shown as nodes, with lines representing the connections between them.

The neural network architecture provides insight into how the network is structured, helping you visualize:

- The input and output layers.
- The number of neurons in the hidden layers.
- The connections (weights) between each layer.

## Training Loss Graph

The **Training Loss Graph** provides a dynamic view of how the loss function evolves during training:

- The **x-axis** represents the number of epochs.
- The **y-axis** represents the cost (loss) value at each epoch.
- As the training progresses, the graph typically shows a decrease in the cost, indicating that the model is learning and improving its performance.

### Interpreting the Loss Graph

- **Decreasing Loss**: A downward slope in the graph indicates that the model is improving, i.e., the error is decreasing over time.
- **Flat or Increasing Loss**: If the graph flattens or the loss increases, it could indicate that the model is not learning effectively, possibly due to a high learning rate or insufficient training time.

## Example Dataset

The neural network is initially configured to work with the **Banknote Authentication dataset** from UCI. This dataset is used for binary classification of banknote authenticity.

The dataset includes:

- **Features**: Variance, Skewness, Curtosis, Entropy.
- **Target**: Authenticity of the banknote (1 = authentic, 0 = not authentic).

You can modify the dataset or use your own data for training.

## Future Enhancements

- Support for different types of datasets.
- Additional activation and cost functions.
- Implementation of more complex neural network architectures (e.g., multiple hidden layers).
- Early stopping to avoid overfitting.
- Improved real-time loss graph visualization with interactive plots.


## Acknowledgements

- **UCI Machine Learning Repository** for providing the dataset.
- **Tkinter** for the graphical interface.
- **Matplotlib** for visualizing the loss graph and neural network architecture.

