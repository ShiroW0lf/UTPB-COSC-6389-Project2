# Neural Network GUI

## Usage

### Step 1: Choose Activation and Cost Functions
In the GUI, you can choose:

- **Activation Function**: Sigmoid, ReLU, or Tanh.
- **Cost Function**: Mean Squared Error or Binary Cross-Entropy.

### Step 2: Set Hyperparameters
Set the following hyperparameters:

- **Number of Epochs**: The number of training iterations.
- **Learning Rate**: The rate at which the model adjusts its weights during training.
- **Hidden Layers**: Define the structure of hidden layers as a comma-separated list. For example, entering `4,10` will create:
  - Input Layer: Determined by the input feature size (X.shape[1]).
  - Hidden Layer 1: 4 nodes.
  - Output Layer: 10 nodes.
  - **Expected Visualization**: The Input Layer, Hidden Layer 1, and Output Layer should all appear. Connections should be properly drawn between each layer.

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
- **Hidden Layers**: Input box to define the number of neurons in the hidden layers, provided as a comma-separated list. 

### Training Controls:
- **Start Training Button**: Starts the training process, which will run in the background and update the UI in real-time.
- **Reset Button**: Resets the training parameters and clears the progress graphs for a fresh training run.

### Training Progress Display:
- **Epoch/Cost Label**: Displays the current epoch and the cost value (loss) during training.
- **Training Status**: A label to show the current status of the training (e.g., "Training in Progress").

### Loss Graph:
The loss graph updates in real-time to show how the cost value changes over the epochs. This helps in visualizing the model's performance and diagnosing issues like underfitting or overfitting.

### Neural Network Architecture Graph:
The architecture of the neural network is displayed visually. Layers are shown as nodes (circles), and the connections between them are represented as lines. The visualization will update as the training progresses.

# Neural Network Architecture Graph

The architecture of the neural network is displayed visually. Layers are shown as nodes (circles), and the connections between them are represented as lines. The visualization will update as the training progresses, showing the activation values of each node in real-time. Each layer is color-coded to differentiate between the **Input Layer**, **Hidden Layers**, and **Output Layer**.

## Neural Network Architecture

The neural network consists of:

- **Input Layer**: The input features of the dataset, displayed in **blue** with nodes representing the features.
- **Hidden Layers**: A user-defined number of hidden layers, each with a customizable number of neurons, displayed in **green**. The activations of each neuron in the hidden layers are shown as color gradients, where blue represents low activations and red represents high activations.
- **Output Layer**: The final output of the network, displayed in **red**. The activations of the output node are similarly represented with a color gradient.

### Neural Network Visualization

The neural network architecture is displayed in a dynamic plot that visually represents the structure and activations of the network:

- **Layer Labels**: Each layer is clearly labeled as "Input Layer," "Hidden Layer X," and "Output Layer."
- **Activation Visualization**: Each node’s color intensity corresponds to its activation value for a given input. Low activation is shown in **blue**, while high activation is shown in **red**.
- **Connections**: Lines between layers represent the weights connecting the neurons. The connections are drawn as subtle gray lines, showing how neurons in one layer are linked to those in the next layer.

### Insights from the Visualization

The neural network architecture provides insight into:

- **Layer Structure**: The input, hidden, and output layers are visually distinguished by color, helping to understand the flow of data through the network.
- **Neuron Activations**: The color gradients within the nodes display the activation levels of the neurons in real-time during training, indicating how information is processed at each layer.
- **Connections (Weights)**: The connections between layers represent the weights learned by the network, visually showing how neurons interact and influence each other.


## Training Loss Graph

The Training Loss Graph provides a dynamic view of how the loss function evolves during training:

- The x-axis represents the number of epochs.
- The y-axis represents the cost (loss) value at each epoch.

As the training progresses, the graph typically shows a decrease in the cost, indicating that the model is learning and improving its performance.

### Interpreting the Loss Graph:
- **Decreasing Loss**: A downward slope in the graph indicates that the model is improving, i.e., the error is decreasing over time.
- **Flat or Increasing Loss**: If the graph flattens or the loss increases, it could indicate that the model is not learning effectively, possibly due to a high learning rate or insufficient training time.

## Example Dataset

The neural network is initially configured to work with the **Banknote Authentication** dataset from UCI. This dataset is used for binary classification of banknote authenticity.

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

- UCI Machine Learning Repository for providing the dataset.
- Tkinter for the graphical interface.
