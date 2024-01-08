# AutoMLP-GA
Automates the optimization of the topology of Multi-Layer Perceptrons using Genetic Algorithms in PyTorch for classification on MNIST dataset 

## Overview
### Tuned Parameters
AutoMLP-GA evolves MLP architectures by tuning the following parameters:
- **Number of Neurons:** Adjusts the number of neurons in each layer.
- **Number of Layers:** Determines the depth of the network by varying the number of hidden layers.
- **Activation Functions:** Experiments with different activation functions (e.g., ReLU, Tanh, Sigmoid, LeakyReLU) for each layer.
- **Initial Learning Rate:** Optimizes the initial learning rate for the training algorithm.
- **Optimizer Type:** Selects the type of optimizer, such as Adam, SGD, AdamW and RMSprop.
- **Learning Rate decay** Selects the type of learning rate decay, such as LinearLR, CosineAnnelaing and ExponentialLR.

These parameters are crucial in defining the structure and learning capability of the MLPs. By evolving these aspects, AutoMLP-GA aims to discover the most effective network configurations for specific datasets and tasks.

## Installation

To set up the AutoGAMLP tool, follow these steps:

1. **Clone the Repository:**
```python
git clone https://github.com/yourusername/AutoMLP-GA.git
cd AutoMLP-GA
```
2. **Install Python:**
Ensure that Python (version 3.6 or higher) is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

3. **Install Dependencies:**
AutoGAMLP requires PyTorch and a few other libraries. Install them using pip:
```python
pip install torch torchvision tqdm
```

## Usage

Using AutoGAMLP involves running the `main.py` script, which orchestrates the process of evolving neural networks using genetic algorithms. Here’s how to use it:

1. **Configure Parameters:**
Before running `main.py`, you might want to configure certain parameters in the script or in a separate configuration file, such as the number of generations, population size, and network parameter choices.

2. **Run the Evolution Process:**
Execute the main script to start the evolution process:
```python
python main.py
```
## Composition

The `Network` class in `network.py` is designed to define and handle a neural network based on the parameters provided (like the number of neurons, number of layers, activation function, etc.). The `Optimizer` class in `optimizer.py` manages the evolutionary process, creating a population of these networks, breeding them, mutating them, and selecting the fittest networks over generations.

### To ensure full compatibility, here's a quick recap of how they work together:

### Network Initialization and Random Creation:
- In `network.py`, the `Network` class can create a random network configuration using `create_random`.
- In `optimizer.py`, the `create_population` method uses `Network` to create an initial population of randomly configured networks.

### Breeding and Mutation:
- The `breed` and `mutate` methods in `optimizer.py` handle the crossover and mutation of networks. After these operations, the network's structure (like layers, neurons, activation functions) is updated.
- The `Network` class has methods (`create_set` and `create_network`) to update its structure based on these new configurations.

### Fitness Evaluation:
- The `fitness` method in `optimizer.py` evaluates networks based on their accuracy, which is set and updated in the `Network` class during training.

### Training:
- Training is handled outside of these classes, typically in a script like `train.py`, where each network is trained, and its performance (accuracy) is evaluated.

## GA specifications
### Selection
The selection method used in the `Optimizer` class is a combination of Truncation Selection and Random Selection:
- **Truncation Selection:** After each generation, a certain percentage of the best-performing networks (as determined by their fitness, which in this case is accuracy) are retained for the next generation. This is controlled by the `retain` attribute.
- **Random Selection:** In addition to the top performers, there is also a chance (`random_select` probability) to select some networks that are not among the top performers. This method introduces diversity into the gene pool, preventing premature convergence to a local optimum.

### Crossover
The crossover method in `breed` function seems to implement a form of Uniform Crossover:
- **Uniform Crossover:** In this method, for each network parameter (like the number of neurons, layers, activation functions), the child network randomly inherits the value of that parameter from either one of its two parents. This method treats each gene (parameter) independently and gives equal chance for a gene to be inherited from either parent.

### Mutation
The mutation method in the `mutate` function is a basic form of Random Mutation:
- **Random Mutation:** Here, a network is chosen to undergo mutation with a certain probability (`mutate_chance`). When a mutation occurs, one of the network's parameters is randomly selected and then randomly altered (by picking a new value for that parameter from the available choices).

