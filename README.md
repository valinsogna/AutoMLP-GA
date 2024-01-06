# AutoMLP-GA
Automates the optimization of the topology of Multi-Layer Perceptrons using Genetic Algorithms in PyTorch for classification on MNIST dataset 

## Overview
### Tuned Parameters
AutoMLP-GA evolves MLP architectures by tuning the following parameters:
- **Number of Neurons:** Adjusts the number of neurons in each layer.
- **Number of Layers:** Determines the depth of the network by varying the number of hidden layers.
- **Activation Functions:** Experiments with different activation functions (e.g., ReLU, Tanh, Sigmoid, LeakyReLU) for each layer.
- **Learning Rate:** (If included in parameter choices) Optimizes the learning rate for the training algorithm.
- **Optimizer Type:** (If included in parameter choices) Selects the type of optimizer, such as Adam, SGD, etc.
- **Learning Rate decay** 

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
