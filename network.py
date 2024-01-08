"""Class that represents the network to be evolved."""
import random
import logging
import torch.nn as nn
from train import train_and_score


class Network():
    """Represent a network and let us operate on it.

    This is designed for a simple feedforward neural network.
    """

    def __init__(self, nn_param_choices=None):
        """Initialize our network.

        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [128, 256, 512, 768, 1024]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['ReLU', 'ELU', 'Tanh']
                optimizer (list): ['adamw']
        """
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dict): represents MLP network parameters
        self.model = None

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])
        self.create_network()

    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters
        """
        self.network = network
        self.create_network()

    def create_network(self):
        """Construct a PyTorch network from our parameters."""
        layers = []
        input_size = 784  # MNIST images are 28x28 pixels

        # Creating layers based on the network parameters
        for i in range(self.network['nb_layers']):
            layers.append(nn.Linear(input_size, self.network['nb_neurons']))
            layers.append(getattr(nn, self.network['activation'])())
            layers.append(nn.Dropout(0.2))  # Fixed dropout value
            input_size = self.network['nb_neurons']

        # Output layer
        layers.append(nn.Linear(self.network['nb_neurons'], 10))  # 10 classes for MNIST

        self.model = nn.Sequential(*layers)

    def train(self, dataset, debug=False):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        Note: The actual training process should be implemented here or called from here.
        This placeholder does not actually train the model.
        """
        if self.accuracy == 0.:
            # Implement training logic here
            self.accuracy = train_and_score(self, dataset, debug=debug)
        

    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))

# Helper function to get the PyTorch activation function
def get_activation(name):
    return getattr(nn, name)

class DebugLayer(nn.Module):
    def __init__(self, layer, name=""):
        super().__init__()
        self.layer = layer
        self.name = name

    def forward(self, x):
        print(f"Input shape to {self.name}: {x.shape}")
        x = self.layer(x)
        print(f"Output shape from {self.name}: {x.shape}")
        return x