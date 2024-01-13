import logging
import argparse
import random
import os
import datetime
from tqdm import tqdm
from network import Network
from utils import save_and_plot_results

# Setup logging
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
result_dir = os.path.join("results", f"run_{timestamp}")
os.makedirs(result_dir, exist_ok=True)
log_file = os.path.join(result_dir, 'log.txt')
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    filename=log_file
)

def train_networks(networks, dataset, debug=False):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        if debug:
            print(f"Training network {networks.index(network)+1}/{len(networks)}")
        network.train(dataset, debug=debug)
        pbar.update(1)
    pbar.close()

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks
    """
    logging.info('-'*80)
    logging.info("Printing top 10% networks:")
    #print(len(networks))
    #print(networks)
    for network in networks:
        network.print_network()

def generate_random_networks(num_networks, nn_param_choices):
    """Generate a specified number of random networks.

    Args:
        num_networks (int): Number of networks to generate.
        nn_param_choices (dict): The parameter choices

    Returns:
        networks (list): A list of network objects
    """
    networks = []
    neuron_distribution = []
    layer_distribution = []
    activation_distribution = []
    optimizer_distribution = []
    lr_scheduler_distribution = []
    initial_lr_distribution = []
    dropout_distribution = []
    batch_size_distribution = []

    for _ in range(num_networks):
        network = Network(nn_param_choices)
        network.create_random()

        # Collect parameter distributions
        neuron_distribution.append(network.network['nb_neurons'])
        layer_distribution.append(network.network['nb_layers'])
        activation_distribution.append(network.network['activation'])
        optimizer_distribution.append(network.network['optimizer'])
        lr_scheduler_distribution.append(network.network['lr_scheduler'])
        initial_lr_distribution.append(network.network['initial_lr'])
        dropout_distribution.append(network.network['dropout'])
        batch_size_distribution.append(network.network['batch_size'])

        networks.append(network)

        distributions = {
            'neuron_distribution': [neuron_distribution],
            'layer_distribution': [layer_distribution],
            'activation_distribution': [activation_distribution],
            'optimizer_distribution': [optimizer_distribution],
            'lr_scheduler_distribution': [lr_scheduler_distribution],
            'initial_lr_distribution': [initial_lr_distribution],
            'dropout_distribution': [dropout_distribution],
            'batch_size_distribution': [batch_size_distribution]
        }
        save_and_plot_results(networks, distributions, result_dir, 1)  # '1' for a single generation

    return networks

def main():
    parser = argparse.ArgumentParser(description='Train a set of random networks.')
    parser.add_argument('--num_networks', type=int, required=True, help='Number of networks to generate and train.')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use for training.')
    args = parser.parse_args()

    nn_param_choices = {
        'nb_neurons': [128, 256, 384, 512, 640, 768, 896, 1024],
        'nb_layers': [1, 2, 3, 4, 5, 6],
        'activation': ['ReLU', 'ELU', 'Tanh', 'LeakyReLU', 'Sigmoid'],
        'optimizer': ['adam', 'adamw', 'sgd', 'rmsprop'],
        'lr_scheduler': ['cosine', 'exponential', 'linear', 'none'],
        'initial_lr': [0.1, 0.01, 0.001, 0.0003, 0.0001],
        'batch_size': [32, 64, 128, 256],
        'dropout': [0, 0.1, 0.2, 0.3, 0.4]
    }

    networks = generate_random_networks(args.num_networks, nn_param_choices)

    train_networks(networks, args.dataset)

    print_networks(networks)


if __name__ == '__main__':
    main()