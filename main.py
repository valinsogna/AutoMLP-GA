"""Entry point to evolving the neural network. Start here."""
import logging
from optimizer import Optimizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np
import argparse
import pickle
from utils import save_and_plot_results

# Create a unique directory to save results
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
result_dir = os.path.join("results", f"run_{timestamp}")
os.makedirs(result_dir, exist_ok=True)

# Setup logging inside the generate function after result_dir is created
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

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.
    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def generate(generations, population, nn_param_choices, dataset, debug=False):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating
    """
    # Data collection for plots
    avg_fitness_over_generations = []
    best_fitness_over_generations = []
    fitness_values_each_generation = []  # For diversity plot
    neuron_distribution = []  # parameter distribution plot
    layer_distribution = []  # parameter distribution plot
    activation_distribution = []  # parameter distribution plot
    optimizer_distribution = []  # parameter distribution plot
    lr_scheduler_distribution = []  # parameter distribution plot
    initial_lr_distribution = []  # parameter distribution plot

    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population, debug=debug)

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" % (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, dataset, debug=debug)

        # Record data for plots
        average_accuracy = get_average_accuracy(networks)
        avg_fitness_over_generations.append(average_accuracy)

        best_network = max(networks, key=lambda net: net.accuracy)
        best_fitness_over_generations.append(best_network.accuracy)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)

        # Collect data for plots
        fitness_values_each_generation.append([net.accuracy for net in networks])

        neuron_distribution.append([net.network['nb_neurons'] for net in networks])
        layer_distribution.append([net.network['nb_layers'] for net in networks])
        activation_distribution.append([net.network['activation'] for net in networks])
        optimizer_distribution.append([net.network['optimizer'] for net in networks])
        lr_scheduler_distribution.append([net.network['lr_scheduler'] for net in networks])
        initial_lr_distribution.append([net.network['initial_lr'] for net in networks])

        # Evolve, except on the last iteration.
        if i != generations - 1:
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    diversity = [np.std(fitness) for fitness in fitness_values_each_generation]

    distributions = {
        'avg_fitness_over_generations': avg_fitness_over_generations,
        'best_fitness_over_generations': best_fitness_over_generations,
        'fitness_values_each_generation': fitness_values_each_generation,
        'diversity': diversity,
        'neuron_distribution': neuron_distribution,
        'layer_distribution': layer_distribution,
        'activation_distribution': activation_distribution,
        'optimizer_distribution': optimizer_distribution,
        'lr_scheduler_distribution': lr_scheduler_distribution,
        'initial_lr_distribution': initial_lr_distribution
    }

    # External function to save results and plot distributions
    save_and_plot_results(networks, distributions, result_dir, generations)
    
    # Print out the top 10% networks.
    if len(networks) < 10:
        print_networks(networks)
    else:
        print_networks(networks[:int(len(networks)/10)])



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

def main():
    """Evolve a network."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evolve a network.')
    parser.add_argument('--gen', type=int, default=10, help='Number of generations to evolve.')
    parser.add_argument('--pop', type=int, default=20, help='Population size in each generation.')
    args = parser.parse_args()

    generations = args.gen  # Number of times to evolve the population.
    population = args.pop  # Number of networks in each generation.
    dataset = 'mnist'  # Use MNIST dataset

    nn_param_choices = {
        'nb_neurons': [128, 256, 384, 512, 640, 768, 896, 1024],
        'nb_layers': [1, 2, 3, 4, 5, 6],
        'activation': ['ReLU', 'ELU', 'Tanh', 'LeakyReLU', 'Sigmoid'],
        'optimizer': ['adam', 'adamw', 'sgd', 'rmsprop'],
        'lr_scheduler': ['cosine', 'exponential', 'linear', 'none'],
        'initial_lr': [0.1, 0.01, 0.001, 0.0003, 0.0001],
    }

    logging.info("Evolution in %d generations with population %d" % (generations, population))

    generate(generations, population, nn_param_choices, dataset, debug=False)


if __name__ == '__main__':
    main()
