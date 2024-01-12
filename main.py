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

    # Save network architectures to file in txt followd by the fitness value
    with open(os.path.join(result_dir, 'network_architectures.txt'), 'w') as f:
        for network in networks:
            f.write(str(network.network) + '\n')
            f.write(str(network.accuracy) + '\n')
            f.write('-'*80 + '\n')
    
    # Print out the top 10% networks.
    if len(networks) < 10:
        print_networks(networks)
    else:
        print_networks(networks[:int(len(networks)/10)])

    diversity = [np.std(fitness) for fitness in fitness_values_each_generation]

    # Save results to file in pickle format
    with open(os.path.join(result_dir, 'avg_fitness_over_generations.pkl'), 'wb') as f:
        pickle.dump(avg_fitness_over_generations, f)

    with open(os.path.join(result_dir, 'best_fitness_over_generations.pkl'), 'wb') as f:
        pickle.dump(best_fitness_over_generations, f)

    with open(os.path.join(result_dir, 'fitness_values_each_generation.pkl'), 'wb') as f:
        pickle.dump(fitness_values_each_generation, f)

    with open(os.path.join(result_dir, 'neuron_distribution.pkl'), 'wb') as f:
        pickle.dump(neuron_distribution, f)

    with open(os.path.join(result_dir, 'layer_distribution.pkl'), 'wb') as f:
        pickle.dump(layer_distribution, f)

    with open(os.path.join(result_dir, 'activation_distribution.pkl'), 'wb') as f:
        pickle.dump(activation_distribution, f)

    with open(os.path.join(result_dir, 'optimizer_distribution.pkl'), 'wb') as f:
        pickle.dump(optimizer_distribution, f)

    with open(os.path.join(result_dir, 'lr_scheduler_distribution.pkl'), 'wb') as f:
        pickle.dump(lr_scheduler_distribution, f)

    with open(os.path.join(result_dir, 'initial_lr_distribution.pkl'), 'wb') as f:
        pickle.dump(initial_lr_distribution, f)


    # Create bins for histograms
    neuron_bins = create_bins(networks[0].nn_param_choices['nb_neurons'])
    layer_bins = create_bins(networks[0].nn_param_choices['nb_layers'])
    initial_lr_bins = create_bins(networks[0].nn_param_choices['initial_lr'] , bin_width=0.0001) 

    # Plot Fitness vs. Generations
    plt.figure(figsize=(10, 5))
    plt.scatter(range(1, generations + 1), avg_fitness_over_generations, label='Average Fitness')
    plt.plot(range(1, generations + 1), avg_fitness_over_generations)
    plt.scatter(range(1, generations + 1), best_fitness_over_generations, label='Best Fitness')
    plt.plot(range(1, generations + 1), best_fitness_over_generations)
    plt.xticks(range(1, generations + 1))  # Set x-axis ticks starting from 1
    plt.xlabel('Generations')
    plt.ylabel('Fitness (Accuracy)')
    plt.title('Fitness over Generations')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'fitness_over_generations.png'))

    # Plot Diversity of Population
    plt.figure(figsize=(10, 5))
    plt.scatter(range(1, generations + 1), diversity, label='Diversity (Std Dev of Fitness)')
    plt.plot(range(1, generations + 1), diversity)
    plt.xticks(range(1, generations + 1))  # Set x-axis ticks starting from 1
    plt.xlabel('Generations')
    plt.ylabel('Diversity')
    plt.title('Diversity of Population over Generations')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'diversity_over_generations.png'))

    # Plot for Neuron Distribution
    plot_histogram(
        distributions=neuron_distribution,
        bins=neuron_bins,
        title='Distribution of Neurons Over Generations',
        xlabel='Number of Neurons',
        ylabel='Frequency',
        filename=os.path.join(result_dir, 'neuron_distribution.png')

    )

    # Plot for Layer Distribution
    plot_histogram(
        distributions=layer_distribution,
        bins=layer_bins,
        title='Distribution of Layers Over Generations',
        xlabel='Number of Layers',
        ylabel='Frequency',
        filename=os.path.join(result_dir, 'layer_distribution.png')
    )


    # Plot Activation Function Distribution
    plt.figure()
    for index, generation in enumerate(activation_distribution):
        plt.hist(generation, alpha=0.5, label=f'Gen {index+1}')
    plt.xlabel('Activation Function')
    plt.ylabel('Frequency')
    plt.title('Distribution of Activation Functions Over Generations')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'activation_distribution.png'))
    

    # Plot Optimizer Distribution
    plt.figure()
    for index, generation in enumerate(optimizer_distribution):
        plt.hist(generation, alpha=0.5, label=f'Gen {index+1}')
    plt.xlabel('Optimizer')
    plt.ylabel('Frequency')
    plt.title('Distribution of Optimizers Over Generations')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'optimizer_distribution.png'))

    # Plot Learning Rate Scheduler Distribution
    plt.figure()
    for index, generation in enumerate(lr_scheduler_distribution):
        plt.hist(generation, alpha=0.5, label=f'Gen {index+1}')
    plt.xlabel('Learning Rate Scheduler')
    plt.ylabel('Frequency')
    plt.title('Distribution of Learning Rate Schedulers Over Generations')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'lr_scheduler_distribution.png'))

    # Plot Initial Learning Rate Distribution
    plt.figure()
    for index, generation in enumerate(initial_lr_distribution):
        plt.hist(generation, alpha=0.5, label=f'Gen {index+1}')
    plt.xlabel('Initial Learning Rate')
    plt.ylabel('Frequency')
    plt.title('Distribution of Initial Learning Rates Over Generations')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'initial_lr_distribution.png'))


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
        'batch_size': [32, 64, 128, 256],
        'dropout': [0, 0.1, 0.2, 0.3, 0.4]
    }

    logging.info("Evolution in %d generations with population %d" % (generations, population))

    generate(generations, population, nn_param_choices, dataset, debug=False)

# Function to create bins centered around the given values
def create_bins(centers, bin_width=None):
    if bin_width is not None:  # For initial learning rate or any other fixed width requirement
        bins = np.array(centers) - bin_width/2
        return np.append(bins, bins[-1] + bin_width)
    else:  # For neurons and layers where the bin width is not fixed
        bins = []
        for i in range(len(centers)-1):
            bins.append((centers[i] + centers[i+1]) / 2)
        return [centers[0] - (bins[0] - centers[0])] + bins + [centers[-1] + (centers[-1] - bins[-1])]

# Function to plot histogram with appropriate bins
def plot_histogram(distributions, bins, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 5))
    for i, distribution in enumerate(distributions):
        plt.hist(distribution, bins=bins, alpha=0.5, label=f'Gen {i+1}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)

if __name__ == '__main__':
    main()
