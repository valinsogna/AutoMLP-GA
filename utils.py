# utils.py
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_fitness_over_generations(distributions, result_dir, generations):
    avg_fitness = distributions['avg_fitness_over_generations']
    best_fitness = distributions['best_fitness_over_generations']
    
    plt.figure(figsize=(10, 5))
    plt.scatter(range(1, generations + 1), avg_fitness, label='Average Fitness')
    plt.plot(range(1, generations + 1), avg_fitness)
    plt.scatter(range(1, generations + 1), best_fitness, label='Best Fitness')
    plt.plot(range(1, generations + 1), best_fitness)
    plt.xticks(range(1, generations + 1))  # Set x-axis ticks starting from 1
    plt.xlabel('Generations')
    plt.ylabel('Fitness (Accuracy)')
    plt.title('Fitness over Generations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'fitness_over_generations.png'))

def save_network_architectures(networks, result_dir):
    with open(os.path.join(result_dir, 'network_architectures.txt'), 'w') as f:
        for network in networks:
            f.write(str(network.network) + '\n')
            f.write(str(network.accuracy) + '\n')
            f.write('-'*80 + '\n')

def save_fitness_distributions(distributions, result_dir):
    for key, value in distributions.items():
        with open(os.path.join(result_dir, f'{key}.pkl'), 'wb') as f:
            pickle.dump(value, f)

def plot_all_distributions(distributions, result_dir, generations):
    for key, value in distributions.items():
        if key not in ['avg_fitness_over_generations', 'best_fitness_over_generations', 'fitness_values_each_generation', 'diversity']:
            plot_bar_chart(
                distributions=value,
                title=f'Distribution of {key.capitalize()} Over Generations',
                xlabel=key.capitalize().replace('_', ' '),
                ylabel='Frequency',
                filename=os.path.join(result_dir, f'{key}.png'),
                generations=generations
            )

# Plot function used in the utils module
def plot_bar_chart(distributions, title, xlabel, ylabel, filename, generations):
    plt.figure(figsize=(10, 5))
    categories = sorted(set(sum(distributions, [])))  # Get unique sorted values
    category_indices = np.arange(len(categories))

    bar_width = 0.8 / generations  # Dynamically size the bars based on number of generations

    for i, distribution in enumerate(distributions):
        counts = [distribution.count(category) for category in categories]
        plt.bar(category_indices + i*bar_width, counts, width=bar_width, label=f'Gen {i+1}')

    plt.xticks(category_indices + bar_width / 2, categories)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)


def save_and_plot_results(networks, distributions, result_dir, generations):
    # Save network architectures and fitness to file
    save_network_architectures(networks, result_dir)
    # Save fitness distributions to file
    save_fitness_distributions(distributions, result_dir)
    if 'avg_fitness_over_generations' in distributions and 'best_fitness_over_generations' in distributions:
        plot_fitness_over_generations(distributions, result_dir, generations)
    if 'diversity' in distributions:
        plt.figure(figsize=(10, 5))
        plt.scatter(range(1, len(distributions['diversity']) + 1), distributions['diversity'], label='Diversity (Std Dev of Fitness)')
        plt.plot(range(1, len(distributions['diversity']) + 1), distributions['diversity'])
        plt.xticks(range(1, len(distributions['diversity']) + 1))  # Set x-axis ticks starting from 1
        plt.xlabel('Generations')
        plt.ylabel('Diversity')
        plt.title('Diversity of Population over Generations')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'diversity_over_generations.png'))
    # Plot all distributions
    plot_all_distributions(distributions, result_dir, generations)