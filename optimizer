"""
Class that holds a genetic algorithm for evolving a network.
"""
from functools import reduce
from operator import add
import random
from network import Network

class Optimizer():
    """Class that implements genetic algorithm for MLP optimization."""

    def __init__(self, nn_param_choices, retain=0.4,
                 random_select=0.1, mutate_chance=0.2):
        """Create an optimizer.

        Args:
            nn_param_choices (dict): Possible network parameters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated
        """
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices

    def create_population(self, count):
        """Create a population of random networks.

        Args:
            count (int): Number of networks to generate, aka the
                size of the population

        Returns:
            (list): Population of network objects
        """
        pop = []
        for _ in range(count):
            network = Network(self.nn_param_choices)
            network.create_random()
            pop.append(network)

        return pop

    def fitness(self, network):
        """Return the accuracy, which is our fitness function."""
        return network.accuracy

    def grade(self, pop):
        """Find average fitness for a population.

        Args:
            pop (list): The population of networks

        Returns:
            (float): The average accuracy of the population
        """
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float(len(pop))

    def breed(self, mother, father):
        """Make two children from parts of their parents.

        Args:
            mother (Network): Network object
            father (Network): Network object

        Returns:
            (list): Two network objects
        """
        children = []
        for _ in range(2):

            child = {}

            for param in self.nn_param_choices:
                child[param] = random.choice(
                    [mother.network[param], father.network[param]]
                )

            network = Network(self.nn_param_choices)
            network.create_set(child)

            if self.mutate_chance > random.random():
                network = self.mutate(network)

            children.append(network)

        return children

    def mutate(self, network):
        """Randomly mutate one part of the network.

        Args:
            network (Network): The network to mutate

        Returns:
            (Network): A mutated network object
        """
        mutation = random.choice(list(self.nn_param_choices.keys()))
        network.network[mutation] = random.choice(self.nn_param_choices[mutation])
        network.create_network()  # Rebuild the network with new mutation
        return network

    def evolve(self, pop):
        """Evolve a population of networks.

        Args:
            pop (list): A list of network objects

        Returns:
            (list): The evolved population of networks
        """
        graded = [(self.fitness(network), network) for network in pop]
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        retain_length = int(len(graded) * self.retain)
        parents = graded[:retain_length]

        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        while len(children) < desired_length:
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)

            if male != female:
                male = parents[male]
                female = parents[female]
                babies = self.breed(male, female)

                for baby in babies:
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)
        return parents
