import logging
import random
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())


class Creature(ABC):
    """
    Base class for all creatures. User-defined subclasses must implement these methods.
    """
    reverse_fitness = False  # bigger the better by default
    def __init__(self):
        self.dna = None
        self._output = None

    def save_to_file(self, path):
        with open(path, 'wb') as f:
            np.save(f, self.dna)

    @staticmethod
    def load_from_file(filepath):
        dna = np.load(filepath)
        return Creature.create_creature(dna=dna)

    @staticmethod
    @abstractmethod
    def create_creature(dna=None):
        """
        Factory method to create a new creature.
        This method should be implemented by the user.
        """
        pass

    @staticmethod
    @abstractmethod
    def crossover(parent1, parent2):
        """
        Crossover method for combining two parent creatures' DNA into a child.
        This method should be implemented by the user.
        """
        pass

    @abstractmethod
    def mutate(self) -> None:
        """
        Mutate the creature's DNA. This method should be implemented by the user.
        """
        pass

    @abstractmethod
    def fitness(self, expected_output=None) -> None:
        """
        Compute the creature's fitness based on the problem at hand.
        This method should be implemented by the user.
        """
        pass

    @abstractmethod
    def process(self, input_=None) -> None:
        """
        Process the creature based on the input (e.g., data, environment).
        This method should be implemented by the user.
        """
        pass

    @abstractmethod
    def get_fitness(self) -> float:
        """
        Return the creature's fitness score.
        This method should be implemented by the user.
        """
        pass

    def get_output(self):
        return self._output

class GeneticAlgorithm:
    SELECTION_METHODS = ("elitism", "roulette")

    def __init__(self, population_size: int, creature_class: Creature, reverse_fitness: bool=False, selection_methods: tuple[str] = ("elitism", "roulette")):
        self.best_creature = None
        self.best_output = None
        self.best_fitness = None
        self.selection_methods = []
        for method in selection_methods:
            if method not in self.SELECTION_METHODS:
                raise ValueError(f"Selection method {method} not supported")
            self.selection_methods.append(method)
        self.population_size = population_size
        self.population = []
        self.reverse_fitness = creature_class.reverse_fitness

        self.creature_class = creature_class
        # Create initial population using the user-defined creature class
        for _ in range(population_size):
            self.population.append(self.creature_class.create_creature())



    @staticmethod
    def elitism(population_fitness_sorted: list, percentage: float=0.05, new_population=None) -> list:
        if new_population is None:
            new_population = []

        new_population.extend(population_fitness_sorted[:int(len(population_fitness_sorted) * percentage)])
        return new_population

    @staticmethod
    def roulette(population_fitness_sorted: list, creature_class: Creature, max_size: int = 10, new_population=None) -> list:
        if new_population is None:
            new_population = []
        total_fitness = sum(creature.get_fitness() for creature in population_fitness_sorted)
        if total_fitness == 0:
            logger.warning("Total fitness is zero; roulette selection cannot proceed.")
            return new_population

        while len(new_population) < max_size:
            pick1 = random.uniform(0, total_fitness)
            current = 0
            parent1 = None
            for creature in population_fitness_sorted:
                current += creature.get_fitness()
                if current >= pick1:
                    parent1 = creature
                    break

            pick2 = random.uniform(0, total_fitness)
            current = 0
            parent2 = None
            for creature in population_fitness_sorted:
                current += creature.get_fitness()
                if current >= pick2:
                    parent2 = creature
                    break

            if parent1 == parent2:
                continue

            offspring = creature_class.crossover(parent1, parent2)
            offspring.mutate()
            new_population.append(offspring)

        return new_population[:max_size]  # Ensure we don’t exceed max_size

    def evolve(self, input_=None, expected_output=None):
        for i in range(self.population_size):
            self.population[i].process(input_=input_)
            self.population[i].fitness(expected_output=expected_output)

        sorted_population = sorted(self.population, key=lambda x: x.get_fitness(), reverse=not self.reverse_fitness)

        self.best_fitness = sorted_population[0].get_fitness()
        self.best_output = np.round(sorted_population[0].get_output(), 3)
        self.best_creature = sorted_population[0]
        
        new_population = []
        for method in self.selection_methods:
            if method == "elitism":
                new_population.extend(GeneticAlgorithm.elitism(sorted_population, percentage=0.2))
            elif method == "roulette":
                new_population.extend(GeneticAlgorithm.roulette(sorted_population, self.creature_class, max_size=self.population_size-len(new_population)))

        self.population = new_population
        assert len(self.population) == self.population_size, f"Population size is not equal to population_size -> {len(self.population)} is not {self.population_size}"

    def epoch(self, inputs_=None, expected_outputs=None):
        for index, (input_, expected_output) in enumerate(zip(inputs_, expected_outputs), start=1):

            self.evolve(input_, expected_output)

