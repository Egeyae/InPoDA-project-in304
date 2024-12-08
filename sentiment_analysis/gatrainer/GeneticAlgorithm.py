import random
import pickle
import logging
from typing import Any

from numpy import round as np_round, ndarray, dtype, floating
from .Creature import Creature
from numpy._typing import _64Bit

try:
    import cupy as np

    HAS_GPU = True
except ImportError:
    import numpy as np

    HAS_GPU = False


class GeneticAlgorithm:
    SELECTION_METHODS = ("elitism", "roulette")

    def __init__(self, population_size: int, creature_class: Creature, reverse_fitness: bool = False,
                 selection_methods: tuple[str] = ("elitism", "roulette"), batch_size: int = 1,
                 elitism_percentage: float = 0.2):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.addHandler(logging.NullHandler())

        self.population_size = population_size
        self.creature_class = creature_class
        self.reverse_fitness = reverse_fitness or creature_class.reverse_fitness
        self.batch_size = batch_size

        # Validate selection methods
        self.selection_methods = []
        for method in selection_methods:
            if method not in self.SELECTION_METHODS:
                raise ValueError(f"Selection method {method} is not supported.")
            self.selection_methods.append(method)

        self.elitism_percentage = elitism_percentage

        # Create the initial population
        self.population = [self.creature_class.create_creature() for _ in range(population_size)]

        # Attributes to track the best creature
        self.best_creature = None
        self.best_output = None
        self.best_fitness = None

    @staticmethod
    def load_from_file(file_path: str):
        try:
            with open(file_path, "rb") as file:
                ga = pickle.load(file)
            if not isinstance(ga, GeneticAlgorithm):
                raise TypeError("The loaded object is not an instance of GeneticAlgorithm.")
            return ga
        except (FileNotFoundError, PermissionError) as e:
            raise IOError(f"Error loading file: {e}")

    def save_to_file(self, file_path: str):
        try:
            with open(file_path, 'wb') as file:
                pickle.dump(self, file)
        except (FileNotFoundError, PermissionError) as e:
            raise IOError(f"Error saving file: {e}")

    @staticmethod
    def elitism(sorted_population: list, percentage: float = 0.05, new_population=None) -> list:
        """Select a percentage of the top-performing population."""
        if new_population is None:
            new_population = []
        elite_count = max(1, int(len(sorted_population) * percentage))
        new_population.extend(sorted_population[:elite_count])
        return new_population

    @staticmethod
    def roulette(sorted_population: list, creature_class: Creature, max_size: int = 10,
                 new_population=None) -> list:
        """Perform roulette wheel selection."""
        if new_population is None:
            new_population = []

        total_fitness = sum(creature.get_fitness() for creature in sorted_population)
        if total_fitness == 0:
            return new_population

        def pick_parent():
            """Pick a parent using roulette wheel selection."""
            pick = random.uniform(0, total_fitness)
            cumulative = 0
            for creature in sorted_population:
                cumulative += creature.get_fitness()
                if cumulative >= pick:
                    return creature

        while len(new_population) < max_size:
            parent1, parent2 = pick_parent(), pick_parent()
            if parent1 is parent2:
                continue

            offspring = creature_class.crossover(parent1, parent2)
            offspring.mutate()
            new_population.append(offspring)

        return new_population[:max_size]

    def compute_fitness(self, input_, output) -> ndarray[Any, dtype[floating[_64Bit]]]:
        """Calculate the fitness of each creature."""
        pop_fitness_array = np.zeros(self.population_size)

        for _ in range(self.batch_size):
            for i, creature in enumerate(self.population):
                try:
                    creature.process(input_)
                    creature.fitness(output)
                    pop_fitness_array[i] += creature.get_fitness()
                except AttributeError as e:
                    raise AttributeError(f"Error in fitness evaluation for creature: {e}")

        return pop_fitness_array

    def evolve(self, input_=None, expected_output=None):
        """Run one generation of the genetic algorithm."""
        input_ = np.asarray(input_)
        expected_output = np.asarray(expected_output)

        fitness = self.compute_fitness(input_, expected_output)

        # Sort population by fitness
        self.population = [creature for creature, _ in
                           sorted(zip(self.population, fitness), reverse=not self.reverse_fitness, key=lambda x: x[1])]

        # Track the best creature
        self.best_creature = self.population[0]
        self.best_fitness = self.best_creature.get_fitness()
        self.logger.debug(f"Current best fitness: {self.best_fitness}")
        self.best_output = np_round(self.best_creature.get_output(), 16)

        # Create a new population using selection methods
        new_population = []
        for method in self.selection_methods:
            if method == "elitism":
                new_population = self.elitism(self.population, percentage=self.elitism_percentage, new_population=new_population)
            elif method == "roulette":
                new_population = self.roulette(self.population, self.creature_class, max_size=self.population_size,
                                               new_population=new_population)

        if len(new_population) != self.population_size:
            raise ValueError(f"Population size mismatch: {len(new_population)} != {self.population_size}")

        # Mutate new population
        for creature in new_population:
            creature.mutate()

        self.population = new_population

    def epoch(self, inputs_: list = None, expected_outputs: list = None, patience: int = 10, delta: float = 0.0001):
        """Run multiple generations for a batch of inputs and expected outputs."""
        random.shuffle(inputs_)
        random.shuffle(expected_outputs)
        for i, (input_, expected_output) in enumerate(zip(inputs_ or [], expected_outputs or [])):
            self.logger.debug(f"Evolving generation {i}")
            self.evolve(input_=input_, expected_output=expected_output)
            self.logger.info(f"Generation {i} done. Fitness: {self.best_fitness} / Output: {self.best_output} / Expected: {expected_output}")
