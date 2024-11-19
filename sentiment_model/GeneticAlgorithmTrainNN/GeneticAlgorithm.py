import random
import pickle
import logging
from numpy import round as np_round
from Creature import Creature

try:
    import cupy as np

    HAS_GPU = True
except ImportError:
    import numpy as np

    HAS_GPU = False

# configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.NullHandler())


class GeneticAlgorithm:
    SELECTION_METHODS = ("elitism", "roulette")

    def __init__(self, population_size: int, creature_class: Creature, reverse_fitness: bool = False,
                 selection_methods: tuple[str] = ("elitism", "roulette")):
        if not issubclass(creature_class, Creature):
            raise TypeError("creature_class must be a subclass of Creature.")

        self.population_size = population_size
        self.creature_class = creature_class
        self.reverse_fitness = reverse_fitness or creature_class.reverse_fitness

        # we check if the selection methods used are supported
        # for the moment, only elitism and roulette methods are supported
        self.selection_methods = []
        for method in selection_methods:
            if method not in self.SELECTION_METHODS:
                raise ValueError(f"Selection method {method} is not supported.")
            self.selection_methods.append(method)

        # we create the initial population
        self.population = [self.creature_class.create_creature() for _ in range(population_size)]

        # attributes to keep track of the best creature and its stats
        self.best_creature = None
        self.best_output = None
        self.best_fitness = None

    @staticmethod
    def load_from_file(file_path: str):
        with open(file_path, "rb") as file:
            ga = pickle.load(file)
        # we check that we loaded the correct type of object
        if isinstance(ga, GeneticAlgorithm):
            return ga

    def save_to_file(self, file_path: str):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

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
            logger.warning("Total fitness is zero; roulette selection cannot proceed.")
            return new_population

        def pick_parent():
            """
            Small method to pick a parent out of the population
            Uses the roulette wheel selection method (https://en.wikipedia.org/wiki/Fitness_proportionate_selection)
            """
            pick = random.uniform(0, total_fitness)
            cumulative = 0
            for creature in sorted_population:
                cumulative += creature.get_fitness()
                if cumulative >= pick:
                    return creature

        while len(new_population) < max_size:
            # we pick 2 parents
            parent1, parent2 = pick_parent(), pick_parent()
            if parent1 is parent2:
                continue  # avoid crossover with the same parent

            # we reproduce them
            offspring = creature_class.crossover(parent1, parent2)
            # mutate the offspring for genetic diversity
            offspring.mutate()
            # off we go
            new_population.append(offspring)

        return new_population[:max_size]

    def compute_fitness(self, input_, output):
        population_size = len(self.population)
        i = 0
        while i < population_size:
            self.population[i].process(input_)
            self.population[i].fitness(output)
            i += 1

    def evolve(self, input_=None, expected_output=None):
        """Run one generation of the genetic algorithm."""
        # convert to the good type (cupy array or numpy array) depending on the available devices
        input_ = np.asarray(input_)
        expected_output = np.asarray(expected_output)

        # evaluate fitness for the population
        self.compute_fitness(input_, expected_output)

        # sort population by fitness
        self.population = [creature for creature in
                           sorted(self.population, reverse=not self.reverse_fitness, key=lambda x: x.get_fitness())]

        # track the best creature (for an interface)
        self.best_creature = self.population[0]
        self.best_fitness = self.best_creature.get_fitness()
        self.best_output = np_round(self.best_creature.get_output(), 4)  # we avoid not necessary details

        # create a new population using selection methods
        new_population = []
        for method in self.selection_methods:
            if method == "elitism":
                new_population = self.elitism(self.population, percentage=0.2, new_population=new_population)
            elif method == "roulette":
                new_population = self.roulette(self.population, self.creature_class,
                                               max_size=self.population_size,
                                               new_population=new_population)

        if len(new_population) != self.population_size:  # in case an error occurred in the selection methods
            raise ValueError(f"Population size mismatch: {len(new_population)} != {self.population_size}")

        # mutate the population for genetic diversity
        for creature in self.population:
            creature.mutate()

        self.population = new_population

    def epoch(self, inputs_: list = None, expected_outputs: list = None):
        """Run multiple generations for a batch of inputs and expected outputs."""
        for input_, expected_output in zip(inputs_ or [], expected_outputs or []):
            self.evolve(input_, expected_output)
