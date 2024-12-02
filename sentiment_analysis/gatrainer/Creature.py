import pickle
from abc import ABC, abstractmethod

# if CUDA is installed, then cupy can be used to make the program faster
try:
    import cupy as np

    HAS_GPU = True
except ImportError:
    # if gpu acceleration is not available, then we simply use numpy (at the cost of performances)
    import numpy as np

    HAS_GPU = False


class Creature(ABC):
    """
    Base class for all creatures.
    For the different use cases, the programmer will define a subclass of Creature, providing specific functions, fitness computation, etc
    """
    reverse_fitness = False  # by default, higher fitness is better
    batch_size = 1

    def __init__(self):
        self.dna = None
        self._output = None
        self._fitness = None

    def save_to_file(self, path):
        """Save DNA to a file."""
        with open(path, "wb") as f:
            pickle.dump(self.dna, f)

    @staticmethod
    def load_from_file(file_path):
        """Load DNA from a file and create a creature."""
        with open(file_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    @abstractmethod
    def create_creature(dna=None):
        """Factory method to create a new creature."""
        pass

    @staticmethod
    @abstractmethod
    def crossover(parent1, parent2):
        """Combine the DNA of two parents to create offspring."""
        pass

    @abstractmethod
    def mutate(self):
        """Mutate the creature's DNA."""
        pass

    @abstractmethod
    def fitness(self, expected_output=None):
        """Compute the creature's fitness."""
        pass

    @abstractmethod
    def process(self, input_=None):
        """Process input and compute output."""
        pass

    @abstractmethod
    def get_fitness(self) -> float:
        """Return the fitness score."""
        pass

    def get_output(self):
        """Return the current computed output."""
        return self._output

    @abstractmethod
    def test_input(self, user_input):
        """Provide a simple test process on user input and return the output"""
        pass
