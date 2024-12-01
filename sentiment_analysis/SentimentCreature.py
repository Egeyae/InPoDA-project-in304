import random
import logging
import numpy as np
from abc import ABC
from gatrainer.Creature import Creature

# Check for GPU availability
try:
    import cupy as np

    HAS_GPU = True
except ImportError:
    import numpy as np

    HAS_GPU = False

# Set up logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.info(f"GPU available: {HAS_GPU}")


# SentimentCreature class definition
class SentimentCreature(Creature, ABC):
    reverse_fitness = True  # We do 1/x so the closer to 1, the better
    batch_size = 3

    def __init__(self, layers, dna=None, noDNA=False):
        super().__init__()
        self.layers = layers
        self.mutation_rate = 0.01
        self.mutation_strength = 0.5
        self._fitness = None
        self._output = np.empty(layers[-1])

        # If DNA is provided, use it. Otherwise, create random DNA or leave it as None
        if dna is not None:
            self.dna = dna
        elif not noDNA:
            self.build_random_dna()
        else:
            self.dna = None

    def build_random_dna(self):
        """Generate random DNA for the creature."""
        self.dna = []
        for i in range(len(self.layers) - 1):
            # Create weights and biases for each layer
            W = np.random.randn(self.layers[i], self.layers[i + 1])
            b = np.random.randn(self.layers[i + 1])
            self.dna.extend(W.flatten())
            self.dna.extend(b)

        self.dna = np.array(self.dna)

    @staticmethod
    def crossover(parent1, parent2):
        """Perform one-point crossover between two parents."""
        size = parent1.dna.shape[0]
        point = random.randint(1, size - 1)
        new_dna = np.concatenate((parent1.dna[:point], parent2.dna[point:]))
        return SentimentCreature(parent1.layers, new_dna)

    def mutate(self):
        """Mutate the creature's DNA with the given mutation rate and strength."""
        for i in range(self.dna.shape[0]):
            if random.random() < self.mutation_rate:
                self.dna[i] += random.uniform(-self.mutation_strength, self.mutation_strength)

    def fitness(self, expected_output=None):
        """Evaluate the creature's fitness."""
        #print(expected_output, self._output)
        if expected_output is not None:
            self._fitness = np.linalg.norm(expected_output - self._output)

    def process(self, input_=None):
        """Process the input through the neural network layers."""
        idx = 0
        for i in range(len(self.layers) - 1):
            in_size = self.layers[i]
            out_size = self.layers[i + 1]

            # Reshape weights and biases
            W_size = in_size * out_size
            Weights = self.dna[idx:idx + W_size].reshape(in_size, out_size)
            idx += W_size

            Biases = self.dna[idx:idx + out_size]
            idx += out_size

            # Apply sigmoid activation function
            input_ = 1 / (1 + np.exp(-(np.dot(input_, Weights) + Biases)))

        self._output = input_  # The last input is the output

    def get_fitness(self) -> float:
        """Return the fitness of the creature."""
        return float(self._fitness) if self._fitness is not None else None

    @staticmethod
    def create_creature(dna=None):
        """Create a new SentimentCreature with the given DNA."""
        return SentimentCreature(layers=[768, 400, 400, 100, 2], dna=dna)

    @staticmethod
    def load_from_file(filepath):
        """Load a creature from a file."""
        dna = np.load(filepath)
        return SentimentCreature.create_creature(dna=dna)

    def test_input(self, user_input):
        """Test the creature with the given user input."""
        input_ = np.array(eval(user_input.strip()))
        self.process(input_)
        return self._output
