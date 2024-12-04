import pickle
import random
import logging
import numpy as np
from abc import ABC
from sentiment_analysis.gatrainer.Creature import Creature

# Check for GPU availability
try:
    import cupy as cp

    HAS_GPU = True
    np = cp  # Alias CuPy as NumPy for seamless integration
except ImportError:
    import numpy as np

    HAS_GPU = False

# Set up logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
logger.info(f"GPU available: {HAS_GPU}")


# SentimentCreature class definition
class SentimentCreature(Creature, ABC):
    reverse_fitness = True  # We do 1/x so the closer to 1, the better
    batch_size = 16  # Increased batch size for better GPU utilization

    def __init__(self, layers, dna=None, noDNA=False):
        super().__init__()
        self.layers = layers
        self.mutation_rate = 0.01
        self.mutation_strength = 0.5
        self._fitness = None
        self._output = np.empty(layers[-1])

        # Use provided DNA or generate random DNA
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
        mutation_mask = np.random.rand(self.dna.shape[0]) < self.mutation_rate
        mutations = np.random.uniform(-self.mutation_strength, self.mutation_strength, size=self.dna.shape)
        self.dna += mutation_mask * mutations

    def fitness(self, expected_output=None):
        """Evaluate the creature's fitness."""
        if expected_output is not None:
            self._fitness = np.linalg.norm(expected_output - self._output)

    def process(self, input_):
        """Process the input through the neural network."""
        input_ = np.array(input_, dtype=np.float32)  # Ensure compatibility with GPU if CuPy is used
        idx = 0

        for i in range(len(self.layers) - 1):
            in_size = self.layers[i]
            out_size = self.layers[i + 1]

            # Extract weights and biases
            W_size = in_size * out_size
            W = self.dna[idx:idx + W_size].reshape(in_size, out_size)
            idx += W_size

            b = self.dna[idx:idx + out_size]
            idx += out_size

            # Forward pass with sigmoid activation
            input_ = 1 / (1 + np.exp(-np.dot(input_, W) - b))

        self._output = input_  # Final output

    def get_fitness(self) -> float:
        """Return the fitness of the creature."""
        return float(self._fitness) if self._fitness is not None else None

    @staticmethod
    def create_creature(dna=None):
        """Create a new SentimentCreature with the given DNA."""
        return SentimentCreature(layers=[768, 400, 200, 2], dna=dna)

    @staticmethod
    def load_from_file(file_path):
        """Load a SentimentCreature instance from a file."""
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def test_input(self, user_input):
        """Test the creature with the given user input."""
        input_ = np.array(eval(user_input.strip()), dtype=np.float32)
        self.process(input_)
        return self._output
