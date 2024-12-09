import logging
import pickle
import random
from abc import ABC

from sentiment_analysis.gatrainer.Creature import Creature

# Check for GPU availability
try:
    import cupy as cp

    HAS_GPU = True
    np = cp
except ImportError:
    import numpy as np

    HAS_GPU = False

# Set up logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
logger.info(f"GPU available: {HAS_GPU}")


class SentimentCreature(Creature, ABC):
    reverse_fitness = True
    batch_size = 1

    def __init__(self, layers, dna=None, noDNA=False):
        super().__init__()
        self.layers = layers
        self.mutation_rate = 0.01
        self.mutation_strength = 0.5
        self._fitness = None
        self._output = np.empty((1,), dtype=np.float32)  # Single scalar output as array

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

    def fitness_binary_crossentropy(self, expected_output=None, epsilon=1e-10):
        """Evaluate the creature's fitness using Binary Cross-Entropy loss."""
        if expected_output is not None:
            # Clip values to avoid log(0)
            clipped_output = np.clip(self._output, epsilon, 1 - epsilon)

            # Scale expected output to match range [0, 1]
            scaled_expected = expected_output / 4

            # Calculate Binary Cross-Entropy Loss
            bce_loss = -np.sum(
                scaled_expected * np.log(clipped_output) + (1 - scaled_expected) * np.log(1 - clipped_output)
            )
            self._fitness = bce_loss

    def fitness_hybrid(self, expected_output=None, alpha=0.5, epsilon=1e-10):
        """Evaluate the creature's fitness using a hybrid loss function."""
        if expected_output is not None:
            # Clip values to avoid log(0)
            clipped_output = np.clip(self._output, epsilon, 1 - epsilon)

            # Scale expected output to match range [0, 1]
            scaled_expected = expected_output / 4

            # Binary Cross-Entropy Component
            bce_loss = -np.sum(
                scaled_expected * np.log(clipped_output) + (1 - scaled_expected) * np.log(1 - clipped_output)
            )

            # Mean Squared Error Component (use unscaled values)
            mse_loss = np.mean((expected_output - self._output) ** 2)

            # Weighted Hybrid Loss
            self._fitness = alpha * bce_loss + (1 - alpha) * mse_loss

    def fitness(self, expected_output=None):
        """Evaluate the creature's fitness."""
        return self.fitness_hybrid(expected_output, epsilon=1e-10)

    def process(self, input_=None):
        """Process the input through the neural network."""
        # input_ = np.array(input_, dtype=np.float64)  # Ensure compatibility with GPU if CuPy is used
        if HAS_GPU:
            input_ = np.asarray(input_, dtype=np.float64)
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

            if HAS_GPU:
                W = np.asarray(W, dtype=np.float64)
                b = np.asarray(b, dtype=np.float64)

            # Forward pass with sigmoid activation
            input_ = 1 / (1 + np.exp(-np.dot(input_, W) - b))

        # Scale the final output to range [0, 4]
        self._output = input_ * 4

    def get_fitness(self) -> float:
        """Return the fitness of the creature."""
        return float(self._fitness) if self._fitness is not None else None

    @staticmethod
    def create_creature(dna=None):
        """Create a new SentimentCreature with the given DNA."""
        return SentimentCreature(layers=[768, 400, 200, 1], dna=dna)

    @staticmethod
    def load_from_file(file_path):
        """Load a SentimentCreature instance from a file."""
        with open(file_path, "rb") as f:
            return SentimentCreature.create_creature(pickle.load(f))  # Load a DNA as we save DNA and not the object

    def test_input(self, user_input):
        """Test the creature with the given user input."""
        input_ = np.array(eval(user_input.strip()), dtype=np.float32)
        self.process(input_)
        return self._output

    def get_output(self):
        """Return the current computed output."""
        return self._output[0]
