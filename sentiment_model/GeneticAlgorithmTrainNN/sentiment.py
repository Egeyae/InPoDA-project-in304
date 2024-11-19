from abc import ABC

from GeneticAlgorithm import Creature

# GeneticAlgorithm already import cupy or numpy but in case of modifications:
try:
    import cupy as np

    HAS_GPU = True
except ImportError:
    import numpy as np

    HAS_GPU = False
import random
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())

logger.info(f"GPU available: {HAS_GPU}")


class SentimentCreature(Creature, ABC):
    reverse_fitness = True  # we do 1/x so the closer to 1, the better

    def __init__(self, layers, dna=None, noDNA=False):
        super().__init__()

        self.layers = layers
        self.mutation_rate = 0.01
        self.mutation_strength = 0.5

        self._fitness = None
        self._output = np.empty(layers[-1])

        if dna is not None:
            self.dna = dna
        elif not noDNA:
            self.build_random_dna()
        else:
            self.dna = None

    def build_random_dna(self):
        num_layers = len(self.layers) - 1  # -1 because we don't create a layer for the input
        self.dna = []

        for i in range(num_layers):
            W = np.random.randn(self.layers[i], self.layers[i + 1])
            b = np.random.randn(self.layers[i + 1])

            self.dna.extend(W.flatten())
            self.dna.extend(b)
        self.dna = np.array(self.dna)

    @staticmethod
    def crossover(parent1, parent2):
        # simple, one point crossover
        size = parent1.dna.shape[0]
        point = random.randint(1, size - 1)

        new_dna = np.concatenate((parent1.dna[:point], parent2.dna[point:]))

        return SentimentCreature(parent1.layers, new_dna)

    def mutate(self) -> None:
        size = self.dna.shape[0]
        for i in range(size):
            if random.random() < self.mutation_rate:
                self.dna[i] += random.uniform(-self.mutation_strength, self.mutation_strength)

    def fitness(self, expected_output=None) -> None:
        if expected_output is not None:
            self._fitness = np.linalg.norm(expected_output - self._output)

    def process(self, input_=None) -> None:
        num_layers = len(self.layers)
        idx = 0
        input_ = np.array(input_)  # in case of numpy to cupy (shouldn't be the case but...)
        for i in range(num_layers - 1):
            in_size = self.layers[i]
            out_size = self.layers[i + 1]

            W_size = in_size * out_size
            Weights = self.dna[idx:idx + W_size].reshape(in_size,
                                                         out_size)  # take out the matrix flattened at a position and reshape it as a matrix
            idx += W_size

            Biases = self.dna[idx:idx + out_size]
            idx += out_size

            input_ = 1 / (1 + np.exp(-(np.dot(input_, Weights) + Biases)))  # sigmoid activation

        self._output = input_  # the last input is the output

    def get_fitness(self) -> float:
        return float(self._fitness) if self._fitness is not None else None

    @staticmethod
    def create_creature(dna=None):
        return SentimentCreature(layers=[4, 8, 8, 3], dna=dna)

    @staticmethod
    def load_from_file(filepath):
        dna = np.load(filepath)
        return SentimentCreature.create_creature(dna=dna)

    def test_input(self, user_input):
        input_ = np.array(eval(user_input.strip()))
        self.process(input_)
        return self._output


CREATURE = SentimentCreature

with open("iris.data") as f:
    data = [x.split(",") for x in f.read().splitlines()]

INPUTS = [np.array([float(x) for x in z[:4]]) for z in data]


def type_to_array(t):
    if t == "Iris-setosa":
        return np.array([1, 0, 0])
    if t == "Iris-versicolor":
        return np.array([0, 1, 0])
    if t == "Iris-virginica":
        return np.array([0, 0, 1])


OUTPUTS = [type_to_array(z[4]) for z in data]
