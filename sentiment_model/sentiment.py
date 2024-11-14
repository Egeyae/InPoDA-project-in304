from GeneticAlgorithmTrainNN import Creature, GeneticAlgorithm
import numpy as np
import random


class SentimentCreature(Creature):
    reverse_fitness = True # we do 1/x so the closer to 1, the better
    def __init__(self, layers, dna=None):
        super().__init__()

        self.layers = layers
        self.mutation_rate = 0.01
        self.mutation_strength = 0.5

        self._fitness = None
        self._output = np.empty(layers[-1])

        if dna is not None:
            self.dna = dna
        else:
            self.build_random_dna()

    def build_random_dna(self):
        num_layers = len(self.layers) - 1 # -1 because we don't create a layer for the input
        self.dna = []

        for i in range(num_layers):
            W = np.random.randn(self.layers[i], self.layers[i + 1])
            b = np.random.randn(self.layers[i + 1])

            self.dna.extend(W.flatten())
            self.dna.extend(b)
        self.dna = np.array(self.dna)

    @staticmethod
    def crossover(parent1, parent2):
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
        self._fitness = np.linalg.norm(self._output - expected_output)

    def process(self, input_=None) -> None:
        num_layers = len(self.layers)
        idx = 0
        for i in range(num_layers - 1):
            in_size = self.layers[i]
            out_size = self.layers[i + 1]

            W_size = in_size * out_size
            Weights = self.dna[idx:idx + W_size].reshape(in_size, out_size)
            idx += W_size

            Biases = self.dna[idx:idx + out_size]
            idx += out_size

            input_ = 1 / (1+ np.exp(-(np.dot(input_, Weights) + Biases)))

        self._output = input_ # the last input is the output

    def get_fitness(self) -> float:
        return self._fitness

    @staticmethod
    def create_creature(dna=None):
        return SentimentCreature(layers=[4, 8, 8, 3], dna=dna)

    @staticmethod
    def load_from_file(filepath):
        dna = np.load(filepath)
        return IrisCreature.create_creature(dna=dna)