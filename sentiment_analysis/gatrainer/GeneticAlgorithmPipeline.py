import logging
import os
import json

from gatrainer.GeneticAlgorithm import GeneticAlgorithm
from gatrainer.Creature import Creature


class GeneticAlgorithmPipeline:
    """
    A pipeline for training and testing neural networks using Genetic Algorithm (GA).
    Provides an easy-to-use interface for configuring, training, saving, and evaluating GA-based models.
    """
    config: dict = None

    DEFAULT_CONFIG = {
        "population_size": 100,
        "selection_methods": ("elitism", "roulette"),
        "elitism_percentage": 0.2,
        "mutation_rate": 0.05,
        "max_epochs": 100,
        "early_stopping": {"enabled": True, "patience": 20, "min_delta": 1e-4},
        "save_dir": "./models/",
    }

    def __init__(self, creature_class, config: dict=None):
        """
        Initialize the GeneticAlgorithmPipeline with a given creature class and configuration.
        :param creature_class: The class representing the "Creature" to be evolved.
        :param config: A dictionary of configuration parameters.
        """
        self.logger = None
        self.set_logger()
        self.logger.info("Initializing GeneticAlgorithmPipeline...")

        if not issubclass(creature_class, Creature):
            raise TypeError("creature_class must be a subclass of Creature.")

        self.set_config(config)

        self.creature_class = creature_class
        self.ga = GeneticAlgorithm(
            population_size=self.config["population_size"],
            creature_class=creature_class,
            selection_methods=self.config["selection_methods"],
        )
        self.history = []

        # Ensure save directory exists
        os.makedirs(self.config["save_dir"], exist_ok=True)
        self.logger.info("Pipeline initialized successfully.")

    def set_logger(self):
        """
        Set up a logger for the pipeline
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.addHandler(logging.NullHandler())

    def train(self, inputs, expected_outputs):
        """
        Train the Genetic Algorithm on the provided data.
        :param inputs: A list of input data batches.
        :param expected_outputs: A list of corresponding expected outputs for the inputs.
        """
        patience = self.config["early_stopping"]["patience"]
        min_delta = self.config["early_stopping"]["min_delta"]
        best_fitness = None
        no_improvement = 0

        for epoch in range(self.config["max_epochs"]):
            self.ga.epoch(inputs, expected_outputs)

            # Track and log the best fitness
            current_best_fitness = self.ga.best_fitness
            self.logger.info(f"Epoch {epoch + 1}: Best Fitness = {current_best_fitness:.4f}")
            self.history.append(current_best_fitness)

            # Early stopping
            if best_fitness is None or (current_best_fitness - best_fitness) > min_delta:
                best_fitness = current_best_fitness
                no_improvement = 0
            else:
                no_improvement += 1

            if self.config["early_stopping"]["enabled"] and no_improvement >= patience:
                self.logger.info(f"Early stopping at epoch {epoch + 1}.")
                break

    def evaluate(self, test_inputs, test_outputs):
        """
        Evaluate the best creature from the GA on a test dataset.
        :param test_inputs: A list of test input batches.
        :param test_outputs: A list of expected outputs for the test inputs.
        :return: Evaluation metrics such as accuracy or loss.
        """
        self.logger.info("Evaluating the best creature...")
        best_creature = self.ga.best_creature

        metrics = {"accuracy": 0.0}
        for input_, output in zip(test_inputs, test_outputs):
            best_creature.process(input_)
            best_creature.fitness(expected_output=output)
            if self.ga.reverse_fitness:  # 0 is better
                accuracy = 1 / (1 + best_creature.get_fitness())
            else:
                accuracy = best_creature.get_fitness()  # bigger is better

            metrics["accuracy"] += accuracy

        metrics["accuracy"] /= len(test_inputs)

        self.logger.info(f"Evaluation Results: {metrics}")
        return metrics

    def save(self, file_name="genetic_algorithm.pkl"):
        """
        Save the Genetic Algorithm state to a file.
        :param file_name: The name of the file to save the GA state.
        """
        path = os.path.join(self.config["save_dir"], file_name)
        self.ga.save_to_file(path)
        self.logger.info(f"Saved GA to {path}.")

    def load(self, file_path):
        """
        Load a saved Genetic Algorithm state.
        :param file_path: The path to the saved file.
        """
        self.ga = GeneticAlgorithm.load_from_file(file_path)
        self.logger.info(f"Loaded GA from {file_path}.")

    def save_config(self, file_name="ga_config.json"):
        """
        Save the current pipeline configuration to a JSON file.
        :param file_name: The name of the configuration file.
        """
        path = os.path.join(self.config["save_dir"], file_name)
        with open(path, "w") as file:
            json.dump(self.config, file, indent=4)
        self.logger.info(f"Saved configuration to {path}.")

    def load_config(self, file_path):
        """
        Load configuration from a JSON file.
        :param file_path: The path to the configuration file.
        """
        with open(file_path, "r") as file:
            self.config = json.load(file)
        self.logger.info(f"Loaded configuration from {file_path}.")

    def set_config(self, config):
        self.config = self.DEFAULT_CONFIG
        for k in config.keys():
            self.config[k] = config[k]
