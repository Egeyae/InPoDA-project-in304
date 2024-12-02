import logging
import os
import json
from random import randint
from .GeneticAlgorithm import GeneticAlgorithm
from .Creature import Creature
import pickle


# noinspection PyUnresolvedReferences
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
        "model_name": "best_model.pkl",
        "training_sample_size": 5
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

        # if not issubclass(creature_class, Creature):
        #     self.logger.error("creature_class is not a subclass of Creature")
        #     self.logger.error(f"creature_class = {creature_class}")
        #     raise TypeError("creature_class must be a subclass of Creature.")

        self.set_config(config)

        self.creature_class = creature_class
        self.ga = GeneticAlgorithm(
            population_size=self.config["population_size"],
            creature_class=creature_class,
            selection_methods=self.config["selection_methods"],
        )
        self.best_creature = None
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
        try:
            patience = self.config["early_stopping"]["patience"]
            min_delta = self.config["early_stopping"]["min_delta"]
            best_fitness = -1
            no_improvement = 0

            # avoids to run on all the inputs, just a portion of it
            training_sample_size = self.config["training_sample_size"]

            for epoch in range(self.config["max_epochs"]):
                indexes = [randint(0, len(inputs) - 1) for _ in range(training_sample_size)]
                sample_inputs = [inputs[i] for i in indexes]
                sample_outputs = [expected_outputs[i] for i in indexes]
                self.logger.info(f"Starting epoch {epoch}")
                self.ga.epoch(sample_inputs, sample_outputs)

                # Track and log the best fitness
                current_best_fitness = self.ga.best_fitness
                self.logger.info(f"Epoch {epoch + 1}: Best Fitness = {current_best_fitness:.8f}")
                self.history.append(current_best_fitness)

                # Early stopping
                if (current_best_fitness - best_fitness) < min_delta:
                    no_improvement += 1
                else:
                    no_improvement = 0

                if self.config["early_stopping"]["enabled"] and no_improvement >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1} due to no improvement.")
                    break
        except KeyboardInterrupt:
            self.logger.info("Early stopping due to keyboard interrupt.")
            self.best_creature = self.ga.best_creature
        except Exception as e:
            self.logger.error(f"Could not train Genetic Algorithm: {e}")
        else:
            self.best_creature = self.ga.best_creature

    def evaluate(self, test_inputs, test_outputs):
        """
        Evaluate the best creature from the GA on a test dataset.
        :param test_inputs: A list of test input batches.
        :param test_outputs: A list of expected outputs for the test inputs.
        :return: Evaluation metrics such as accuracy or loss.
        """
        self.logger.info("Evaluating the best creature...")
        if not self.best_creature:
            self.logger.error("No best creature found for evaluation. Genetic Algorithm has to be trained first.")
            return

        metrics = {"accuracy": 0.0}
        for input_, output in zip(test_inputs, test_outputs):
            self.best_creature.process(input_)
            self.best_creature.fitness(expected_output=output)
            if self.ga.reverse_fitness:  # 0 is better
                accuracy = 1 / (1 + self.best_creature.get_fitness())
            else:
                accuracy = self.best_creature.get_fitness()  # bigger is better

            metrics["accuracy"] += accuracy

        metrics["accuracy"] /= len(test_inputs)

        self.logger.info(f"Evaluation Results: {metrics}" + " (closer to 1 is better)" if self.ga.reverse_fitness else " (bigger is better)")
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

    def set_config(self, config):
        self.config = self.DEFAULT_CONFIG
        for k in config.keys():
            self.config[k] = config[k]

    def save_best_model(self):
        if not self.best_creature:
            self.logger.error("No best creature found. Genetic Algorithm has to be trained first.")
            return
        self.best_creature.save_to_file(os.path.join(self.config["save_dir"], self.config["model_name"]))

    def load_best_model(self):
        try:
            print("loaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaading")
            self.best_creature = self.creature_class.load_from_file(os.path.join(self.config["save_dir"], self.config["model_name"]))
            self.logger.info("Loaded best model")
        except FileNotFoundError:
            self.best_creature = None
            self.logger.error("Couldn't load best model as it couldn't be found")
        except Exception as e:
            self.logger.error(f"Could not load best model: {e}")
