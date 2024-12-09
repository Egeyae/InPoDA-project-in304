import logging
import os
from random import randint
from .GeneticAlgorithm import GeneticAlgorithm

try:
    from cupy import asarray

    HAS_GPU = True
except Exception:
    HAS_GPU = False


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

    def __init__(self, creature_class, config: dict = None):
        """
        Initialize the GeneticAlgorithmPipeline with a given creature class and configuration.
        :param creature_class: The class representing the "Creature" to be evolved.
        :param config: A dictionary of configuration parameters.
        """
        self.logger = None
        self.set_logger()
        self.logger.info("Initializing GeneticAlgorithmPipeline...")

        self.set_config(config)

        self.creature_class = creature_class
        self.ga = None
        self.best_creature = None
        self.history = []

        # Ensure save directory exists
        os.makedirs(self.config["save_dir"], exist_ok=True)
        self.logger.info("Pipeline initialized successfully.")

    def set_logger(self):
        """Set up a logger for the pipeline."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.addHandler(logging.NullHandler())

    def train(self, inputs, expected_outputs):
        """
        Train the Genetic Algorithm on the provided data.
        :param inputs: A list of input data batches.
        :param expected_outputs: A list of corresponding expected outputs for the inputs.
        """
        if self.ga is None:
            self.logger.debug("No Genetic Algorithm has been initialized. Initializing Genetic Algorithm...")
            self.ga = GeneticAlgorithm(
                population_size=self.config["population_size"],
                creature_class=self.creature_class,
                selection_methods=self.config["selection_methods"],
            )
            self.logger.info("Genetic Algorithm initialized.")
        try:
            patience = self.config["early_stopping"]["patience"]
            min_delta = self.config["early_stopping"]["min_delta"]
            best_fitness = float('inf') if self.ga.reverse_fitness else float('-inf')
            no_improvement = 0

            training_sample_size = self.config["training_sample_size"]

            for epoch in range(self.config["max_epochs"]):
                indexes = [randint(0, len(inputs) - 1) for _ in range(training_sample_size)]
                sample_inputs = [inputs[i] for i in indexes]
                sample_outputs = [expected_outputs[i] for i in indexes]
                self.logger.info(f"Starting epoch {epoch + 1}")
                self.ga.epoch(sample_inputs, sample_outputs)

                current_best_fitness = self.ga.best_fitness
                self.logger.info(f"Epoch {epoch + 1}: Best Fitness = {current_best_fitness:.8f}")
                self.history.append(current_best_fitness)

                # Early stopping logic
                if abs(current_best_fitness - best_fitness) < min_delta:
                    no_improvement += 1
                else:
                    no_improvement = 0
                    best_fitness = current_best_fitness

                if self.config["early_stopping"]["enabled"] and no_improvement >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1} due to no improvement.")
                    break
        except KeyboardInterrupt:
            self.logger.info("Early stopping due to keyboard interrupt.")
        except Exception as e:
            self.logger.error(f"Training error: {e}", exc_info=True)
        finally:
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
            self.logger.error("No best creature found for evaluation. Train the Genetic Algorithm first.")
            return

        metrics = {"accuracy": 0.0}
        for input_, output in zip(test_inputs, test_outputs):
            self.best_creature.process(input_)
            predicted_output = self.best_creature.get_output() # Single scalar output

            # Compute accuracy based on closeness of prediction to actual value
            actual_output = float(output)
            error = abs(predicted_output - actual_output)
            accuracy = max(0, 1 - error / 4)  # Scaled accuracy (1 is perfect, 0 is worst)
            metrics["accuracy"] += accuracy

            self.logger.info(f"Predicted: {predicted_output:.4f}, Actual: {actual_output:.4f}, Accuracy: {accuracy:.4f}")

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

    def set_config(self, config):
        self.config = self.DEFAULT_CONFIG
        if config:
            self.config.update(config)

    def save_best_model(self):
        """Save the best creature to a file."""
        if not self.best_creature:
            self.logger.error("No best creature found. Train the Genetic Algorithm first.")
            return
        path = os.path.join(self.config["save_dir"], self.config["model_name"])
        self.best_creature.save_to_file(path)
        self.logger.info(f"Saved best model to {path}.")

    def load_best_model(self):
        """Load the best creature from a file."""
        try:
            path = os.path.join(self.config["save_dir"], self.config["model_name"])
            self.best_creature = self.creature_class.load_from_file(path)
            self.logger.info("Loaded best model.")
        except FileNotFoundError:
            self.best_creature = None
            self.logger.error("Best model not found.")
        except Exception as e:
            self.logger.error(f"Error loading best model: {e}")
