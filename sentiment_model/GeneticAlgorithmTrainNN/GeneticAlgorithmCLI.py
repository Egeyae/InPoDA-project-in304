import random
import sys
import logging
import os
from datetime import datetime
import importlib.util
from GeneticAlgorithm import GeneticAlgorithm
from numpy import round as np_round
import time


# Log file setup
log_filename = f"logs/ga_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
if not os.path.exists("logs"):
    os.mkdir("logs")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Genetic Algorithm CLI initialized.")


class GeneticAlgorithmCLI:
    def __init__(self):
        self.ga = None
        self.inputs = []
        self.outputs = []
        self.generation_data = []
        self.fitness_data = []
        self.creature_class = None

    def load_creature_io_file(self, file_path):
        """Load the Creature class and IO data from the specified file."""
        try:
            fname = file_path.split("/")[-1].strip() # .split(".")[0]
            spec = importlib.util.spec_from_file_location("custom", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            self.creature_class = getattr(module, "CREATURE", None)
            self.inputs = getattr(module, "INPUTS", None)
            self.outputs = getattr(module, "OUTPUTS", None)

            if not self.creature_class:
                raise AttributeError("The file must contain a 'CREATURE' class.")
            if not self.inputs or not self.outputs:
                raise AttributeError("The file must contain 'INPUTS' and 'OUTPUTS'.")

            print("\n✅ Successfully loaded Creature class, inputs, and outputs.")
            logger.info("Creature class, inputs, and outputs loaded.")
        except Exception as e:
            logger.error(f"Error loading Creature, inputs, and outputs file: {e}")
            print(f"\n❌ Error: {e}")

    def start_training(self, generations: int, population_size: int = None, reset: bool = False):
        """Start the genetic algorithm training."""
        if not self.creature_class or not self.inputs or not self.outputs:
            print("\n❌ Error: Load a valid Creature file and IO data before starting.")
            return

        if not self.ga or reset:
            self.ga = GeneticAlgorithm(
                population_size=population_size,
                creature_class=self.creature_class,
                reverse_fitness=self.creature_class.reverse_fitness
            )

        print("\n🚀 Starting training...")
        logger.info(f"Training started with Population Size: {population_size}, Generations: {generations}")
        self.generation_data = []
        self.fitness_data = []

        try:
            for gen in range(1, generations + 1):
                start = time.process_time()
                index = random.randint(0, len(self.inputs)-1)
                self.ga.evolve(input_=self.inputs[index], expected_output=self.outputs[index])
                best_fitness = self.ga.best_fitness
                best_output = self.ga.best_output
                self.generation_data.append(gen)
                self.fitness_data.append(best_fitness)

                end = time.process_time()
                print(f"🌟 Generation {gen:5} ({end-start:10}s): Best Fitness = {best_fitness:.4f}, Best Output = \t{best_output}\t\t VS \t\t{self.outputs[index]} = Expected")
                logger.info(f"Generation {gen}: Best Fitness={best_fitness}, Best Output={best_output}, Time Elapsed={end - start}")
        except KeyboardInterrupt:
            print("\n❌ Training stopped.")
            logger.info("Training stopped.")
        else:
            print("\n✅ Training complete!")
            logger.info("Training complete.")

    def export_metrics(self, file_path):
        """Export the metrics to a specified file."""
        try:
            with open(file_path, 'w') as f:
                f.write("Generation,Accuracy,Fitness\n")
                for gen, fitness in zip(self.generation_data, self.fitness_data):
                    accuracy = 1 / (1 + fitness)
                    f.write(f"{gen},{accuracy:.4f},{fitness:.4f}\n")
            print(f"\n📂 Metrics exported to {file_path}")
            logger.info(f"Metrics exported to {file_path}")
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            print(f"\n❌ Error: {e}")

    def test_best_creature(self):
        """Test the best creature on a specific input."""
        try:
            if not self.ga:
                print("\n❌ Error: Train a population before testing.")
                return

            choice = int(input("On which sample size of the input do you want to test your best creature?: ").strip())

            results_fitness = []
            results_output = []
            for _ in range(choice):
                index = random.randint(0, len(self.inputs)-1)
                self.ga.best_creature.process(self.inputs[index])
                self.ga.best_creature.fitness(self.outputs[index])
                results_fitness.append(self.ga.best_creature.get_fitness())
                results_output.append(self.ga.best_creature.get_output())
            average_fitness = sum(results_fitness) / len(results_fitness)
            average_output = sum(results_output) / len(results_output)

            print(f"\n🔍 Best Creature Average Fitness: {average_fitness:.6f}, Average Output: {average_output}")
        except Exception as e:
            logger.error(f"Error testing creature: {e}")
            print(f"\n❌ Error: {e}")

    def save_best_creature(self, file_path):
        """Save the best creature to a specified file."""
        try:
            if not self.ga:
                print("\n❌ Error: Train a population before saving.")
                return
            self.ga.best_creature.save_to_file(file_path)
            print(f"\n💾 Best creature saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving creature: {e}")
            print(f"\n❌ Error: {e}")

    def save_algorithm(self, file_path):
        """Save the current genetic algorithm to a specified file."""
        try:
            if not self.ga:
                print("\n❌ Error: Train a population before saving.")
                return
            self.ga.save_algorithm(file_path)
            print(f"\n💾 Algorithm saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving algorithm: {e}")
            print(f"\n❌ Error: {e}")

    def load_algorithm(self, file_path):
        """Load an existing genetic algorithm from a file."""
        try:
            self.ga = GeneticAlgorithm.load_from_file(file_path)
            print(f"\n📂 Algorithm loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading algorithm: {e}")
            print(f"\n❌ Error: {e}")

    def save_metrics(self, file_path):
        """Save training metrics to a specified file."""
        try:
            if not self.ga:
                print("\n❌ Error: Train a population before saving.")
                return
            self.ga.save_metrics(file_path)
            print(f"\n💾 Metrics saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            print(f"\n❌ Error: {e}")


def main():
    cli = GeneticAlgorithmCLI()
    while True:
        print("\n" + "=" * 50)
        print("             🌟 Genetic Algorithm CLI 🌟")
        print("=" * 50)
        print("1. Load Creature + IO File")
        print("2. Load a pre-existing Genetic Algorithm")
        print("3. Start Training")
        print("4. Test Best Creature")
        print("5. Save Best Creature")
        print("6. Save Genetic Algorithm")
        print("7. Export Metrics")
        print("q. Exit")
        print("=" * 50)

        choice = input("Enter your choice: ").strip()

        match choice:
            case "1":
                file_path = input("Enter path to creature file: ").strip()
                cli.load_creature_io_file(file_path)
            case "2":
                file_path = input("Enter path to GA file: ").strip()
                cli.load_algorithm(file_path)
            case "3":
                reset = False
                if cli.ga is not None:
                    reset = input("An algorithm already exists. Reset it? [y]es / [n]o: ").strip().lower() == "y"
                try:
                    population_size = int(input("Enter population size: ").strip()) if cli.ga is None or reset else None
                    generations = int(input("Enter number of generations: ").strip())
                    cli.start_training(generations, population_size, reset)
                except ValueError:
                    print("\n❌ Error: Please provide valid integers.")
            case "4":
                try:
                    cli.test_best_creature()
                except Exception as e:
                    print(f"\n❌ Error: {e}")
            case "5":
                file_path = input("Enter path to save the best creature: ").strip()
                cli.save_best_creature(file_path)
            case "6":
                file_path = input("Enter path to save the algorithm: ").strip()
                cli.save_algorithm(file_path)
            case "7":
                file_path = input("Enter path to save metrics: ").strip()
                cli.export_metrics(file_path)
            case "q":
                confirm = input("Are you sure you want to exit? [y]es / [n]o: ").strip().lower()
                if confirm == "y":
                    print("\n👋 Goodbye!")
                    sys.exit(0)
            case _:
                print("\n❌ Unknown option. Please try again.")


if __name__ == "__main__":
    main()
