import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import logging
import os
from datetime import datetime
import importlib.util
from gatrainer.GeneticAlgorithm import GeneticAlgorithm
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
logger.info("Genetic Algorithm GUI initialized.")


class GeneticAlgorithmGUI:
    def __init__(self, root):
        self.training_thread = None
        self.root = root
        self.root.title("Genetic Algorithm GUI")
        self.ga = None
        self.inputs = []
        self.outputs = []
        self.generation_data = []
        self.fitness_data = []
        self.creature_class = None

        # Configure main GUI layout
        self.setup_gui()

    def setup_gui(self):
        """Sets up the Tkinter GUI layout."""
        frame_top = tk.Frame(self.root)
        frame_top.pack(pady=10)

        tk.Button(frame_top, text="Load Creature + IO File", command=self.load_creature_io_file).grid(row=0, column=0, padx=10)
        tk.Button(frame_top, text="Load Existing Algorithm", command=self.load_algorithm).grid(row=0, column=1, padx=10)
        tk.Button(frame_top, text="Start Training", command=self.start_training_gui).grid(row=0, column=2, padx=10)
        tk.Button(frame_top, text="Test Best Creature", command=self.test_best_creature).grid(row=0, column=3, padx=10)
        tk.Button(frame_top, text="Save Best Creature", command=self.save_best_creature).grid(row=0, column=4, padx=10)

        frame_middle = tk.Frame(self.root)
        frame_middle.pack(pady=10)
        tk.Button(frame_middle, text="Save Algorithm", command=self.save_algorithm).grid(row=0, column=0, padx=10)
        tk.Button(frame_middle, text="Export Metrics", command=self.export_metrics).grid(row=0, column=1, padx=10)

        # Add matplotlib figure for live updates
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Training Progress")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")
        self.line, = self.ax.plot([], [], label="Best Fitness")
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().pack(pady=20)

    def load_creature_io_file(self):
        """Load the Creature class and IO data from a specified file."""
        file_path = filedialog.askopenfilename(title="Select Creature + IO File")
        if not file_path:
            return

        try:
            fname = os.path.basename(file_path)
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

            messagebox.showinfo("Success", f"Successfully loaded {fname}")
            logger.info(f"Loaded Creature class, inputs, and outputs from {file_path}.")
        except Exception as e:
            logger.error(f"Error loading Creature, inputs, and outputs file: {e}")
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def start_training_gui(self):
        """Starts training with user-specified parameters."""
        if not self.creature_class or not self.inputs or not self.outputs:
            messagebox.showerror("Error", "Load a valid Creature file and IO data before starting.")
            return

        try:
            pop_size = int(self.simple_input_dialog("Enter Population Size"))
            generations = int(self.simple_input_dialog("Enter Number of Generations"))
            self.start_training_thread(generations, population_size=pop_size)
            self.update_graph_gui()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer values.")
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

    def start_training(self, generations, population_size=None, reset=False):
        """Start the genetic algorithm training."""
        if not self.ga or reset:
            self.ga = GeneticAlgorithm(
                population_size=population_size,
                creature_class=self.creature_class,
                reverse_fitness=self.creature_class.reverse_fitness
            )

        self.generation_data = []
        self.fitness_data = []

        try:
            for gen in range(1, generations + 1):
                start = time.process_time()
                index = random.randint(0, len(self.inputs)-1)
                self.ga.evolve(input_=self.inputs[index], expected_output=self.outputs[index])
                best_fitness = self.ga.best_fitness

                self.generation_data.append(gen)
                self.fitness_data.append(best_fitness)

                end = time.process_time()
                logger.info(f"Generation {gen}: Best Fitness={best_fitness}, Time Elapsed={end - start}")

                # Update graph
                self.update_graph()

            messagebox.showinfo("Success", "Training complete!")
            logger.info("Training complete.")
        except KeyboardInterrupt:
            logger.info("Training interrupted.")
            messagebox.showwarning("Stopped", "Training interrupted.")

    def start_training_thread(self, generations, population_size=None, reset=False):
        self.training_thread = threading.Thread(target=self.start_training, args=(generations, population_size, reset))
        self.training_thread.start()

    def update_graph(self):
        """Update Matplotlib graph with new data."""
        self.line.set_xdata(self.generation_data)
        self.line.set_ydata(self.fitness_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def update_graph_gui(self):
        if self.training_thread.is_alive():
            self.update_graph()
            self.root.after(200, self.update_graph_gui)

    @staticmethod
    def simple_input_dialog(prompt):
        """Prompt for a simple input."""
        return tk.simpledialog.askstring("Input", prompt)

    def save_best_creature(self):
        file_path = filedialog.asksaveasfilename(title="Save Best Creature")
        if not file_path or not self.ga:
            return
        try:
            self.ga.best_creature.save_to_file(file_path)
            messagebox.showinfo("Success", f"Best creature saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving best creature: {e}")
            messagebox.showerror("Error", f"Failed to save: {e}")

    def save_algorithm(self):
        file_path = filedialog.asksaveasfilename(title="Save Algorithm")
        if not file_path or not self.ga:
            return
        try:
            self.ga.save_algorithm(file_path)
            messagebox.showinfo("Success", f"Algorithm saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving algorithm: {e}")
            messagebox.showerror("Error", f"Failed to save: {e}")

    def load_algorithm(self):
        file_path = filedialog.askopenfilename(title="Load Algorithm")
        if not file_path:
            return
        try:
            self.ga = GeneticAlgorithm.load_from_file(file_path)
            messagebox.showinfo("Success", f"Algorithm loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading algorithm: {e}")
            messagebox.showerror("Error", f"Failed to load: {e}")

    def export_metrics(self):
        file_path = filedialog.asksaveasfilename(title="Export Metrics")
        if not file_path:
            return
        try:
            with open(file_path, 'w') as f:
                f.write("Generation,Fitness\n")
                for gen, fitness in zip(self.generation_data, self.fitness_data):
                    f.write(f"{gen},{fitness:.4f}\n")
            messagebox.showinfo("Success", f"Metrics exported to {file_path}")
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            messagebox.showerror("Error", f"Failed to export: {e}")

    def test_best_creature(self):
        try:
            if not self.ga:
                messagebox.showerror("Error", "Train a population before testing.")
                return

            size = self.simple_input_dialog("Enter test size: ")
            try:
                size = int(size)
            except ValueError:
                messagebox.showerror("Error", "Size must be an integer.")
                return
            fitness = 0
            for i in range(size):
                index = random.randint(0, len(self.inputs) - 1)
                self.ga.best_creature.process(self.inputs[index])
                self.ga.best_creature.fitness(self.outputs[index])

                fitness += self.ga.best_creature.get_fitness()
            messagebox.showinfo("Test Result", f"Average Fitness: {fitness/size} out of {size} tests")
        except Exception as e:
            logger.error(f"Error testing creature: {e}")
            messagebox.showerror("Error", f"Failed to test: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticAlgorithmGUI(root)
    root.mainloop()
