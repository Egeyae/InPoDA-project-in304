"""












#######################################################################################
#######################################################################################

THIS IS A WORK IN PROGRESS, FOR THE MOMENT, PLEASE USE THE CLI INTERFACE

#######################################################################################
#######################################################################################

























"""


import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import importlib.util
import threading

from random import randint
import logging
import os
from datetime import datetime
from GeneticAlgorithm import GeneticAlgorithm  # Ensure this is in your path
import numpy as np

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
    def __init__(self, window_root: tk.Tk):
        self.load_io_btn = None
        self.load_creature_btn = None
        self.io_module_name = None
        self.creature_module_name = None
        self.root = window_root
        self.root.title("Genetic Algorithm Trainer")
        self.root.geometry("1000x600")
        self.root.resizable(True, True)

        self.ga = None
        self.population = None
        self.inputs = []
        self.outputs = []
        self.generation_data = []
        self.accuracy_data = []
        self.running = False

        # Tkinter variables
        self.out_strvar = tk.StringVar(value="Current / Expected")
        self.accuracy_strvar = tk.StringVar(value="Accuracy")
        self.progress_percent_strvar = tk.StringVar(value="Progress: 0%")

        # Plot setup
        self.fig, self.ax = plt.subplots()
        self.accuracy_line, = self.ax.plot([], [], 'r-', label="Best Fitness")
        self.ax.set_title("Accuracy Over Generations")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")
        self.ax.legend(loc="upper right")

        self.create_widgets()

    def stop_training(self):
        self.running = False
        logger.info("Training manually stopped.")

    def start(self):
        self.running = True

    def create_widgets(self):
        # Layout
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        plot_frame = tk.Frame(self.root, padx=10, pady=10)
        plot_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Matplotlib Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # the plot will fill all the reserved space

        # Controls
        ttk.Label(control_frame, text="Genetic Algorithm Trainer", font=("Arial", 16)).grid(row=0, column=0,
                                                                                            columnspan=2, pady=10)

        self.load_creature_btn = ttk.Button(control_frame, text="Load Creature File", command=self.load_creature_file)
        self.load_creature_btn.grid(row=1, column=0, columnspan=1, pady=10)
        self.load_io_btn = ttk.Button(control_frame, text="Load Inputs/Outputs", command=self.load_inputs_outputs)
        self.load_io_btn.grid(row=1, column=1, columnspan=1, pady=10)

        self.create_labeled_entry(
            control_frame,
            "Population Size:",
            "population_size_entry",
            default_value="100",
            row=2, column=0
        )
        self.create_labeled_entry(
            control_frame,
            "Generations:",
            "generations_entry",
            default_value="50",
            row=3, column=0)
        self.create_labeled_entry(
            control_frame,
            "Epochs:",
            "epochs_entry",
            default_value="-1",
            row=4, column=0)

        self.progress_bar = ttk.Progressbar(control_frame, length=200, mode="determinate")
        self.progress_bar.grid(row=5, column=0, columnspan=2, pady=10)
        ttk.Label(control_frame, textvariable=self.progress_percent_strvar).grid(row=6, column=0, columnspan=2, pady=10)

        ttk.Button(control_frame, text="Start Training", command=self.start_training).grid(row=7, column=0,
                                                                                           columnspan=1, pady=10)
        ttk.Button(control_frame, text="Stop Training", command=self.stop_training).grid(row=7, column=1, columnspan=1,
                                                                                         pady=10)

        ttk.Button(control_frame, text="Export Metrics", command=self.export_metrics).grid(row=8, column=0,
                                                                                           columnspan=2, pady=10)

    def create_side_panel(self):
        """Creates a side panel to test the best creature."""
        side_panel = tk.Frame(self.root, borderwidth=1, relief="solid")
        side_panel.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="ns")

        # Label and Entry for custom input
        tk.Label(side_panel, text="Test Best Creature", font=("Arial", 12, "bold")).pack(pady=5)
        tk.Label(side_panel, text="Enter Input:").pack(pady=5)
        self.test_input_entry = tk.Entry(side_panel, width=20)
        self.test_input_entry.pack(pady=20)

        # Test Button
        test_btn = tk.Button(side_panel, text="Test", command=self.test_best_creature)
        test_btn.pack(pady=10)

        # Output Display
        self.test_output_strvar = tk.StringVar(side_panel, "Output: ")
        tk.Label(side_panel, textvariable=self.test_output_strvar, wraplength=200).pack(pady=10)

        # Save best creature
        save_best_creature_btn = tk.Button(side_panel, text="Save Best Creature", command=self.save_best_creature)
        save_best_creature_btn.pack(pady=10)

    def export_metrics(self):
        try:
            file_path = filedialog.asksaveasfilename(
                title="Export Metrics",
                filetypes=[("CSV Files", "*.csv")],
                defaultextension=".csv"
            )
            if not file_path:
                return

            with open(file_path, 'w') as f:
                f.write("Generation,Accuracy,Fitness\n")
                for gen, fitness in zip(self.generation_data, self.accuracy_data):
                    accuracy = 1 / (1 + fitness)
                    f.write(f"{gen},{accuracy:.4f},{fitness:.4f}\n")

            messagebox.showinfo("Success", f"Metrics exported to {file_path}")
            logger.info(f"Metrics exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export metrics: {e}")
            logger.error(f"Error exporting metrics: {e}")

    def save_best_creature(self):
        fpth = filedialog.asksaveasfilename(title="Save Best Creature", filetypes=[("Numpy Array", ".npy")])
        self.ga.best_creature.save_to_file(fpth)
        messagebox.showinfo("Success", f"Saved the current best model to {fpth}")

    def save_population(self):
        if self.ga and self.ga.population:
            fpth = filedialog.asksaveasfilename(title="Save Population", filetypes=[("Numpy Archive", ".npz")])
            arr = [x.dna for x in self.ga.population]
            np.savez_compressed(fpth, *arr)
            messagebox.showinfo("Success", f"Saved the current population to {fpth}")
            return
        messagebox.showwarning("Warning", "Please train a population before saving")

    def load_population(self):
        fpth = filedialog.askopenfilename(title="Load Population", filetypes=[("Numpy Archive", ".npz")])
        try:
            archive = np.load(fpth)
            self.population = [archive[x] for x in archive.files]
            self.population_size_entry.delete(0, tk.END)
            self.population_size_entry.insert(0, f"{len(self.population)}")
        except FileNotFoundError:
            messagebox.showwarning("Warning", "Population file not found")

    def create_labeled_entry(self, parent, label_text, attr_name, default_value="", row=0, column=0, span=2):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=column, columnspan=span, pady=10)
        ttk.Label(frame, text=label_text).pack(side=tk.LEFT, padx=5)
        entry = ttk.Entry(frame)
        entry.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=5)
        entry.insert(0, default_value)
        setattr(self, attr_name, entry)

    @staticmethod
    def validate_integer(input_str: str) -> bool:
        """Ensure input is an integer."""

        if input_str == "" or (input_str[0].lstrip('-+') + input_str[1:]).isdigit():
            return True
        return False

    def load_creature_file(self):
        """Load a user-defined Creature class and data."""
        file_path = filedialog.askopenfilename(title="Select Creature Python File",
                                               filetypes=[("Python files", "*.py")])
        if file_path:
            try:
                spec = importlib.util.spec_from_file_location("custom_module", file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                self.creature_class = getattr(module, "Creature", None)

                if not self.creature_class:
                    raise AttributeError(
                        "The file must contain a Creature class object, derived from GeneticAlgorithmTrainNN.GeneticAlgorithm.Creature if possible")
                messagebox.showinfo("Success", "Successfully loaded Creature class and data.")
                self.creature_module_name = file_path.split("/")[-1].split(".")[0]
                self.load_creature_btn.config(text="Unload creature", command=self.unload_creature_file)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
                logger.error(f"Error loading file: {e}")

    def unload_creature_file(self):
        if not self.creature_module_name is None:
            try:
                print(sys.modules)
                del self.creature_class
                del sys.modules[self.creature_module_name]
                logger.info(f"Unloaded {self.creature_module_name}")
                self.creature_module_name = None
                self.creature_class = None
                self.load_creature_btn.config(text="Load creature", command=self.load_creature_file)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to unload module: {e}")
                logger.error(f"Failed to unload module: {e}")

    def unload_inputs_outputs(self):
        if not (self.io_module_name is None):
            try:
                del self.inputs
                del self.outputs
                del sys.modules[self.io_module_name]
                logger.info(f"Unloaded {self.io_module_name}")
                self.io_module_name = None
                self.inputs = None
                self.outputs = None
                self.load_io_btn.config(text="Load Inputs/Outputs", command=self.load_inputs_outputs)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to unload module: {e}")
                logger.error(f"Failed to unload module: {e}")

    def load_inputs_outputs(self):
        file_path = filedialog.askopenfilename(title="Select Inputs and Outputs Python File",
                                               filetypes=[("Python files", "*.py")])
        if file_path:
            try:
                spec = importlib.util.spec_from_file_location("custom_module", file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                self.inputs = getattr(module, "INPUTS", None)
                self.outputs = getattr(module, "OUTPUTS", None)
                self.io_module_name = file_path
                if not self.inputs or not self.outputs:
                    raise AttributeError("The file must contain 'INPUTS', and 'OUTPUTS'.")
                self.load_io_btn.config(text="Unload Inputs/Outputs", command=self.unload_inputs_outputs)
                messagebox.showinfo("Success", "Successfully loaded inputs and outputs.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def start_training(self):
        """Initialize and start the training process."""
        try:
            population_size = int(self.population_size_entry.get())
            generations = int(self.generations_entry.get())
            epochs = int(self.epochs_entry.get())

            if epochs > 0:
                yes = messagebox.askyesno(title="Epoch or Generations",
                                          message=f"Epochs was specified, do you want to run on {epochs} epochs = {epochs * len(self.inputs)} generations ?")
                if yes:
                    generations = len(self.inputs) * epochs
                    self.generations_entry.delete(0, tk.END)
                    self.generations_entry.insert(0, str(generations))

        except ValueError:
            messagebox.showwarning("Warning",
                                   "Please enter valid integer values for Population Size, Generations, and Epochs.")
            return

        if not hasattr(self, 'creature_class') or not self.inputs or not self.outputs:
            messagebox.showwarning("Warning", "Please load a valid Creature file before starting.")
            return
        self.ga = GeneticAlgorithm(
            population_size=population_size,
            creature_class=self.creature_class,
            reverse_fitness=self.creature_class.reverse_fitness,
        )
        if not self.population is None:
            self.ga.population = self.population
        print(generations)
        self.generations = generations
        self.progress_bar['maximum'] = self.generations
        logger.info(
            f"Training started with Population Size: {population_size}, Generations: {generations}, Epochs: {epochs}")
        self.generation_data = []
        self.accuracy_data = []
        self.progress_bar['value'] = 0
        self.start()

        # Create a log of the initial population
        for idx, creature in enumerate(self.ga.population):
            logger.debug(f"Initial Population - Creature {idx + 1}: Fitness={creature.get_fitness()}")

        threading.Thread(target=self.run_genetic_algorithm).start()

    def run_genetic_algorithm(self):
        """Run the genetic algorithm in a separate thread to keep GUI responsive."""
        for gen in range(1, self.generations + 1):
            if not self.running:
                logger.warning(f"Training stopped manually at Generation {gen}")
                break

            index = randint(0, len(self.inputs) - 1)
            self.ga.evolve(input_=self.inputs[index], expected_output=self.outputs[index])
            logger.info(f"Generation {gen}: Best Fitness={self.ga.best_fitness}, Best Output={self.ga.best_output}")

            self.out_strvar.set(f"Current: {self.ga.best_output} / Expected: {self.outputs[index]}")
            self.accuracy_strvar.set(f"Accuracy: {1 / (1 + self.ga.best_fitness)}")
            self.update_progress(gen, self.generations)
            self.update_plot()
        logger.info("Training completed.")
        messagebox.showinfo("Training Complete", "Genetic Algorithm training is complete!")

    def update_progress(self, step, total):
        progress = int((step / total) * 100)
        self.progress_bar['value'] = step
        self.progress_percent_strvar.set(f"Progress: {progress}%")

        self.generation_data.append(step)
        self.accuracy_data.append(self.ga.best_fitness)
        logger.info(f"Progress updated: {progress}%")

    def update_plot(self):
        self.ax.clear()
        self.ax.plot(self.generation_data, self.accuracy_data, 'r-', label="Best Fitness")
        self.ax.set_title("Accuracy Over Generations")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")
        self.ax.legend(loc="upper right")
        self.canvas.draw()

    def test_best_creature(self):
        """Test the best creature using a custom input."""
        test_input = self.test_input_entry.get()
        try:
            if not test_input:
                raise ValueError("Please enter a valid input.")

            # Convert the input string to the expected input format of the GA
            test_input = eval(test_input)  # Use `ast.literal_eval` if the input format is strict

            # Evaluate the best creature
            self.ga.best_creature.process(test_input)
            best_creature_output = np.round(self.ga.best_creature.get_output())
            self.test_output_strvar.set(f"Output: {best_creature_output}")
        except Exception as e:
            self.test_output_strvar.set(f"Error: {str(e)}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticAlgorithmGUI(root)
    app.run()
