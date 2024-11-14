import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import importlib.util
import threading
from random import randint

from matplotlib.pyplot import title

from GeneticAlgorithm import GeneticAlgorithm  # Make sure this module is in your path
import numpy as np

class GeneticAlgorithmGUI:
    def __init__(self, root: tk.Tk):
        self.generations = None
        self.creature_class = None
        self.root = root
        self.root.title("Genetic Algorithm GUI")
        self.root.resizable(False, False)

        self.ga = None
        self.population: list = None
        self.inputs = []
        self.outputs = []
        self.fig, self.ax = plt.subplots()
        self.accuracy_line, = self.ax.plot([], [], 'r-')
        self.accuracy_line.set_label("Accuracy")
        self.generation_data = []
        self.accuracy_data = []

        self.running = False

        self.out_strvar = tk.StringVar(self.root, "Current / Expected")
        self.accuracy_strvar = tk.StringVar(self.root, "Accuracy")
        self.load_strvar = tk.StringVar(self.root, "Load Creature File")

        # Initialize UI components
        self.create_widgets()

    def stop(self):
        self.running = False

    def start(self):
        self.running = True

    def create_widgets(self):
        # Frame for Controls
        control_frame = tk.Frame(self.root)
        control_frame.grid(row=0, column=0, pady=10)

        # Load and Save Population
        save_population_btn = tk.Button(control_frame, text="Save Population", command=self.save_population)
        save_population_btn.grid(row=0, column=0, pady=10)

        load_population_btn = tk.Button(control_frame, text="Load Population", command=self.load_population)
        load_population_btn.grid(row=0, column=1, pady=10)

        # Load Creature File Button
        load_file_btn = tk.Button(control_frame, textvariable=self.load_strvar, command=self.load_creature_file)
        load_file_btn.grid(row=1, column=0, padx=10, pady=5, columnspan=2)

        # Population Size Entry
        self.create_labeled_entry(control_frame, "Population Size:", 2, 1, default_value="1000")

        # Generations Entry
        self.create_labeled_entry(control_frame, "Generations:", 3, 1, default_value="100")

        # Epochs Entry
        self.create_labeled_entry(control_frame, "Epochs: (-1 to work on Generations instead)", 4, 1,
                                  default_value=f"{-1}")

        # Start Training Button
        start_btn = tk.Button(control_frame, text="Start Training", command=self.start_training)
        start_btn.grid(row=5, column=0, columnspan=1, pady=10)

        # Stop Training Button
        stop_btn = tk.Button(control_frame, text="Stop Training", command=self.stop)
        stop_btn.grid(row=5, column=1, columnspan=1, pady=10)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(control_frame, length=300, mode="determinate")
        self.progress_bar.grid(row=7, column=0, columnspan=2, pady=10)

        # Matplotlib Plot Area
        plot_frame = tk.Frame(self.root)
        plot_frame.grid(row=1, column=0, pady=10)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=10, pady=10)

        output_label = tk.Label(plot_frame, textvariable=self.out_strvar)
        output_label.grid(row=1, column=0, columnspan=5, pady=10)

        current_accuracy_label = tk.Label(plot_frame, textvariable=self.accuracy_strvar)
        current_accuracy_label.grid(row=1, column=6, columnspan=5, pady=10)

        # Side Panel for Testing Best Creature
        self.create_side_panel()

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

    def create_labeled_entry(self, frame, label_text, row, column, default_value=""):
        """Helper function to create a labeled entry with default value and validation."""
        label = tk.Label(frame, text=label_text)
        label.grid(row=row, column=column - 1, padx=5, pady=5, sticky="e")
        entry = tk.Entry(frame, validate="key")
        entry.grid(row=row, column=column, padx=5, pady=5)
        entry.insert(0, default_value)
        entry.config(validatecommand=(self.root.register(self.validate_integer), '%P'))
        setattr(self, f"{label_text.split(':')[0].lower().replace(" ", "_")}_entry", entry)

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
                self.inputs = getattr(module, "INPUTS", None)
                self.outputs = getattr(module, "OUTPUTS", None)

                if not self.creature_class or not self.inputs or not self.outputs:
                    raise AttributeError("The file must contain 'Creature', 'INPUTS', and 'OUTPUTS'.")
                messagebox.showinfo("Success", "Successfully loaded Creature class and data.")
                self.load_strvar.set("Load Creature File (loaded)")
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
        self.ga = GeneticAlgorithm(population_size=population_size, creature_class=self.creature_class)
        if not self.population is None:
            self.ga.population = self.population
        self.generations = generations
        self.progress_bar['maximum'] = self.generations

        self.generation_data = []
        self.accuracy_data = []
        self.progress_bar['value'] = 0

        self.start()
        threading.Thread(target=self.run_genetic_algorithm).start()

    def run_genetic_algorithm(self):
        """Run the genetic algorithm in a separate thread to keep GUI responsive."""
        for gen in range(1, self.generations + 1):
            if not self.running:
                break

            index = randint(0, len(self.inputs) - 1)
            self.ga.evolve(input_=self.inputs[index], expected_output=self.outputs[index])

            self.out_strvar.set(f"Current: {self.ga.best_output} / Expected: {self.outputs[index]}")
            self.accuracy_strvar.set(f"Accuracy: {1 / (1 + self.ga.best_fitness)}")
            self.progress_bar['value'] = gen
            self.update_progress(gen)
            self.update_plot()

        messagebox.showinfo("Training Complete", "Genetic Algorithm training is complete!")

    def update_progress(self, step):
        """Update generation and accuracy data for plotting."""
        self.generation_data.append(step)
        self.accuracy_data.append(self.ga.best_fitness)

    def update_plot(self):
        """Update the plot with new data."""
        self.accuracy_line.set_data(self.generation_data, self.accuracy_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_title("Accuracy Over Generations")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Accuracy")
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
        """Start Tkinter main loop."""
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticAlgorithmGUI(root)
    app.run()
