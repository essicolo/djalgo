import os
import itertools
import warnings
import json
import numpy as np
from numba import jit # Just-in-time compilation for faster execution of the Mandelbrot fractal generator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.axes
from . import utils

class Fibonacci:
    """
    A class that generates a Fibonacci sequence.

    Args:
        scale (list or None): A list of values to scale the generated sequence to. If None, the sequence is not scaled.
        length (int): The length of the generated sequence.

    Attributes:
        scale (list or None): A list of values to scale the generated sequence to. If None, the sequence is not scaled.
        length (int): The length of the generated sequence.

    Raises:
        UserWarning: If the length is greater than 10 and the scale is not provided, a warning is raised.

    """

    def __init__(self, scale=None, length=6):
        self.scale = scale
        self.length = length

        if (self.length > 10) and (self.scale is None):
            warnings.warn("Fibonacci numbers grow rapidly. Since the outcome is scaled to the specified scale, the first note will be repeated a large amount of times, hence creating a monotone piece.", UserWarning)
        
    def generate(self):
        """
        Generates a Fibonacci sequence.

        Returns:
            list: The generated Fibonacci sequence.

        """
        fibonacci_number = [1, 1]  # Initialize the Fibonacci sequence with the first two numbers
        i = 2
        while i < self.length:
            # Fibonacci sequence
            fibonacci_next_number = fibonacci_number[i-1] + fibonacci_number[i-2]
            fibonacci_number.append(fibonacci_next_number)

            # Increment loop
            i += 1

        if self.scale is None:
            signal = fibonacci_number
        else:
            signal = np.min(self.scale) + (np.array(fibonacci_number) - 1) / (np.max(np.array(fibonacci_number)) - 1) * (np.max(self.scale) - np.min(self.scale))
            signal = signal.tolist()
            signal = [utils.nearest_neighbor(i, self.scale) for i in signal]
        
        return signal

class CellularAutomata:
    """
    A class for simulating one-dimensional cellular automata based on a specific rule set.

    Args:
        rule_number (int or str): Rule number for the cellular automaton, must be between 0 and 255.
        width (int): Number of cells in the automaton's width.
        initial_state (list of int, optional): Initial state of the automaton. Defaults to all zeros with a central one.

    Attributes:
        width (int): The cellular automaton's width.
        rule_number (str): The rule number, zero-padded to three digits.
        initial_state (list of int): The initial state of the automaton.
        state (list of int): The current state of the automaton.
        rules (list of tuple): The rules loaded from a JSON file.
    """
    
    def __init__(self, rule_number, width, initial_state=None):
        """
        Initializes the CellularAutomata class with a rule number, width, and optionally an initial state.

        Parameters:
            rule_number (int or str): The rule number for the cellular automaton, must be between 0 and 255.
            width (int): The number of cells in one row of the automaton.
            initial_state (list, optional): Initial binary state of the automaton. Defaults to a list of zeros with a single one in the center.

        Raises:
            ValueError: If the rule number is not within the valid range (0 to 255).
        """
        self.width = width
        if int(rule_number) < 0 or int(rule_number) > 255:
            raise ValueError("Rule number must be an integer between 0 and 255.")
        if isinstance(rule_number, int):
            rule_number = str(rule_number).zfill(3)
        self.rule_number = rule_number
        self.rules = self.load_rules(rule_number)
        self.initial_state = [0] * width if initial_state is None else initial_state
        self.state = self.initial_state.copy()

    def load_rules(self, rule_number):
        """
        Loads the rules from a JSON file based on the rule number.

        Parameters:
            rule_number (str): The rule number as a string, zero-padded to three digits.

        Returns:
            list: A list of tuples representing the rules for the cellular automaton.
        """
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, 'data', 'ca1D_rules.json')

        with open(file_path, 'r') as file:
            rules_json = json.load(file)
        rule_conditions = rules_json[rule_number]
        return [tuple(condition) for condition in rule_conditions]

    def update_state(self):
        """
        Updates the state of the automaton based on its rules for one generation.
        """
        new_state = [0] * self.width
        for i in range(self.width):
            neighborhood = (self.state[(i - 1) % self.width],
                            self.state[i],
                            self.state[(i + 1) % self.width])
            new_state[i] = 1 if neighborhood in self.rules else 0
        self.state = new_state

    def validate_strips(self, strips):
        """
        Validates that the strips are correctly formatted and within the valid range.

        Parameters:
            strips (list of tuples): Each tuple should specify the start and end indices of a strip in the automaton.

        Raises:
            ValueError: If strips are not properly formatted or indices are out of range.
        """
        if isinstance(strips, tuple):
            strips = [strips]
        if not all(isinstance(strip, tuple) for strip in strips) or not all(len(strip) == 2 for strip in strips) or not all(isinstance(cell, int) for strip in strips for cell in strip):
            raise ValueError("Strips must be a tuple of two integers or a list of tuples with two integers each.")
        if not all(0 <= strip[0] < strip[1] <= self.width for strip in strips): 
            raise ValueError("Strip values must be within the width of the automata.")

    def validate_values(self, values, strips):
        """
        Validates that the values are provided as dictionaries, and there is a dictionary for each strip.

        Parameters:
            values (list or dict): List of dictionaries mapping indices to pitches or other data.
            strips (list of tuples): List of strip ranges to validate against.

        Raises:
            ValueError: If values are not provided as a list of dictionaries or their count doesn't match the strips.
        """
        if isinstance(values, dict):
            values = [values]
        if not isinstance(values, list) or not all(isinstance(val_dict, dict) for val_dict in values):
            raise ValueError("Values must be provided as a dictionnary or a list of dictionaries mapping indices to pitches or other data.")
        if len(values) != len(strips):
            raise ValueError("The number of value dictionaries must match the number of strips.")

    def generate_01(self, iterations, strips=None):
        """
        Generates a binary (0 or 1) evolution of the automaton over a specified number of iterations, optionally for specific strips.

        Parameters:
            iterations (int): Number of iterations to evolve the automaton.
            strips (list of tuples, optional): Specific sections of the automaton to evolve.

        Returns:
            list: A list representing the evolution of the automaton, either as a whole or just the specified strips.
        """
        if strips:
            self.validate_strips(strips)
        self.state = self.initial_state.copy()
        evolution = [self.state.copy()]
        for _ in range(iterations - 1):
            self.update_state()
            evolution.append(self.state.copy())
        
        if strips:
            strip_evolutions = []
            for strip in strips:
                strip_evolution = [row[strip[0]:strip[1]] for row in evolution]
                strip_evolutions.append(strip_evolution)
            return strip_evolutions
        
        return evolution
    
    def generate(self, iterations, strips, values):
        """
        Generates the evolution of the automaton, mapping binary states (0 or 1) to specific values based on provided mappings.

        Parameters:
            iterations (int): Number of iterations to evolve the automaton.
            strips (list of tuples): Sections of the automaton for which to generate values.
            values (list of dicts): Mappings from indices in each strip to specific values.

        Returns:
            list: A list of evolutions for each strip, with binary states mapped to specified values.
        """

        # Validate strips and value mapping
        self.validate_strips(strips)
        self.validate_values(values, strips)
        if isinstance(values, dict):
            values = [values]

        # Generate the binary evolution for specified strips
        evolution_01 = self.generate_01(iterations, strips)
        values_evolution = []
        for strip_evolution, value_dict in zip(evolution_01, values):  # Process each strip with its corresponding value map
            strip_values = []
            for row in strip_evolution:
                row_values = []
                for i, cell in enumerate(row):
                    if cell == 1 and i in value_dict:
                        row_values.append(value_dict[i])
                if row_values:
                    if len(row_values) == 1:
                        strip_values.append(row_values[0])  # Single value
                    else:
                        strip_values.append(row_values)  # Multiple values forming a chord
                else:
                    strip_values.append(None)
            values_evolution.append(strip_values)
        if len(values_evolution) == 1:
            return values_evolution[0]
        else:
            return values_evolution

    def plot(self, iterations, ax=None, strips=None, extract_strip=False, title=None, show_axis=True):
        """
        Plots the evolution of the cellular automaton.

        Parameters:
            iterations (int): Number of generations to simulate.
            ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new figure is created.
            strips (list of tuples, optional): Ranges to highlight or exclusively plot.
            extract_strip (bool): If True, only the specified strips are plotted each in separate subplots.
            title (str, optional): Title for the plot. Default is based on the rule number.
            show_axis (bool): Whether to show axis labels and grid.

        Returns:
            matplotlib.axes.Axes: The axis with the plot.
        """

        # Handle based on the extraction flag
        if not extract_strip:
            if title is None:
                title = f"Cellular Automata Evolution for Rule {self.rule_number}"
            # Display the complete evolution
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 8))
            evolution = np.array(self.generate_01(iterations)).T
            im = ax.imshow(evolution, cmap='binary', aspect='equal')
            ax.invert_yaxis()
            ax.set_title(title)
            ax.set_ylabel("Cell Position")
            ax.set_xlabel("Generation")

            # Add strips if provided
            if strips:
                label_offset = 3
                for index, strip in enumerate(strips):
                    rect = patches.Rectangle(
                        (0, strip[0]),
                        iterations, strip[1] - strip[0], linewidth=1, edgecolor='none', facecolor='#88888880'
                    )
                    ax.add_patch(rect)
                    ax.text(
                        1, strip[0] + label_offset, f'Strip {index + 1}',
                        color='white', fontsize=10, verticalalignment='top',
                        bbox=dict(facecolor='#333333', edgecolor='none', pad=2)
                    )
        else:
            # Handle multiple strips, each in its own subplot
            if not strips or len(strips) == 1:
                fig, axs = plt.subplots(figsize=(12, 8))  # Use a new axis for a single strip
            else:
                fig, axs = plt.subplots(len(strips), 1, figsize=(12, 2 * len(strips)**(0.5) ))
                axs = axs.flatten()

            if isinstance(axs, matplotlib.axes.Axes):
                axs = [axs]

            strip_evolutions = self.generate_01(iterations, strips)
            for i, strip_data in enumerate(strip_evolutions):
                axs[i].imshow(np.array(strip_data).T, cmap='binary', aspect='equal')
                axs[i].invert_yaxis()
                if title is None:
                    strip_title = f"Strip {i + 1} Evolution" 
                else:
                    strip_title = title[i]
                axs[i].set_title(strip_title)
                axs[i].set_ylabel("Cell Position")
                axs[i].set_xlabel("Generation")

        # Optionally turn off axis
        if not show_axis:
            ax.axis('off')

        # If this function created the figure and it's handling a subplot setup, adjust the layout
        if extract_strip:
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.5)  # Adjust horizontal space if needed

        return ax
    

@jit
def generate_mandelbrot_jit(x_range, y_range, dimensions, max_iter):
    x_lin = np.linspace(x_range[0], x_range[1], dimensions[0])
    y_lin = np.linspace(y_range[0], y_range[1], dimensions[1])
    output = np.zeros(dimensions, dtype=np.int32)  # Initialize the output array

    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            x = x_lin[i]
            y = y_lin[j]
            C = complex(x, y)  # Ensure using complex number
            Z = 0 + 0j  # Initialize Z as a complex zero
            count = 0  # Initialize escape time count

            while abs(Z) < 2 and count < max_iter:
                Z = Z**2 + C
                count += 1
            
            output[i, j] = count  # Set the output based on the escape time
    
    return output

class Mandelbrot:
    def __init__(self, scale=None, start_note_index=0, dimensions=(800, 800), max_iter=1000, x_range=(-2.0, 1.0), y_range=(-1.5, 1.5)):
        if isinstance(dimensions, int):
            dimensions = (dimensions, dimensions)
        if not isinstance(dimensions, tuple) or not isinstance(max_iter, int):
            raise ValueError("Dimensions must be a tuple and max_iter must be an integer.")
        if dimensions[0] <= 0 or dimensions[1] <= 0 or max_iter <= 0:
            raise ValueError("Dimensions and max_iter must be positive.")
        
        self.scale = scale
        self.start_note_index = start_note_index
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.x_range = x_range
        self.y_range = y_range

    def generate_mandelbrot(self):
        return generate_mandelbrot_jit(self.x_range, self.y_range, self.dimensions, self.max_iter)
    
    def generate(self, method='horizontal', line_index=0):
        data = self.generate_mandelbrot()
        if method == 'horizontal':
            return data[line_index, :]
        elif method == 'vertical':
            return data[:, line_index]
        elif method == 'diagonal':
            return np.diagonal(data)
        elif method == 'random':
            flat_data = data.flatten()
            return np.random.choice(flat_data, size=100, replace=False)

    def plot(self, ax=None, figsize=(10, 10), zoom_rect=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        fractal = self.generate_mandelbrot()
        im = ax.imshow(
            fractal.T,
            extent=(self.x_range[0], self.x_range[1], self.y_range[0], self.y_range[1]),
            cmap='hot', aspect='equal', origin='lower'
        )
        ax.set_title('Mandelbrot Set')
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        #plt.colorbar(im, ax=ax, orientation='vertical')
        if zoom_rect:
            rect = patches.Rectangle(
                (zoom_rect[0][0], zoom_rect[1][0]),
                zoom_rect[0][1] - zoom_rect[0][0], zoom_rect[1][1] - zoom_rect[1][0],
                linewidth=1, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
        return ax


class LogisticMap:
    """
    Represents a logistic map generator.

    Attributes:
        scale (list or None): A list of values to map the generated logistic map values to. If None, the raw values will be used.
        start_note_index (int): The starting index in the scale list.
        length (int): The length of the generated logistic map signal.
        r (float): The r parameter of the logistic map. Determines the chaos of the generated sequence.
        x0 (float): The initial x value for the logistic map.
        center_offset (bool): Whether to offset the generated values by the center of the scale list.
    """

    def __init__(self, scale, start_note_index=0, length=10, r=3.9, x0=0.5, center_offset=False):
        self.scale = scale
        self.start_note_index = start_note_index
        self.length = length
        self.r = r
        self.x0 = x0
        self.center_offset = center_offset

    def generate(self):
        logistic_map_values = []
        x = self.x0

        for _ in range(self.length):
            x = self.r * x * (1 - x)
            logistic_map_values.append(x)

        if self.scale is None:
            signal = logistic_map_values
        else:
            signal = []
            current_note_index = self.start_note_index
            for offset in itertools.islice(itertools.cycle(logistic_map_values), self.length):
                if self.center_offset:
                    current_note_index = (current_note_index + int(offset * len(self.scale))) - len(self.scale) // 2
                else:
                    current_note_index = (current_note_index + int(offset * len(self.scale))) % len(self.scale)
                signal.append(self.scale[current_note_index])

        return signal