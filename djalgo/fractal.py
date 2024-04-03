import itertools
import warnings
import json
import numpy as np
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
    Represents a cellular automaton.

    Args:
        scale (list): A list of values representing the possible states of each cell.
        length (int, optional): The number of generations to generate. Defaults to 10.
        init (list, optional): The initial state of the automaton. Defaults to None.
        rules (str or list, optional): The rules to apply for the automaton. Defaults to 'default_rule'.

    Raises:
        ValueError: If the specified rule is not found in the JSON file.

    Attributes:
        scale (list): A list of values representing the possible states of each cell.
        length (int): The number of generations to generate.
        init (list): The initial state of the automaton.
        rules (list): The rules to apply for the automaton.
        array (list): The generated cellular automaton.

    """

    def __init__(self, scale, length=10, init=None, rules='default_rule'):
        self.scale = scale
        self.length = length

        if isinstance(rules, str):
            with open("data/ca1D_rules.json", "r") as file:
                ca1D_rules = json.load(file)
                if rules in ca1D_rules:
                    rules = ca1D_rules[rules]
                else:
                    raise ValueError(f"Rule {rules} not found in JSON file.")

        self.rules = rules

        if init is None:
            init = [0] * len(scale)
            init[len(scale) // 2] = 1
        self.init = init

        self.array = self.create_ca1D()

    def create_ca1D(self):
        """
        Creates a 1D cellular automaton.

        Returns:
            list: The generated cellular automaton.

        """
        ncells = len(self.scale)
        nrows = self.length
        array = [[0] * ncells for _ in range(nrows)]
        array[0] = self.init
        for i in range(1, nrows):
            for j in range(ncells):
                ca_at = [array[i - 1][j - 1], array[i - 1][j], array[i - 1][(j + 1) % ncells]]
                if ca_at in self.rules:
                    array[i][j] = 1
        return array

    def generate(self):
        """
        Generates a signal based on the cellular automaton.

        Returns:
            list: The generated signal.

        """
        nevents = self.length
        ncells = len(self.scale)
        signal = []
        for i in range(nevents):
            notes_i = []
            for j in range(ncells):
                if self.array[i][j] == 1:
                    notes_i.append(self.scale[j])
            if len(notes_i) == 0:
                signal.append(None)
            elif len(notes_i) == 1:
                signal.append(notes_i[0])
            else:
                signal.append(notes_i)
        return signal


class Mandelbrot:
    """
    Represents a Mandelbrot fractal generator.
    
    Attributes:
        scale (list or None): A list of values to map the generated fractal values to. If None, the raw values will be used.
        start_note_index (int): The starting index in the scale list.
        length (int): The length of the generated fractal signal.
        max_iter (int): The maximum number of iterations to perform for each point in the fractal.
        x_min (float): The minimum x-coordinate value of the fractal.
        x_max (float): The maximum x-coordinate value of the fractal.
        y_min (float): The minimum y-coordinate value of the fractal.
        y_max (float): The maximum y-coordinate value of the fractal.
        center_offset (bool): Whether to offset the generated values by the center of the scale list.
    """

    def __init__(self, scale, start_note_index=0, length=10, max_iter=1000, x_min=-2.0, x_max=1.0, y_min=-1.5, y_max=1.5, center_offset=False):
        self.scale = scale
        self.start_note_index = start_note_index
        self.length = length
        self.max_iter = max_iter
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.center_offset = center_offset
    
    def generate(self):        
        mandelbrot_values = []
        
        for i in range(self.length):
            x = self.x_min + (self.x_max - self.x_min) * ((i % int(self.length ** 0.5)) / (self.length ** 0.5))
            y = self.y_min + (self.y_max - self.y_min) * ((i // int(self.length ** 0.5)) / (self.length ** 0.5))
            
            z = 0 + 0j
            c = complex(x, y)
            for _ in range(self.max_iter):
                if abs(z) > 2.0:
                    break 
                z = z * z + c
            
            if self.scale is None:
                mandelbrot_value = _  # Append the raw value to the list
            else:
                if self.center_offset:
                    mandelbrot_value = (_ % len(self.scale)) - len(self.scale) // 2
                else:
                    mandelbrot_value = _ % len(self.scale)
                
            mandelbrot_values.append(mandelbrot_value)
        
        if self.scale is None:
            signal = mandelbrot_values
        else:
            signal = []
            current_note_index = self.start_note_index
            for offset in itertools.islice(itertools.cycle(mandelbrot_values), self.length):
                current_note_index = (current_note_index + offset) % len(self.scale)
                signal.append(self.scale[current_note_index])
            
        return signal