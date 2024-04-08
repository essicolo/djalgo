import random
import numpy as np
from . import utils

def isorhythm(durations, pitches):
    """
    Merges durations and pitches until both ends coincide, then sets offsets according to successive durations.

    Args:
        durations (list): The first list.
        pitches (list): The second list.

    Returns:
        list: The zipped list.
    """
    lcm = np.lcm(len(pitches), len(durations))

    p_repeated = (pitches * (lcm // len(pitches)))[:lcm]
    d_repeated = (durations * (lcm // len(durations)))[:lcm]
    o = [1] * lcm
    notes = list(zip(p_repeated, d_repeated, o))
    notes = utils.set_offsets_according_to_durations(notes)

    return notes

class Rhythm:
    """
    A class used to represent a Rhythm.

    ...

    Attributes
    ----------
    measure_length : int
        the length of the measure
    output : str
        the output type, either 'duration' or 'time' (default 'duration')

    Methods
    -------
    _convert_to_output(durations):
        Converts the durations to the specified output type.
    random(durations, max_iter=100, seed=None):
        Generates a random rhythm that fits within the measure length.
    fibonacci(base=1):
        Generates a rhythm based on the Fibonacci sequence.
    """

    def __init__(self, measure_length, durations):
        """
        Constructs all the necessary attributes for the Rhythm object.

        Parameters
        ----------
            measure_length : int
                the length of the measure
            output : str
                the output type, either 'duration' or 'time' (default 'duration')
        """
        self.measure_length = measure_length
        self.durations = durations

    def random(self, seed=None, rest_probability=0, max_iter=100):
        """
        Generate a random rhythm as a list of (duration, offset) tuples.

        Args:
            duration (list): List of possible durations.
            measure_length (float): Total length of the measure.
            rest_probability (float): Probability of a rest (i.e., removing a tuple).
            max_iter (int): Maximum number of iterations to generate the rhythm.

        Returns:
            list: List of (duration, offset) tuples representing the rhythm.
        """
        random.seed(seed)
        rhythm = []
        total_length = 0.0
        n_iter = 0
        while total_length < self.measure_length:
            if n_iter >= max_iter:
                print('Max iterations reached. The sum of the durations is not equal to the measure length.')
                break # Avoid infinite loops
            d = random.choice(self.durations)
            if total_length + d > self.measure_length:
                continue
            if random.random() < rest_probability:
                continue
            rhythm.append((d, total_length))
            total_length += d
            n_iter += 1
        return rhythm
    
    def darwin(self, seed=None, population_size=10, max_generations=50, mutation_rate=0.1):
        ga = GeneticRhythm(seed, population_size, self.measure_length, max_generations, mutation_rate, self.durations)
        best_rhythm = ga.generate()
        return best_rhythm

    def fibonacci(self, base=1, reverse=False):
        fib = Fibonacci(self.measure_length, self.durations, reverse=reverse)
        return fib.generate()


class GeneticRhythm:
    def __init__(self, seed, population_size, measure_length, max_generations, mutation_rate, durations):
        """
        Initializes a Rhythm Genetic Algorithm instance.
        
        Args:
            population_size (int): The number of rhythms in each generation.
            measure_length (int): The total length of the rhythm to be generated.
            max_generations (int): The maximum number of generations to evolve.
            mutation_rate (float): The probability of mutating a given rhythm.
            durations (list): List of possible note durations.
        """
        random.seed(seed)
        self.population_size = population_size
        self.measure_length = measure_length
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.durations = durations
        self.population = self.initialize_population()

    def initialize_population(self):
        """Initializes a population of random rhythms."""
        population = []
        for _ in range(self.population_size):
            rhythm = self.create_random_rhythm()
            population.append(rhythm)
        return population

    def create_random_rhythm(self):
        """
        Creates a random rhythm ensuring it respects the measure length and has no overlapping notes.
        
        Returns:
            list: A list of (duration, offset) tuples representing the rhythm.
        """
        rhythm = []
        total_length = 0
        while total_length < self.measure_length:
            remaining = self.measure_length - total_length
            note_length = random.choice(self.durations)
            if note_length <= remaining:
                rhythm.append((note_length, total_length))
                total_length += note_length
        return rhythm

    def evaluate_fitness(self, rhythm):
        """
        Evaluates the fitness of a rhythm based on how close it is to the total measure length.
        
        Args:
            rhythm (list): The rhythm to evaluate, represented as a list of (duration, offset) tuples.
        
        Returns:
            int: The fitness score of the rhythm.
        """
        total_length = sum(note[0] for note in rhythm)  # Use note[0] for duration
        return abs(self.measure_length - total_length)

    def select_parents(self):
        """
        Selects two parents for reproduction using a simple random selection approach.
        
        Returns:
            tuple: Two selected parent rhythms for crossover.
        """
        parent1 = random.choice(self.population)
        parent2 = random.choice(self.population)
        return parent1 if self.evaluate_fitness(parent1) < self.evaluate_fitness(parent2) else parent2

    def crossover(self, parent1, parent2):
        """
        Performs crossover between two parent rhythms to produce a new child rhythm.
        
        Args:
            parent1 (list): The first parent rhythm.
            parent2 (list): The second parent rhythm.
        
        Returns:
            list: The new child rhythm generated from the parents.
        """
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return self.ensure_measure_length(child)

    def ensure_measure_length(self, rhythm):
        """
        Ensures that the rhythm respects the measure length, adjusting if necessary.
        
        Args:
            rhythm (list): The rhythm to check and adjust.
        
        Returns:
            list: The adjusted rhythm.
        """
        total_length = sum(note[0] for note in rhythm)  # Changed to note[0] for duration since we're working with (duration, offset) now
        if total_length > self.measure_length:
            rhythm.pop()  # Remove the last note if the total duration exceeds the measure length
        return rhythm

    def mutate(self, rhythm):
        """
        Performs a mutation on a rhythm with a certain probability, ensuring no note overlap.
        
        Args:
            rhythm (list): The rhythm to mutate.
        
        Returns:
            list: The mutated rhythm.
        """
        if random.random() < self.mutation_rate:
            index = random.randint(0, len(rhythm) - 2)  # Avoid mutating the last note for simplicity
            duration, offset = rhythm[index]
            next_offset = self.measure_length if index == len(rhythm) - 1 else rhythm[index + 1][1]
            max_new_duration = next_offset - offset
            new_durations = [d for d in self.durations if d <= max_new_duration]
            if new_durations:
                new_duration = random.choice(new_durations)
                rhythm[index] = (new_duration, offset)
        return rhythm

    def generate(self):
        """
        Executes the genetic algorithm, evolving the rhythms over generations.
        
        Returns:
            list: The best rhythm found after the last generation, sorted by ascending offset.
        """
        for generation in range(self.max_generations):
            new_population = []
            for _ in range(self.population_size):
                parent1 = self.select_parents()
                parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                child_sorted = sorted(child, key=lambda x: x[1])  # x[1] est l'offset dans le tuple (duration, offset)
                new_population.append(child_sorted)
            self.population = new_population

        best_rhythm = min(self.population, key=self.evaluate_fitness)
        best_rhythm_sorted = sorted(best_rhythm, key=lambda x: x[1])  # Trier par offset ascendant
        return best_rhythm_sorted


class Fibonacci:
    def __init__(self, measure_length, possible_durations, max_fibonacci=21, reverse=False):
        """
        Initializes the Fibonacci class to generate rhythms based on the Fibonacci sequence.

        Args:
            measure_length (float): The total length of the measure.
            possible_durations (list): List of possible note durations.
            max_fibonacci (int): The maximum Fibonacci number to consider for the sequence.
            reverse (bool): If True, the generated rhythm will have increasing durations.
        """
        self.measure_length = measure_length
        self.possible_durations = possible_durations
        self.max_fibonacci = max_fibonacci
        self.reverse = reverse
        self.fib_sequence = self.generate_fibonacci_sequence()

    def generate_fibonacci_sequence(self):
        """Generates a Fibonacci sequence up to the maximum Fibonacci number."""
        fib_sequence = [1, 1]
        while fib_sequence[-1] + fib_sequence[-2] <= self.max_fibonacci:
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
        return fib_sequence

    def map_fibonacci_to_durations(self):
        """Maps Fibonacci numbers to musical durations based on the list of possible durations."""
        duration_sequence = []
        for fib_number in reversed(self.fib_sequence):
            possible_matches = [d for d in self.possible_durations if d <= fib_number]
            if possible_matches:
                duration_sequence.append(max(possible_matches))
        return duration_sequence if not self.reverse else reversed(duration_sequence)

    def generate(self):
        """Generates a rhythm based on the Fibonacci sequence and fits it into the measure."""
        duration_sequence = self.map_fibonacci_to_durations()
        rhythm = []
        total_length = 0
        offset = 0
        for duration in duration_sequence:
            if total_length + duration <= self.measure_length:
                rhythm.append((duration, offset))
                total_length += duration
                offset += duration
            else:
                break  # Stop if adding another note would exceed the measure length
        return list(rhythm)  # Ensure it's a list, especially if reversed
