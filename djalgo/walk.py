import random
import numpy as np


class Chain:
    """
    A class representing a chain of sequences generated using random walks.

    Attributes:
        walk_range (list): The range of step choices for the random walk. Defaults to [-1, 0, 1].
        walk_start (int): The starting position for the random walk. Defaults to the middle value of walk_range.
        walk_probability (list or object): The probability distribution for choosing a step in the random walk.
            Can be a list of step choices or a random variable object. Defaults to [-1, 0, 1].
        branching_probability (float): The probability of branching at each step. Defaults to 0.0.
        merging_probability (float): The probability of merging sequences with the same last value. Defaults to 0.0.
    """

    def __init__(self, walk_range=None, walk_start=None, walk_probability=None, round_to=None, branching_probability=0.0, merging_probability=0.0):
        self.walk_range = walk_range
        self.walk_start = walk_start if walk_start is not None else (self.walk_range[1] - self.walk_range[0]) // 2
        self.walk_probability = walk_probability or [-1, 0, 1]  # Default step choices
        self.branching_probability = branching_probability
        self.merging_probability = merging_probability
        self.round_to = round_to

    def generate(self, length, seed=None):
        """
        Generates a chain of sequences using random walks.

        Args:
            length (int): The length of each sequence in the chain. Defaults to 10.
            seed (int): The seed value for the random number generator. Defaults to None.

        Returns:
            list: A list of sequences generated using random walks.
        """
        self.length = length
        random.seed(seed)
        np.random.seed(seed)
        sequences = [[self.walk_start]]
        for _ in range(self.length - 1):
            new_sequences = []
            for seq in sequences:
                if seq[-1] is not None:  # Check if sequence is active
                    # Find the index of the last value in the scale
                    last_value = seq[-1]

                    if isinstance(self.walk_probability, list):
                        step = random.choice(self.walk_probability)  # Choose a random step
                    elif hasattr(self.walk_probability, 'rvs'):
                        step = self.walk_probability.rvs()  # Draw a sample and round to nearest integer
                    
                    if self.round_to is not None:
                        step = round(step, self.round_to)

                    next_value = last_value + step  # Calculate next index

                    # Handle boundary conditions for next_value
                    if self.walk_range is not None:
                        if next_value < self.walk_range[0]:
                            next_value = self.walk_range[0]
                        elif next_value > self.walk_range[1]:
                            next_value = self.walk_range[1]

                    # Branching decision
                    if random.random() < self.branching_probability:
                        # Create new branch with None values up to the current length
                        new_branch = [None for _ in range(len(seq))]  
                        new_branch.append(next_value)  # Start new branch from next value value
                        new_sequences.append(new_branch)  # Add new branch
                        seq.append(next_value)  # Continue current sequence with value value
                    else:
                        seq.append(next_value)  # Continue without branching with value value
                    
            # Handle merging
            if random.random() < self.merging_probability:
                unique_values = set(s[-1] for s in sequences if s[-1] is not None)
                for value in unique_values:
                    value_seqs = [s for s in sequences if s[-1] == value]
                    if len(value_seqs) > 1:
                        # Keep the longest sequence, close others
                        longest_seq = max(value_seqs, key=len)
                        for s in value_seqs:
                            if s != longest_seq:
                                s[len(s):] = [None] * (self.length - len(s))  # Extend with None without overwriting last value
            sequences.extend(new_sequences)  # Incorporate new branches
            
        # Ensure all sequences have proper length
        for seq in sequences:
            if len(seq) < self.length:
                seq.extend([None] * (self.length - len(seq)))  # Fill with None to reach desired length

        return sequences  # Return the updated sequences
    

class Kernel:
    """
    A class representing a kernel for generating sequences.

    Attributes:
        walk_around (float): The mean value around which the sequence will walk.
        data (ndarray): The input data used for generating the sequence.
        length_scale (float): The length scale parameter of the kernel.
        amplitude (float): The amplitude parameter of the kernel.
    """

    def __init__(self, walk_around=0.0, length_scale=1.0, amplitude=1.0):
        self.walk_around = walk_around
        self.length_scale = length_scale
        self.amplitude = amplitude

    def rbf_kernel(self, dimension, length_scale):
        """
        Computes the radial basis function (RBF) kernel matrix.

        Args:
            dimension (int): The dimension of the kernel matrix.
            length_scale (float): The length scale parameter of the kernel.

        Returns:
            ndarray: The RBF kernel matrix.
        """
        covariance_matrix = np.zeros((dimension, dimension))
        for i in range(dimension):
            for j in range(dimension):
                distance_squared = (i - j) ** 2
                covariance_matrix[i, j] = np.exp(-distance_squared / (2 * length_scale ** 2))
                
        return covariance_matrix

    def generate(self, length=10, data=None, nsamples=1, seed=None):
        """
        Generates a sequence using the kernel.

        Args:
            length (int): The length of the sequence.
            data (ndarray): The input data used for generating the sequence.
            nsamples (int): The number of samples to generate.
            seed (int): The seed value for random number generation.

        Returns:
            list: The generated sequence.
        """
        self.length = length
        if data is not None:
            if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 2:
                raise ValueError("data must be a two-dimensional NumPy array with two columns.")
        if not (isinstance(self.length_scale, (float, int)) and self.length_scale > 0):
            raise ValueError("length_scale must be a positive int or float.")
        if not (isinstance(self.amplitude, (float, int)) and self.amplitude > 0):
            raise ValueError("amplitude must be a positive int or float.")
        if not (isinstance(nsamples, int) and nsamples > 0):
            raise ValueError("nsamples must be a positive integer.")
        
        self.data = data

        random.seed(seed)
        np.random.seed(seed)

        if self.data is None:
            kernel_cov = self.walk_around + self.amplitude * self.rbf_kernel(dimension = self.length, length_scale = self.length_scale)
            
            sequence = []
            for _ in range(nsamples):
                sequence.append(np.random.multivariate_normal(
                    mean=np.repeat(self.walk_around, self.length),
                    cov=kernel_cov
                    ).tolist()
                )
        else:
            try:
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import RBF
            except ImportError:
                raise ImportError("It seems that scikit-learn is not installed. Please install it using pip install scikit-learn or using your favorite installer.")
            x = np.linspace(0, self.data[:, 0].max(), self.length)[:, np.newaxis]
            kernel = self.amplitude * RBF(length_scale=self.length_scale) # , length_scale_bounds = (self.length_scale*0.9, self.length_scale*1.1)

            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)
            gp.fit(data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1))
            sequence = [
                x.flatten().tolist(),
                gp.sample_y(x, n_samples=nsamples).T.tolist()
            ]
        
        return sequence
    
class CelestialBody:
    """
    Represents a celestial body in space.

    Attributes:
        distance (float): The distance of the celestial body from its parent body.
        orbital_speed (float): The orbital speed of the celestial body.
        phase (float, optional): The phase of the celestial body's orbit (default is 0).
        moons (list, optional): A list of CelestialBody objects representing the moons of the celestial body (default is None).
    """

    def __init__(self, distance, orbital_speed, phase=0, moons=None):
        """
        Initializes a new instance of the CelestialBody class.

        Args:
            distance (float): The distance of the celestial body from its parent body.
            orbital_speed (float): The orbital speed of the celestial body.
            phase (float, optional): The phase of the celestial body's orbit (default is 0).
            moons (list, optional): A list of CelestialBody objects representing the moons of the celestial body (default is None).
        """
        self.distance = distance
        self.orbital_speed = orbital_speed
        self.phase = phase
        self.moons = moons if moons else []

    def position(self, times):
        """
        Calculates the position of the celestial body at the given times.

        Args:
            times (array-like): An array-like object containing the times at which to calculate the position.

        Returns:
            tuple: A tuple containing the x and y coordinates of the celestial body's position at the given times.
        """
        # Vectorize the position calculations
        angles = self.orbital_speed * times + self.phase
        x = self.distance * np.cos(angles)
        y = self.distance * np.sin(angles)
        return x, y

    def simulate(self, times, parent_position=None):
        """
        Simulates the motion of the celestial body over the given times.

        Args:
            times (array-like): An array-like object containing the times at which to simulate the motion.
            parent_position (tuple, optional): A tuple containing the x and y coordinates of the parent body's position (default is None).

        Returns:
            list: A list of tuples representing the distances and angles of the celestial body at the given times.
        """
        if parent_position is None:
            parent_position = (np.zeros_like(times), np.zeros_like(times))

        x, y = self.position(times)
        x_total = parent_position[0] + x
        y_total = parent_position[1] + y

        results = []
        for moon in self.moons:
            results.extend(moon.simulate(times, (x_total, y_total)))

        if not self.moons:
            distances = np.sqrt(x_total**2 + y_total**2)
            angles = np.arctan2(y_total, x_total) * 180 / np.pi  # Convert to degrees
            return [(distances, angles % 360)]  # Ensure angles are from 0 to 360
        return results

class SolarSystem:
    """
    Represents a solar system.

    Attributes:
        planets (list): A list of celestial bodies in the solar system.
    """

    def __init__(self):
        """
        Initializes a new instance of the SolarSystem class.
        """
        self.planets = []

    def add_planet(self, planet):
        """
        Adds a new planet to the solar system.

        Args:
            planet (CelestialBody): A CelestialBody object representing the planet to add.
        """
        self.planets.append(planet)

    def simulate(self, times):
        """
        Simulates the motion of all planets in the solar system.

        Args:
            times (list): A list of time points at which to simulate the motion.

        Returns:
            list: A list of simulation results for all planets.
        """
        results = []
        for planet in self.planets:
            results.extend(planet.simulate(times))
        return results