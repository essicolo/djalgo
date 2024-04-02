import random
import numpy as np


class Chain:
    def __init__(self, walk_range=None, walk_start=None, walk_probability=None, length=10, branching_probability=0.0, merging_probability=0.0):
        self.walk_range = walk_range or [-1, 0, 1]  # Default step choices
        self.walk_start = walk_start if walk_start is not None else (self.walk_range[1] - self.walk_range[0]) // 2
        self.walk_probability = walk_probability or [-1, 0, 1]  # Default step choices
        self.length = length
        self.branching_probability = branching_probability
        self.merging_probability = merging_probability

    def generate(self, seed=None):
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
                    
                    next_value = last_value + step  # Calculate next index

                    # Handle boundary conditions for next_value
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
    def __init__(self, walk_around=0.0, length=10, data=None, length_scale=1.0, amplitude=1.0):
        self.walk_around = walk_around
        self.length = length
        self.data = data
        self.length_scale = length_scale
        self.amplitude = amplitude

    def rbf_kernel(self, dimension, length_scale):
        covariance_matrix = np.zeros((dimension, dimension))
        for i in range(dimension):
            for j in range(dimension):
                distance_squared = (i - j) ** 2
                covariance_matrix[i, j] = np.exp(-distance_squared / (2 * length_scale ** 2))
                
        return covariance_matrix

    def generate(self, data=None, nsamples=1, seed=None):
        if data is not None:
            if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 2:
                raise ValueError("data must be a two-dimensional NumPy array with two columns.")
        if not (isinstance(self.length_scale, float) and self.length_scale > 0):
            raise ValueError("length_scale must be a positive float.")
        if not (isinstance(self.amplitude, float) and self.amplitude > 0):
            raise ValueError("amplitude must be a positive float.")
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
            x = np.linspace(0, self.length-1, self.length)[:, np.newaxis]
            kernel = self.amplitude * RBF(length_scale=self.length_scale, length_scale_bounds = (self.length_scale*0.95, self.length_scale*1.05))

            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)
            gp.fit(data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1))
            sequence = gp.sample_y(x, n_samples=nsamples).T.tolist()
        
        return sequence