import numpy as np
from scipy.signal import correlate, find_peaks


class Index:
    """
    A class that performs various analysis on a list of values.

    Args:
        values (list): A list of numerical values.
        weights (list, optional): A list of weights corresponding to the values. Defaults to None.

    Attributes:
        values (list): A list of numerical values.
        weights (list): A list of weights corresponding to the values.

    """

    def __init__(self, values, weights=None): 
        if weights is None:
            self.weights = [1] * len(values)
        else:
            self.weights = weights
        
        # Pair values and weights to filter out None values jointly
        cleaned_data = [(v, w) for v, w in zip(values, self.weights) if v is not None]
        self.values, self.weights = zip(*cleaned_data) if cleaned_data else ([], [])

        # Initialize positions assuming sequential values
        self.positions = list(range(len(self.values)))

    def gini(self):
        """
        Calculates the Gini index of the values.

        Returns:
            float: The Gini index.
        """
        if not self.values or len(self.values) <= 1:
            return None
        values = [v * w for v, w in zip(self.values, self.weights)]
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumulative_sum = sum((i+1) * val for i, val in enumerate(sorted_values))
        total_sum = sum(sorted_values)
        gini = (2 * cumulative_sum) / (n * total_sum) - (n + 1) / n
        return gini

    def balance(self):
        """
        Calculates the balance index of the values.

        Returns:
            float: The balance index.

        """
        if not self.values or len(self.values) <= 1:
            return None
        weighted_positions = sum(pos * weight for pos, weight in zip(self.positions, self.weights))
        total_weight = sum(self.weights)
        center_of_mass = weighted_positions / total_weight
        total_length_of_cycle = max(self.positions) + (self.weights[self.positions.index(max(self.positions))])
        ideal_center = total_length_of_cycle / 2
        balance_index = abs(center_of_mass - ideal_center) / ideal_center
        return balance_index

    def autocorrelation(self):
        """
        Calculates the autocorrelation of the values.

        Returns:
            list: The autocorrelation values.

        """
        n = len(self.values)
        result = correlate(self.values, self.values, mode='full', method='fft')
        result = result[n-1:] / np.array([n - abs(i) for i in range(n)])
        return result.tolist()

    def motif(self):
        """
        Calculates the motif score of the values.

        Returns:
            float: The motif score.

        """
        if not self.values:  # Check if the list is empty
            return 0  # Or return another appropriate default value
        autocorr = self.autocorrelation()
        peaks, _ = find_peaks(autocorr)
        if len(peaks) > 1:
            motif_lengths = np.diff(peaks)  # Distances between peaks as potential motif lengths
            motif_length = np.median(motif_lengths)  # Use the median as a common motif length
        else:
            return 0  # Return zero score if no motif length is identified
        motif_length = int(motif_length)
        motif_counts = {}
        for i in range(len(self.values) - motif_length + 1):
            motif = tuple(self.values[i:i + motif_length])
            motif_counts[motif] = motif_counts.get(motif, 0) + 1
        # Score based on the frequency of repeated motifs
        motif_score = sum(count for count in motif_counts.values() if count > 1) / len(self.values)
        return motif_score

    def dissonance(self, scale):
        """
        Calculates the dissonance of the values with respect to a scale.

        Args:
            scale (list): A list of values representing a musical scale.

        Returns:
            float: The dissonance.

        """
        if not self.values:  # Check if the list is empty
            return 0  # Or return another appropriate default value
        n = 0
        for v in self.values:
            if v not in scale:
                n += 1
        return n/len(self.values)

    def rhythmic(self, measure_length):
        """
        Calculates the rhythmic score of the values.

        Args:
            measure_length (float): The length of a measure.

        Returns:
            float: The rhythmic score.

        """
        scores = []
        current_measure_duration = 0.0
        for duration in self.values:
            current_measure_duration += duration
            if current_measure_duration > measure_length:
                # Here we consider overflow as a negative, so we reset for the next measure
                scores.append(0)  # Score of 0 for overflowed measure, adjust based on your scoring preference
                current_measure_duration = duration  # Start counting the new measure
            elif current_measure_duration == measure_length:
                scores.append(1)  # Perfect fit for this measure
                current_measure_duration = 0.0  # Reset for the next measure

        # Handle last measure if it doesn't reach full measure_length but no more durations are available
        if 0 < current_measure_duration <= measure_length:
            # The closer the last measure's total duration is to the measure_length, the better
            scores.append(current_measure_duration / measure_length)

        # Return the average score if there are scores, else return 0 (indicating no fit)
        return np.mean(scores) if scores else 0
    
    def fibonacci_index(self):
        """
        Calculates a Fibonacci index to evaluate how closely the sequence matches a Fibonacci sequence.

        Returns:
            float: The Fibonacci index, lower values indicate closer match to Fibonacci sequence.
        """
        if len(self.values) < 3:
            return float('inf')  # Not enough data to compute Fibonacci likeness

        # Calculate ratios of consecutive numbers
        ratios = [self.values[i] / self.values[i-1] for i in range(1, len(self.values)) if self.values[i-1] != 0]

        # Calculate how these ratios deviate from the golden ratio
        golden_ratio = (1 + np.sqrt(5)) / 2
        deviations = [abs(ratio - golden_ratio) for ratio in ratios]

        # Calculate an index as the average of these deviations
        fibonacci_index = sum(deviations) / len(deviations)
        return fibonacci_index