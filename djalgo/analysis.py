import numpy as np
from scipy.signal import correlate, find_peaks


class Analysis:
    """
    This class provides methods for analyzing a list of values, whether they are pitches, durations or offsets.

    Args:
        values (list): A list of numerical values.

    Attributes:
        values (list): The list of values provided during initialization.
    """
    def __init__(self, values):
        self.values = [v for v in values if v is not None] # Remove None values

    def gini(self):
        """
        Calculate the Gini index, a measure of inequality.
        
        Returns:
            float or None: The Gini index of the values if more than one value is present; otherwise, None.
        """
        if not self.values or len(self.values) <= 1:
            return None
        sorted_values = sorted(self.values)
        n = len(sorted_values)
        cumulative_sum = sum((i+1) * val for i, val in enumerate(sorted_values))
        total_sum = sum(sorted_values)
        gini = (2 * cumulative_sum) / (n * total_sum) - (n + 1) / n
        return gini

    def balance(self):
        """
        Compute the balance based on the values.

        Returns:
            float or None: The balance index of the values if more than one value is present; otherwise, None.
        """
        if not self.values or len(self.values) <= 1:
            return None
        sum_values = sum(self.values)
        center_of_mass = sum(i * v for i, v in enumerate(self.values)) / sum_values
        ideal_center = (len(self.values) - 1) / 2
        balance = abs(center_of_mass - ideal_center) / ideal_center
        return balance
    
    def autocorrelation(self):
        """
        Calculate the autocorrelation of the values using the Fast Fourier Transform (FFT).

        Returns:
            list: A list containing the autocorrelation coefficients.
        """
        n = len(self.values)
        result = correlate(self.values, self.values, mode='full', method='fft')
        result = result[n-1:] / np.array([n - abs(i) for i in range(n)])
        return result.tolist()
    
    def motif(self):
        """
        Identify and score repeating motifs in the values.

        Returns:
            float: The score based on the frequency of repeated motifs. Zero if no motifs are found.
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
        Calculate the dissonance, which measures the proportion of values not aligning with a given scale.

        Args:
            scale (list): A list of values representing the desired scale.

        Returns:
            float: The dissonance score, representing the fraction of values not in the scale.
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
        Calculate how well the durations fit into bars of a given length, measure by measure.

        Args:
            measure_length (float): Length of a bar in the same units as the durations.

        Returns:
            float: An average score representing the rhythmic fit for each measure (closer to 1 is better).
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

