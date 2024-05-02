from . import utils

class Minimalism:
    """
    Main class for musical minimalism.
    """

    class Process:
        """
        Class for the process of musical minimalism.
        """

        def __init__(self, operation='additive', direction='forward', repetition=0):
            """
            Initializes the process with the specified operation and direction.

            Args:
                operation (str): The operation to be used. Can be 'additive' or 'subtractive'.
                direction (str): The direction of the operation. Can be 'forward', 'backward', 'inward' or 'outward'.
                repetition (int): The number of times the process is repeated.
            """

            if operation in ['additive', 'subtractive']:
                self.operation = operation
            else:
                raise ValueError("Invalid output type. Choose 'additive' or 'subtractive'.")
            if direction in ['forward', 'backward', 'inward', 'outward']:
                self.direction = direction
            else:
                raise ValueError("Invalid output type. Choose 'forward', 'backward', 'inward' or 'outward'.")
            
            if repetition < 0 or not isinstance(repetition, int):
                raise ValueError("Invalid repetition value. Must be an integer greater of equal to 0.")
            self.repetition = repetition
            
        # Additive operations
        # -------------------
        def additive_forward(self):
            """
            Applies the additive forward operation to the sequence with repetition, as::

                [
                    'C4',
                    'C4', 'D4',
                    'C4', 'D4', 'E4',
                    'C4', 'D4', 'E4', 'F4',
                    'C4', 'D4', 'E4', 'F4', 'G4',
                    'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 
                    'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 
                    'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'
                ]

            Returns:
                list: The processed sequence.
            """
            processed = []
            for i in range(len(self.sequence)):
                segment = [self.sequence[j] for j in range(i + 1)]
                for _ in range(self.repetition+1):
                    processed.extend(segment)
            return processed
        
        def additive_backward(self):
            """
            Applies the additive backward operation to the sequence with repetition, as::

                [
                    'C5',
                    'B4', 'C5',
                    'A4', 'B4', 'C5',
                    'G4', 'A4', 'B4', 'C5',
                    'F4', 'G4', 'A4', 'B4', 'C5',
                    'E4', 'F4', 'G4', 'A4', 'B4', 'C5',
                    'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5',
                    'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5',
                ]

            Returns:
                list: The processed sequence.
            """
            processed = []
            for i in range(len(self.sequence), 0, -1): 
                segment = [self.sequence[j] for j in range(i - 1, len(self.sequence))]
                for _ in range(self.repetition+1): 
                    processed.extend(segment)
            return processed

        def additive_inward(self):
            """
            Applies the additive inward operation to the sequence, as::

                [
                    'C4',                                     'C5',
                    'C4', 'D4',                         'B4', 'C5',
                    'C4', 'D4', 'E4',             'A4', 'B4', 'C5',
                    'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'
                ]

            Returns:
                list: The processed sequence.
            """

            n = len(self.sequence)
            processed = []
            segment = []
            for i in range((n + 1) // 2):  # Ensuring we cover all elements
                if i < n - i - 1:  # Avoid adding the middle element twice
                    segment = [self.sequence[:i+1], self.sequence[(n - i-1):]]
                else:
                    segment = [self.sequence]
                for _ in range(self.repetition+1):
                    processed.extend(segment)
            
            processed = [item for sublist in processed for item in sublist]
            return processed
        
        def additive_outward(self):
            """
            Applies the additive outward operation to the sequence with repetition, as::

                [
                                    'F4', 'G4',
                                'E4', 'F4', 'G4', 'A4',
                        'D4', 'E4', 'F4', 'G4', 'A4', 'B4',
                    'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'
                ]

            Returns:
                list: The processed sequence.
            """

            n = len(self.sequence)
            processed = []

            # Check if the sequence is odd or even and find the middle
            if n % 2 == 0:  # Even
                mid_left = n // 2 - 1
                mid_right = n // 2
                # Start from the middle two elements and expand outwards
                for i in range(n // 2):
                    segment = self.sequence[mid_left - i : mid_right + i + 1]
                    for _ in range(self.repetition+1):
                        processed.extend(segment)
            else:  # Odd
                mid = n // 2
                # Start from the middle element and expand outwards
                for i in range(mid + 1):
                    segment = self.sequence[mid - i : mid + i + 1]
                    for _ in range(self.repetition+1):
                        processed.extend(segment)

            return processed
        
        # Subtractive operations
        # ----------------------
        def subtractive_forward(self):
            """
            Applies the subtractive forward operation to the sequence, as::

                [
                    'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5',
                    'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5',
                    'E4', 'F4', 'G4', 'A4', 'B4', 'C5',
                    'F4', 'G4', 'A4', 'B4', 'C5',
                    'G4', 'A4', 'B4', 'C5',
                    'A4', 'B4', 'C5',
                    'B4', 'C5',
                    'C5'
                ]

            Returns:
                list: The processed sequence.
            """

            processed = []
            for i in range(len(self.sequence)):
                # Each time, remove one more note from the beginning
                segment = self.sequence[i:]
                for _ in range(self.repetition+1):  # Here we consider if repetition is needed
                    processed.extend(segment)  # Directly append the segment without extending it, because we're dealing with segments now
            return processed
        
        def subtractive_backward(self):
            """
            Applies the subtractive backward operation to the sequence, as::

                [
                    'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5',
                    'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4',
                    'C4', 'D4', 'E4', 'F4', 'G4', 'A4',
                    'C4', 'D4', 'E4', 'F4', 'G4',
                    'C4', 'D4', 'E4', 'F4',
                    'C4', 'D4', 'E4',
                    'C4', 'D4',
                    'C4'
                ]

            Returns:
                list: The processed sequence.
            """

            processed = []
            for i in range(len(self.sequence), 0, -1):  # Start from the full sequence and decrement
                # Each time, create a segment that removes one more note from the end
                segment = self.sequence[:i]
                for _ in range(self.repetition+1):  # Consider if repetition is needed
                    processed.extend(segment)  # Append the segment directly
            return processed
        
        def subtractive_inward(self):
            """
            Applies the subtractive inward operation to the sequence, as::

                [
                    'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5',
                        'D4', 'E4', 'F4', 'G4', 'A4', 'B4',
                                'E4', 'F4', 'G4', 'A4',
                                    'F4', 'G4',
                ]

            Returns:
                list: The processed sequence.
            """

            n = len(self.sequence)
            steps = n // 2
            # Compute the steps for subtraction, which will be half the length of the sequence rounded down
            processed = self.sequence * (self.repetition + 1)

            # Add segments, removing elements from both ends, also repeated
            for i in range(1, steps + 1):
                segment = self.sequence[i: n - i]  # Remove elements from both ends
                if segment:  # Check if there's anything left to add.
                    processed += segment * (self.repetition + 1)
            return processed
        
        def subtractive_outward(self):
            """
            Applies the subtractive outward operation to the sequence, as::

                [
                    'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5',
                    'C4', 'D4', 'E4',           , 'A4', 'B4', 'C5',
                    'C4', 'D4',                         'B4', 'C5',
                    'C4',                                     'C5',
                ]

            Returns:
                list: The processed sequence.
            """

            segment = self.sequence
            processed = segment * (self.repetition+1)
            while len(segment) > 2:
                segment = segment[1:-1]  # remove the first and last elements
                for _ in range(self.repetition+1):
                    processed.extend(segment)
            return processed

        
        def generate(self, sequence):
            """
            Generates the processed sequence based on the operation and direction.

            Args:
                sequence (list): The sequence to be processed.

            Returns:
                list: The processed sequence.
            """
            self.sequence = sequence
            
            if self.operation == 'additive' and self.direction == 'forward':
                processed = self.additive_forward()
            elif self.operation == 'additive' and self.direction == 'backward':
                processed = self.additive_backward()
            elif self.operation == 'additive' and self.direction == 'inward':
                processed = self.additive_inward()
            elif self.operation == 'additive' and self.direction == 'outward':
                processed = self.additive_outward()
            elif self.operation == 'subtractive' and self.direction == 'forward':
                processed = self.subtractive_forward()
            elif self.operation == 'subtractive' and self.direction == 'backward':
                processed = self.subtractive_forward()
            elif self.operation == 'subtractive' and self.direction == 'inward':
                processed = self.subtractive_inward()
            elif self.operation == 'subtractive' and self.direction == 'outward':
                processed = self.subtractive_outward()

            new_processed = []
            current_offset = 0
            
            # Adjust offsets based on the duration
            for note in processed:
                new_note = (note[0], note[1], current_offset)
                new_processed.append(new_note)
                current_offset += note[1]

            return new_processed

    class Tintinnabuli:
        """
        Class for the Tintinnabuli style of musical minimalism.
        """

        def __init__(self, t_chord, direction='down', rank=0):
            """
            Initializes the Tintinnabuli style with the specified t-chord, direction, and rank.

            Args:
                t_chord (list): The t-chord to be used.
                direction (str): The direction of the operation. Can be 'up', 'down', 'any' or 'alternate'.
                rank (int): The rank of the t-chord.
            """

            if direction in ['up', 'down', 'any', 'alternate']:
                self.direction = direction
            else:
                raise ValueError("Invalid output type. Choose 'up', 'down', 'any' or 'alternate'.")
            
            if self.direction == 'alternate':
                self.is_alternate = True
                self.direction = 'up'
            else:
                self.is_alternate = False
            
            self.t_chord = t_chord
            if not isinstance(rank, int) or rank < 0:
                raise ValueError("Rank must be a non-negative integer lower or equal to the length of the t-chord.")
            else:
                self.rank = rank
            
            if self.rank >= len(self.t_chord):
                self.rank = len(self.t_chord) - 1
                print("Rank exceeds the length of the t-chord. Defaulting to the last note of the t-chord.")

        def generate(self, sequence):
            """
            Generates the t-voice based on the t-chord, direction, and rank.

            Args:
                sequence (list): The m-voice sequence.

            Returns:
                list: The t-voice sequence.
            """
            
            self.sequence = sequence  # the m-voice
            if utils.check_input(self.sequence) == 'list of tuples':
                pitch_sequence = [note[0] for note in self.sequence]
            elif utils.check_input(self.sequence) == 'list':
                pitch_sequence = self.sequence
            else:
                raise ValueError("Invalid input type. Must be a list of tuples or a list.")
            
            t_voice = []
            for m in pitch_sequence:
                if m is None:
                    t_voice.append(None)
                    continue
                differences = [t - m for t in self.t_chord]
                sorted_differences = sorted(enumerate(differences), key=lambda x: abs(x[1]))

                # Ensure rank is within the range of available notes
                effective_rank = self.rank  # Default to the specified rank
                
                if self.direction in ['up', 'down']:
                    filtered_differences = [(index, value) for index, value in sorted_differences if value >= 0] if self.direction == 'up' else [(index, value) for index, value in sorted_differences if value <= 0]
                    if not filtered_differences:
                        # If there are no notes in the desired direction, default to the extreme note in that direction
                        t_voice_i = max(self.t_chord) if self.direction == 'up' else min(self.t_chord)
                    else:
                        # Adjust rank if it exceeds the length of filtered differences
                        if effective_rank >= len(filtered_differences):
                            effective_rank = len(filtered_differences) - 1  # Use the last available note
                        
                        chosen_index = filtered_differences[effective_rank][0]
                        t_voice_i = self.t_chord[chosen_index]
                elif self.direction == 'any':
                    # Adjust rank if it exceeds the length of sorted differences
                    if effective_rank >= len(sorted_differences):
                        effective_rank = len(sorted_differences) - 1  # Use the last available note
                    
                    chosen_index = sorted_differences[effective_rank][0]
                    t_voice_i = self.t_chord[chosen_index]

                # Change direction if alternate
                if self.is_alternate:
                    self.direction = 'down' if self.direction == 'up' else 'up'
                
                t_voice.append(t_voice_i)
            
            if utils.check_input(self.sequence) == 'list of tuples':
                t_voice = [(t_voice[i], note[1], note[2]) for i, note in enumerate(self.sequence)]

            return t_voice
