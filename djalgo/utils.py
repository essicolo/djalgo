def round_to_list(value, scale):
    """
    Rounds the given value to the nearest value in the scale list.

    Args:
        value (float): The value to be rounded.
        scale (list): A list of values to round to.

    Returns:
        float: The value from the scale list that is closest to the given value.
    """
    return min(scale, key=lambda x: abs(x - value))

def get_octave(midi_note):
    return midi_note // 12 - 1

def get_sharp(string):
    dict_flat = {'D-': 'C#', 'E-': 'D#', 'G-': 'F#', 'A-': 'G#', 'B-': 'A#'}
    if string in dict_flat.keys():
        string = dict_flat[string]
    return string

def get_degree_from_note(note, scale_list, tonic_midi):
    ref_index = scale_list.index(tonic_midi)

    # If the note is in the scale_type
    if note in scale_list:
        # Find its index and compute the degree
        note_index = scale_list.index(note)
        degree = note_index - ref_index
    else:
        # If the note is not in the scale_type, find the two notes it falls between
        upper_note = round_to_list(note, scale_list)
        upper_index = scale_list.index(upper_note)
        lower_index = upper_index - 1 if upper_index > 0 else upper_index
        lower_note = scale_list[lower_index]

        # Compute the degree as the weighted average of the degrees of the two notes
        distance_to_upper = upper_note - note
        distance_to_lower = note - lower_note
        upper_weight = 1 - distance_to_upper / (distance_to_upper + distance_to_lower)
        lower_weight = 1 - distance_to_lower / (distance_to_upper + distance_to_lower)
        upper_degree = upper_index - ref_index
        lower_degree = lower_index - ref_index
        degree = upper_degree * upper_weight + lower_degree * lower_weight

    return degree

def get_note_from_degree(degree, scale_list, tonic_midi):
    ref_index = scale_list.index(tonic_midi)
    note_index = round(ref_index + degree)  # round to nearest integer

    # If the degree is within the scale
    if 0 <= note_index < len(scale_list):
        note = scale_list[note_index]
    else:
        # If the degree is not within the scale, find the two notes it falls between
        lower_index = max(0, min(note_index, len(scale_list) - 1))
        upper_index = min(len(scale_list) - 1, max(note_index, 0))
        lower_note = scale_list[lower_index]
        upper_note = scale_list[upper_index]

        # Compute the note as the weighted average of the two notes
        distance_to_upper = upper_index - note_index
        distance_to_lower = note_index - lower_index
        upper_weight = 1 - distance_to_upper / (distance_to_upper + distance_to_lower)
        lower_weight = 1 - distance_to_lower / (distance_to_upper + distance_to_lower)
        note = upper_note * upper_weight + lower_note * lower_weight

    return note

def fill_gaps_with_rests(notes, parent_offset=0.0):
    """
    Analyze a list of notes (each note is a (pitch, duration, offset) tuple)
    and insert rests (None, duration, offset) to fill gaps between notes.

    Args:
        notes (list): The list of notes to be processed.
        parent_offset (float): The offset to consider from the parent list, used in recursion.

    Returns:
        list: The modified list with gaps filled with rests.
    """
    last_offset = 0.0  # Keep track of the offset after the last note or rest
    filled_notes = []

    for note in notes:
        pitch, duration, offset = note
        current_offset = parent_offset + offset
        if current_offset > last_offset:
            # There is a gap that needs to be filled with a rest
            gap_duration = current_offset - last_offset
            rest_to_insert = (None, gap_duration, last_offset - parent_offset)
            filled_notes.append(rest_to_insert)  # Adjust offset for insertion

        filled_notes.append(note)
        last_offset = max(last_offset, current_offset + duration)  # Update the last offset

    return filled_notes

def adjust_note_durations_to_prevent_overlaps(notes):
    """
    Adjust the durations of notes in a list (each note is a (pitch, duration, offset) tuple)
    to prevent overlaps, while keeping their offsets intact.

    Args:
        notes (list): The list of notes to be adjusted.

    Returns:
        list: The modified list with adjusted note durations.
    """
    # Ensure the list is sorted by offset
    notes.sort(key=lambda note: note[2])

    for i in range(len(notes) - 1):  # Loop through all notes except the last one
        current_note = notes[i]
        next_note = notes[i + 1]
        # Calculate the current end of the note
        current_note_end = current_note[2] + current_note[1]

        # If the current note ends after the next note starts, adjust its duration
        if current_note_end > next_note[2]:
            # Adjust duration to avoid overlap
            new_duration = next_note[2] - current_note[2]
            notes[i] = (current_note[0], new_duration, current_note[2])

    return notes

def insert_rests(notes, measure_length=4.0):
    result = []
    current_offset = 0.0

    for note in notes:
        pitch, duration, offset = note

        # If the offset of the current note is greater than the current offset
        # plus the duration of the previous note, insert a rest
        if offset > current_offset:
            rest_duration = offset - current_offset
            rest = (None, rest_duration, current_offset)
            result.append(rest)

        result.append(note)
        current_offset = offset + duration

    # Check if there is a rest at the end of the last measure
    if current_offset % measure_length != 0:
        rest_duration = measure_length - (current_offset % measure_length)
        rest = (None, rest_duration, current_offset)
        result.append(rest)

    return result

def repair_notes(s: list) -> list:
    """
    Apply the fill_gaps_with_rests and adjust_note_durations_to_prevent_overlaps functions to a stream.

    Args:
        s (stream.Stream): The music21 stream to be processed.

    Returns:
        stream.Stream: The modified stream with gaps filled and note durations adjusted.
    """
    return adjust_note_durations_to_prevent_overlaps(fill_gaps_with_rests(s))

def abc_to_midi(note):
    # Mapping of note names to MIDI numbers
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = int(note[-1]) + 1
    key = notes.index(note[:-1])
    midi = 12 * octave + key
    return midi

def midi_to_abc(midi):
    # Mapping of MIDI numbers to note names
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = midi // 12 - 1
    key = midi % 12
    note = notes[key] + str(octave)
    return note

def no_overlap(notes, adjust='offsets'):
    """
    Adjusts the offsets of the notes to prevent overlap.

    Args:
        notes (list): A list of tuples, where each tuple contains a note (pitch),
                      a duration (quarterLength), and an offset.

    Returns:
        list: The list of notes with adjusted offsets or durations.
    """
    adjusted_notes = []
    current_offset = 0

    for pitch, duration, _ in notes:
        adjusted_notes.append((pitch, duration, current_offset))
        current_offset += duration

    return adjusted_notes

def check_input(input_list):
    """
    Checks if the input is a list of simple elements or a list of tuples.

    Args:
        input_list (list): The input to check.

    Returns:
        str: 'list' if the input is a list of simple elements,
             'list of tuples' if the input is a list of tuples,
             'unknown' otherwise.
    """
    if all(isinstance(i, tuple) for i in input_list):
        return 'list of tuples'
    elif all(not isinstance(i, tuple) for i in input_list):
        return 'list'
    else:
        return 'unknown'

def scale_list(numbers, to_min, to_max, min_numbers=None, max_numbers=None):
    """
    Scale a list of numbers so that its range is between min_value and max_value.

    Args:
        numbers (list): List of numbers to scale.
        min_value (float): Minimum value of the scaled list.
        max_value (float): Maximum value of the scaled list.

    Returns:
        list: Scaled list of numbers.
    """
    if min_numbers is None:
        min_numbers = min(numbers)
    if max_numbers is None:
        max_numbers = max(numbers)

    return [(num - min_numbers) * (to_max - to_min) / (max_numbers - min_numbers) + to_min for num in numbers]

def offset_list_of_notes(list, by):
    """
    Offset the notes in a list by a given amount.

    Args:
        list (list): List of notes to offset.
        by (float): Amount to offset the notes.

    Returns:
        list: List of notes with adjusted offsets.
    """
    return [(pitch, duration, offset + by) for pitch, duration, offset in list]


def quantize_notes(notes, measure_length, time_resolution):
    """
    Quantize the durations and offsets of notes in musical phrases.

    Args:
        notes (list): List of musical phrases, where each phrase is a list of tuples (pitch, duration, offset).
        measure_length (float): The total duration of a measure, typically in quarter notes.
        time_resolution (float): The smallest time unit for quantization, typically in quarter notes.

    Returns:
        list: The quantized musical phrases.
    """
    quantized_notes = []
    for note in notes:
        pitch, duration, offset = note
        quantized_offset = round(offset / time_resolution) * time_resolution
        measure_end = ((quantized_offset // measure_length) + 1) * measure_length
        quantized_duration = min(duration, measure_end - quantized_offset)
        quantized_notes.append((pitch, quantized_duration, quantized_offset))
    return quantized_notes