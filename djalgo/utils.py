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

def get_degree_from_pitch(pitch, scale_list, tonic_pitch):

    if isinstance(pitch, str):
        pitch = cde_to_midi(pitch)
    if isinstance(tonic_pitch, str):
        tonic_pitch = cde_to_midi(tonic_pitch) 

    tonic_index = scale_list.index(tonic_pitch)

    # If the pitch is in the mode
    if pitch in scale_list:
        # Find its index and compute the degree
        pitch_index = scale_list.index(pitch)
        degree = pitch_index - tonic_index
    else:
        # If the pitch is not in the mode, find the two pitches it falls between
        upper_pitch = round_to_list(pitch, scale_list)
        upper_index = scale_list.index(upper_pitch)
        lower_index = upper_index - 1 if upper_index > 0 else upper_index
        lower_pitch = scale_list[lower_index]

        # Compute the degree as the weighted average of the degrees of the two pitches
        distance_to_upper = upper_pitch - pitch
        distance_to_lower = pitch - lower_pitch
        upper_weight = 1 - distance_to_upper / (distance_to_upper + distance_to_lower)
        lower_weight = 1 - distance_to_lower / (distance_to_upper + distance_to_lower)
        upper_degree = upper_index - tonic_index
        lower_degree = lower_index - tonic_index
        degree = upper_degree * upper_weight + lower_degree * lower_weight

    return degree

def get_pitch_from_degree(degree, scale_list, tonic_pitch):
    tonic_index = scale_list.index(tonic_pitch)
    pitch_index = round(tonic_index + degree)  # round to nearest integer

    # If the degree is within the scale
    if 0 <= pitch_index < len(scale_list):
        pitch = scale_list[pitch_index]
    else:
        # If the degree is not within the scale, find the two pitches it falls between
        lower_index = max(0, min(pitch_index, len(scale_list) - 1))
        upper_index = min(len(scale_list) - 1, max(pitch_index, 0))
        lower_pitch = scale_list[lower_index]
        upper_pitch = scale_list[upper_index]

        # Compute the pitch as the weighted average of the two pitches
        distance_to_upper = upper_index - pitch_index
        distance_to_lower = pitch_index - lower_index
        if distance_to_upper + distance_to_lower == 0:
            upper_weight = lower_weight = 0.5
        else:
            upper_weight = 1 - distance_to_upper / (distance_to_upper + distance_to_lower)
            lower_weight = 1 - distance_to_lower / (distance_to_upper + distance_to_lower)
        pitch = upper_pitch * upper_weight + lower_pitch * lower_weight

    return pitch

def set_offsets_according_to_durations(notes):
    """
    Adjusts the offsets of the notes based on their durations.

    Args:
        notes (list): A list of tuples, where each tuple contains a note (pitch),
                      a duration (quarterLength), and an offset.

    Returns:
        list: The list of notes with adjusted offsets.
    """
    if len(notes[0]) == 2:
        notes = [(note[0], note[1], 0) for note in notes]
    adjusted_notes = []
    current_offset = 0

    for pitch, duration, _ in notes:
        adjusted_notes.append((pitch, duration, current_offset))
        current_offset += duration

    return adjusted_notes

def fill_gaps_with_rests(notes, parent_offset=0.0):
    """
    Analyze a sorted list of notes (each note is a (pitch, duration, offset) tuple)
    and insert rests (None, duration, offset) to fill gaps between notes. Notes are
    sorted by offset before processing to ensure accurate gap detection and filling.

    Args:
        notes (list): The list of notes to be processed, not necessarily sorted.
        parent_offset (float): The offset to consider from the parent sequence, used in recursion.

    Returns:
        list: The modified list with gaps filled with rests, ensuring continuity.
    """
    # Sort notes by offset to ensure correct processing order
    notes_sorted = sorted(notes, key=lambda x: x[2])

    last_offset = 0.0  # Keep track of the offset after the last note or rest
    filled_notes = []

    for note in notes_sorted:
        pitch, duration, offset = note
        current_offset = parent_offset + offset
        if current_offset > last_offset:
            # There is a gap that needs to be filled with a rest
            gap_duration = current_offset - last_offset
            rest_to_insert = (None, gap_duration, last_offset - parent_offset)
            filled_notes.append(rest_to_insert)  # Insert the rest to fill the gap

        filled_notes.append(note)
        last_offset = max(last_offset, current_offset + duration)  # Update last offset for the next iteration

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


def repair_notes(s: list) -> list:
    """
    Apply the fill_gaps_with_rests and adjust_note_durations_to_prevent_overlaps functions to a stream.

    Args:
        s (stream.Stream): The music21 stream to be processed.

    Returns:
        stream.Stream: The modified stream with gaps filled and note durations adjusted.
    """
    return adjust_note_durations_to_prevent_overlaps(fill_gaps_with_rests(s))


def cde_to_midi(pitch):
    # Mapping of note names to MIDI numbers with sharps
    pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    flat_to_sharp = {
        'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
        'Cb': 'B',  # Special handling for 'Cb', converting it directly to 'B'
    }

    octave = 4  # Default octave if not specified in the pitch string

    # Check and convert flat notes to sharp notes
    if 'b' in pitch:
        note = pitch[:-1]  # Exclude the octave number if present
        if note in flat_to_sharp:
            pitch = flat_to_sharp[note] + pitch[-1]  # Append the octave number back if it was present

    # Extract the note (with sharp) and octave from the pitch
    if len(pitch) > 2 or pitch[1].isdigit():
        note, octave = pitch[:-1], int(pitch[-1])
    else:
        note = pitch[0]

    midi = 12 * (octave + 1) + pitches.index(note)
    return midi


def midi_to_cde(midi):
    # Mapping of MIDI numbers to note names
    pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = midi // 12 - 1
    key = midi % 12
    pitch = pitches[key] + str(octave)
    return pitch


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
    if min_numbers == max_numbers:
        return [(min_numbers + max_numbers) / 2] * len(numbers)
    else:
        return [(num - min_numbers) * (to_max - to_min) / (max_numbers - min_numbers) + to_min for num in numbers]


def offset_track(track, by):
    """
    Offset the notes in a list by a given amount.

    Args:
        track (list): List of notes to offset.
        by (float): Amount to offset the notes.

    Returns:
        list: List of notes with adjusted offsets.
    """
    return [(pitch, duration, offset + by) for pitch, duration, offset in track]


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
        quantized_duration = round(duration / time_resolution) * time_resolution
        quantized_duration = min(quantized_duration, measure_end - quantized_offset)
        quantized_notes.append((pitch, quantized_duration, quantized_offset))
    quantized_notes = [note for note in quantized_notes if note[1] > 0]  # remove notes with zero duration
    return quantized_notes


def find_closest_pitch_at_measure_start(notes, measure_length):
    """
    Finds the closest pitch at the beginning of each measure.

    Args:
        notes (list of tuples): A list of tuples where each tuple is (pitch, duration, offset).
        measure_length (float): The length of a measure.

    Returns:
        list: A list of pitches, each representing the closest pitch at the start of a measure.
    """
    # Filter out notes with None offset or pitch
    notes = [note for note in notes if note[2] is not None and note[0] is not None]

    # Sort the notes by offset to ensure they are in order
    notes_sorted_by_offset = sorted(notes, key=lambda x: x[2])

    # Find the maximum offset to determine how many measures we have
    max_offset = max(notes_sorted_by_offset, key=lambda x: x[2])[2]
    num_measures = int(max_offset // measure_length) + 1

    closest_pitches = []

    for measure_num in range(num_measures):
        measure_start = measure_num * measure_length
        closest_pitch = None
        closest_distance = float('inf')

        for pitch, duration, offset in notes_sorted_by_offset:
            # Calculate the distance from the start of the measure to the note's offset
            distance = measure_start - offset

            # If the note starts before the measure and is closer than any note we've looked at before
            if distance >= 0 and distance < closest_distance:
                closest_distance = distance
                closest_pitch = pitch

            # If we've passed the current measure start, we can break out of the loop
            if offset > measure_start:
                break

        if closest_pitch is not None:
            closest_pitches.append(closest_pitch)

    return closest_pitches

def tune(pitch, scale):
    """
    Adjust the pitch of a note to the nearest pitch within the given scale.

    Args:
        pitch (int): a MIDI pitch number to tune.
        scale (list): A list of pitches

    Returns:
        pitch: A tuned MIDI pitch number.
    """
    return min(scale, key=lambda x: abs(x - pitch))

def ql_to_seconds(ql, bpm):
    """
    Convert a duration in quarter-length units to seconds.

    Args:
        ql (float): Duration in quarter-length units.
        bpm (float): Beats per minute.

    Returns:
        float: Duration in seconds.
    """
    return 60 / bpm * ql

def fibonacci(a = 0, b = 1, base = 0, scale = 1):
    """
    Generate a Fibonacci iterator.

    This function generates a Fibonacci iterator that yields the next Fibonacci number in the sequence.

    Args:
        a (int): The first number in the Fibonacci sequence (default is 0).
        b (int): The second number in the Fibonacci sequence (default is 1).
        base (int): The base value to be added to each Fibonacci number (default is 0).
        scale (int): The scale factor to be multiplied with each Fibonacci number (default is 1).

    Yields:
        int: The next Fibonacci number in the sequence.
    """
    while True:
        yield base + scale * a
        a, b = b, a + b

# Instrument mapping (from https://raw.githubusercontent.com/FoxLisk/midifier/b8b276fe3ff9b8fe159b9dc4046c1d9f0e62ea29/midifier/instruments.py)
instrument_mapping = {
  'Acoustic Grand Piano': 0,
  'Bright Acoustic Piano': 1,
  'Electric Grand Piano': 2,
  'Honky-tonk Piano': 3,
  'Electric Piano 1': 4,
  'Electric Piano 2': 5,
  'Harpsichord': 6,
  'Clavinet': 7,
  'Celesta': 8,
  'Glockenspiel': 9,
  'Music Box': 10,
  'Vibraphone': 11,
  'Marimba': 12,
  'Xylophone': 13,
  'Tubular Bells': 14,
  'Dulcimer': 15,
  'Drawbar Organ': 16,
  'Percussive Organ': 17,
  'Rock Organ': 18,
  'Church Organ': 19,
  'Reed Organ': 20,
  'Accordion': 21,
  'Harmonica': 22,
  'Tango Accordion': 23,
  'Acoustic Guitar (nylon)': 24,
  'Acoustic Guitar (steel)': 25,
  'Electric Guitar (jazz)': 26,
  'Electric Guitar (clean)': 27,
  'Electric Guitar (muted)': 28,
  'Overdriven Guitar': 29,
  'Distortion Guitar': 30,
  'Guitar Harmonics': 31,
  'Acoustic Bass': 32,
  'Electric Bass (finger)': 33,
  'Electric Bass (pick)': 34,
  'Fretless Bass': 35,
  'Slap Bass 1': 36,
  'Slap Bass 2': 37,
  'Synth Bass 1': 38,
  'Synth Bass 2': 39,
  'Violin': 40,
  'Viola': 41,
  'Cello': 42,
  'Contrabass': 43,
  'Tremolo Strings': 44,
  'Pizzicato Strings': 45,
  'Orchestral Harp': 46,
  'Timpani': 47,
  'String Ensemble 1': 48,
  'String Ensemble 2': 49,
  'Synth Strings 1': 50,
  'Synth Strings 2': 51,
  'Choir Aahs': 52,
  'Voice Oohs': 53,
  'Synth Choir': 54,
  'Orchestra Hit': 55,
  'Trumpet': 56,
  'Trombone': 57,
  'Tuba': 58,
  'Muted Trumpet': 59,
  'French Horn': 60,
  'Brass Section': 61,
  'Synth Brass 1': 62,
  'Synth Brass 2': 63,
  'Soprano Sax': 64,
  'Alto Sax': 65,
  'Tenor Sax': 66,
  'Baritone Sax': 67,
  'Oboe': 68,
  'English Horn': 69,
  'Bassoon': 70,
  'Clarinet': 71,
  'Piccolo': 72,
  'Flute': 73,
  'Recorder': 74,
  'Pan Flute': 75,
  'Blown bottle': 76,
  'Shakuhachi': 77,
  'Whistle': 78,
  'Ocarina': 79,
  'Lead 1 (square)': 80,
  'Lead 2 (sawtooth)': 81,
  'Lead 3 (calliope)': 82,
  'Lead 4 (chiff)': 83,
  'Lead 5 (charang)': 84,
  'Lead 6 (voice)': 85,
  'Lead 7 (fifths)': 86,
  'Lead 8 (bass + lead)': 87,
  'Pad 1 (new age)': 88,
  'Pad 2 (warm)': 89,
  'Pad 3 (polysynth)': 90,
  'Pad 4 (choir)': 91,
  'Pad 5 (bowed)': 92,
  'Pad 6 (metallic)': 93,
  'Pad 7 (halo)': 94,
  'Pad 8 (sweep)': 95,
  'FX 1 (rain)': 96,
  'FX 2 (soundtrack)': 97,
  'FX 3 (crystal)': 98,
  'FX 4 (atmosphere)': 99,
  'FX 5 (brightness)': 100,
  'FX 6 (goblins)': 101,
  'FX 7 (echoes)': 102,
  'FX 8 (sci-fi)': 103,
  'Sitar': 104,
  'Banjo': 105,
  'Shamisen': 106,
  'Koto': 107,
  'Kalimba': 108,
  'Bagpipe': 109,
  'Fiddle': 110,
  'Shanai': 111,
  'Tinkle Bell': 112,
  'Agogo': 113,
  'Steel Drums': 114,
  'Woodblock': 115,
  'Taiko Drum': 116,
  'Melodic Tom': 117,
  'Synth Drum': 118,
  'Reverse Cymbal': 119,
  'Guitar Fret Noise': 120,
  'Breath Noise': 121,
  'Seashore': 122,
  'Bird Tweet': 123,
  'Telephone Ring': 124,
  'Helicopter': 125,
  'Applause': 126,
  'Gunshot': 127,
}