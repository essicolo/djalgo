import math
from typing import List, Tuple, Union

import difflib
from .utils import instrument_mapping 

try:
    import music21 as m21
except ImportError:
    m21 = None

try:
    import pretty_midi as pm
except ImportError:
    pm = None

try:
    import mido
except ImportError:
    mido = None

try:
    import miditoolkit
except ImportError:
    miditoolkit = None


def get_program_number(instrument_name):
    """Find the closest program number for a given instrument name."""
    if instrument_name in instrument_mapping:
        return instrument_mapping[instrument_name]
    closest_match = difflib.get_close_matches(instrument_name, instrument_mapping.keys(), n=1)
    if closest_match:
        return instrument_mapping[closest_match[0]]
    return 0  # Default to piano if no close match is found


def convert(notes, to, tempo=120, time_signature=None, key=None, clef="treble", title=None):
    """
    Convert a list of musical notes to the specified format.

    Args:
        notes (list): List of musical notes.
        to (str): Format to convert the notes to. Options are "music21", "mido", or "pretty_midi".
        tempo (int): Tempo in BPM (default 120).
        time_signature (str): Time signature (default None).
        key (str): Key signature (default None).
        clef (str): Clef to use (default None).
        title (str): Title of the piece (default None).

    Returns:
        object: Converted musical notes in the specified format.

    Raises:
        ValueError: If the specified format is not supported.
    """
    if to == 'music21':
        if m21 is None:
            raise ImportError("The music21 library is not installed. Please install it to use this function.")
        return to_music21(notes, tempo, time_signature, key)
    elif to == 'mido':
        if mido is None:
            raise ImportError("The mido library is not installed. Please install it to use this function.")
        return to_mido(notes, tempo)
    elif to == 'pretty_midi' or to == 'pretty-midi':
        if pm is None:
            raise ImportError("The pretty_midi library is not installed. Please install it to use this function.")
        return to_prettymidi(notes, tempo)
    elif to == 'miditoolkit':
        if pm is None:
            raise ImportError("The miditoolkit library is not installed. Please install it to use this function.")
        return to_miditoolkit(notes, tempo)
    elif to == 'abc':
        return to_abc(notes, key, clef, time_signature, title, tempo)
    else:
        raise ValueError('Format not supported. Please use "music21", "mido", "pretty_midi" or "miditoolkit".')


# Music21 conversion
# ------------------
def tuple_to_music21_element(note_tuple):
    """Convert a single note tuple to a music21 Note, Chord, or Rest."""
    if note_tuple[0] is None:  # Rest
        rest = m21.note.Rest()
        rest.duration.quarterLength = note_tuple[1]
        return rest
    elif isinstance(note_tuple[0], list):  # Chord
        # Filter out None values from the chord
        valid_pitches = [pitch for pitch in note_tuple[0] if pitch is not None]
        if not valid_pitches:  # If no valid pitches, return a Rest instead
            rest = m21.note.Rest()
            rest.duration.quarterLength = note_tuple[1]
            return rest
        chord = m21.chord.Chord(valid_pitches)
        chord.duration.quarterLength = note_tuple[1]
        return chord
    else:  # Single note
        note = m21.note.Note(int(note_tuple[0]))
        note.duration.quarterLength = note_tuple[1]
        return note

def sequence_to_music21_stream(notes, bpm=120):
    """Convert a sequence of musical notes to a music21 Stream, with a tempo mark."""
    stream = m21.stream.Part()
    stream.append(m21.tempo.MetronomeMark(number=bpm))  # Set tempo mark at the beginning of the stream
    for note_tuple in notes:
        element = tuple_to_music21_element(note_tuple)
        stream.insert(note_tuple[2], element)  # Insert element at the specified offset in quarter lengths
    return stream

def to_music21(notes, bpm=120, time_signature=None, key_signature=None):
    """Convert notes to music21 format based on their structure, with BPM defining the tempo."""
    score = m21.stream.Score()
    if time_signature:
        score.insert(0, m21.meter.TimeSignature(time_signature))
    if key_signature:
        score.insert(0, m21.key.KeySignature(key_signature))
    if isinstance(notes, tuple) and len(notes) == 3:  # Single note, rest, or chord
        score.append(sequence_to_music21_stream([notes], bpm))
    elif isinstance(notes, list):
        if all(isinstance(note, tuple) and len(note) == 3 for note in notes):  # List of notes
            score.append(sequence_to_music21_stream(notes, bpm))
        else:  # List of lists
            for note_sequence in notes:
                score.append(sequence_to_music21_stream(note_sequence, bpm))
    elif isinstance(notes, dict):  # Dictionary of lists
        for key, value in notes.items():
            part = sequence_to_music21_stream(value, bpm)
            part.id = str(key)  # Optionally set part ID to the dictionary key
            score.insert(0, part)
    return score




# Pretty-midi conversion
# ----------------------
def tuple_to_prettymidi_element(note_tuple, bpm=120, velocity=64):
    """Convert a single note tuple to a list of PrettyMIDI Note objects."""
    elements = []
    if isinstance(note_tuple, (list, tuple)) and note_tuple[0] is None:  # Rest, so no note is created
        return elements
    elif isinstance(note_tuple, (list, tuple)) and isinstance(note_tuple[0], list):  # Chord
        for pitch in note_tuple[0]:
            note = pm.Note(velocity=64, pitch=pitch, start=note_tuple[2]*60/bpm, end=(note_tuple[2]+note_tuple[1])*60/bpm)
            elements.append(note)
    else:  # Single note
        if isinstance(note_tuple, (list, tuple)):
            note = pm.Note(velocity=velocity, pitch=note_tuple[0], start=note_tuple[2]*60/bpm, end=(note_tuple[2]+note_tuple[1])*60/bpm)
            elements.append(note)
    return elements

def sequence_to_prettymidi_instrument(notes, bpm=120, velocity=64, program=0):
    """Convert a sequence of musical notes to a PrettyMIDI instrument."""
    instrument = pm.Instrument(program=program)
    if isinstance(notes, int):  # Check if notes is an integer
        notes = [notes] 
    for note_tuple in notes:
        for note in tuple_to_prettymidi_element(note_tuple, bpm, velocity):
            instrument.notes.append(note)
    return instrument

def sequences_to_prettymidi(parts, bpm=120, velocity=64):
    """Convert multiple sequences of musical notes to a PrettyMIDI object."""
    return [sequence_to_prettymidi_instrument(part_notes, bpm, velocity) for part_notes in parts]

def to_prettymidi(notes, bpm=120, velocity=64):
    """Convert notes to PrettyMIDI format based on their structure."""
    pm_object = pm.PrettyMIDI()
    if isinstance(notes, tuple) and len(notes) == 3:  # Single note, rest, or chord
        pm_object.instruments.append(sequence_to_prettymidi_instrument([notes], bpm, velocity))
    elif isinstance(notes, list):  
        if all(isinstance(note, tuple) and len(note) == 3 for note in notes):  # List of notes
            pm_object.instruments.append(sequence_to_prettymidi_instrument(notes, bpm, velocity))
        else:  # List of lists
            for note_sequence in notes:
                pm_object.instruments.append(sequence_to_prettymidi_instrument(note_sequence, bpm, velocity))
    elif isinstance(notes, dict):  # Dictionary of lists
        for key, value in notes.items():
            program = get_program_number(key) if isinstance(key, str) else key
            instrument = sequence_to_prettymidi_instrument(value, bpm, velocity, program)
            pm_object.instruments.append(instrument)
    return pm_object

# Mido conversion
# ---------------
def tuple_to_mido_messages(note_tuple, bpm=120, velocity=64):
    """Convert a single note tuple to a list of Mido messages."""
    messages = []
    if isinstance(note_tuple, (list, tuple)) and note_tuple[0] is None:  # Rest, so no note is created
        return messages
    elif isinstance(note_tuple, (list, tuple)) and isinstance(note_tuple[0], list):  # Chord
        for pitch in note_tuple[0]:
            # Assuming note_tuple structure is (pitches, duration, offset)
            note_on = mido.Message('note_on', note=pitch, velocity=velocity, time=int(note_tuple[2]*60/bpm))
            note_off = mido.Message('note_off', note=pitch, velocity=velocity, time=int(note_tuple[1]*60/bpm))
            messages.extend([note_on, note_off])
    else:  # Single note
        if isinstance(note_tuple, (list, tuple)):
            note_on = mido.Message('note_on', note=note_tuple[0], velocity=velocity, time=int(note_tuple[2]*60/bpm))
            note_off = mido.Message('note_off', note=note_tuple[0], velocity=velocity, time=int(note_tuple[1]*60/bpm))
            messages.extend([note_on, note_off])
    return messages

def sequence_to_mido_track(notes, bpm=120, velocity=64, program=0):
    """Convert a sequence of musical notes to a Mido track."""
    track = mido.MidiTrack()
    track.append(mido.Message('program_change', program=program, time=0))
    if isinstance(notes, int):  # Check if notes is an integer
        notes = [notes] 
    for note_tuple in notes:
        for message in tuple_to_mido_messages(note_tuple, bpm, velocity):
            track.append(message)
    return track

def to_mido(notes, bpm=120, velocity=64):
    """Convert notes to Mido format based on their structure."""
    mido_object = mido.MidiFile()
    if isinstance(notes, tuple) and len(notes) == 3:  # Single note, rest, or chord
        mido_object.tracks.append(sequence_to_mido_track([notes], bpm, velocity))
    elif isinstance(notes, list):  
        if all(isinstance(note, tuple) and len(note) == 3 for note in notes):  # List of notes
            mido_object.tracks.append(sequence_to_mido_track(notes, bpm, velocity))
        else:  # List of lists
            for note_sequence in notes:
                mido_object.tracks.append(sequence_to_mido_track(note_sequence, bpm, velocity))
    elif isinstance(notes, dict):  # Dictionary of lists
        for key, value in notes.items():
            program = get_program_number(key) if isinstance(key, str) else key
            track = sequence_to_mido_track(value, bpm, velocity, program)
            mido_object.tracks.append(track)
    return mido_object

# Miditoolkit conversion
# ----------------------
def tuple_to_miditoolkit_notes(note_tuple, bpm=120, velocity=64):
    """Convert a single note tuple to a list of miditoolkit Notes."""
    notes = []
    if isinstance(note_tuple, (list, tuple)) and note_tuple[0] is None:  # Rest, so no note is created
        return notes
    elif isinstance(note_tuple, (list, tuple)) and isinstance(note_tuple[0], list):  # Chord
        for pitch in note_tuple[0]:
            # Assuming note_tuple structure is (pitches, duration, offset)
            start = int(note_tuple[2]*60/bpm)
            end = int((note_tuple[2]+note_tuple[1])*60/bpm)
            note = miditoolkit.midi.containers.Note(pitch, start, end, velocity)
            notes.append(note)
    else:  # Single note
        if isinstance(note_tuple, (list, tuple)):
            start = int(note_tuple[2]*60/bpm)
            end = int((note_tuple[2]+note_tuple[1])*60/bpm)
            note = miditoolkit.midi.containers.Note(note_tuple[0], start, end, velocity)
            notes.append(note)
    return notes

def sequence_to_miditoolkit_instrument(notes, bpm=120, velocity=64, program=0):
    """Convert a sequence of musical notes to a miditoolkit Instrument."""
    instrument = miditoolkit.midi.containers.Instrument(program)
    if isinstance(notes, int):  # Check if notes is an integer
        notes = [notes] 
    for note_tuple in notes:
        for note in tuple_to_miditoolkit_notes(note_tuple, bpm, velocity):
            instrument.notes.append(note)
    return instrument

def to_miditoolkit(notes, bpm=120, velocity=64):
    """Convert notes to miditoolkit format based on their structure."""
    midi_obj = miditoolkit.midi.parser.MidiFile()
    midi_obj.ticks_per_beat = 480  # Set ticks per beat
    if isinstance(notes, tuple) and len(notes) == 3:  # Single note, rest, or chord
        instrument = sequence_to_miditoolkit_instrument([notes], bpm, velocity)
        midi_obj.instruments.append(instrument)
    elif isinstance(notes, list):  
        if all(isinstance(note, tuple) and len(note) == 3 for note in notes):  # List of notes
            instrument = sequence_to_miditoolkit_instrument(notes, bpm, velocity)
            midi_obj.instruments.append(instrument)
        else:  # List of lists
            for note_sequence in notes:
                instrument = sequence_to_miditoolkit_instrument(note_sequence, bpm, velocity)
                midi_obj.instruments.append(instrument)
    elif isinstance(notes, dict):  # Dictionary of lists
        for key, value in notes.items():
            program = get_program_number(key) if isinstance(key, str) else key
            instrument = sequence_to_miditoolkit_instrument(value, bpm, velocity, program)
            midi_obj.instruments.append(instrument)
    return midi_obj


# ABC conversion
# --------------

def to_abc(
    tracks: Union[List[Tuple[int, float, float]], List[List[Tuple[int, float, float]]]],
    key: str = "C",
    clef: str = "treble",
    time_signature: str = "4/4",
    title: str = "Untitled",
    tempo: int = 120
) -> str:
    """
    Convert MIDI-like notation to ABC notation.

    Args:
    tracks: List of notes (midi_pitch, duration, offset) or list of such lists for multiple tracks
    key: Key signature (default "C")
    clef: Clef to use (default "treble")
    time_signature: Time signature (default "4/4")
    title: Title of the piece (default "Untitled")
    tempo: Tempo in BPM (default 120)

    Returns:
    ABC notation as a string
    """
    if not isinstance(tracks[0], list):
        tracks = [tracks]

    midi_to_abc_note = {
        60: "C", 62: "D", 64: "E", 65: "F", 67: "G", 69: "A", 71: "B", 72: "c"
    }

    def duration_to_abc(duration: float) -> str:
        if duration == 1:
            return ""
        elif duration == 0.5:
            return "/2"
        elif duration == 0.25:
            return "/4"
        elif duration == 0.125:
            return "/8"
        elif duration == 2:
            return "2"
        elif duration == 4:
            return "4"
        else:
            return f"{int(duration * 4)}/4"

    abc_output = [
        f"X:1",
        f"T:{title}",
        f"M:{time_signature}",
        f"L:1/4",
        f"Q:1/4={tempo}",
        f"K:{key}",
        f"%%score {' '.join([f'T{i+1}' for i in range(len(tracks))])}"
    ]

    for track_num, track in enumerate(tracks, 1):
        abc_output.append(f"V:T{track_num} clef={clef}")
        measure = []
        measure_duration = 0
        total_duration = 4  # Assuming 4/4 time signature

        for note in track:
            midi_pitch, duration, _ = note
            abc_note = midi_to_abc_note.get(midi_pitch, "C")  # Default to C if not found
            abc_duration = duration_to_abc(duration)
            measure.append(f"{abc_note}{abc_duration}")
            measure_duration += duration

            if measure_duration >= total_duration:
                abc_output.append(" ".join(measure) + " |")
                measure = []
                measure_duration = 0

        # Add any remaining notes in the last measure
        if measure:
            abc_output.append(" ".join(measure) + " |")

    return "\n".join(abc_output)