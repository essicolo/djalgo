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

# To do...
#try:
#    import scamp
#except ImportError:
#    scamp = None

def convert(notes, to, time_signature=None, key_signature=None):
    """
    Convert a list of musical notes to the specified format.

    Args:
        notes (list): List of musical notes.
        to (str): Format to convert the notes to. Options are "music21", "mido", or "pretty_midi".

    Returns:
        object: Converted musical notes in the specified format.

    Raises:
        ValueError: If the specified format is not supported.
    """
    if to == 'music21':
        if m21 is None:
            raise ImportError("The music21 library is not installed. Please install it to use this function.")
        return to_music21(notes, time_signature, key_signature)
    elif to == 'mido':
        if mido is None:
            raise ImportError("The mido library is not installed. Please install it to use this function.")
        return to_mido(notes)
    elif to == 'pretty_midi':
        if pm is None:
            raise ImportError("The pretty_midi library is not installed. Please install it to use this function.")
        return to_prettymidi(notes)
    else:
        raise ValueError('Format not supported. Please use "music21", "mido" or "pretty_midi".')


# Music21 conversion
# ------------------
def to_music21(notes, time_signature=None, key_signature=None):
    """Convert notes to music21 format based on their structure."""
    if isinstance(notes, tuple):  # Single note, rest, or chord
        return tuple_to_music21_element(notes)
    elif isinstance(notes[0], list):  # List of lists
        return sequences_to_music21_score(notes, time_signature, key_signature)
    else:  # Single list of notes
        return sequence_to_music21_part(notes, time_signature, key_signature)

def tuple_to_music21_element(note_tuple):
    """Convert a single note tuple to a music21 element (Note, Rest, or Chord)."""
    if note_tuple[0] is None:  # it's a rest
        element = m21.note.Rest()
        element.duration = m21.duration.Duration(note_tuple[1])
    elif isinstance(note_tuple[0], list):  # it's a chord
        element = [item for item in note_tuple[0] if item is not None] # Remove None values
        element = m21.chord.Chord(element)
        element.duration.quarterLength = note_tuple[1]
    else:  # it's a note
        element = m21.note.Note()
        element.pitch.midi = note_tuple[0]
        element.duration.quarterLength = note_tuple[1]
    element.offset = note_tuple[2]
    return element

def sequence_to_music21_part(notes, time_signature=None, key_signature=None):
    """Convert a sequence of musical notes to a music21 Part, ensuring no trailing measures."""
    p = m21.stream.Part()
    if time_signature:
        ts = m21.meter.TimeSignature(time_signature)
        p.append(ts)
    if key_signature:
        ks = m21.key.KeySignature(m21.key.pitchToSharps(key_signature))
        p.append(ks)

    last_offset = 0.0  # Track the last note's end time to prevent overlaps and trailing measures
    for note_tuple in notes:
        element = tuple_to_music21_element(note_tuple)
        p.append(element)
        last_offset = max(last_offset, element.offset + element.duration.quarterLength)

    # Ensure the Part does not extend beyond the last note or rest by trimming or filling the final measure
    p.makeMeasures(inPlace=True)  # Organize notes into measures
    if p.getElementsByClass(m21.stream.Measure):  # Check if there are measures
        last_measure = p.getElementsByClass(m21.stream.Measure).last()  # Get the last measure
        if last_measure and last_measure.barDuration.quarterLength > last_measure.duration.quarterLength:  # If the last measure is underfilled
            fill_rest = m21.note.Rest(quarterLength=last_measure.barDuration.quarterLength - last_measure.duration.quarterLength)
            last_measure.append(fill_rest)  # Fill the remaining time in the last measure with a rest
    return p


def sequences_to_music21_score(parts, time_signature=None, key_signature=None):
    """Convert multiple sequences of musical notes to a music21 Score."""
    s = m21.stream.Score()
    for part_notes in parts:
        part = sequence_to_music21_part(part_notes, time_signature, key_signature)
        s.insert(0, part)
    return s


# Pretty-midi conversion
# ----------------------
def to_prettymidi(notes):
    """Convert notes to PrettyMIDI format based on their structure."""
    pm_object = pm.PrettyMIDI()
    if isinstance(notes, tuple):  # Single note, rest, or chord
        pm_object.instruments.append(sequence_to_prettymidi_instrument([notes]))
        return pm_object
    elif isinstance(notes[0], list):  # List of lists
        pm_object.instruments.append(sequences_to_prettymidi(notes))
        return pm_object
    else:  # Single list of notes
        pm_object.instruments.append(sequence_to_prettymidi_instrument(notes))
        return pm_object

def tuple_to_prettymidi_element(note_tuple):
    """Convert a single note tuple to a list of PrettyMIDI Note objects."""
    elements = []
    if note_tuple[0] is None:  # Rest, so no note is created
        return elements
    elif isinstance(note_tuple[0], list):  # Chord
        for pitch in note_tuple[0]:
            # Assuming note_tuple structure is (pitches, start, end)
            note = pm.Note(velocity=64, pitch=pitch, start=note_tuple[1], end=note_tuple[2])
            elements.append(note)
    else:  # Single note
        note = pm.Note(velocity=64, pitch=note_tuple[0], start=note_tuple[1], end=note_tuple[2])
        elements.append(note)
    return elements

def sequence_to_prettymidi_instrument(notes):
    """Convert a sequence of musical notes to a PrettyMIDI instrument."""
    instrument = pm.Instrument(program=0)  # Default to piano; modify as needed
    for note_tuple in notes:
        for note in tuple_to_prettymidi_element(note_tuple):
            instrument.notes.append(note)
    return instrument

def sequences_to_prettymidi(parts):
    """Convert multiple sequences of musical notes to a PrettyMIDI object."""
    instruments = []
    for part_notes in parts:
        instrument = sequence_to_prettymidi_instrument(part_notes)
        instruments.instruments.append(instrument)
    return instruments


# Mido conversion
# ---------------
def to_mido(notes, channel=0, velocity=64):
    """Convert notes to Mido format based on their structure."""
    if isinstance(notes, tuple):  # Single note, rest, or chord
        return sequence_to_mido_track([notes], channel, velocity)  # Wrap it in a list as a single part
    elif isinstance(notes[0], list):  # List of lists
        return sequences_to_mido_midi(notes, channel, velocity)
    else:  # Single list of notes
        return sequence_to_mido_track(notes, channel, velocity)

def tuple_to_mido_messages(note_tuple, channel=0, velocity=64):
    """Convert a single note tuple to Mido messages."""
    messages = []
    if note_tuple[0] is None:  # it's a rest
        # Mido doesn't need explicit rest messages; just no note on/off messages during this time
        return messages
    elif isinstance(note_tuple[0], list):  # it's a chord
        for pitch in note_tuple[0]:
            # Note on and note off messages for each note in the chord
            messages.append(mido.Message('note_on', channel=channel, note=pitch, velocity=velocity, time=int(note_tuple[1]*1000)))  # time for note_on is start time
            messages.append(mido.Message('note_off', channel=channel, note=pitch, velocity=velocity, time=int(note_tuple[2]*1000)))  # time for note_off is end time
    else:  # it's a single note
        messages.append(mido.Message('note_on', channel=channel, note=note_tuple[0], velocity=velocity, time=int(note_tuple[1]*1000)))
        messages.append(mido.Message('note_off', channel=channel, note=note_tuple[0], velocity=velocity, time=int(note_tuple[2]*1000)))
    return messages

def sequence_to_mido_track(notes, channel=0, velocity=64):
    """Convert a sequence of musical notes to a Mido track."""
    track = mido.MidiTrack()
    for note_tuple in notes:
        for msg in tuple_to_mido_messages(note_tuple, channel, velocity):
            track.append(msg)
    return track

def sequences_to_mido_midi(parts, channel=0, velocity=64):
    """Convert multiple sequences of musical notes to a Mido MidiFile object."""
    mid = mido.MidiFile()
    for part_notes in parts:
        track = sequence_to_mido_track(part_notes, channel, velocity)
        mid.tracks.append(track)
    return mid

# SCAMP conversion
# ---------------
# To do...
#def to_scamp(notes, tempo=120, time_signature=(4, 4), key_signature=None):
#    """Convert notes to SCAMP format based on their structure."""
#    pass