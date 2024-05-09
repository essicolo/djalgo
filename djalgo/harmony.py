import random
import itertools
from . import utils
from . import harmony

class MusicTheoryConstants():
    """
    The Base class defines a set of musical scales, intervals, and notes.
    - scale_to_triad method returns the intervals for a triad based on the given scale intervals.
    - note_to_triad method converts a note to a triad based on the given scale intervals.
    """
    chromatic_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    scale_intervals = {
        'major': [0, 2, 4, 5, 7, 9, 11],  # Ionian
        'minor': [0, 2, 3, 5, 7, 8, 10],  # Aeolian
        'diminished': [0, 2, 3, 5, 6, 8, 9, 11],
        'major pentatonic': [0, 2, 4, 7, 9],
        'minor pentatonic': [0, 3, 5, 7, 10],
        'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'lydian': [0, 2, 4, 6, 7, 9, 11],
        'mixolydian': [0, 2, 4, 5, 7, 9, 10],
        'dorian': [0, 2, 3, 5, 7, 9, 10],
        'phrygian': [0, 1, 3, 5, 7, 8, 10],
        'locrian': [0, 1, 3, 5, 6, 8, 10],
        'harmonic minor': [0, 2, 3, 5, 7, 8, 11],
        'melodic minor ascending': [0, 2, 3, 5, 7, 9, 11],
        'melodic minor descending': [0, 2, 3, 5, 7, 8, 10],  # same as natural minor
    }
    intervals = {'P1': 0, 'm2': 1, 'M2': 2, 'm3': 3, 'M3': 4, 'P4': 5, 'P5': 7, 'm6': 8, 'M6': 9, 'm7': 10, 'M7': 11, 'P8': 12}

    @staticmethod
    def convert_flat_to_sharp(note):
        # Mapping of flat notes to their equivalent sharp notes
        flat_to_sharp = {
            'Bb': 'A#', 'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#',
            'B-': 'A#', 'D-': 'C#', 'E-': 'D#', 'G-': 'F#', 'A-': 'G#'
        }
        return flat_to_sharp.get(note, note) 

    @staticmethod
    def scale_to_triad(scale):
        """Returns the intervals for a triad based on the given scale intervals."""
        return [scale[i] for i in [0, 2, 4]]  # root, third, fifth

    
class Scale(MusicTheoryConstants):
    """
    Represents a musical scale.

    Args:
        tonic (str): The tonic note of the scale.
        mode (str or list): The type of scale. Defaults to 'major'. If a list is provided, it represents a custom scale.

    Raises:
        ValueError: If the tonic note is not a valid note or if the scale type is not a valid scale.

    Attributes:
        tonic (str): The tonic note of the scale.
        mode (str or list): The type of scale.
    """

    def __init__(self, tonic, mode='major'):
        if tonic not in self.chromatic_scale:
            tonic = self.convert_flat_to_sharp(tonic)
            if tonic not in self.chromatic_scale:
                raise ValueError(f"'{tonic}' is not a valid tonic note. Select one among '{self.chromatic_scale}'.")
        self.tonic = tonic

        if isinstance(mode, list):
            self.scale_intervals['custom'] = mode
            mode = 'custom'
        elif mode not in self.scale_intervals.keys():
            raise ValueError(f"'{mode}' is not a valid scale. Select one among '{self.scale_intervals.keys()}' or a list of half steps such as [0, 2, 4, 5, 7, 9, 11] for a major scale.")
            
        self.mode = mode

    def generate(self):
        """
        Generates the full range of the scale.
    
        Returns:
            list: A list of MIDI note numbers representing the full range of the scale.
        """
        tonic_note = self.chromatic_scale.index(self.tonic)
        scale = self.scale_intervals.get(self.mode, self.scale_intervals['major'])
    
        full_range_scale = []
        added_notes = set()  # Keep track of added notes
    
        for octave in range(11):
            for interval in scale:
                note = (tonic_note + interval) % 12 + octave * 12
                if note <= 127 and note not in added_notes:
                    full_range_scale.append(note)
                    added_notes.add(note)
    
        full_range_scale.sort()
        return full_range_scale



class Progression(MusicTheoryConstants):
    """A class representing a musical progression generator based on the circle of fifths (or any other interval)."""

    def __init__(self, tonic_pitch='C4', circle_of='P5', type='chords', radius=[3, 3, 1], weights=None):
        """
        Initialize a Progression object.

        Args:
            tonic_pitch (str): The tonic pitch of the progression. Defaults to 'C4'.
            circle_of (str): The interval to form the circle of fifths. Defaults to 'P5'.
            type (str): The type of progression to generate. Can be 'chords' or 'pitches'. Defaults to 'chords'.
            radius (list): A list defining the range for major, minor, and diminished chords. Defaults to [3, 3, 1].
            weights (list): The weights for selecting chord types. If not provided, the radius values will be used as weights.

        Raises:
            ValueError: If the circle_of value is not among the available intervals.
            ValueError: If the type value is not 'chords' or 'pitches'.
        """
        self.tonic_midi = utils.cde_to_midi(tonic_pitch)
        self.circle_of = circle_of
        self.type = type
        self.radius = radius
        self.weights = weights if weights else radius

        if self.circle_of not in self.intervals.keys():
            raise ValueError(f"Select a circle_of among {self.intervals.keys()}.")
        if self.type not in ['chords', 'pitches']:
            raise ValueError("Type must either be 'pitches' or 'chords'.")

    def compute_circle(self):
        """
        Compute chords based on the circle of fifths, thirds, etc., within the specified radius.

        Returns:
            tuple: A tuple containing lists of root MIDI notes for major, minor, and diminished chords.
        """
        n_semitones = self.intervals[self.circle_of]
        circle_notes = [self.tonic_midi]
        for _ in range(max(self.radius)):
            next_note = (circle_notes[-1] + n_semitones) % 12 + (circle_notes[-1] // 12) * 12
            circle_notes.append(next_note)

        major_roots = circle_notes[:self.radius[0]]
        minor_roots = circle_notes[:self.radius[1]]
        diminished_roots = circle_notes[:self.radius[2]]

        return major_roots, minor_roots, diminished_roots

    def generate_chord(self, root_note_midi, chord_type):
        """
        Generate a chord based on root MIDI note and chord type.

        Args:
            root_note_midi (int): The root MIDI note of the chord.
            chord_type (str): The type of chord to generate. Can be 'major', 'minor', or 'diminished'.

        Returns:
            list: A list of MIDI notes representing the generated chord.
        """
        chord_intervals = {
            'major': [0, 4, 7],
            'minor': [0, 3, 7],
            'diminished': [0, 3, 6]
        }

        intervals = chord_intervals.get(chord_type, [0, 4, 7])

        chord_notes = [(root_note_midi + interval) for interval in intervals]
        chord_notes = [note if note <= 127 else note - 12 for note in chord_notes]

        return chord_notes

    def generate(self, length=4, seed=None):
        """
        Generate a musical progression.

        Args:
            length (int): The length of the progression in number of chords. Defaults to 4.
            seed (int): The seed value for the random number generator. Defaults to None.

        Returns:
            list: A list of lists, where each inner list represents a chord in the progression.
        """
        if seed is not None:
            random.seed(seed)

        major_roots, minor_roots, diminished_roots = self.compute_circle()

        chord_roots = [major_roots, minor_roots, diminished_roots]
        progression = []
        for _ in range(length):
            chord_type_index = random.choices(range(3), weights=self.weights, k=1)[0]
            if chord_roots[chord_type_index]:
                root_note_midi = random.choice(chord_roots[chord_type_index])
                chord_type = ['major', 'minor', 'diminished'][chord_type_index]
                if isinstance(root_note_midi, list):
                    print('Warning: root_note_midi was a list, taking first element.')
                    root_note_midi = root_note_midi[0]
                chosen_chord = self.generate_chord(root_note_midi, chord_type)
                progression.append(chosen_chord)

        return progression


class Voice(MusicTheoryConstants):
    """
    A class to represent a musical voice.
    """  

    def __init__(self, mode='major', tonic='C', degrees=[0, 2, 4]):
        """
        Constructs all the necessary attributes for the voice object.

        Parameters
        ----------
            mode : str, optional
                The type of the scale (default is 'major').
            tonic : str, optional
                The tonic note of the scale (default is 'C').
            degrees : list, optional
                Relative degrees for chord formation (default is [0, 2, 4]).
        """
        self.tonic = tonic
        self.scale = harmony.Scale(tonic, mode).generate()  # a list of MIDI notes for the scale
        self.degrees = degrees  # relative degrees for chord formation

    def pitch_to_chord(self, pitch):
        """
        Convert a MIDI note to a chord based on the scale using the specified degrees.

        Parameters
        ----------
            pitch : int
                The MIDI note to convert.

        Returns
        -------
            list
                A list of MIDI notes representing the chord.
        """
        # to get the degree, I need a the tonic in the right octave, i.e. the tonic midi pitch
        octave = utils.get_octave(pitch)
        tonic_cde_pitch = self.tonic + str(octave)
        tonic_midi_pitch = utils.cde_to_midi(tonic_cde_pitch)

        # the degrees of the whole scale
        scale_degrees = [utils.get_degree_from_pitch(p, scale_list=self.scale, tonic_pitch=tonic_midi_pitch) for p in self.scale]
        pitch_degree = utils.get_degree_from_pitch(pitch, scale_list=self.scale, tonic_pitch=tonic_midi_pitch)
        pitch_degree = int(round(pitch_degree)) # round the degree if the pitch is out of scale

        chord = []
        for degree in self.degrees:
            absolute_degree = pitch_degree + degree
            absolute_index = scale_degrees.index(absolute_degree)
            chord.append(self.scale[absolute_index])
        return chord  # Chord is now directly from the scale

    def generate(self, notes, durations=None, arpeggios=False):
        """
        Generate chords or arpeggios based on the given notes.

        Args:
            notes (list or tuple): The notes to generate chords or arpeggios from.
            durations (list, optional): The durations of each note. If not provided, defaults to [1].
            arpeggios (bool, optional): If True, generate arpeggios instead of chords. Defaults to False.

        Returns:
            list: The generated chords or arpeggios.

        """
        
        if isinstance(notes, tuple):
            notes = [notes]
        if isinstance(notes[0], int): # if notes are in fact pitches
            if durations is None:
                durations = [1]
            durations_cycle = itertools.cycle(durations)
            current_offset = 0
            for i,p in enumerate(notes):
                d = next(durations_cycle)
                notes[i] = (p, d, current_offset)
                current_offset = current_offset + d
        
        chords = [(self.pitch_to_chord(p), d, o) for p, d, o in notes]
        
        if not arpeggios:
            return chords
        else:
            arpeggios_p = []
            for n in chords:
                pitches = n[0]
                for p in pitches:
                    arpeggios_p.append(p)
            arpeggios_n = []
            durations_cycle = itertools.cycle(durations) # reset cycle
            current_offset = 0
            for p in arpeggios_p:
                d = next(durations_cycle)
                arpeggios_n.append((p, d, current_offset))
                current_offset = current_offset + d
            return arpeggios_n



class Ornament(MusicTheoryConstants):

    def __init__(
            self, type='grace_note', tonic=None, mode=None, by=1.0,
            grace_note_type='acciaccatura', grace_pitches=None, trill_rate=0.125, arpeggio_degrees=None, slide_length=4.0
        ):
        """
        Initializes an Ornament object.

        Args:
            type (str): The type of ornament to be processed. Supported types include 'grace_note', 'trill', and 'mordent'.
            tonic (str): The tonic note for the scale.
            mode (str): The type of scale to generate.
            by (float): The pitch step for the trill.
            grace_note_type (str): Specifies the type of grace note ('acciaccatura' or 'appoggiatura') if applicable.
            grace_pitches (list): The list of pitches for the grace note.
            trill_rate (float): The duration of each individual note in the trill.
            arpeggio_degrees (list of integers): degrees in the scale to run the arpeggio
            slide_length (float): length of the slide
        """
        self.type = type
        if tonic and mode:
            self.tonic_index = self.chromatic_scale.index(tonic)  # Index in chromatic scale
            self.scale = self.generate_scale(tonic, mode)  # This will be a list of MIDI notes for the scale
            if arpeggio_degrees:
                self.arpeggio_voice = Voice(mode=mode, tonic=tonic, degrees=arpeggio_degrees)
            else:
                self.arpeggio_voice = None
        else:
            self.scale = None
            self.arpeggio_voice = None
        self.by = by
        self.grace_note_type = grace_note_type
        self.grace_pitches = grace_pitches
        self.trill_rate = trill_rate
        self.slide_length = slide_length
        

    def generate_scale(self, tonic, mode):
        """
        Generate a complete scale based on the tonic and scale type. This function is the same as the one in the Voice class.

        Args:
            tonic (str): The tonic note for the scale.
            mode (str): The type of scale to generate.
        
        Returns:
            list: A list of MIDI notes for the complete scale.
        """
        scale_pattern = self.scale_intervals[mode]
        scale_notes = [(self.tonic_index + interval) % 12 for interval in scale_pattern]  # Pitch classes
        complete_scale = []  # This will store the full scale in MIDI numbers
        for octave in range(-1, 10):  # Covering all MIDI octaves
            for note in scale_notes:
                midi_note = 12 * octave + note
                if 0 <= midi_note <= 127:  # Valid MIDI range
                    complete_scale.append(midi_note)
        return complete_scale

    def add_grace_note(self, notes, note_index):
        """
        Adds a grace note (either acciaccatura or appoggiatura) to a specified note in the list.

        Args:
            notes (list): The list of notes to be processed.
            note_index (int): The index of the note to which the trill will be added.

        Returns:
            list: The list of notes with the specified grace note added.
        """
        main_pitch, main_duration, main_offset = notes[note_index]
        ornament_pitch = random.choice(self.grace_pitches)
        if self.grace_note_type == 'acciaccatura':
            # Acciaccatura is very brief, does not alter the main note's start time.
            grace_duration = main_duration * 0.125  # Typically very short
            modified_main = (main_pitch, main_duration, main_offset + grace_duration)
            new_notes = notes[:note_index] + [(ornament_pitch, grace_duration, main_offset), modified_main] + notes[note_index + 1:]
        elif self.grace_note_type == 'appoggiatura':
            # Appoggiatura takes half the time of the main note and delays its start.
            grace_duration = main_duration / 2
            modified_main = (main_pitch, grace_duration, main_offset + grace_duration)
            new_notes = notes[:note_index] + [(ornament_pitch, grace_duration, main_offset), modified_main] + notes[note_index + 1:]
        else:
            # If neither, return the list unchanged
            new_notes = notes
        return new_notes

    def add_trill(self, notes, note_index):
        """
        Simulates a trill ornament by alternating between the original pitch and one step above.

        Args:
            notes (list): The list of notes to be processed.
            note_index (int): The index of the note to which the trill will be added.

        Returns:
            list: The list of notes with the specified trill applied to the specified note.
        """
        main_pitch, main_duration, main_offset = notes[note_index]
        trill_notes = []
        current_offset = main_offset
        
        # Determine the pitch to trill with based on the scale or semitone adjustment
        if self.scale and main_pitch in self.scale:
            pitch_index = self.scale.index(main_pitch)
            trill_pitch = self.scale[(pitch_index + int(round(self.by))) % len(self.scale)]  # Ensure the index wraps around the scale
        else:
            trill_pitch = main_pitch + self.by  # Default step if no scale is given or pitch is not in scale

        # Generate the sequence of trill notes to insert
        while current_offset < main_offset + main_duration:
            trill_notes.append((main_pitch, self.trill_rate, current_offset))
            trill_notes.append((trill_pitch, self.trill_rate, current_offset + self.trill_rate))
            current_offset += 2 * self.trill_rate

        # Insert the trill notes into the original list, replacing the original note at note_index
        new_notes = notes[:note_index] + trill_notes + notes[note_index + 1:]
        return new_notes


    def add_mordent(self, notes, note_index):
        """
        Simulates a mordent ornament by rapidly alternating between the original pitch and one step defined in `self.by`.

        Args:
            notes (list): The list of notes to be processed.
            note_index (int): The index of the note to which the trill will be added.

        Returns:
            list: A list containing the notes that make up the mordent.
        """
        main_pitch, main_duration, main_offset = notes[note_index]
        if self.scale and main_pitch in self.scale:
            pitch_index = self.scale.index(main_pitch)
            mordent_pitch = self.scale[pitch_index + int(round(self.by))]
        else:
            # If no scale is provided, default to moving a semitone down for a lower mordent
            # or a semitone up for an upper mordent
            mordent_pitch = main_pitch + self.by

        # The mordent splits the duration into three parts
        part_duration = main_duration / 3
        mordent_notes = [
            (main_pitch, part_duration, main_offset),
            (mordent_pitch, part_duration, main_offset + part_duration),
            (main_pitch, part_duration, main_offset + 2 * part_duration)
        ]
        new_notes = notes[:note_index] + mordent_notes + notes[note_index + 1:]
        return new_notes

    def add_arpeggiation(self, notes, note_index, voice):
        """
        Applies arpeggiation to a chord at a specified index in the list of notes using the degrees from a Voice instance.

        Args:
            notes (list): The list of notes to be processed.
            note_index (int): The index of the note to which the arpeggiation will be added.
            voice (Voice): An instance of the Voice class.

        Returns:
            list: The list of notes with the specified arpeggiation applied to the specified chord.
        """
        root_note = notes[note_index][0]
        main_duration, main_offset = notes[note_index][1], notes[note_index][2]
        
        # Generate the arpeggio notes based on the Voice's degrees
        arpeggio_notes = []
        for degree in voice.degrees:
            # Calculate pitch from the scale degree
            scale_degree_index = (voice.scale.index(root_note) + degree) % len(voice.scale)
            arpeggio_pitch = voice.scale[scale_degree_index]

            # Add an arpeggio note for each degree
            note_duration = main_duration / len(voice.degrees)
            note_offset = main_offset + voice.degrees.index(degree) * note_duration
            arpeggio_notes.append((arpeggio_pitch, note_duration, note_offset))

        # Replace the original note with the arpeggio notes
        new_notes = notes[:note_index] + arpeggio_notes + notes[note_index + 1:]
        return new_notes
    
    def add_turn(self, notes, note_index):
        """
        Simulates a turn ornament by playing the note above, the note itself, the note below, and returning to the note.

        Args:
            notes (list): The list of notes to be processed.
            note_index (int): The index of the note to which the turn will be added.

        Returns:
            list: The list of notes with the specified turn applied to the specified note.
        """
        main_pitch, main_duration, main_offset = notes[note_index]
        part_duration = main_duration / 4  # Splitting the total duration among the four notes of the turn

        if self.scale and main_pitch in self.scale:
            pitch_index = self.scale.index(main_pitch)
            upper_pitch = self.scale[(pitch_index + 1) % len(self.scale)]
            lower_pitch = self.scale[(pitch_index - 1 + len(self.scale)) % len(self.scale)]
        else:
            upper_pitch = main_pitch + self.by  # Assuming 'by' is the interval step
            lower_pitch = main_pitch - self.by

        turn_notes = [
            (upper_pitch, part_duration, main_offset),
            (main_pitch, part_duration, main_offset + part_duration),
            (lower_pitch, part_duration, main_offset + 2 * part_duration),
            (main_pitch, part_duration, main_offset + 3 * part_duration)
        ]
        
        new_notes = notes[:note_index] + turn_notes + notes[note_index + 1:]
        return new_notes

    def add_slide(self, notes, note_index, slide_length=4):
        """
        Simulates a slide from the current note to the next by incrementally changing the pitch.

        Args:
            notes (list): The list of notes to be processed.
            note_index (int): The index of the note from which the slide will start.
            slide_length (int): The number of steps in the slide.

        Returns:
            list: The list of notes with the specified slide applied.
        """
        if note_index < len(notes) - 1:  # Ensure there is a following note to slide into
            start_pitch, _, start_offset = notes[note_index]
            end_pitch, end_duration, end_offset = notes[note_index + 1]

            # Calculate pitch steps for the slide, assuming a linear slide for simplicity
            pitch_step = (end_pitch - start_pitch) / slide_length
            slide_duration = (end_offset - start_offset) / slide_length
            
            slide_notes = [(int(start_pitch + pitch_step * i), slide_duration, start_offset + slide_duration * i) for i in range(slide_length)]
            
            # Insert the slide notes, remove the original note and the next one since they are now part of the slide
            new_notes = notes[:note_index] + slide_notes + notes[note_index + 2:]
        else:
            # No following note to slide into, so return the original notes unchanged
            new_notes = notes

        return new_notes


    def generate(self, notes, note_index=None):
        """
        Applies the specified ornamentation action and type to the list of notes.

        Args:
            notes (list): The list of notes to be processed.
            note_index (int): The index of the note to ornament. If None, a note will be chosen randomly.

        Returns:
            list: The list of notes with the specified ornamentation applied.
        """
        if note_index is None:
            note_index = random.randint(0, len(notes) - 1)
        
        if self.type == 'grace_note':
            return self.add_grace_note(notes, note_index)
        elif self.type == 'trill':
            return self.add_trill(notes, note_index)
        elif self.type == 'mordent':
            return self.add_mordent(notes, note_index)
        elif self.type == 'arpeggio':
            return self.add_arpeggiation(notes, note_index, self.arpeggio_voice)
        elif self.type == 'turn':
            return self.add_turn(notes, note_index)
        elif self.type == 'slide':
            return self.add_slide(notes, note_index, self.slide_length)
        else:
            return notes  # Return original if ornament type is not recognized or required data is missing

