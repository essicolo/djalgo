import os
import json
import music21 as m21
import tensorflow as tf
import numpy as np
import random
import glob

def scan_midi_files(directory, max_files=None):
    """
    Scans the specified directory for MIDI files using glob with a while loop.

    Args:
        directory (str): The directory to scan for MIDI files.
        max_files (int, optional): The maximum number of files to scan. If None, all files are scanned.

    Returns:
        list: The list of MIDI files found.
    """
    search_pattern = os.path.join(directory, '**', '*.mid*')
    midi_files = []

    # Utiliser glob.iglob pour obtenir un itérateur
    for file in glob.iglob(search_pattern, recursive=True):
        midi_files.append(file)
        if max_files is not None and len(midi_files) >= max_files:
            break

    return midi_files



class DataProcessor:
    def __init__(self, sequence_length_i, sequence_length_o, n_instruments, scale_mins=None, scale_maxs=None):
        self.sequence_length_i = sequence_length_i
        self.sequence_length_o = sequence_length_o
        self.total_sequence_length = sequence_length_i + sequence_length_o
        self.n_instruments = n_instruments
        self.numerical_indices = slice(0, 3) # hard coded, pitch, duration, tick_delta
        self.instrument_encoding = {}
        self.scale_mins = scale_mins
        self.scale_maxs = scale_maxs

    def encode_instruments(self, part):
        if part.partName not in self.instrument_encoding:
            if len(self.instrument_encoding) < self.n_instruments:
                self.instrument_encoding[part.partName] = len(self.instrument_encoding)
            else:
                return np.zeros(self.n_instruments)  # Return an empty one-hot vector
        index = self.instrument_encoding.get(part.partName, 0)
        one_hot = np.zeros(self.n_instruments)
        one_hot[index] = 1
        return one_hot

    def extract_features(self, notes, instrument_vector):
        pitches = []
        durations = []
        tick_deltas = []
        
        # Ensure that notes are sorted correctly by their offsets
        notes.sort(key=lambda x: x.offset)

        # Use the first note's offset as the starting point for delta calculations
        previous_offset = notes[0].offset

        for i, note in enumerate(notes):
            current_offset = note.offset  # Handle both chords and notes uniformly
            if note.isChord:
                # If it's a chord, process each note in the chord
                for chord_note in note.notes:
                    if len(pitches) < self.total_sequence_length:
                        pitches.append(chord_note.pitch.midi)
                        durations.append(chord_note.duration.quarterLength)
                        tick_deltas.append(current_offset - previous_offset)  # Delta calculation
                        previous_offset = current_offset  # Update previous offset after processing the chord
            elif note.isNote:
                if len(pitches) < self.total_sequence_length:
                    pitches.append(note.pitch.midi)
                    durations.append(note.duration.quarterLength)
                    tick_deltas.append(current_offset - previous_offset)
                    previous_offset = current_offset

        # Adjust the first tick delta to zero for the sequence start
        if tick_deltas:
            tick_deltas[0] = 0

        # Combine features and tile the instrument vector
        features = np.column_stack((pitches, durations, tick_deltas))
        instrument_features = np.tile(instrument_vector, (len(pitches), 1))
        return np.column_stack((features, instrument_features))
    
    def extract_features_rest(self, notes, instrument_vector):
        pitches = []
        durations = []
        is_rests = []
        
        # Ensure that notes are sorted correctly by their offsets
        notes.sort(key=lambda x: x.offset)

        for i, note in enumerate(notes):
            if note.isChord:
                # If it's a chord, process each note in the chord
                for chord_note in note.notes:
                    if len(pitches) < self.total_sequence_length:
                        pitches.append(chord_note.pitch.midi)
                        durations.append(chord_note.duration.quarterLength)
                        is_rests.append(0)  # Not a rest
            elif note.isNote:
                if len(pitches) < self.total_sequence_length:
                    pitches.append(note.pitch.midi)
                    durations.append(note.duration.quarterLength)
                    is_rests.append(0)  # Not a rest
            elif note.isRest:
                if len(pitches) < self.total_sequence_length:
                    pitches.append(0)  # No pitch for a rest
                    durations.append(note.duration.quarterLength)
                    is_rests.append(1)  # It's a rest

        # Combine features and tile the instrument vector
        features = np.column_stack((pitches, durations, is_rests))
        instrument_features = np.tile(instrument_vector, (len(pitches), 1))
        return np.column_stack((features, instrument_features))


    def midi_files_to_sequences(self, midi_files):
        all_sequences = []
        for midi_file in midi_files:
            score = m21.converter.parse(midi_file)
            for part in score.parts:
                instrument_vector = self.encode_instruments(part)
                if instrument_vector is None:
                    continue

                notes = list(part.flatten().notesAndRests)
                notes.sort(key=lambda note: note.offset)
                # Iterate through the notes to extract all possible sequences of the defined length
                for i in range(len(notes) - self.total_sequence_length + 1):
                    sequence = notes[i:i + self.total_sequence_length]
                    features = self.extract_features_rest(sequence, instrument_vector)
                    if features.shape[0] == self.total_sequence_length:  # Ensure only complete sequences are added
                        all_sequences.append(features)

        return np.array(all_sequences, dtype=object)

    def compute_scaling_parameters(self, sequences):
        self.scale_mins = np.nanmin(sequences[:, :, self.numerical_indices], axis=(0, 1))
        self.scale_maxs = np.nanmax(sequences[:, :, self.numerical_indices], axis=(0, 1))
        self.scale_mins[0] = 0  # Min pitch
        self.scale_maxs[0] = 127  # Max pitch

    def scale_numerical_features(self, sequences):
        if self.scale_mins is None or self.scale_maxs is None:
            self.compute_scaling_parameters(sequences)
        sequences[:, :, self.numerical_indices] = (sequences[:, :, self.numerical_indices] - self.scale_mins) / (self.scale_maxs - self.scale_mins)
        return sequences

    def prepare_data(self, midi_files):
        self.sequences = self.midi_files_to_sequences(midi_files)
        if len(self.sequences) == 0:
            raise ValueError("No sequences were extracted. Check the input data, maybe just a wrong path definition.")
        if self.n_instruments == 1:
            self.sequences = self.sequences[:, :, self.numerical_indices] # remove the instrument feature if only one instrument is scanned
        self.sequences_sc = self.scale_numerical_features(self.sequences.copy())
        sequences_input = np.array(self.sequences_sc[:, :self.sequence_length_i, :], dtype=np.float32)
        sequences_output = self.sequences_sc[:, self.sequence_length_i:, :]
        sequences_output_list = [np.array(sequences_output[:, :, i], dtype=np.float32) for i in range(self.numerical_indices.stop)]
        sequences_output_list.append(
            np.array(sequences_output[:, :, self.numerical_indices.stop:], dtype=np.float32)
        )
        return sequences_input, tuple(sequences_output_list)


@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super().get_config()  # Make sure the super call is appropriate.
        config.update({
            'position': self.position,
            'd_model': self.d_model
        })
        return config

class ModelManager:
    def __init__(self, sequence_length_i=32, sequence_length_o=8, n_instruments=1, model_type='lstm', n_layers=3, n_units=128, dropout=0.2, batch_size=32, learning_rate=0.005, n_heads=2, loss_weights=None):
        """
        sequence_length_o is not 1, even though we work in autoregression. Predicting multiple steps ahead even though subsequent steps are ignored is called teacher forcing.
        """

        self.sequence_length_i = sequence_length_i
        self.sequence_length_o = sequence_length_o
        self.n_instruments = n_instruments
        self.num_numeric_features = 3  # pitch, duration, tick_delta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        if loss_weights is None:
            # Set default loss weights
            self.loss_weights = {'pitch': 1.0, 'duration': 1.0, 'tick_delta': 1.0}
            if n_instruments > 1:
                self.loss_weights['instrument_index'] = 1.0  # Add instrument loss weight 
        else:
            self.loss_weights = loss_weights
        self.model = self._create_default_model(n_layers, n_units, dropout, n_heads, model_type, n_instruments)

        self.data_processor = DataProcessor(sequence_length_i, sequence_length_o, n_instruments)
        self.model = self._create_default_model(n_layers, n_units, dropout, n_heads, model_type, n_instruments)

    def _create_default_model(self, n_layers, n_units, dropout, n_heads, model_type, n_instruments):
        if n_instruments < 2:
            n_features_onehot = self.num_numeric_features
        else:
            n_features_onehot = self.num_numeric_features + n_instruments
        input_shape = (self.sequence_length_i, n_features_onehot)
        inputs = tf.keras.Input(shape=input_shape)

        x = inputs
        if model_type == 'transformer':
            positional_encoding_layer = PositionalEncoding(self.sequence_length_i, n_features_onehot)
            x = positional_encoding_layer(x)
            for i in range(n_layers):
                attention_output = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=n_units)(x, x)
                x = tf.keras.layers.Dropout(dropout)(x)
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
                ff_output = tf.keras.Sequential([
                    tf.keras.layers.Dense(n_units, activation='relu'),
                    tf.keras.layers.Dense(x.shape[-1]),
                ])(x)
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff_output)
        else:
            for i in range(n_layers):
                if model_type == 'lstm':
                    x = tf.keras.layers.LSTM(n_units, return_sequences=True, dropout=dropout)(x)
                elif model_type == 'gru':
                    x = tf.keras.layers.GRU(n_units, return_sequences=True, dropout=dropout)(x)

        x = x[:, -self.sequence_length_o:]

        outputs = [
            #tf.keras.layers.Dense(1, name=f"{feature}")(x) for feature in ["pitch", "duration", "tick_delta"]
            tf.keras.layers.Dense(1, name=f"{feature}")(x) for feature in ["pitch", "duration", "is_rest"]
        ]
        if n_instruments > 1:
            instruments_output = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(n_instruments, activation='softmax'), name='instrument_index'
            )(x)
            outputs.append(instruments_output)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.learning_rate,
            decay_steps=10000,
            decay_rate=0.96,
            staircase=True
        )
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
        
        loss_dict = {'pitch': 'mean_squared_error',
                    'duration': 'mean_squared_error',
                    #'tick_delta': 'mean_squared_error'}
                    'is_rest': 'categorical_crossentropy'}
        if n_instruments > 1:
            loss_dict['instrument_index'] = 'categorical_crossentropy'
        
        model.compile(optimizer=optimizer,
                    loss=loss_dict,
                    loss_weights=self.loss_weights,
                    metrics={'instrument_index': 'accuracy'} if n_instruments > 1 else None)
        return model
    
    def fit(self, midi_files, epochs=10):
        inputs, targets = self.data_processor.prepare_data(midi_files)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(self.batch_size)
        history = self.model.fit(dataset, epochs=epochs)
        return history
        
    def scale_back(self, predicted_outputs):
        unscaled_outputs = []
        for i, p in enumerate(predicted_outputs):
            if i < self.data_processor.numerical_indices.stop:  # Handle numerical features
                min_val = self.data_processor.scale_mins[i]
                max_val = self.data_processor.scale_maxs[i]
                unscaled_outputs.append(p * (max_val - min_val) + min_val)
            else:  # Handle one-hot encoded instruments
                exp_logits = np.exp(p)
                softmax_output = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                unscaled_outputs.append(softmax_output)
        return unscaled_outputs

    def generate(self, midi_file_path, length=10):
        if isinstance(midi_file_path, str):
            midi_file_path = [midi_file_path]
        inputs, _ = self.data_processor.prepare_data(midi_file_path)
        if inputs.size == 0:
            print("No sequences extracted, possibly too few notes.")
            return None

        current_sequences = inputs[0:1, :self.sequence_length_i, :]  # Initial input
        generated_sequences = [[] for _ in range(self.n_instruments)]

        for _ in range(length):
            predictions = self.model.predict(current_sequences, verbose=0)
            scaled_predictions = self.scale_back(predictions)

            for i in range(self.n_instruments):
                pitch = int(round(scaled_predictions[0][0, -1, 0]))  # Assumes pitch is the first feature
                duration = scaled_predictions[1][0, -1, 0]  # Assumes duration is the second feature
                offset = scaled_predictions[2][0, -1, 0]  # Assumes offset is the third feature
                if self.n_instruments > 1:
                    instrument_probabilities = scaled_predictions[3][0, -1, :]  # Instrument classification
                    chosen_instrument = np.argmax(instrument_probabilities)
                    if chosen_instrument == i:
                        generated_sequences[i].append((pitch, duration, offset))
                else:
                    generated_sequences[i].append((pitch, duration, offset))

                # Update the sequence for the next prediction
                next_step_features = []
                for i in range(len(predictions)):
                    if i < self.data_processor.numerical_indices.stop:
                        next_step_features.append(predictions[i][0, 0, 0])
                    else:
                        next_step_features.extend([p for p in predictions[i][0, 0, :]])
                next_step_features = np.array(next_step_features, dtype=np.float32).reshape(1, 1, -1)
                current_sequences = np.concatenate([current_sequences[:, 1:, :], next_step_features], axis=1)
        
        if len(generated_sequences) == 1:
            return generated_sequences[0]
        else:
            return generated_sequences

    def save(self, filepath):
        self.model.save(filepath)

    @staticmethod
    def load(filepath):
        return tf.keras.models.load_model(filepath, custom_objects={'PositionalEncoding': PositionalEncoding})