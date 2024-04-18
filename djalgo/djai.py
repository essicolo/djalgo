import os
import json
import pretty_midi
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
    def __init__(self, sequence_length_i, sequence_length_o, num_instruments):
        self.sequence_length_i = sequence_length_i
        self.sequence_length_o = sequence_length_o
        self.num_instruments = num_instruments
        # feature indices metadata
        self.feature_config = {
            'numerical_features': slice(0, 4),  # assuming first 4 features are numerical
            'instrument_features': slice(4, None)  # assuming features from index 4 onwards are instrument
        }
        self.numerical_indices = self.feature_config['numerical_features']
        self.instrument_indices = self.feature_config['instrument_features']
        self.means = None
        self.stds = None

    def extract_features(self, notes, instrument_index, midi_data):
        # Get tempo changes and tick per beat
        tempo_changes = midi_data.get_tempo_changes()
        tick_per_beat = midi_data.resolution
        
        # Interpolate tempo changes to find the tempo at each note start time
        tempos = np.interp([note.start for note in notes], tempo_changes[0], tempo_changes[1])
        ticks_to_quarters = 60.0 / (tempos / tick_per_beat)  # Converts ticks to quarter note length

        # Calculate features using interpolated ticks to quarters for each note
        pitches = [note.pitch for note in notes]
        durations = [(note.end - note.start) / tpb for note, tpb in zip(notes, ticks_to_quarters)]
        offsets = [note.start / tpb for note, tpb in zip(notes, ticks_to_quarters)]
        time_deltas = [offsets[i] - (offsets[i-1] + durations[i-1]) if i > 0 else offsets[0] for i in range(len(notes))]
        
        instrument_indices = [instrument_index] * len(notes)
        return np.stack([pitches, durations, offsets, time_deltas, instrument_indices], axis=1)

    def midi_files_to_sequences(self, midi_files):
        all_sequences = []
        for midi_file in midi_files:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            for instrument in midi_data.instruments:
                notes = instrument.notes
                if len(notes) < self.sequence_length_i + self.sequence_length_o:
                    continue
                for i in range(len(notes) - self.sequence_length_i - self.sequence_length_o + 1):
                    sequence = notes[i:i + self.sequence_length_i + self.sequence_length_o]
                    features = self.extract_features(sequence, instrument.program, midi_data)
                    all_sequences.append(features)
        return np.array(all_sequences)

    def compute_scaling_parameters(self, sequences):
        self.means = np.nanmean(sequences[:, :, self.numerical_indices], axis=(0, 1))
        self.stds = np.nanstd(sequences[:, :, self.numerical_indices], axis=(0, 1))

    def scale_numerical_features(self, numerical_sequences):
        if self.means is None or self.stds is None:
            self.compute_scaling_parameters(numerical_sequences)
        sequences_scaled = (numerical_sequences - self.means) / self.stds
        return sequences_scaled
    
    def preprocess_data(self, sequences):
        numerical_sequences = sequences[:, :, self.numerical_indices]
        scaled_numerical_sequences = self.scale_numerical_features(numerical_sequences)
        instrument_sequences = sequences[:, :, self.instrument_indices]

        # One-hot encoding for instrument indices for both inputs and outputs
        instrument_indices = instrument_sequences.astype(int).reshape(-1)
        one_hot_instruments = np.eye(self.num_instruments)[instrument_indices]
        one_hot_instruments = one_hot_instruments.reshape(instrument_sequences.shape[0], instrument_sequences.shape[1], self.num_instruments)

        #print("scaled sequences shape:", scaled_numerical_sequences.shape)
        #print("one hot instruments shape:", one_hot_instruments.shape)

        # Concatenate scaled features with one-hot encoded instrument indices
        numerical_input_data = scaled_numerical_sequences[:, :self.sequence_length_i, self.numerical_indices]
        nunerical_output_data = scaled_numerical_sequences[:, self.sequence_length_i:, self.numerical_indices]
        instrument_input_data = one_hot_instruments[:, :self.sequence_length_i]
        instrument_output_data = one_hot_instruments[:, self.sequence_length_i:]
        inputs = np.concatenate([numerical_input_data, instrument_input_data], axis=-1)
        outputs = [nunerical_output_data[:, :, i] for i in range(nunerical_output_data.shape[2])]
        outputs.append(instrument_output_data)  # Add one-hot encoded instrument index as the last output

        return inputs, tuple(outputs)
    
    def prepare_data(self, midi_files):
        sequences = self.midi_files_to_sequences(midi_files)
        inputs, outputs = self.preprocess_data(sequences)
        return inputs, outputs


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
    def __init__(self, sequence_length_i=30, sequence_length_o=10, num_instruments=2, model_type='lstm', n_layers=3, n_units=128, dropout=0.2, batch_size=32, learning_rate=0.005, num_heads=2, loss_weights=None):
        self.sequence_length_i = sequence_length_i
        self.sequence_length_o = sequence_length_o
        self.num_instruments = num_instruments
        self.num_features = 5  # pitch, duration, offset, time_delta, instrument_index
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_weights = loss_weights if loss_weights else {'pitch': 1.0, 'duration': 1.0, 'offset': 1.0, 'time_delta': 100.0, 'instrument_index': 100.0}
        self.data_processor = DataProcessor(sequence_length_i, sequence_length_o, num_instruments)
        self.model = self._create_default_model(n_layers, n_units, dropout, num_heads, model_type, num_instruments)
        
    def _create_default_model(self, n_layers, n_units, dropout, num_heads, model_type, num_instruments):
        n_features_onehot = self.num_features - 1 + num_instruments
        input_shape = (self.sequence_length_i, n_features_onehot)
        inputs = tf.keras.Input(shape=input_shape)

        x = inputs
        if model_type == 'transformer':
            positional_encoding_layer = PositionalEncoding(self.sequence_length_i, n_features_onehot)
            x = positional_encoding_layer(x)
            for i in range(n_layers):
                attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=n_units)(x, x)
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

        # Using direct slicing here
        x = x[:, -self.sequence_length_o:]

        outputs = [
            tf.keras.layers.Dense(1, name=f"{feature}")(x) for feature in ["pitch", "duration", "offset", "time_delta"]
        ]
        if num_instruments > 1:
            instruments_output = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(num_instruments, activation='softmax'), name='instrument_index'
            )(x)
        else:
            instruments_output = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(1, activation='sigmoid'), name='instrument_index'
            )(x)
        outputs.append(instruments_output)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss={'pitch': 'mean_squared_error',
                            'duration': 'mean_squared_error',
                            'offset': 'mean_squared_error',
                            'time_delta': 'mean_squared_error',
                            'instrument_index': 'categorical_crossentropy' if num_instruments > 1 else 'binary_crossentropy'},
                      loss_weights=self.loss_weights,
                      metrics={'instrument_index': 'accuracy'})
        return model
    
    def fit(self, midi_files, epochs=10):
        inputs, targets = self.data_processor.prepare_data(midi_files)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(self.batch_size)
        history = self.model.fit(dataset, epochs=epochs)
        return history
    
    def scaled_back(self, predicted_outputs):
        rescaled_outputs = predicted_outputs.copy()
        numerical_indices = self.data_processor.numerical_indices
        means = self.data_processor.means
        stds = self.data_processor.stds
        for i in range(predicted_outputs.shape[2]):  # Loop over the last dimension (features)
            if i in range(numerical_indices.start, numerical_indices.stop): 
                rescaled_outputs[:, :, i] = predicted_outputs[:, :, i] * stds[i - numerical_indices.start] + means[i - numerical_indices.start]
        return rescaled_outputs

    def sequences_to_track_list(self, predicted_sequences):
        tracks = []
        instrument_indices = self.data_processor.instrument_indices
        numerical_indices = self.data_processor.numerical_indices
        for i in range(self.num_instruments):
            track = []
            for j in range(predicted_sequences.shape[1]):
                max_instrument_index = np.argmax(predicted_sequences[0, j, instrument_indices])
                if max_instrument_index == i:
                    pitch = int(round(predicted_sequences[0, j, numerical_indices.start]))
                    duration = predicted_sequences[0, j, numerical_indices.start + 1]
                    offset = predicted_sequences[0, j, numerical_indices.start + 2]
                    track.append((pitch, duration, offset))
            tracks.append(track)

        return tracks
    
    def generate(self, midi_file_path, length=10):
        if isinstance(midi_file_path, str):
            midi_file_path = [midi_file_path]
        inputs, _ = self.data_processor.prepare_data(midi_file_path)
        if inputs.size == 0:
            print("No sequences extracted, possibly too few notes.")
            return None
        input_data = inputs[0:1, :self.sequence_length_i, :] # 0:1 for only use the first sequence
        predictions = self.model.predict(input_data, verbose=0)
        predictions = np.concatenate(predictions, axis=2) # the prediction has one output for each feature (necessary for loss calculation)
        if predictions.shape[1] > length:
            predictions = predictions[:, :length, :]
        else:
            total_steps = length - predictions.shape[1]
            for _ in range(total_steps):
                input_data = np.concatenate((input_data[:, 1:, :], predictions[:, -1:, :]), axis=1)
                next_step = np.concatenate(self.model.predict(input_data, verbose=0), axis=2)
                predictions = np.concatenate((predictions, next_step), axis=1)

        predictions = self.scaled_back(predictions)
        #print("Print predictions:", predictions)
        tracks = self.sequences_to_track_list(predictions)
        return tracks

    def save(self, filepath):
        self.model.save(filepath)

    @staticmethod
    def load(filepath):
        return tf.keras.models.load_model(filepath, custom_objects={'PositionalEncoding': PositionalEncoding})
    
