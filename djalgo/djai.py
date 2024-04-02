import os
import json
import pretty_midi
import tensorflow as tf
import numpy as np
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

    for file in glob.iglob(search_pattern, recursive=True):
        midi_files.append(file)
        if max_files is not None and len(midi_files) >= max_files:
            break

    return midi_files


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional encoding layer for Transformer models.

    Args:
        position (int): The maximum sequence length.
        d_model (int): The dimensionality of the model.

    Attributes:
        pos_encoding (tf.Tensor): The positional encoding tensor.
    """

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        """
        Calculates the angles for the positional encoding.

        Args:
            position (tf.Tensor): The position tensor.
            i (tf.Tensor): The index tensor.
            d_model (int): The dimensionality of the model.

        Returns:
            tf.Tensor: The angles tensor.

        """
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        """
        Generates the positional encoding tensor.

        Args:
            position (int): The maximum sequence length.
            d_model (int): The dimensionality of the model.

        Returns:
            tf.Tensor: The positional encoding tensor.

        """
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        """
        Adds the positional encoding to the inputs.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor.

        """
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    
class DjFlow:
    """
    DjFlow class represents a music generation model using deep learning techniques.

    Args:
        sequence_length_i (int): The length of the input sequence.
        sequence_length_o (int): The length of the output sequence.
        num_instruments (int): The number of instruments in the music.
        model_type (str): The type of the model architecture to use. Possible values are 'lstm', 'gru', and 'transformer'.
        n_layers (int): The number of layers in the model.
        n_units (int or list): The number of units in each layer. If int, the same number of units will be used for all layers. If list, each element represents the number of units in each layer.
        dropout (float or list): The dropout rate for each layer. If float, the same dropout rate will be used for all layers. If list, each element represents the dropout rate for each layer.
        batch_size (int): The batch size for training.
        learning_rate (float): The learning rate for the optimizer.
        num_heads (int): The number of attention heads in the transformer model.
        loss_weights (dict): The weights for each loss function. If None, default weights will be used.

    Attributes:
        sequence_length_i (int): The length of the input sequence.
        sequence_length_o (int): The length of the output sequence.
        num_instruments (int): The number of instruments in the music.
        num_features (int): The number of features used to represent each note.
        model_type (str): The type of the model architecture used.
        n_layers (int): The number of layers in the model.
        n_units (list): The number of units in each layer.
        dropout (list): The dropout rate for each layer.
        batch_size (int): The batch size for training.
        learning_rate (float): The learning rate for the optimizer.
        num_heads (int): The number of attention heads in the transformer model.
        loss_weights (dict): The weights for each loss function.
        model (tf.keras.Model): The music generation model.

    Methods:
        midi_files_to_sequences(midi_files): Converts MIDI files to input sequences for training.
        extract_features(notes, instrument_index): Extracts features from a list of notes.
        compute_scaling_parameters(sequences): Computes the scaling parameters for feature scaling.
        scale_features(sequences): Scales the features of the input sequences.
        scale_back_features(sequences_scaled): Scales back the features of the input sequences.
        prepare_data(midi_files): Prepares the input data for training.
        _create_default_model(): Creates the default music generation model.
        fit(midi_files, epochs): Trains the model with the given data.
        predict(initial_midi, num_predictions): Generates music predictions based on the initial MIDI file.

    """

    def __init__(self, sequence_length_i=30, sequence_length_o=10, num_instruments=2, model_type='lstm', n_layers=3, n_units=128, dropout=0.2, batch_size=32, learning_rate=0.005, num_heads=2, loss_weights=None):
        self.sequence_length_i = sequence_length_i
        self.sequence_length_o = sequence_length_o
        self.num_instruments = num_instruments
        self.num_features = 5 # pitch, duration, offset, time_delta, instrument_index
        self.model_type = model_type
        self.n_layers = n_layers
        self.n_units = [n_units] * n_layers if isinstance(n_units, int) else n_units
        self.dropout = [dropout] * n_layers if isinstance(dropout, float) else dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_heads = num_heads
        self.loss_weights = loss_weights if loss_weights else {'pitch': 1.0, 'duration': 1.0, 'offset': 1.0, 'time_delta': 100.0, 'instrument_index': 100.0}
        self.model = self._create_default_model()

    # Rest of the code...
class DjFlow:
    """
    
    """
    def __init__(self, sequence_length_i=30, sequence_length_o=10, num_instruments=2, model_type='lstm', n_layers=3, n_units=128, dropout=0.2, batch_size=32, learning_rate=0.005, num_heads=2, loss_weights=None):
        self.sequence_length_i = sequence_length_i
        self.sequence_length_o = sequence_length_o
        self.num_instruments = num_instruments
        self.num_features = 5 # pitch, duration, offset, time_delta, instrument_index
        self.model_type = model_type
        self.n_layers = n_layers
        self.n_units = [n_units] * n_layers if isinstance(n_units, int) else n_units
        self.dropout = [dropout] * n_layers if isinstance(dropout, float) else dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_heads = num_heads
        self.loss_weights = loss_weights if loss_weights else {'pitch': 1.0, 'duration': 1.0, 'offset': 1.0, 'time_delta': 100.0, 'instrument_index': 100.0}
        self.model = self._create_default_model()

    def midi_files_to_sequences(self, midi_files):
        all_sequences = []
        for midi_file in midi_files:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            for instrument in midi_data.instruments:
                notes = instrument.notes
                if len(notes) < self.sequence_length_i + self.sequence_length_o:
                    continue  # Skip instruments with not enough notes
                for i in range(len(notes) - self.sequence_length_i - self.sequence_length_o + 1):
                    sequence = notes[i:i + self.sequence_length_i + self.sequence_length_o]
                    features = self.extract_features(sequence, instrument.program)
                    all_sequences.append(features)
        return np.array(all_sequences)

    def extract_features(self, notes, instrument_index):
        pitches = [note.pitch for note in notes]
        durations = [note.end - note.start for note in notes]
        offsets = [note.start for note in notes]
        time_deltas = [notes[i].start - (notes[i-1].end if i > 0 else 0) for i in range(len(notes))]
        instrument_indices = [instrument_index] * len(notes)
        return np.stack([pitches, durations, offsets, time_deltas, instrument_indices], axis=1)

    def compute_scaling_parameters(self, sequences):
        # Excluding instrument index from scaling
        self.means = np.nanmean(sequences[:, :, :-1], axis=(0, 1))
        self.stds = np.nanstd(sequences[:, :, :-1], axis=(0, 1))

    def scale_features(self, sequences):
        sequences_scaled = sequences.copy()
        sequences_scaled[:, :, :-1] = (sequences[:, :, :-1] - self.means) / self.stds
        sequences_scaled[np.isnan(sequences_scaled)] = -999  # Replace NaNs with a placeholder
        return sequences_scaled

    def scale_back_features(self, sequences_scaled):
        sequences = sequences_scaled.copy()
        sequences[:, :, :-1] = (sequences[:, :, :-1] * self.stds) + self.means
        sequences[sequences == -999] = np.nan  # Replace placeholders with NaNs
        return sequences

    def prepare_data(self, midi_files):
        sequences = self.midi_files_to_sequences(midi_files) 
        self.compute_scaling_parameters(sequences)
        sequences_scaled = self.scale_features(sequences)
        
        continuous_features = sequences_scaled[:, :, :-1]
        instrument_indices = sequences_scaled[:, :, -1]

        # One-hot encoding for instrument indices for both inputs and outputs
        instrument_indices_one_hot = tf.keras.utils.to_categorical(instrument_indices, num_classes=self.num_instruments)
        
        # Preparing one-hot encoded input data by concatenating continuous features with one-hot encoded instrument indices
        input_data_continuous = continuous_features[:, :self.sequence_length_i, :]
        input_data_categorical = instrument_indices_one_hot[:, :self.sequence_length_i, :]
        input_data = np.concatenate([input_data_continuous, input_data_categorical], axis=-1)
        
        # Preparing separate outputs for each continuous feature and one-hot encoded categorical output
        output_data_cont = continuous_features[:, self.sequence_length_i:, :]
        output_data_cat = instrument_indices_one_hot[:, self.sequence_length_i:, :]
        outputs = [output_data_cont[:, :, i] for i in range(output_data_cont.shape[2])]  # Continuous features outputs
        outputs.append(output_data_cat)  # Categorical output
        
        self.dataset = tf.data.Dataset.from_tensor_slices((input_data, tuple(outputs))).batch(self.batch_size)

    def _create_default_model(self):
        n_features_onehot = self.num_features - 1 + self.num_instruments  # Excluding instrument index from one-hot encoding
        input_shape = (self.sequence_length_i, n_features_onehot)  # Input features shape
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Masking(mask_value=-999., input_shape=input_shape)(inputs)
        if self.model_type == 'transformer':
            positional_encoding_layer = PositionalEncoding(self.sequence_length_i, n_features_onehot)
            x = positional_encoding_layer(x)
            for i in range(self.n_layers):
                attention_output = tf.keras.layers.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.n_units[i]
                )(x, x)
                x = tf.keras.layers.Dropout(self.dropout[i])(x)
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
                ff_output = tf.keras.Sequential([
                    tf.keras.layers.Dense(self.n_units[i], activation='relu'),
                    tf.keras.layers.Dense(x.shape[-1]),
                ])(x)
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff_output)
        else:
            for i in range(self.n_layers):
                if self.model_type == 'lstm':
                    x = tf.keras.layers.LSTM(self.n_units[i], return_sequences=True, dropout=self.dropout[i])(x)
                elif self.model_type == 'gru':
                    x = tf.keras.layers.GRU(self.n_units[i], return_sequences=True, dropout=self.dropout[i])(x)

        if self.num_instruments == 1:
            instruments_layer_activation = 'sigmoid'
        else:
            instruments_layer_activation = 'softmax'

        if self.num_instruments <= 2:
            instrument_loss_function = 'binary_crossentropy'
        else:
            instrument_loss_function = 'categorical_crossentropy'

        x = tf.keras.layers.Lambda(lambda x: x[:, -self.sequence_length_o:])(x)
        pitch_output = tf.keras.layers.Dense(1, name='pitch')(x)
        duration_output = tf.keras.layers.Dense(1, name='duration')(x)
        offset_output = tf.keras.layers.Dense(1, name='offset')(x)
        time_delta_output = tf.keras.layers.Dense(1, name='time_delta')(x)
        instrument_index_output = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.num_instruments, activation=instruments_layer_activation),
            name='instrument_index'
        )(x)

        model = tf.keras.Model(inputs=inputs, outputs=[pitch_output, duration_output, offset_output, time_delta_output, instrument_index_output])
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      loss={'pitch': 'mean_squared_error',
                            'duration': 'mean_squared_error',
                            'offset': 'mean_squared_error',
                            'time_delta': 'mean_squared_error',
                            'instrument_index': instrument_loss_function},
                      loss_weights=self.loss_weights,
                      metrics={'instrument_index': 'accuracy'})

        return model


    def fit(self, midi_files, epochs=10):
        """
        Train the model with the given data.

        Parameters:
        epochs (int): The number of epochs to train for.

        Returns:
        History: The history object generated by the training process. This contains information about the training process, such as the loss and accuracy at each epoch.
        """
        self.prepare_data(midi_files)
        history = self.model.fit(self.dataset, epochs=epochs)
        return history

    def predict(self, initial_midi, num_predictions=20):
        if self.model is None:
            raise ValueError("The model has not been trained yet. Please call `fit` before calling `predict`.")
        sequences = self.midi_files_to_sequences([initial_midi])
        sequences_scaled = self.scale_features(sequences)
        initial_data = np.concatenate([
            sequences_scaled[:, :, :-1],
            tf.keras.utils.to_categorical(sequences_scaled[:, :, -1], num_classes=self.num_instruments)
        ], axis=-1)
        initial_data = initial_data[np.newaxis, :self.sequence_length_i, :]  # Select only the needed initial sequence

        all_predicted_features = []

        while len(all_predicted_features) < num_predictions:
            prediction = self.model.predict(initial_data)
            for i in range(prediction[0].shape[1]):
                if len(all_predicted_features) < num_predictions:
                    combined_prediction = np.concatenate([
                        prediction[j][0, i, :] for j in range(len(prediction) - 1)
                    ] + [np.argmax(prediction[-1][0, i, :], axis=-1)])
                    all_predicted_features.append(combined_prediction)
                else:
                    break

            # Prepare the next input sequence using the latest predictions
            next_input_sequence = np.concatenate([
                initial_data[:, 1:, :4],  # Continuous features from the next step
                tf.keras.utils.to_categorical(initial_data[:, 1:, -1], num_classes=self.num_instruments)
            ], axis=-1)
            next_prediction_sequence = np.concatenate([
                prediction[j][:, i+1:i+2, :] for j in range(len(prediction) - 1)
            ] + [np.expand_dims(np.argmax(prediction[-1][:, i+1, :], axis=-1), axis=-1)], axis=-1)
            next_input_sequence = np.concatenate((next_input_sequence, next_prediction_sequence), axis=1)

            initial_data = np.concatenate((initial_data, next_input_sequence), axis=1)[:, -self.sequence_length_i:, :]

        predicted_scaled_features = np.array(all_predicted_features)
        predicted_continuous = self.scale_back_features(predicted_scaled_features[:, :4])
        predicted_instruments = np.argmax(predicted_scaled_features[:, 4:], axis=1)
        predicted_ds = np.concatenate([predicted_continuous, np.expand_dims(predicted_instruments, axis=1)], axis=-1)

        return predicted_ds
    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        model_saved_successfully = False
        
        # Attempt to save the full model first
        full_model_path = os.path.join(model_dir, 'full_model')
        try:
            self.model.save(full_model_path)
            print(f"Full model saved successfully at {full_model_path}.")
            model_saved_successfully = True
        except Exception as e:
            print(f"Unable to save the full model. Reason: {e}")
            print("Attempting to save model weights and architecture separately...")
        
        # Save model weights
        weights_path = os.path.join(model_dir, 'model_weights.h5')
        try:
            self.model.save_weights(weights_path)
            print(f"Model weights saved at {weights_path}.")
        except Exception as e:
            print(f"Failed to save model weights. Reason: {e}")
        
        # Save the model architecture as JSON
        model_json_path = os.path.join(model_dir, 'model_architecture.json')
        try:
            model_json = self.model.to_json()
            with open(model_json_path, 'w') as json_file:
                json_file.write(model_json)
            print(f"Model architecture saved at {model_json_path}.")
        except Exception as e:
            print(f"Failed to save model architecture. Reason: {e}")
        
        # Save other model parameters
        params_path = os.path.join(model_dir, 'model_params.json')
        model_params = {
            'sequence_length_i': self.sequence_length_i,
            'sequence_length_o': self.sequence_length_o,
            'num_instruments': self.num_instruments,
            'means': self.means.tolist(),
            'stds': self.stds.tolist(),
            'model_type': self.model_type,
            'n_layers': self.n_layers,
            'n_units': self.n_units,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'loss_weights': self.loss_weights,
        }

        with open(params_path, 'w') as json_file:
            json.dump(model_params, json_file)
        print(f"Model parameters saved to {params_path}.")

        # Loading instructions
        if model_saved_successfully:
            print("\nTo load this model in the future, use tf.keras.models.load_model('path/to/full_model').")
        else:
            print("\nTo load this model in the future:")
            print("1. Load the architecture with: with open('path/to/model_architecture.json', 'r') as json_file:")
            print("     json_config = json_file.read()")
            print("   model = tf.keras.models.model_from_json(json_config)")
            print("2. Load the weights with: model.load_weights('path/to/model_weights.h5').")
            print("Remember to compile the model after loading.")

    @classmethod
    def load(cls, model_dir):
        # Load model parameters
        params_path = os.path.join(model_dir, 'model_params.json')
        with open(params_path, 'r') as json_file:
            model_params = json.load(json_file)

        # Check if full model or weights only were saved
        if model_params.get('save_method') == 'full':
            # Load the full model
            full_model_path = os.path.join(model_dir, 'full_model')
            model = tf.keras.models.load_model(full_model_path)
            print(f"Full model loaded from {full_model_path}.")
        else:
            # Load model architecture and weights separately
            model_json_path = os.path.join(model_dir, 'model_architecture.json')
            with open(model_json_path, 'r') as json_file:
                model_json = json_file.read()
            model = tf.keras.models.model_from_json(model_json)

            weights_path = os.path.join(model_dir, 'model_weights.h5')
            model.load_weights(weights_path)
            print(f"Model architecture and weights loaded separately from {model_json_path} and {weights_path}, respectively.")

        # Re-instantiate the class with loaded parameters
        instance = cls(
            sequence_length_i=model_params['sequence_length_i'],
            sequence_length_o=model_params['sequence_length_o'],
            num_instruments=model_params['num_instruments'],
            # Add other parameters as necessary
        )
        instance.model = model
        instance.means = np.array(model_params['means'])
        instance.stds = np.array(model_params['stds'])
        return instance