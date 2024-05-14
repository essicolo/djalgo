from importlib.util import find_spec
import warnings
djai_spec = find_spec(['torch', 'miditok'])
if djai_spec is None:
    warnings.warn("Install both PyTorch and MidiTok to use Djai.")

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator

class ModelManager:
    def __init__(self, sequence_length_input, sequence_length_output, model_type, nn_units, dropout, batch_size, learning_rate, n_heads=None):
        self.seq_len_input = sequence_length_input
        self.seq_len_output = sequence_length_output
        self.model_type = model_type
        self.nn_units = nn_units
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_heads = n_heads

        # Initialize the MidiTok tokenizer with custom configuration
        TOKENIZER_PARAMS = {
            "pitch_range": (21, 109),
            "beat_res": {(0, 4): 8, (4, 12): 4},
            "num_velocities": 32,
            "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
            "use_chords": True,
            "use_rests": True,
            "use_tempos": True,
            "use_time_signatures": False,
            "use_programs": False,
            "num_tempos": 32,
            "tempo_range": (40, 250),
        }
        config = TokenizerConfig(**TOKENIZER_PARAMS)
        self.tokenizer = REMI(config)
        self.input_size = len(self.tokenizer.vocab)

        #print(f"Tokenizer vocabulary size (input_size): {self.input_size}")

        # Create the model
        if model_type == 'transformer':
            self.model = TransformerModel(nn_units, dropout, n_heads, self.input_size)
        elif model_type == 'lstm':
            self.model = LSTMModel(nn_units, dropout, self.input_size)
        elif model_type == 'gru':
            self.model = GRUModel(nn_units, dropout, self.input_size)
        else:
            raise ValueError("Unsupported model type. Choose 'lstm', 'gru', or 'transformer'.")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def fit(self, midi_files_path, epochs):
        dataset_chunks_dir = Path(midi_files_path)
        dataset = DatasetMIDI(
            files_paths=list(dataset_chunks_dir.glob("**/*.mid")),
            tokenizer=self.tokenizer,
            max_seq_len=self.seq_len_input
        )
        collator = DataCollator(self.tokenizer.pad_token_id, copy_inputs_as_labels=True)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collator)

        self.model.train()
        for epoch in range(epochs):
            for batch in data_loader:
                inputs, targets = batch['input_ids'], batch['labels']

                # One-hot encode inputs
                inputs_one_hot = torch.nn.functional.one_hot(inputs, num_classes=self.input_size).float()
                if torch.isnan(inputs_one_hot).any() or torch.isinf(inputs_one_hot).any():
                    print("Input contains NaN or Inf values.")

                if self.model_type == 'transformer':
                    attention_mask = batch['attention_mask']
                    attention_mask_bool = attention_mask.bool()
                    outputs = self.model(inputs_one_hot, attention_mask=attention_mask_bool)
                else:
                    outputs = self.model(inputs_one_hot)  # No attention mask for other types

                loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print("Loss contains NaN or Inf values.")
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        return self.model

    def generate(self, primer_file, length):
        # Tokenize the primer file using the correct method
        midi = self.tokenizer.encode(Path(primer_file))
        primer_tokens = midi[0].ids  # Assuming single-track MIDI
        primer_one_hot = torch.nn.functional.one_hot(torch.tensor(primer_tokens), num_classes=self.input_size).float().unsqueeze(0)

        # Simplified prediction method
        self.model.eval()
        generated = primer_one_hot
        for _ in range(length):
            output = self.model(generated)
            next_token = torch.argmax(output[:, -1, :], dim=-1, keepdim=True)
            next_one_hot = torch.nn.functional.one_hot(next_token, num_classes=self.input_size).float()
            generated = torch.cat((generated, next_one_hot), dim=1)

        # Convert generated tokens back to MIDI
        generated_tokens = torch.argmax(generated, dim=-1).squeeze().tolist()
        generated_midi = self.tokenizer.decode([generated_tokens])
        generated_midi.dump_midi(Path("output.mid"))

    def save(self, file_path):
        torch.save(self.model.state_dict(), file_path)


class LSTMModel(nn.Module):
    def __init__(self, nn_units, dropout, input_size):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.LSTM(input_size=input_size if i == 0 else nn_units[i-1],
                    hidden_size=nn_units[i],
                    batch_first=True,
                    dropout=dropout if i < len(nn_units) - 1 else 0)  # dropout for all but the last layer
            for i in range(len(nn_units))
        ])
        self.fc = nn.Linear(nn_units[-1], input_size)

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return self.fc(x)


class GRUModel(nn.Module):
    def __init__(self, nn_units, dropout, input_size):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.GRU(input_size=input_size if i == 0 else nn_units[i-1],
                   hidden_size=nn_units[i],
                   batch_first=True,
                   dropout=dropout if i < len(nn_units) - 1 else 0)  # dropout for all but the last layer
            for i in range(len(nn_units))
        ])
        self.fc = nn.Linear(nn_units[-1], input_size)

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return self.fc(x)


class TransformerModel(nn.Module):
    def __init__(self, nn_units, dropout, n_heads, input_size):
        super().__init__()
        self.embedding = nn.Linear(input_size, nn_units[0])  # Project input_size to the embedding dimension
        nn.init.xavier_uniform_(self.embedding.weight)  # Xavier initialization
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=nn_units[0], nhead=n_heads, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=len(nn_units))
        self.fc = nn.Linear(nn_units[-1], input_size)
        nn.init.xavier_uniform_(self.fc.weight)  # Xavier initialization

    def forward(self, x, attention_mask=None):
        x = self.embedding(x)  # Project input to the embedding dimension
        if attention_mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        else:
            x = self.transformer_encoder(x)
        return self.fc(x)
