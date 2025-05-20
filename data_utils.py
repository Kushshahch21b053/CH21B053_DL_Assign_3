import numpy as np
import pandas as pd

# Load train, validation, and test data from TSV files into pandas DataFrames
def load_data(train_path, val_path, test_path):
    """
    Read TSV files and assign column names: 'roman', 'dev', 'count'.
    """
    train = pd.read_csv(train_path, sep='\t', names=['roman', 'dev', 'count'])
    val   = pd.read_csv(val_path,   sep='\t', names=['roman', 'dev', 'count'])
    test  = pd.read_csv(test_path,  sep='\t', names=['roman', 'dev', 'count'])
    return train, val, test

# Build mappings from characters to indices and vice-versa
def build_vocab(sequences):
    """
    Create a vocabulary mapping from characters and add a padding token.
    """
    vocab = set(char for seq in sequences for char in seq)
    idx = {c: i+1 for i, c in enumerate(sorted(vocab))}
    idx['<pad>'] = 0
    rev_idx = {i: c for c, i in idx.items()}
    return idx, rev_idx

# Encode list of strings into numpy arrays with padding
def encode_sequences(sequences, idx, maxlen=None):
    """
    Convert strings to indices with zero-padding.
    """
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    arr = np.zeros((len(sequences), maxlen), dtype=int)
    for i, seq in enumerate(sequences):
        for j, c in enumerate(seq):
            arr[i, j] = idx.get(c, 0)
    return arr

# Prepare decoder inputs by shifting target sequences right
def prepare_decoder_inputs(y, pad_token=0):
    """
    Prepend pad_token to sequences for teacher forcing.
    """
    decoder_input = np.concatenate(
        [np.full((y.shape[0], 1), pad_token, dtype=int), y[:, :-1]],
        axis=1
    )
    return decoder_input
