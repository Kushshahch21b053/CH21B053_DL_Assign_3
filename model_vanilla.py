import tensorflow as tf
from tensorflow.keras import layers, Model, Input

def build_seq2seq(src_vocab, tgt_vocab, embed_dim, hidden_dim,
                  enc_layers, dec_layers, cell_type, dropout):
    """
    Build a configurable vanilla seq2seq model.
    """
    # Select the RNN cell type based on the provided string.
    RNN = {'SimpleRNN': layers.SimpleRNN,
           'GRU': layers.GRU,
           'LSTM': layers.LSTM}[cell_type]

    # Encoder: Process the source sequence.
    enc_inputs = Input(shape=(None,), name='encoder_inputs')
    x = layers.Embedding(src_vocab, embed_dim, mask_zero=True)(enc_inputs)
    for i in range(enc_layers):
        # Only the last layer does not return sequences.
        return_seq = (i < enc_layers - 1)
        x = RNN(hidden_dim, return_sequences=return_seq, dropout=dropout)(x)
    # Use a single state for SimpleRNN, and a tuple for LSTM/GRU.
    enc_states = [x] if cell_type == 'SimpleRNN' else [x, x]

    # Decoder: Generate the target sequence.
    dec_inputs = Input(shape=(None,), name='decoder_inputs')
    y = layers.Embedding(tgt_vocab, embed_dim, mask_zero=True)(dec_inputs)
    for _ in range(dec_layers):
        y = RNN(hidden_dim, return_sequences=True, dropout=dropout)(y, initial_state=enc_states)
    outputs = layers.TimeDistributed(layers.Dense(tgt_vocab, activation='softmax'))(y)

    # Build and compile the model.
    model = Model([enc_inputs, dec_inputs], outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
