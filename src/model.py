import tensorflow as tf
from tensorflow.keras.layers import(
    Input,
    Embedding,
    LSTM, GRU, SimpleRNN,
    Dense
)
from tensorflow.keras.models import Model

_CELL_MAP = {
    'RNN': SimpleRNN,
    'LSTM': LSTM,
    'GRU':  GRU 
} # Dictionary to map cell types to their respective classes

def build_se2seq_model(
        imput_vocab_size: int,
        target_vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        cell_type: str = 'LSTM',
        num_layers: int = 1,
        share_embeddings: bool = False,
) -> Model:
    """
    Builds a sequence-to-sequence model with an encoder and decoder.
    """

    # --- Encoder --- 
    encoder_inputs = Input(shape=(None,), name='encoder_inputs') # Input layer for encoder
    encoder_embed = Embedding(
        input_dim=imput_vocab_size, 
        output_dim=embed_dim, 
        mask_zero=True,  # Ignore padding (0) tokens in input sequences during training
        name='encoder_embedding'
    )(encoder_inputs)

    encoder_outputs = encoder_embed # Initialize encoder outputs with the embedding layer output
    encoder_states = [] 
    cell = _CELL_MAP[cell_type] # Get the cell class based on the provided cell type

    for layer in range(num_layers):
        return_state = True # Return the final state of the RNN cell
        go_backwards = False # Process the input sequence in the forward direction
        enc_rnn = cell(
            units=hidden_dim,
            return_sequences=(layer < num_layers - 1), # Return sequences for all but the last layer
            return_state=return_state,
            name=f'encoder_{cell_type.lower()}_{layer+1}',
        )
        if layer == num_layers-1:
            # Last layer: Keep the full state
            outputs_and_states = enc_rnn(encoder_outputs) # Get the outputs and states
            encoder_outputs, *states = outputs_and_states
        else:
            # Intermediate: Only sequences
            encoder_outputs = enc_rnn(encoder_outputs) # Get the outputs
        encoder_states += states

    # --- Decoder ---
    decoder_inputs = Input(shape=(None,), name='decoder_inputs') # Input layer for decoder

    # Share embeddings if specified
    if share_embeddings:
        decoder_embed_layer = encoder_embed._keras_layer # Reuse the encoder embedding layer
    else:
        decoder_embed_layer = Embedding(
            input_dim=target_vocab_size, 
            output_dim=embed_dim, 
            mask_zero=True,  # Ignore padding (0) tokens in input sequences during training
            name='decoder_embedding'
        )
    decoder_embed = decoder_embed_layer(decoder_inputs) # Apply the embedding layer to decoder inputs

    decoder_outputs = decoder_embed # Initialize decoder outputs with the embedding layer output
    states = encoder_states # Use the encoder states as initial states for the decoder

    for layer in range(num_layers):
        dec_rnn = cell(
            units=hidden_dim,
            return_sequences=True,
            return_state=True, # Return the final state of the RNN cell
            name=f'decoder_{cell_type.lower()}_{layer+1}',
        )
        outputs_and_states = dec_rnn(
            decoder_outputs, 
            initial_state=states[layer*len(states)//num_layers:(layer+1)*len(states)//num_layers] # Use the corresponding encoder states as initial states for the decoder
        )
        decoder_outputs, *dec_states = outputs_and_states
        states[layer*len(states)//num_layers:(layer+1)*len(states)//num_layers] = dec_states # Update the states with the decoder states

    # Final projection
    decoder_dense = Dense(
        target_vocab_size, 
        activation='softmax', 
        name='output_projection'
    )
    decoder_outputs = decoder_dense(decoder_outputs) # Apply the dense layer to decoder outputs

    # Build the model
    model = Model(
        [encoder_inputs, decoder_inputs], 
        decoder_outputs, 
        name='seq2seq_model'
    )
    return model
   
            
