import tensorflow as tf
from tensorflow.keras import layers, Model, Input

def build_attention_model(src_vocab, tgt_vocab, embed_dim,
                          hidden_dim, dropout):
    """
    Build a single‐layer encoder–decoder model with additive attention.
    """
    
    # -------------------------
    # Encoder
    # -------------------------
    # Define encoder input layer, where the shape is [batch_size, sequence_length]
    enc_in = Input(shape=(None,), name='enc_in')
    
    # Embedding layer to convert integer tokens to dense vectors
    enc_emb = layers.Embedding(src_vocab, embed_dim, mask_zero=False)(enc_in)
    
    # LSTM layer processing the embedded encoder input.
    # return_sequences=True to output the full sequence for attention mechanism.
    # return_state=True to get the last hidden and cell states.
    enc_out, state_h, state_c = layers.LSTM(
        hidden_dim, return_sequences=True,
        return_state=True, dropout=dropout
    )(enc_emb)

    # -------------------------
    # Decoder
    # -------------------------
    # Define decoder input layer.
    dec_in = Input(shape=(None,), name='dec_in')
    
    # Embedding layer for the decoder inputs.
    dec_emb = layers.Embedding(tgt_vocab, embed_dim, mask_zero=False)(dec_in)
    
    # LSTM layer for decoder that produces output sequences.
    # Initializes state with encoder's final hidden and cell states.
    dec_seq, _, _ = layers.LSTM(
        hidden_dim, return_sequences=True,
        return_state=True, dropout=dropout
    )(dec_emb, initial_state=[state_h, state_c])

    # -------------------------
    # Attention
    # -------------------------
    # Calculate the context vectors using additive attention.
    # The attention layer takes the decoder sequence as query and encoder outputs as value.
    context = layers.AdditiveAttention()([dec_seq, enc_out])
    
    # Combine the attention context vectors with the decoder outputs.
    merged = layers.Concatenate()([context, dec_seq])
    
    # -------------------------
    # Output Layer
    # -------------------------
    # A TimeDistributed Dense layer with softmax activation over the target vocabulary.
    # It produces a probability distribution over the vocabulary for each time-step.
    outputs = layers.TimeDistributed(
        layers.Dense(tgt_vocab, activation='softmax')
    )(merged)

    # Define the model with encoder and decoder inputs mapping to outputs.
    model = Model([enc_in, dec_in], outputs)
    
    # Compile the model with Adam optimizer and sparse categorical crossentropy loss.
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
