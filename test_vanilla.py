import os
import numpy as np
import tensorflow as tf
from data_utils import load_data, build_vocab, encode_sequences, prepare_decoder_inputs
from model_vanilla import build_seq2seq
from evaluate import compute_exact_match
from config import TRAIN_PATH, VAL_PATH, TEST_PATH

# Best hyperparameters found via sweep
best_cfg = {
    'embed_dim': 256,
    'hidden_dim': 64,
    'enc_layers': 1,
    'dec_layers': 3,
    'cell_type': 'GRU',
    'dropout': 0.2,
    'batch_size': 32
}
epochs = 10

def main():
    # Load splits
    train, val, test = load_data(TRAIN_PATH, VAL_PATH, TEST_PATH)
    
    # Build vocabularies
    # Note: For both source ('roman') and target ('dev') sequences in training split.
    src_idx, _ = build_vocab(train['roman'])
    tgt_idx, _ = build_vocab(train['dev'])

    # Encode sequences
    # Convert the texts into numerical representations using the constructed vocabulary
    X_train = encode_sequences(train['roman'], src_idx)
    y_train = encode_sequences(train['dev'],    tgt_idx)
    X_test  = encode_sequences(test['roman'],  src_idx)
    y_test  = encode_sequences(test['dev'],    tgt_idx)

    # Prepare decoder inputs for teacher forcing
    # It shifts the target sequences appropriately to use as inputs for the decoder during training
    dec_in_train = prepare_decoder_inputs(y_train)
    dec_in_test  = prepare_decoder_inputs(y_test)

    # Train on GPU if available
    with tf.device('/GPU:0'):
        # Build the sequence-to-sequence model with the specified hyperparameters
        model = build_seq2seq(
            src_vocab=len(src_idx),  # Size of source vocabulary
            tgt_vocab=len(tgt_idx),  # Size of target vocabulary
            **best_cfg              # Unpacks hyperparameters
        )
        
        # Fit the model
        # x: list containing encoder and decoder inputs; y: target outputs (expanded dims for compatibility)
        model.fit(
            [X_train, dec_in_train],
            y_train[..., None],
            epochs=epochs,
            batch_size=best_cfg['batch_size']
        )

        # Save the fully trained model to H5 file
        os.makedirs('models', exist_ok=True)
        model.save('models/best_seq2seq_vanilla.h5')  # Saves architecture+weights+optimizer state[3]

        # Predict on test set
        # Generate probability distributions over the vocabulary for each timestep in the output sequence
        probs = model.predict(
            [X_test, dec_in_test],
            batch_size=best_cfg['batch_size']
        )

    # Compute exact-match accuracy
    # Convert probability distributions to predicted token indices using argmax and compare with ground truth
    pred_ids = np.argmax(probs, axis=-1)
    acc = compute_exact_match(pred_ids, y_test)
    print(f'Test exact-match seq accuracy: {acc:.4f}')

if __name__ == '__main__':
    main()
