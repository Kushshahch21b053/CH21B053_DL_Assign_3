import numpy as np
import tensorflow as tf
from data_utils import load_data, build_vocab, encode_sequences, prepare_decoder_inputs
from model_vanilla import build_seq2seq
from evaluate import compute_exact_match
from config import TRAIN_PATH, VAL_PATH, TEST_PATH

# Define best hyperparameters for the vanilla model (from previous sweep)
best_cfg = {
    'embed_dim': 256,
    'hidden_dim': 64,
    'enc_layers': 1,
    'dec_layers': 3,
    'cell_type': 'GRU',
    'dropout': 0.2,
    'batch_size': 32
}

# Number of training epochs
epochs = 10

def main():
    """Retrain the vanilla sequence-to-sequence model on training set and evaluate test accuracy."""
    # Load train, validation, and test datasets
    train, val, test = load_data(TRAIN_PATH, VAL_PATH, TEST_PATH)
    
    # Build vocabulary for encoder (source) and decoder (target) using training data
    src_idx, _ = build_vocab(train['roman'])
    tgt_idx, _ = build_vocab(train['dev'])

    # Encode the sequences based on the built vocabularies
    X_train = encode_sequences(train['roman'], src_idx)
    y_train = encode_sequences(train['dev'], tgt_idx)
    X_test = encode_sequences(test['roman'], src_idx)
    y_test = encode_sequences(test['dev'], tgt_idx)

    # Prepare decoder inputs by shifting target sequences
    dec_in_train = prepare_decoder_inputs(y_train)
    dec_in_test = prepare_decoder_inputs(y_test)

    # Use GPU for training if available
    with tf.device('/GPU:0'):
        # Build the sequence-to-sequence model with the given configuration
        model = build_seq2seq(
            src_vocab=len(src_idx),
            tgt_vocab=len(tgt_idx),
            **best_cfg
        )
        # Train the model with the training data
        model.fit(
            [X_train, dec_in_train],
            y_train[..., None],
            epochs=epochs,
            batch_size=best_cfg['batch_size']
        )
        # Predict probabilities for the test data
        probs = model.predict([X_test, dec_in_test],
                              batch_size=best_cfg['batch_size'])
    
    # Convert probabilities to predicted token indices by taking argmax over the vocabulary axis
    pred_ids = np.argmax(probs, axis=-1)
    # Compute the exact-match sequence accuracy
    acc = compute_exact_match(pred_ids, y_test)
    print(f'Test exact-match seq accuracy: {acc:.4f}')

if __name__ == '__main__':
    main()
