import os
import numpy as np
import tensorflow as tf
from data_utils import load_data, build_vocab, encode_sequences, prepare_decoder_inputs
from model_attention import build_attention_model
from evaluate import compute_exact_match
from generate_predictions import save_predictions
from config import TRAIN_PATH, VAL_PATH, TEST_PATH

# Best attention hyperparameters from your sweep
best_attn_cfg = {
    'embed_dim': 256,
    'hidden_dim': 64,
    'dropout': 0.0,
    'batch_size': 32
}
epochs = 10  # Number of training epochs

def main():
    # 1) Load the train, validation, and test datasets from file paths.
    train_df, val_df, test_df = load_data(TRAIN_PATH, VAL_PATH, TEST_PATH)

    # 2) Build vocabularies for the source (roman) and target (dev) languages.
    #    idx_tgt will be used to decode predictions later.
    src_idx, _ = build_vocab(train_df['roman'])
    tgt_idx, idx_tgt = build_vocab(train_df['dev'])

    # 3) Encode the sequences from texts to numerical representations.
    X_train = encode_sequences(train_df['roman'], src_idx)
    y_train = encode_sequences(train_df['dev'], tgt_idx)
    X_test  = encode_sequences(test_df['roman'], src_idx)
    y_test  = encode_sequences(test_df['dev'], tgt_idx)

    # 4) Prepare decoder-input sequences for teacher forcing,
    #    by shifting the target sequences appropriately.
    dec_in_train = prepare_decoder_inputs(y_train)
    dec_in_test  = prepare_decoder_inputs(y_test)

    # 5) Train the attention model on GPU if available.
    with tf.device('/GPU:0'):
        # Build the attention-based sequence-to-sequence model with given hyperparameters.
        model = build_attention_model(
            src_vocab=len(src_idx),
            tgt_vocab=len(tgt_idx),
            **best_attn_cfg
        )

        # Fit (train) the model on the training data.
        model.fit(
            [X_train, dec_in_train],
            y_train[..., None],  # Add an extra dimension for the output
            epochs=epochs,
            batch_size=best_attn_cfg['batch_size']
        )

        # 6) Save the trained attention model for later use.
        os.makedirs('models', exist_ok=True)
        model.save('models/best_seq2seq_attention.h5')

        # 7) Generate and save predictions for the test set.
        #    Predictions are saved to the 'predictions_attention' directory.
        save_predictions(
            model,
            X_test,
            y_test,
            idx_tgt,
            out_dir='predictions_attention',
            batch_size=best_attn_cfg['batch_size']
        )

        # 8) Compute softmax probabilities from the model for final accuracy measure.
        probs = model.predict(
            [X_test, dec_in_test],
            batch_size=best_attn_cfg['batch_size']
        )

    # 9) Determine the predicted token indices from the probabilities and 
    #    compute the exact-match sequence accuracy.
    pred_ids = np.argmax(probs, axis=-1)
    acc = compute_exact_match(pred_ids, y_test)
    print(f'Attention test exact-match seq accuracy: {acc:.4f}')

if __name__ == '__main__':
    main()
