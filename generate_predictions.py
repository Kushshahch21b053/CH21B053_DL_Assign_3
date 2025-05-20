import os
import numpy as np

def save_predictions(model, X, y, idx_tgt, out_dir, batch_size=32):
    """
    Generate and save one prediction file per example.

    Parameters:
    - model: The trained model used for making predictions.
    - X: Input data for the model.
    - y: Ground truth output sequences.
    - idx_tgt: A mapping (e.g., list or dict) from index to target character.
    - out_dir: Directory to save predicted text files.
    - batch_size: Number of samples per prediction batch.
    """
    # Ensure that the output directory exists or create it.
    os.makedirs(out_dir, exist_ok=True)
    
    # Prepare decoder input by concatenating a column of zeros (as the start token)
    # with the ground truth input sequence shifted to the right (excluding last token).
    dec_input = np.concatenate(
        [np.zeros((X.shape[0], 1), dtype=int), y[:, :-1]],
        axis=1
    )
    
    # Run model prediction using encoder input X and constructed decoder input.
    # The output `probs` are the probabilities for each token in the output sequence.
    probs = model.predict([X, dec_input], batch_size=batch_size, verbose=0)
    
    # Convert probabilities to predicted token indices using argmax.
    pred_ids = np.argmax(probs, axis=-1)
    
    for i in range(len(X)):
        # Convert token indices to characters. Skip index 0 which is assumed to be a padding token.
        pred_str = ''.join(idx_tgt[t] for t in pred_ids[i] if t != 0)
        # Write the prediction string to a file named '<index>.txt' in out_dir.
        with open(os.path.join(out_dir, f'{i}.txt'), 'w', encoding='utf-8') as f:
            f.write(pred_str)
