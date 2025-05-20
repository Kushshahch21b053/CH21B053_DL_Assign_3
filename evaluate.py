import numpy as np

def compute_exact_match(pred_ids, y_true):
    """Compute sequence-level exact match accuracy under teacher forcing."""
    # Identify the first position of padding (assumed to be 0) in each sequence.
    lengths = np.argmax((y_true == 0).astype(int), axis=1)
    # If no padding is found, use the full sequence length.
    lengths = np.where(lengths == 0, y_true.shape[1], lengths)
    
    # Check if the predictions match the ground truth up to the identified sequence lengths.
    matches = [
        np.all(pred_ids[i, :lengths[i]] == y_true[i, :lengths[i]])
        for i in range(len(y_true))
    ]
    
    # Return the proportion of exact matches.
    return np.mean(matches)
