import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from data_utils import load_data, build_vocab, encode_sequences, prepare_decoder_inputs
from generate_predictions import save_predictions
from config import TRAIN_PATH, VAL_PATH, TEST_PATH

def highlight(row):
    """Highlight correct rows in green, incorrect in red."""
    return ['background-color: lightgreen' if row.Correct else 'background-color: lightcoral'] * 4

def main():
    # 1) Load dataset and vocabularies
    train_df, val_df, test_df = load_data(TRAIN_PATH, VAL_PATH, TEST_PATH)
    src_idx, _     = build_vocab(train_df['roman'])
    tgt_idx, idx_tgt = build_vocab(train_df['dev'])

    # 2) Load best‐found vanilla model
    model = load_model('best_seq2seq_vanilla.h5')

    # 3) Encode test inputs and prepare decoder tokens
    X_test = encode_sequences(test_df['roman'], src_idx)
    y_test = encode_sequences(test_df['dev'],    tgt_idx)
    dec_in_test = prepare_decoder_inputs(y_test)

    # 4) Save all test‐set predictions to predictions_vanilla/
    save_predictions(model, X_test, y_test, idx_tgt, out_dir='predictions_vanilla')

    # 5) Sample 10 random examples and collect predictions
    np.random.seed(42)
    sample_ids = np.random.choice(len(test_df), size=10, replace=False)
    rows = []
    for i in sample_ids:
        inp = test_df['roman'].iat[i]
        ref = test_df['dev'].iat[i]

        Xi = X_test[i:i+1]
        di = dec_in_test[i:i+1]
        probs = model.predict([Xi, di], verbose=0)[0]
        pred_ids = np.argmax(probs, axis=-1)
        pred = ''.join(idx_tgt[t] for t in pred_ids if t != 0)

        rows.append({
            'Input':      inp,
            'Reference':  ref,
            'Prediction': pred,
            'Correct':    (ref == pred)
        })

    # 6) Build styled DataFrame and save to HTML
    result_df = pd.DataFrame(rows)
    styled = (
        result_df.style
                 .apply(highlight, axis=1)
                 .set_properties(**{'text-align':'center','font-size':'110%'})
                 .hide(axis='index')
    )
    os.makedirs('reports', exist_ok=True)
    styled.to_html('reports/sample_predictions_vanilla.html')

    print("● Saved full predictions to ./predictions_vanilla/")
    print("● Saved sample grid to ./reports/sample_predictions_vanilla.html")

if __name__ == '__main__':
    main()
