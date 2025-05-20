import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
from data_utils import load_data, build_vocab, encode_sequences, prepare_decoder_inputs
from config import TRAIN_PATH, VAL_PATH, TEST_PATH

# Paths to your retrained models
VANILLA_MODEL_PATH = 'models/best_seq2seq_vanilla.h5'
ATTN_MODEL_PATH    = 'models/best_seq2seq_attention.h5'

# Hidden size used in your attention model (must match retrain_attention.py)
ATTN_HIDDEN_DIM = 64

def highlight_correct(row):
    return ['background-color: lightgreen' if row.Correct 
            else 'background-color: lightcoral'] * len(row)

def main():
    # 1) Load data + build vocabs
    train_df, val_df, test_df = load_data(TRAIN_PATH, VAL_PATH, TEST_PATH)
    src_idx, idx_src = build_vocab(train_df['roman'])
    tgt_idx, idx_tgt = build_vocab(train_df['dev'])

    # 2) Encode test set + prepare decoder inputs
    X_test = encode_sequences(test_df['roman'], src_idx)
    y_test = encode_sequences(test_df['dev'],    tgt_idx)
    dec_in_test = prepare_decoder_inputs(y_test)

    # 3) Load models
    vanilla = load_model(VANILLA_MODEL_PATH)
    attn    = load_model(ATTN_MODEL_PATH)

    # 4) Get vanilla predictions & exact‐match flags
    van_probs      = vanilla.predict([X_test, dec_in_test], batch_size=32, verbose=0)
    pred_van_ids   = np.argmax(van_probs, axis=-1)
    matches_van    = [(pred_van_ids[i] == y_test[i]).all() for i in range(len(y_test))]

    # 5) Get attention predictions & exact‐match flags
    attn_probs      = attn.predict([X_test, dec_in_test], batch_size=32, verbose=0)
    pred_attn_ids   = np.argmax(attn_probs, axis=-1)
    matches_attn    = [(pred_attn_ids[i] == y_test[i]).all() for i in range(len(y_test))]

    os.makedirs('reports', exist_ok=True)

    # --- A) 10‐row sample grid for attention predictions ---
    np.random.seed(42)
    sample_idx = np.random.choice(len(test_df), size=10, replace=False)
    rows = []
    for i in sample_idx:
        inp       = test_df['roman'].iat[i]
        ref       = test_df['dev'].iat[i]
        pred_ids  = pred_attn_ids[i]
        pred      = ''.join(idx_tgt[t] for t in pred_ids if t != 0)
        correct   = (pred == ref)
        rows.append({
            'Input': inp, 'Reference': ref,
            'Prediction': pred, 'Correct': correct
        })
    df_sample = pd.DataFrame(rows)
    styled_sample = (
        df_sample.style
        .apply(highlight_correct, axis=1)
        .set_properties(**{'text-align':'center','font-size':'110%'})
        .hide(axis='index')
    )
    styled_sample.to_html('reports/sample_predictions_attention.html')

    # --- B) Examples fixed by attention but missed by vanilla ---
    corrected = [i for i in range(len(y_test))
                 if (not matches_van[i]) and matches_attn[i]]
    rows2 = []
    for i in corrected[:10]:
        inp = ''.join(idx_src[t] for t in X_test[i] if t != 0)
        ref = ''.join(idx_tgt[t] for t in y_test[i] if t != 0)
        van = ''.join(idx_tgt[t] for t in pred_van_ids[i] if t != 0)
        att = ''.join(idx_tgt[t] for t in pred_attn_ids[i] if t != 0)
        rows2.append({
            'Input': inp, 'Reference': ref,
            'Vanilla': van, 'Attention': att
        })
    df_corr = pd.DataFrame(rows2)
    styled_corr = (
        df_corr.style
        .set_properties(**{'text-align':'center','font-size':'110%'})
        .hide(axis='index')
    )
    styled_corr.to_html('reports/corrections_attention.html')

    # --- C) 3×3 attention heatmaps ---
    # 1) Identify sub‐layers
    emb_layers  = [l for l in attn.layers if isinstance(l, layers.Embedding)]
    enc_emb, dec_emb = emb_layers
    lstm_layers = [l for l in attn.layers if isinstance(l, layers.LSTM)]
    enc_lstm, dec_lstm = lstm_layers
    attn_layer   = next(l for l in attn.layers if isinstance(l, layers.AdditiveAttention))
    concat_layer = next(l for l in attn.layers if isinstance(l, layers.Concatenate))
    td_layer     = next(l for l in attn.layers 
                       if isinstance(l, keras.layers.TimeDistributed))
    # 2) Build encoder‐inference model
    enc_input   = attn.input[0]
    enc_emb_out = enc_emb(enc_input)
    enc_out, h0, c0 = enc_lstm(enc_emb_out)
    encoder_inf = keras.Model(enc_input, [enc_out, h0, c0])
    # 3) Build decoder‐inference model
    dec_token  = keras.Input(shape=(1,), name='dec_token')
    hs_in      = keras.Input(shape=(ATTN_HIDDEN_DIM,), name='hs')
    cs_in      = keras.Input(shape=(ATTN_HIDDEN_DIM,), name='cs')
    enc_out_in = keras.Input(
        shape=(None, ATTN_HIDDEN_DIM), name='enc_out')
    dec_emb2   = dec_emb(dec_token)
    dec_seq2, h2, c2 = dec_lstm(
        dec_emb2, initial_state=[hs_in, cs_in])
    ctx, scores = attn_layer(
        [dec_seq2, enc_out_in], return_attention_scores=True)
    merged2   = concat_layer([ctx, dec_seq2])
    dec_out2  = td_layer(merged2)
    decoder_inf = keras.Model(
        [dec_token, hs_in, cs_in, enc_out_in],
        [dec_out2, h2, c2, scores]
    )
    # 4) Select 9 examples with length >1
    lengths = np.argmax((y_test == 0).astype(int), axis=1)
    lengths = np.where(lengths == 0, y_test.shape[1], lengths)
    candidates = [i for i in range(len(y_test)) if lengths[i] > 1]
    selected = candidates[:9]
    # 5) Plot
    fig, axes = plt.subplots(3, 3, figsize=(10,10))
    for ax, i in zip(axes.flatten(), selected):
        enc_o, h, c = encoder_inf.predict(
            X_test[i:i+1], verbose=0)
        scores_seq = []
        token = np.array([[0]])
        for _ in range(lengths[i]):
            out, h, c, sc = decoder_inf.predict(
                [token, h, c, enc_o], verbose=0)
            scores_seq.append(sc[0,0,:])
            token = np.array([[np.argmax(out[0,0,:])]])
        A = np.stack(scores_seq)
        im = ax.imshow(A, aspect='auto', cmap='viridis')
        ax.set_title(''.join(idx_src[t] for t in X_test[i] if t!=0))
        ax.set_xlabel('Input pos')
        ax.set_ylabel('Output step')
    fig.colorbar(im, ax=axes.ravel().tolist(),
                 fraction=0.02, pad=0.04)
    plt.tight_layout(rect=[0,0,0.92,1.0])
    plt.savefig('reports/attention_heatmaps.png')
    plt.close(fig)

    # Summary
    print("reports/sample_predictions_attention.html")
    print("reports/corrections_attention.html")
    print("reports/attention_heatmaps.png")

if __name__ == '__main__':
    main()
