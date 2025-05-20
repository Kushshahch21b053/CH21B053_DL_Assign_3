import os
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
from data_utils import load_data, build_vocab, encode_sequences, prepare_decoder_inputs
from config import TRAIN_PATH, VAL_PATH, TEST_PATH

def cstr(ch, color):
    """Wrap a character `ch` in a <span> with background `color`."""
    return f"<span style='padding:2px 4px; background-color:{color};'>{ch}</span>"

def get_clr(weight):
    """
    Map a normalized attention weight in [0,1] to a GitHub-like green palette.
    """
    palette = ['#ebedf0','#c6e48b','#7bc96f','#239a3b','#196127']
    idx = int(np.clip(weight, 0, 1) * (len(palette)-1))
    return palette[idx]

def build_inference_models(attn_model, hidden_dim):
    """
    Given a trained attention seq2seq `attn_model`, extract its sub-layers
    and build:
     - encoder_inf: maps encoder_input -> (enc_out, h, c)
     - decoder_inf: maps (token, h, c, enc_out) -> (dec_out, h2, c2, scores)
    """
    # 1) extract embedding & LSTM layers
    emb_layers  = [l for l in attn_model.layers if isinstance(l, layers.Embedding)]
    enc_emb, dec_emb = emb_layers

    lstm_layers = [l for l in attn_model.layers if isinstance(l, layers.LSTM)]
    enc_lstm, dec_lstm = lstm_layers

    # 2) attention, concat & time‐distributed layers
    attn_layer   = next(l for l in attn_model.layers if isinstance(l, layers.AdditiveAttention))
    concat_layer = next(l for l in attn_model.layers if isinstance(l, layers.Concatenate))
    td_layer     = next(l for l in attn_model.layers if isinstance(l, keras.layers.TimeDistributed))

    # Build encoder inference
    enc_input   = attn_model.input[0]                             # encoder Input
    enc_emb_out = enc_emb(enc_input)                              # embed
    enc_out, h0, c0 = enc_lstm(enc_emb_out)                       # unpack
    encoder_inf = keras.Model(enc_input, [enc_out, h0, c0])

    # Build decoder inference
    dec_token   = keras.Input(shape=(1,), name='dec_tok')         # one step token
    hs_in       = keras.Input(shape=(hidden_dim,), name='hs')
    cs_in       = keras.Input(shape=(hidden_dim,), name='cs')
    enc_out_in  = keras.Input(shape=(None, hidden_dim), name='enc_out')

    dec_emb2    = dec_emb(dec_token)
    dec_seq2, h2, c2 = dec_lstm(dec_emb2, initial_state=[hs_in, cs_in])
    ctx, scores      = attn_layer([dec_seq2, enc_out_in], return_attention_scores=True)
    merged2          = concat_layer([ctx, dec_seq2])
    dec_out2         = td_layer(merged2)

    decoder_inf = keras.Model(
        [dec_token, hs_in, cs_in, enc_out_in],
        [dec_out2, h2, c2, scores]
    )

    return encoder_inf, decoder_inf

def visualize_connectivity(idx, 
                           test_df,
                           X_test, y_test, 
                           encoder_inf, decoder_inf, 
                           idx_src, idx_tgt):
    """
    Generate an HTML fragment visualizing, for example `idx`,
    which input character the model attends to when predicting
    each output character.
    """
    inp  = test_df['roman'].iat[idx]
    out  = test_df['dev'].iat[idx]

    # run encoder‐inf
    enc_o, h, c = encoder_inf.predict(X_test[idx:idx+1], verbose=0)

    # step through decoder
    token = np.array([[0]])   # start‐token
    rows = []
    for t, true_ch in enumerate(out):
        dec_out, h, c, sc = decoder_inf.predict([token, h, c, enc_o], verbose=0)
        w = sc[0,0]                # shape (T_in,)
        norm = w / (w.sum() + 1e-8)
        # color‐span each input character
        spans = [cstr(ch, get_clr(p)) for ch,p in zip(inp, norm)]
        pred_id = np.argmax(dec_out[0,0,:])
        token    = np.array([[pred_id]])
        # build row
        html_row = "".join(spans) + f" &rarr; <b>{out[t]}</b>"
        rows.append(html_row)
    # wrap in a <div>
    block = "<br/>\n".join(rows)
    return f"<div style='margin-bottom:1em'>{block}</div>"

def main():
    parser = argparse.ArgumentParser(
        description="Visualize attention connectivity for test examples"
    )
    parser.add_argument(
        "--num_examples", type=int, default=5,
        help="How many test‐examples to visualize"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64,
        help="Hidden dimension of your attention LSTM (must match retrain)"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="models/best_seq2seq_attention.h5",
        help="Path to your saved attention model"
    )
    args = parser.parse_args()

    # load data & vocab
    _, _, test_df = load_data(TRAIN_PATH, VAL_PATH, TEST_PATH)
    # need train vocab for idx_src/idx_tgt
    train_df, _, _ = load_data(TRAIN_PATH, VAL_PATH, TEST_PATH)
    src_idx, idx_src = build_vocab(train_df['roman'])
    tgt_idx, idx_tgt = build_vocab(train_df['dev'])

    # encode test set
    X_test = encode_sequences(test_df['roman'], src_idx)
    y_test = encode_sequences(test_df['dev'],    tgt_idx)

    # load model & build inf models
    attn_model   = load_model(args.model_path)
    encoder_inf, decoder_inf = build_inference_models(attn_model, args.hidden_dim)

    # generate HTML
    os.makedirs("reports", exist_ok=True)
    html_fragments = []
    for i in range(min(args.num_examples, len(test_df))):
        frag = visualize_connectivity(
            idx=i,
            test_df=test_df,
            X_test=X_test,
            y_test=y_test,
            encoder_inf=encoder_inf,
            decoder_inf=decoder_inf,
            idx_src=idx_src,
            idx_tgt=idx_tgt
        )
        html_fragments.append(f"<h4>Example {i}</h4>\n{frag}")

    html_body = "\n<hr/>\n".join(html_fragments)
    html_page = f"""
    <html><head><meta charset="utf-8"><title>Attention Connectivity</title></head>
    <body>
      <h1>Attention Connectivity Visualisation</h1>
      {html_body}
    </body></html>
    """
    with open("reports/connectivity.html", "w", encoding="utf-8") as f:
        f.write(html_page)

    print("Wrote connectivity visualisation → reports/connectivity.html")

if __name__ == "__main__":
    main()
