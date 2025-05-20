import argparse
import numpy as np
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger
from data_utils import load_data, build_vocab, encode_sequences, prepare_decoder_inputs
from model_vanilla import build_seq2seq
from config import TRAIN_PATH, VAL_PATH, TEST_PATH

# Sweep configuration using Bayesian optimization for hyperparameters.
sweep_config = {
    "method": "bayes",
    "name": "vanilla_seq2seq_sweep",
    "metric": {"name": "val_seq_accuracy", "goal": "maximize"},
    "parameters": {
        "embed_dim": {"values": [16, 32, 64, 256]},
        "hidden_dim": {"values": [16, 32, 64, 256]},
        "enc_layers": {"values": [1, 2, 3]},
        "dec_layers": {"values": [1, 2, 3]},
        "cell_type": {"values": ["SimpleRNN", "GRU", "LSTM"]},
        "dropout": {"values": [0.0, 0.2, 0.3]},
        "batch_size": {"values": [32, 64, 128]},
        "epochs": {"value": 2}
    }
}

def train():
    # Obtain configuration from wandb
    cfg = wandb.config

    # Load data splits
    train_df, val_df, _ = load_data(TRAIN_PATH, VAL_PATH, TEST_PATH)

    # Build vocabularies for source and target languages
    src_idx, _ = build_vocab(train_df["roman"])
    tgt_idx, _ = build_vocab(train_df["dev"])

    # Encode the input sequences
    X_train = encode_sequences(train_df["roman"], src_idx)
    y_train = encode_sequences(train_df["dev"], tgt_idx)
    X_val   = encode_sequences(val_df["roman"], src_idx)
    y_val   = encode_sequences(val_df["dev"], tgt_idx)

    # Prepare decoder inputs (usually shifted targets)
    dec_in_train = prepare_decoder_inputs(y_train)
    dec_in_val   = prepare_decoder_inputs(y_val)

    # Use GPU for model training if available
    with tf.device("/GPU:0"):
        # Build seq2seq model with the hyperparameters from wandb config
        model = build_seq2seq(
            src_vocab=len(src_idx),
            tgt_vocab=len(tgt_idx),
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            enc_layers=cfg.enc_layers,
            dec_layers=cfg.dec_layers,
            cell_type=cfg.cell_type,
            dropout=cfg.dropout
        )

        # Train the model with training data and validate on validation data
        model.fit(
            [X_train, dec_in_train],
            y_train[..., None],
            validation_data=([X_val, dec_in_val], y_val[..., None]),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            callbacks=[WandbMetricsLogger()]
        )

        # Get predictions on validation set
        pred_probs = model.predict([X_val, dec_in_val], batch_size=cfg.batch_size, verbose=0)
        pred_ids   = np.argmax(pred_probs, axis=-1)

        # Determine actual sequence lengths (assumes 0 is the end token)
        lengths = np.argmax((y_val == 0).astype(int), axis=1)
        lengths = np.where(lengths == 0, y_val.shape[1], lengths)

        # Calculate sequence-exact-match accuracy
        matches = [np.all(pred_ids[i, :lengths[i]] == y_val[i, :lengths[i]]) for i in range(len(y_val))]
        val_seq_acc = float(np.mean(matches))
        wandb.log({"val_seq_accuracy": val_seq_acc})

def main():
    # Parse command-line arguments for training configuration
    parser = argparse.ArgumentParser(description="Train vanilla seq2seq (local or sweep)")
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--enc_layers", type=int, default=1)
    parser.add_argument("--dec_layers", type=int, default=1)
    parser.add_argument("--cell_type", choices=["SimpleRNN", "GRU", "LSTM"], default="LSTM")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--project", type=str, default="dakshina-seq2seq")
    parser.add_argument("--sweep", action="store_true",
                        help="Launch a W&B hyperparameter sweep")
    parser.add_argument("--count", type=int, default=20,
                        help="Number of runs for the sweep")
    args = parser.parse_args()

    if args.sweep:
        # Start a sweep run with wandb using the sweep configuration
        sweep_id = wandb.sweep(sweep_config, project=args.project)
        wandb.agent(sweep_id, function=train, count=args.count)
    else:
        # Regular single run initializing wandb with CLI settings
        wandb.init(project=args.project, config=vars(args), reinit=True)
        train()
        wandb.finish()

if __name__ == "__main__":
    main()
