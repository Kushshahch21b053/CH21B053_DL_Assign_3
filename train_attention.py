import argparse
import numpy as np
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger
from data_utils import load_data, build_vocab, encode_sequences, prepare_decoder_inputs
from model_attention import build_attention_model
from config import TRAIN_PATH, VAL_PATH, TEST_PATH

# Sweep configuration for hyperparameter optimization using Bayesian optimization
sweep_config_attn = {
    "method": "bayes",
    "name": "attn_seq2seq_sweep",
    "metric": {"name": "val_seq_accuracy", "goal": "maximize"},
    "parameters": {
        "embed_dim": {"values": [64, 128, 256]},
        "hidden_dim": {"values": [32, 64, 128]},
        "dropout": {"values": [0.0, 0.2, 0.3]},
        "batch_size": {"values": [32, 64]},
        "epochs": {"value": 2}  # Fixed number of epochs for the sweep experiments
    }
}

def train_attn():
    # Get hyperparameters configuration from wandb
    cfg = wandb.config

    # Load training, validation (and test) datasets
    train_df, val_df, _ = load_data(TRAIN_PATH, VAL_PATH, TEST_PATH)

    # Build vocabularies for source (roman) and target (dev) texts
    src_idx, _ = build_vocab(train_df["roman"])
    tgt_idx, _ = build_vocab(train_df["dev"])
    
    # Encode text sequences into integer sequences
    X_train = encode_sequences(train_df["roman"], src_idx)
    y_train = encode_sequences(train_df["dev"], tgt_idx)
    X_val   = encode_sequences(val_df["roman"], src_idx)
    y_val   = encode_sequences(val_df["dev"], tgt_idx)
    
    # Prepare decoder input sequences (e.g., shifted sequences for teacher forcing)
    dec_in_train = prepare_decoder_inputs(y_train)
    dec_in_val   = prepare_decoder_inputs(y_val)

    # Use GPU if available for training
    with tf.device("/GPU:0"):
        # Build the attention-based seq2seq model with specified hyperparameters
        model = build_attention_model(
            src_vocab=len(src_idx),
            tgt_vocab=len(tgt_idx),
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout
        )
        
        # Train the model with given training data and validation data
        model.fit(
            [X_train, dec_in_train],
            y_train[..., None],  # Add an extra dimension for target sequence labels
            validation_data=([X_val, dec_in_val], y_val[..., None]),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            callbacks=[WandbMetricsLogger()]  # Log metrics to wandb during training
        )
        
        # Predict on validation data
        pred_probs = model.predict([X_val, dec_in_val], batch_size=cfg.batch_size, verbose=0)
        pred_ids = np.argmax(pred_probs, axis=-1)  # Get index with highest probability for each token

        # Determine sequence lengths by finding the first zero (assumed padding)
        lengths = np.argmax((y_val == 0).astype(int), axis=1)
        lengths = np.where(lengths == 0, y_val.shape[1], lengths)
        
        # Check if predicted sequence matches the true sequence for each sample
        matches = [np.all(pred_ids[i, :lengths[i]] == y_val[i, :lengths[i]]) for i in range(len(y_val))]
        val_seq_acc = float(np.mean(matches))
        
        # Log validation sequence accuracy
        wandb.log({"val_seq_accuracy": val_seq_acc})

def main():
    # Parse command-line arguments for customizing training run parameters
    parser = argparse.ArgumentParser(description="Train attention seq2seq (local or sweep)")
    parser.add_argument("--embed_dim", type=int, default=128, help="Dimension of the embedding layer")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Dimension of the hidden layer")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for regularization")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--project", type=str, default="dakshina-seq2seq-attn", help="WandB project name")
    parser.add_argument("--sweep", action="store_true", help="Launch a W&B hyperparameter sweep")
    parser.add_argument("--count", type=int, default=20, help="Number of sweep runs to execute")
    args = parser.parse_args()

    if args.sweep:
        # Start a hyperparameter sweep on WandB
        sweep_id = wandb.sweep(sweep_config_attn, project=args.project)
        wandb.agent(sweep_id, function=train_attn, count=args.count)
    else:
        # Initialize a regular wandb run, train the model and finish logging
        wandb.init(project=args.project, config=vars(args), reinit=True)
        train_attn()
        wandb.finish()

if __name__ == "__main__":
    main()
