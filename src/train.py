import argparse
from model import build_seq2seq_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_vocab_size', type=int, default=5000)
    p.add_argument('--target_vocab_size', type=int, default=5000)
    p.add_argument('--embed_dim', type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--cell_type', choices=['RNN','LSTM','GRU'], default='LSTM')
    p.add_argument('--num_layers', type=int, default=1)
    return p.parse_args()

def main():
    args = parse_args()
    model = build_seq2seq_model(
        input_vocab_size=args.input_vocab_size,
        target_vocab_size=args.target_vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        cell_type=args.cell_type,
        num_layers=args.num_layers
    )
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    # TODO: load data, fit(), save model, etc.

if __name__ == '__main__':
    main()