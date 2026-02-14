from sys import argv
from argparse import ArgumentParser
from .train_bpe import train_bpe
from datetime import datetime

if __name__ == '__main__':
    parser = ArgumentParser(
        prog='run_train_bpe',
        description='Runs the BPE tokenizer algorithm using the specified corpus'
    )

    parser.add_argument(
        'file',
        help="The file path of the corpus"
    )
    parser.add_argument(
        '-v',
        '--vocab_size',
        type=int,
        default=10000,
        help="The target vocab size. This will determine how many merge iterations will be performed and how many tokens will be created.")
    parser.add_argument(
        '-s',
        '--special_token',
        action='append',
    )

    args = parser.parse_args()
    
    path = args.file
    vocab_size = args.vocab_size
    special_tokens = args.special_token

    if not special_tokens:
        special_tokens = ['<|endoftext|>']

    print(f"Training tokenizer, corpus: {path}, vocab size: {vocab_size}, special tokens: {special_tokens}")

    started = datetime.now()
    train_bpe(path, vocab_size, special_tokens)
    elapsed = datetime.now() - started
    print(f"Completed tokenizer training in {elapsed.total_seconds()}s")