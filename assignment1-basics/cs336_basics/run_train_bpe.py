from sys import argv
from argparse import ArgumentParser
from .train_bpe import train_bpe
from datetime import datetime
import json
import os

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
        action='append'
    )
    parser.add_argument(
        '-o',
        '--output',
        help="The folder where to store the generated vocab and merges files",
        default="output"
    )

    args = parser.parse_args()
    
    path = args.file
    vocab_size = args.vocab_size
    special_tokens = args.special_token
    output_dir = args.output

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    base_name, ext = os.path.splitext(
        os.path.basename(path)
    )
    
    vocab_file = os.path.join(output_dir, f"{base_name}-vocab.json")
    merges_file = os.path.join(output_dir, f"{base_name}-merges.json")

    if not special_tokens:
        special_tokens = ['<|endoftext|>']

    print(f"Training tokenizer, corpus: {path}, vocab size: {vocab_size}, special tokens: {special_tokens}")

    started = datetime.now()
    vocab, merges = train_bpe(path, vocab_size, special_tokens)
    elapsed = datetime.now() - started
    print(f"Completed tokenizer training in {elapsed.total_seconds()}s")

    vocab_decoded = { k: str(v) for k,v in vocab.items() }
    with open(vocab_file, 'w') as f:
        json.dump(vocab_decoded, f)
        print(f"Saved vocab JSON file at {vocab_file}")
    
    merges_decoded = [(str(a), str(b)) for a, b in merges]
    with open(merges_file, 'w') as f:
        json.dump(merges_decoded, f)
        print(f"Saved merges JSON file at {merges_file}")

