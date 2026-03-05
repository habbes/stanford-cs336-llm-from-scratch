from argparse import ArgumentParser
from datetime import datetime
import numpy as np
from .tokenizer import Tokenizer
from .bpe_common import tokens_to_np_array, save_tokens_array

if __name__ == '__main__':
    parser = ArgumentParser(
        prog='run_tokenizer',
        description='Tokenizes a text document and saves the tokens the specified file'
    )

    parser.add_argument(
        '-v',
        '--vocab',
        help="The file path of the serializer tokenizer BPE vocabulary",
        required=True
    )
    parser.add_argument(
        '-m',
        '--merges',
        help="The file path of the serializer tokenizer BPE merges",
        required=True
    )
    parser.add_argument(
        '-c',
        '--corpus',
        required=True,
        help="The path to the text corpus from which documents will be sampled.")
    parser.add_argument(
        '--special_token',
        action='append',
        help="Special tokens. Defaults to '<|endoftext|>'."
    )
    parser.add_argument(
        '-o',
        '--output',
        help='The output file where to store the generated tokens',
        required=True
    )

    args = parser.parse_args()
    
    vocab_path = args.vocab
    merges_path = args.merges
    corpus_path = args.corpus
    special_tokens = args.special_token
    output = args.output

    if not special_tokens:
        special_tokens = ['<|endoftext|>']

    print(vars(args))

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    elapsed = None
    token_count = 0
    with open(corpus_path, 'r') as f:
        started = datetime.now()
        tokens_iter = tokenizer.encode_iterable(f)
        array = tokens_to_np_array(tokens_iter)
        save_tokens_array(array, output)
        elapsed = datetime.now() - started
        token_count = array.size

    print(f"Tokenization took {elapsed.total_seconds()} seconds. Generated {token_count} tokens.")
    print(f"Generated tokens saved to {output}")