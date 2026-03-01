from argparse import ArgumentParser
from datetime import datetime
from .tokenizer import Tokenizer
from .doc_sampler import sample_docs

if __name__ == '__main__':
    parser = ArgumentParser(
        prog='estimate_tokenizer_ratio',
        description='Estimates the compression ratio (bytes/token) of tokenizer by tokenizing a sample of documents.'
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
        '-s',
        '--separator',
        help="The document separator word. Defaults to '<|endoftext|>'.",
        default='<|endoftext|>'
    )
    parser.add_argument(
        '-n',
        '--num_samples',
        help="The number of samples to extract from the corpus. Default is 10.",
        type=int,
        default=10
    )
    parser.add_argument(
        '-r',
        '--random_samples',
        help="Whether to extra samples randomly. By default, samples are extracted sequentially from the beginning of the corpus.",
        action='store_true',
        default=False
    )
    parser.add_argument(
        '-p',
        '--print_samples',
        help='Whether to print the extracted samples.',
        action='store_true',
        default=False
    )

    args = parser.parse_args()
    
    vocab_path = args.vocab
    merges_path = args.merges
    corpus_path = args.corpus
    separator = args.separator
    num_samples = args.num_samples
    random_samples = args.random_samples
    special_tokens = args.special_token
    print_samples = args.print_samples

    if not special_tokens:
        special_tokens = ['<|endoftext|>']

    print(vars(args))

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    samples = sample_docs(corpus_path, num_samples, separator, random=random_samples)
    if (print_samples):
        print("Extracted Samples:")
        print(samples)

    sample_bytes = samples.encode('utf-8')
    doc_len = len(samples)
    bytes_len = len(sample_bytes)
    num_extracted_samples = samples.count(separator)
    print(f"Extracted {num_extracted_samples} samples, total length {doc_len} chars, {bytes_len} bytes")

    started = datetime.now()
    tokens = tokenizer.encode(samples)
    elapsed = datetime.now() - started

    num_tokens = len(tokens)

    print(f"Text size: {bytes_len} bytes, Tokens: {num_tokens}, ratio: {bytes_len/num_tokens} bytes/token")
    print(f"Tokenization took {elapsed.total_seconds()} seconds, ratio: {bytes_len/elapsed.total_seconds()} bytes/seconds")

 
