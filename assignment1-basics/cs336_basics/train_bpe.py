from naive_bpe import naive_bpe

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Trains a Byte Pair Encoding (BPE) tokenizer.

    Args:
        input_path (str): Path to the text file with training data.
        vocab_size (int): Positive integer that defines the maximum final vocabulary size
            (including the initial byte vocabulary, vocabulary items produced from merging,
            and special tokens).
        special_token (list[str]): A list of strings to add to the vocabulary. These special tokens
            do not otherwise affect BPE training.
    Returns:
        tuple: A tuple containing:
            - A dictionary containing the tokenizer vocabulary, a mapping from
                `int` (token ID in the vocabulary) to `bytes` (token bytes).
            - A list of BPE merges produced from training. Each list item is a
                `tuple` of `bytes` (<token1>, <token2>) representing that <token1>
                was merged with <token2>. The merges are ordered by order of creation.
    """

    # TODO: Chunk input file for efficiency
    # TODO: use a more efficient bpe implementation
    with open(input_path, 'r', encoding='utf-8') as f:
        corpus = f.read()
        (vocab, merges) = naive_bpe(corpus, vocab_size, special_tokens)

        vocab_dict = {i: token for i, token in enumerate(vocab)}
        return vocab_dict, merges