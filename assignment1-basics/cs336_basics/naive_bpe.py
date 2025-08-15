# This is a naive implementation of BPE (Byte Pair Encoding)
# to make sure I understand the algorithm well before
# implementing a more efficient one that can properly
# handle large corpora.

import regex as re

# The original BPE implementation of Sennrich et al.[2016] pre-tokenizes by simply splitting on whitespace(i.e.,s.split(" ")).
# In contrast, we’ll use a regex-based pre-tokenizer (used by GPT-2;Radford et al.,2019)
# See: https://github.com/openai/tiktoken/pull/234/files 
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def naive_bpe(corpus: str, num_merges: int, special_tokens: list[bytes], pretoken_regex: str = PAT) -> tuple[list[bytes], dict[tuple[bytes], int]]:
    """
    Naive BPE implementation.
    
    Args:
        corpus (str): The input text corpus.
        num_merges (int): Number of merge operations to perform.
        special_tokens (list[bytes]): List of special tokens to initialize the vocabulary.
        pretoken_regex (str): Regular expression for pretokenization.

    Returns:
        tuple: A tuple containing the vocabulary and the pretokenized cache.
    """
    # Keeps track of merges that occurred
    # Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>.
    # The merges should be ordered by order of creation
    merges: list[tuple[bytes, bytes]] = []
    vocab = initialize_vocab(special_tokens)

    corpus_segments = split_on_special_tokens(corpus, special_tokens)
    print("Number of corpus segments after splitting on special tokens:", len(corpus_segments))

    for i, segment in enumerate(corpus_segments):
        print("Processing segment:", i + 1, "of", len(corpus_segments))
        if not segment:
            print("Skipping empty segment")
            continue

        # print("Processing segment:", segment)
        pretokenized_cache = pretokenize(segment, pretoken_regex)
        merge_pairs(vocab, pretokenized_cache, num_merges, merges)
    
    # print("output vocab", vocab);
    return vocab, merges

def initialize_vocab(special_tokens: list[bytes]) -> list[bytes]:
    """
    Initialize a vocabulary of tokens including the specified
    special tokens. In BPE, we initialize the vocabulary
    by assigning each byte value (0-255) a unique token,
    therefore each unicode character can be represented
    by one or more tokens (including spaces, punctuation, etc.).

    Args:
        special_tokens (list[bytes]): List of special tokens to include in the vocabulary.
    
    Returns:
        list[bytes]: A list of bytestring tokens representing the vocabulary. The index
        of each item in the list corresponds to its numerical token ID.
    """
    vocab = [s if isinstance(s, bytes) else s.encode('utf-8') for s in special_tokens] + [chr(i).encode('utf-8') for i in range(256)]
    return vocab

def remove_special_tokens(corpus: str, special_tokens: list[bytes|str]) -> str:
    """
    Remove special tokens from the corpus.
    """
    for token in special_tokens:
        corpus = corpus.replace(token.decode("utf-8") if isinstance(token, bytes) else token, "")
    return corpus

def split_on_special_tokens(corpus: str, escaped_special_tokens: list[bytes|str]) -> list[str]:
    """
    Split the corpus on special tokens so that we don't merge across
    document boundaries.
    """
    escaped_special_tokens = [token.decode("utf-8").replace('|', '\\|') if isinstance(token, bytes) else token.replace('|', '\\|') for token in escaped_special_tokens]
    return re.split("|".join(escaped_special_tokens), corpus)

def pretokenize(corpus: str, pretoken_regex: str = PAT) -> dict[tuple[bytes], int]:
    """
    Once you have a vocabulary, you could, in principle, count how often bytes occur next to each
    other in your text and begin merging them starting with the most frequent pair of bytes.
    However, this is quite computationally expensive, since we'd have to go take a full pass over the corpus each time we merge.
    In addition, directly merging bytes across the corpus may result in tokens that differ only in punctuation (e.g., dog!vs.dog.).
    These tokens would get completely different token IDs, even though they are likely to have high semantic similarity (since they differ only in punctuation).
    To avoid this, we pre-tokenize the corpus. You can think of this as a coarse-grained tokenization over
    the corpus that helps us count how often pairs of characters appear. For example, the word 'text' might be a
    pre-token that appears 10 times. In this case, when we count how often the characters 't' and 'e' appear next to each other,
    we will see that the word 'text' has 't' and 'e' adjacent and we can increment their count by 10 instead of looking through the corpus.
    Since we're training a byte-level BPE model, each pre-token is represented as a sequence of UTF-8 bytes
    """
    pretokens = re.finditer(pretoken_regex, corpus)
    cache: dict[tuple[bytes], int] = {}
    for match in pretokens:
        token = match.group(0)
        encoded_token = token.encode("utf-8")
        token_key = tuple(encoded_token[i:i+1] for i in range(len(encoded_token)))
        if token_key not in cache:
            cache[token_key] = 0
        cache[token_key] += 1
    
    # print("pretokenized cache size", len(cache))
    # print("pretokenized cache", cache)
    return cache

def merge_pairs(vocab: list[bytes], pretokenized_cache: dict[tuple[bytes], int], num_merges: int, merges: list[tuple[bytes, bytes]]) -> tuple[list[bytes], dict[tuple[bytes], int]]:
    old_cache = pretokenized_cache
    for merge_step in range(num_merges):
        # print("Running merge iteration", merge_step)
        best_pair, best_count = find_best_pair(old_cache)
            
        # print("Best pair of merge", merge_step, ":", best_pair, "with count", best_count)
        if best_pair is None:
            # print("No more pairs to merge, stopping early.")
            return # Or return from method?
        
        vocab.append(b"".join(best_pair))

        # merge the best pair in the pretokenized cache
        new_cache = merge_token_pair(best_pair, old_cache)

        # Keep track of the merges that occurred since
        # since it's required by the assignment.
        merges.append(best_pair)

        # print("New cache size after merge", merge_step, ":", len(new_cache))
        # print("New cache after merge", merge_step, ":", new_cache)
        old_cache = new_cache

    return old_cache


def find_best_pair(token_cache: dict[tuple[bytes], int]) -> tuple[tuple[bytes, bytes], int]:
    pair_counts = {}
    best_pair: tuple[bytes, bytes] = None
    best_count: int = 0
    for token_key, count in token_cache.items():
        for i in range(len(token_key) - 1):
            pair = (token_key[i], token_key[i + 1])
            if pair not in pair_counts:
                pair_counts[pair] = count
            else:
                pair_counts[pair] += count
            
            if best_pair is None:
                best_pair = pair
                best_count = pair_counts[pair]
            elif pair_counts[pair] > best_count:
                best_pair = pair
                best_count = pair_counts[pair]
            elif pair_counts[pair] == best_count:
                if pair > best_pair:
                    best_pair = pair
                    best_count = pair_counts[pair]
    return best_pair, best_count

def merge_token_pair(pair: tuple[bytes, bytes], token_cache: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
    new_cache = {}
    for token_key, count in token_cache.items():
        temp_token_key = []
        i = 0
        while i < len(token_key):
            if i == len(token_key) - 1:
                temp_token_key.append(token_key[i])
                i += 1
                continue
            if token_key[i] == pair[0] and token_key[i + 1] == pair[1]:
                temp_token_key.append(pair[0] + pair[1])
                i += 2
                continue
            
            temp_token_key.append(token_key[i])
            i += 1
        
        new_token_key = tuple(temp_token_key)
        new_cache[new_token_key] = count
    return new_cache

def test_naive_bpe():
    sample_text = """low low low low low
lower lower widest widest widest <|endoftext|>
newest newest newest newest <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> newest newest
    """

    num_merges=6

    vocab, _ = naive_bpe(
        corpus=sample_text,
        num_merges=num_merges,
        special_tokens=['<|endoftext|>'],
        pretoken_regex=r"\w+")

    assert vocab[0] == b"<|endoftext|>"
    assert vocab[257] == b"st"
    assert vocab[258] == b"est"
    assert vocab[259] == b"ow"
    assert vocab[260] == b"low"
    assert vocab[261] == b"west"
    assert vocab[262] == b"ne"
    assert len(vocab) == 256 + 7  # 256 byte values + 1 special token + 6 merges

    vocab = {i: token for i, token in enumerate(vocab)}
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    print("Test passed!")

if __name__ == "__main__": 
    test_naive_bpe()
    