# This is a naive implementation of BPE (Byte Pair Encoding)
# to make sure I understand the algorithm well before
# implementing a more efficient one that can properly
# handle large corpora.

import regex as re

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
    vocab = initialize_vocab(special_tokens)
    pretokenized_cache = pretokenize(corpus, pretoken_regex)
    merged_cache = merge_pairs(vocab, pretokenized_cache, num_merges)
    
    return vocab, merged_cache

def initialize_vocab(special_tokens: list[bytes]) -> list[bytes]:
    vocab = special_tokens + [chr(i).encode('utf-8') for i in range(256)]
    return vocab


def pretokenize(corpus: str, pretoken_regex: str = PAT) -> dict[tuple[bytes], int]:
    pretokens = re.finditer(pretoken_regex, corpus)
    cache: dict[tuple[bytes], int] = {}
    for match in pretokens:
        token = match.group(0)
        encoded_token = token.encode("utf-8")
        token_key = tuple(encoded_token[i:i+1] for i in range(len(encoded_token)))
        if token_key not in cache:
            cache[token_key] = 0
        cache[token_key] += 1
    
    print("pretokenized cache size", len(cache))
    print("pretokenized cache", cache)
    return cache

def merge_pairs(vocab: list[bytes], pretokenized_cache: dict[tuple[bytes], int], num_merges: int) -> tuple[list[bytes], dict[tuple[bytes], int]]:
    old_cache = pretokenized_cache
    for merge_step in range(num_merges):
        print("Running merge iteration", merge_step)
        best_pair, best_count = find_best_pair(old_cache)
            
        print("Best pair of merge", merge_step, ":", best_pair, "with count", best_count)
        
        vocab.append(b"".join(best_pair))

        # merge the best pair in the pretokenized cache
        new_cache = merge_token_pair(best_pair, old_cache)

        print("New cache size after merge", merge_step, ":", len(new_cache))
        print("New cache after merge", merge_step, ":", new_cache)
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
lower lower widest widest widest
newest newest newest newest newest newest
    """

    vocab, _ = naive_bpe(
        corpus=sample_text,
        num_merges=6,
        special_tokens=[b'<|endoftext|>'],
        pretoken_regex=r"\w+")

    assert vocab[0] == b"<|endoftext|>"
    assert vocab[257] == b"st"
    assert vocab[258] == b"est"
    assert vocab[259] == b"ow"
    assert vocab[260] == b"low"
    assert vocab[261] == b"west"
    assert vocab[262] == b"ne"
    assert len(vocab) == 256 + 7  # 256 byte values + 1 special token + 6 merges
    print("Test passed!")

if __name__ == "__main__": 
    test_naive_bpe()
    