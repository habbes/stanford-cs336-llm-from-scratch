# This is a naive implementation of BPE (Byte Pair Encoding)
# to make sure I understand the algorithm well before
# implementing a more efficient one that can properly
# handle large corpora.

import regex as re

# The original BPE implementation of Sennrich et al.[2016] pre-tokenizes by simply splitting on whitespace(i.e.,s.split(" ")).
# In contrast, we’ll use a regex-based pre-tokenizer (used by GPT-2;Radford et al.,2019)
# See: https://github.com/openai/tiktoken/pull/234/files 
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe_core(corpus: str, vocab_size: int, special_tokens: list[bytes], pretoken_regex: str = PAT) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Naive BPE implementation.
    
    Args:
        corpus (str): The input text corpus.
        vocab_size (int): Desired output vocabulary size.
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

    num_merges = vocab_size - len(vocab)

    corpus_segments = split_on_special_tokens(corpus, special_tokens)
    # print("Number of corpus segments after splitting on special tokens:", len(corpus_segments))

    pretokenized_cache = {}
    for i, segment in enumerate(corpus_segments):
        # print("Processing segment:", i + 1, "of", len(corpus_segments), "segment length:", len(segment))
        if not segment:
            print("Skipping empty segment")
            continue
        
        # print("Processing segment:", segment)
        pretokenized_cache = merge_pretokenized_counters_in_place(pretokenized_cache, pretokenize(segment, pretoken_regex))
    
    merge_pairs(vocab, pretokenized_cache, num_merges, merges, debug=False)
    
    # print("Complete segments merges", len(corpus_segments), " vocab length", len(vocab), "merges length:", len(merges));
    vocab_dict = {i: token for i, token in enumerate(vocab)}
    return vocab_dict, merges

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
    vocab = [s if isinstance(s, bytes) else s.encode('utf-8') for s in special_tokens] + [bytes([i]) for i in range(256)]
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

def merge_pretokenized_counters_in_place(pretokens1: dict[tuple[bytes], int], pretokens2: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
    """
    Merges pre-tokens and their counts from the second argument, into the dictionary provided
    by the first argument. Then it returns the updated dictionary. If a pre-token exists
    in both dictionaries, their counts will be added up in the updated dictionary.

    Args:
        pretokens1 (dict[tuple[bytes], int]): The first pretokens cache. This dictionary will be updated in-place
            by merging the items of the second cache into it.
        pretokens2 (dict[tuple[bytes], int]): The second pretokens cache, whic will be merged into the first.
    Returns:
        (dict[tuple[bytes], int]): The updated pretokens cache. This is a reference to the mutated first argument.
    """
    for pretoken, count in pretokens2.items():
        pretokens1[pretoken] = pretokens1.get(pretoken, 0) + count
    
    return pretokens1

def merge_pairs(vocab: list[bytes], pretokenized_cache: dict[tuple[bytes], int], num_merges: int, merges: list[tuple[bytes, bytes]], debug: bool = False) -> None:
    pair_counts = None
    for merge_step in range(num_merges):
        if debug:
            print("Running merge iteration", merge_step, "target num merges:", num_merges)
        best_pair, pair_counts = find_best_pair(pretokenized_cache, pair_counts)
            
        # print("Best pair of merge", merge_step, ":", best_pair, "with count", best_count)
        if best_pair is None:
            if debug:
                print("No more pairs to merge, stopping early.")
            return
        
        vocab.append(b"".join(best_pair))

        if debug:
            print("Adding new token to vocab:", vocab[-1], "at index", len(vocab) - 1)

        # merge the best pair in the pretokenized cache
        # the pretokenized cache is updated with the merged pair
        merge_token_pair(best_pair, pretokenized_cache, pair_counts)
        if debug:
            print("New cache size after merge", merge_step, ":", len(pretokenized_cache))

        # Keep track of the merges that occurred since
        # since it's required by the assignment.
        merges.append(best_pair)

        if debug:
            print("Merged pair:", best_pair, "into token:", vocab[-1])


def find_best_pair(token_cache: dict[tuple[bytes], int], pair_counts: dict[tuple[bytes, bytes], int]|None = None) -> tuple[tuple[bytes, bytes], dict[tuple[bytes, bytes], int]]:
    # If pair_counts is provided, we want to update it instead of rebuilding from scratch for effiency.
    # We expect pair_counts to be None on first invocation, then we'll build the table and it
    # will be passed around and updated in-place after that
    # let's say we have the following token_cache
    # - (m, e, s, t): 3
    # - (m, e, k, l): 5
    # - (e, s, k, l): 4
    # - (l, e, s): 1
    # - (s, t, e, s): 2
    #
    # This produces a pair_counts with the following entries
    #
    # - (e, s): 10
    # - (s, t): 5
    # - (e, k): 5
    # - (k, l): 9
    # - (s, k): 4
    # - (m, e): 8
    # - (l, e): 1
    # - (t, e): 2
    
    
    best_pair = None
    if pair_counts is None:
        best_pair, pair_counts = find_best_pair_from_token_cache(token_cache)
    else:
        best_pair = find_best_pair_from_pair_counts(pair_counts)

    return best_pair, pair_counts

def find_best_pair_from_token_cache(token_cache: dict[tuple[bytes], int]) -> tuple[tuple[bytes, bytes], dict[tuple[bytes, bytes], int]]:
    """
    Finds the most common pair of consecutive bytes in the token cache. It also builds
    and returns a dictionary mapping pairs of byte sequences to their frequency
    """
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
                best_pair, best_count = pair, pair_counts[pair]
            elif pair_counts[pair] > best_count:
                best_pair, best_count = pair, pair_counts[pair]
            elif pair_counts[pair] == best_count:
                if pair > best_pair:
                    best_pair, best_count = pair, pair_counts[pair]

    return best_pair, pair_counts

def find_best_pair_from_pair_counts(pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    best_count = 0
    best_pair: tuple[bytes, bytes] = None
    for pair, count in pair_counts.items():
        if best_pair is None:
            best_pair, best_count = pair, count
        elif count > best_count:
            best_pair, best_count = pair, count
        elif count == best_count:
            if pair > best_pair:
                best_pair, best_count = pair, count
    return best_pair

def merge_token_pair(
        pair: tuple[bytes, bytes],
        token_cache: dict[tuple[bytes], int],
        pair_counts: dict[tuple[bytes, bytes], int]) -> dict[tuple[bytes], int]:
    # We want to replace entries in the token cache that contain the pair with the merged pair
    # e.g. say we have the following entries
    # {
    # (l, o ,w): 5
    # (l, o, w, e, r): 2
    # (w, i, d, e, s, t): 3
    # (n, e, w, e, s, t): 6
    # }
    # and pair = (s, t)
    # then the result will be
    # {
    # (l, o ,w): 5
    # (l, o, w, e, r): 2
    # (w, i, d, e, st): 3
    # (n, e, w, e, st): 6
    # }
    
    merged_pair = pair[0] + pair[1]
    entries_to_replace: dict[tuple[bytes], int] = {}

    # Remove pair from pair counts since we're going to merge the pair into a single token
    # and update other entries
    del pair_counts[pair]
    for token_key, count in token_cache.items():
        # check if the pair existing in this entry
        i = 0
        token_len = len(token_key)
        while i < token_len - 1:
            if token_key[i] == pair[0] and token_key[i + 1] == pair[1]:
                entries_to_replace[token_key] = i
                break
            i += 1
    
    for token_key, index_to_replace in entries_to_replace.items():
        token_len = len(token_key)
        count = token_cache[token_key]
        del token_cache[token_key]

        # Replace the target consecutive tokens by the single merged token object
        temp_new_token = [t for t in token_key[:index_to_replace]]
        temp_new_token.append(merged_pair)

        # we also want to replace the pair_counts accordingly
        # let's say we have the following token_cache
        # - (m, e, s, t): 3
        # - (m, e, k, l): 5
        # - (e, s, k, l): 4
        # - (l, e, s): 1
        # - (s, t, e, s): 2
        #
        # This produces a pair_counts with the following entries
        #
        # - (e, s): 10
        # - (s, t): 5
        # - (e, k): 5
        # - (k, l): 9
        # - (s, k): 4
        # - (m, e): 8
        # - (l, e): 1
        # - (t, e): 2
        #
        # the best/most frequent pair is (e, s)
        # We'll remove the entry (e, s) from the pair counts then
        # We have to update the pairs that overlap with the ones we want to merge
        # e.g. we'll add a new entry (m, es) with count 3, and reduce (m, e)'s count to 5
        # overall, the updated pair_counts will look like
        # - (s, t): 2
        # - (es, t): 3
        # - (e, k): 5
        # - (k, l): 9
        # - (es, k): 4
        # - (m, e): 5
        # - (m, es): 3
        # - (l, es): 1
        # - (t, es): 2

        if index_to_replace > 0:
            pair_to_update = (token_key[index_to_replace - 1], pair[0])
            update_pair_counts_with_merged_pair(pair_counts, pair_to_update, merged_pair, count, index_to_replace=1)
            
                
        if index_to_replace + 2 < token_len:
            pair_to_update = (pair[1], token_key[index_to_replace + 2])
            update_pair_counts_with_merged_pair(pair_counts, pair_to_update, merged_pair, count, index_to_replace=0)

        # Check if there are more occurences to merge, and copy remaining bytes
        i = index_to_replace + 2
        while i < token_len:
            if i == token_len - 1:
                temp_new_token.append(token_key[i])
                i += 1
            elif token_key[i] == pair[0] and token_key[i + 1] == pair[1]:
                temp_new_token.append(merged_pair)

                # Update pair counts
                if i > 0:
                    pair_to_update = (token_key[i - 1], pair[0])
                    update_pair_counts_with_merged_pair(pair_counts, pair_to_update, merged_pair, count, index_to_replace=1)
                if i + 2 < token_len:
                    pair_to_update = (pair[1], token_key[i + 2])
                    update_pair_counts_with_merged_pair(pair_counts, pair_to_update, merged_pair, count, index_to_replace=0)
                
                i += 2
            else:
                temp_new_token.append(token_key[i])
                i += 1

        new_token_key = tuple(temp_new_token)
        token_cache[new_token_key] = count
    return token_cache

def update_pair_counts_with_merged_pair(
        pair_counts: dict[tuple[bytes, bytes], int],
        entry_to_update: tuple[bytes, bytes],
        merged_pair: bytes,
        count: int,
        index_to_replace: int):
    assert index_to_replace == 0 or index_to_replace == 1
    new_entry = (entry_to_update[0], merged_pair) if index_to_replace == 1 else (merged_pair, entry_to_update[1])
    pair_counts[new_entry] = pair_counts.get(new_entry, 0) + count

    # We expect entry_to_update to exist in pair_counts since
    # the entry should have been retrieved from a subsequence of consecutive pairs in the
    # pretokenized cache. And all such pairs should have entries in the pair_counts
    # by definition. If that's not the case then there's a bug earlier in the code.
    replaced_new_count = pair_counts[entry_to_update] - count
    assert replaced_new_count >= 0
    if replaced_new_count == 0:
        del pair_counts[entry_to_update]
    else:
        pair_counts[entry_to_update] = replaced_new_count