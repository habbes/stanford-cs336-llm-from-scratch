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
    pair_index: TokenPairIndex = None
    for merge_step in range(num_merges):
        if debug:
            print("Running merge iteration", merge_step, "target num merges:", num_merges)
        best_pair, pair_index = find_best_pair(pretokenized_cache, pair_index)
            
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
        merge_token_pair(best_pair, pretokenized_cache, pair_index)
        if debug:
            print("New cache size after merge", merge_step, ":", len(pretokenized_cache))

        # Keep track of the merges that occurred since
        # since it's required by the assignment.
        merges.append(best_pair)

        if debug:
            print("Merged pair:", best_pair, "into token:", vocab[-1])

def find_best_pair(token_cache: dict[tuple[bytes], int], pair_index: 'TokenPairIndex' = None) -> tuple[tuple[bytes, bytes], dict[tuple[bytes, bytes], int]]:
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
    
    if pair_index is None:
        pair_index = TokenPairIndex(token_cache)
        return pair_index.get_cached_best_pair(), pair_index
    else:
        return pair_index.compute_best_pair(), pair_index

def merge_token_pair(
        pair: tuple[bytes, bytes],
        token_cache: dict[tuple[bytes], int],
        pair_index: 'TokenPairIndex') -> dict[tuple[bytes], int]:
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
    # entries_to_replace: dict[tuple[bytes], int] = {}

   
    # for token_key, count in token_cache.items():
    #     # check if the pair existing in this entry
    #     i = 0
    #     token_len = len(token_key)
    #     while i < token_len - 1:
    #         if token_key[i] == pair[0] and token_key[i + 1] == pair[1]:
    #             entries_to_replace[token_key] = i
    #             break
    #         i += 1
    
    for token_key, index_to_replace in pair_index.get_words_with_pair(pair): # todo, we should copy the dict so that we can modify
        token_len = len(token_key)
        count = token_cache[token_key]
        del token_cache[token_key]
        pair_index.remove_word(token_key)

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
            # When the overlapping pair to update starts with a token preceding the new token in the pretoken sequence
            # then we retrieve it from the updated pretoken (temp_new_token) rather than the original pretoken (token_key)
            # since the preceding token might have been updated in the new pretoken sequence.
            # This is usually the case when the pair to be merged appears multiple times in the sequence
            # .e.g if we have the pretoken sequence (p, a, i, n, i, n, g) and we're merging (i, n)
            # then we'll have temp_new_token = [p, a, in] at some point, waiting to add the second occurrence of (i, n)
            # When adding the second occurrence, temp_new_token = [p, a, in, in] we want to make sure
            # that pair_to_update = (in, in) based on the updated token sequence, and not (n, in) based on the original sequence.
            pair_to_update = (temp_new_token[-2], pair[0])
            update_pair_counts_with_merged_pair(pair_index, pair_to_update, merged_pair, count, index_to_replace=1)
                
        if index_to_replace + 2 < token_len:
            pair_to_update = (pair[1], token_key[index_to_replace + 2])
            update_pair_counts_with_merged_pair(pair_index, pair_to_update, merged_pair, count, index_to_replace=0)

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
                    pair_to_update = (temp_new_token[-2], pair[0])
                    update_pair_counts_with_merged_pair(pair_index, pair_to_update, merged_pair, count, index_to_replace=1)
                if i + 2 < token_len:
                    pair_to_update = (pair[1], token_key[i + 2])
                    update_pair_counts_with_merged_pair(pair_index, pair_to_update, merged_pair, count, index_to_replace=0)
                
                i += 2
            else:
                temp_new_token.append(token_key[i])
                i += 1
    
        new_token_key = tuple(temp_new_token)
        token_cache[new_token_key] = count
        # need to add each pair of this new word
        # note that this loops over the token again to extra all pairs, redundant work that could be optimize since the word
        # has been fully scanned by now.
        pair_index.add_word_with_pairs(new_token_key)
    
     
    # Remove pair from pair counts since we've merged the pair into a single token
    # and update other entries
    pair_index.remove_pair(pair)
    return token_cache

def update_pair_counts_with_merged_pair(
        pair_index: 'TokenPairIndex',
        entry_to_update: tuple[bytes, bytes],
        merged_pair: bytes,
        count: int,
        index_to_replace: int):
    assert index_to_replace == 0 or index_to_replace == 1
    new_entry = (entry_to_update[0], merged_pair) if index_to_replace == 1 else (merged_pair, entry_to_update[1])
    pair_index.increment_pair_count(new_entry, count)

    # We expect entry_to_update to exist in pair_counts since
    # the entry should have been retrieved from a subsequence of consecutive pairs in the
    # pretokenized cache. And all such pairs should have entries in the pair_counts
    # by definition. If that's not the case then there's a bug earlier in the code.
    replaced_new_count = pair_index.get_pair_count(entry_to_update) - count
    assert replaced_new_count >= 0
    if replaced_new_count == 0:
        pair_index.remove_pair(entry_to_update)
    else:
        pair_index.set_pair_count(entry_to_update, replaced_new_count)


class TokenPairIndex:
    index: dict[tuple[bytes, bytes], tuple[dict[tuple[bytes], int], int]]

    def __init__(self, token_cache: dict[tuple[bytes], int]):
        self.pair_counts: dict[tuple[bytes, bytes], int] = {}
        self.pair_to_words: dict[tuple[bytes, bytes], dict[tuple[bytes], int]] = {}
        self.word_to_pairs: dict[tuple[bytes], set[tuple[bytes, bytes]]] = {}
        self.token_cache = token_cache
        self.best_pair: tuple[bytes, bytes] = None
        self._build_index();
    
    def get_pair_counts(self):
        return self.pair_counts
    
    def get_cached_best_pair(self):
        return self.best_pair
    
    def compute_best_pair(self):
        best_count = 0
        best_pair: tuple[bytes, bytes] = None
        for pair, count in self.pair_counts.items():
            if best_pair is None:
                best_pair, best_count = pair, count
            elif count > best_count:
                best_pair, best_count = pair, count
            elif count == best_count:
                if pair > best_pair:
                    best_pair, best_count = pair, count
        
        self.best_pair = best_pair
        return self.best_pair
    
    def get_pair_count(self, pair: tuple[bytes, bytes]):
        return self.pair_counts.get(pair, 0)

    def increment_pair_count(self, pair: tuple[bytes, bytes], delta: int):
        current_count = self.pair_counts.get(pair, 0)
        self.pair_counts[pair] = current_count + delta
    
    def set_pair_count(self, pair: tuple[bytes, bytes], count: int):
        self.pair_counts[pair] = count
    
    def remove_pair(self, pair: tuple[bytes, bytes]):
        del self.pair_counts[pair]
        words_with_pair = self.pair_to_words.get(pair)
        
        if words_with_pair is not None:
            # create list to avoid modifying dict during iteration
            words_to_remove = list(words_with_pair)
            for word in words_to_remove:
                self.remove_word_link(pair, word)
            del self.pair_to_words[pair]
    
    def remove_word(self, word: tuple[bytes]):
        pairs_in_word = self.word_to_pairs.get(word)
        if pairs_in_word is None:
            return
        
        to_remove = list(pairs_in_word)
        for pair in to_remove:
            self.remove_word_link(pair, word)
        
        del self.word_to_pairs[word]
    
    def remove_word_link(self, pair: tuple[bytes, bytes], word: tuple[bytes]):
        words_with_pair = self.pair_to_words.get(pair, None)
        if words_with_pair is not None:
            del words_with_pair[word]
        
        pairs_in_word = self.word_to_pairs.get(word, None)
        if pairs_in_word is not None:
            pairs_in_word.discard(pair)
    
    def add_word_link(self, pair: tuple[bytes, bytes], word: tuple[bytes], first_index: int):
        words_with_pair = self.pair_to_words.get(pair)
        if not words_with_pair:
            self.pair_to_words[pair] = { word: first_index }
        else:
            words_with_pair[word] = first_index
        
        pairs_in_word = self.word_to_pairs.get(word)
        if not pairs_in_word:
            self.word_to_pairs[word] = set([pair])
        else:
            pairs_in_word.add(pair)
    
    def get_first_index_of_pair_in_word(self, pair: tuple[bytes, bytes], word: tuple[bytes]):
        words_with_pair = self.pair_to_words.get(pair)
        if words_with_pair is None:
            return -1
        
        return words_with_pair.get(word, -1)
    
    def word_contains_pair(self, word: tuple[bytes], pair: tuple[bytes, bytes]):
        pairs_in_word = self.word_to_pairs.get(word, None)
        if pairs_in_word is None:
            return False

        return pair in pairs_in_word
    
    def get_words_with_pair(self, pair: tuple[bytes, bytes]):
        words_with_pair = self.pair_to_words.get(pair)
        if words_with_pair is None:
            return []
        
        return list(words_with_pair.items())
    
    def add_word_with_pairs(self, word: tuple[bytes]):
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            if not self.word_contains_pair(word, pair):
                self.add_word_link(pair, word, i) 
    
    def _build_index(self):
        pair_counts = self.pair_counts
        best_pair: tuple[bytes, bytes] = None
        best_count: int = 0
        for token_key, count in self.token_cache.items():
            for i in range(len(token_key) - 1):
                pair = (token_key[i], token_key[i + 1])
                if pair not in pair_counts:
                    pair_counts[pair] = count
                else:
                    pair_counts[pair] += count
                
                if not self.word_contains_pair(token_key, pair):
                    self.add_word_link(pair, token_key, i)
                
                if best_pair is None:
                    best_pair, best_count = pair, pair_counts[pair]
                elif pair_counts[pair] > best_count:
                    best_pair, best_count = pair, pair_counts[pair]
                elif pair_counts[pair] == best_count:
                    if pair > best_pair:
                        best_pair, best_count = pair, pair_counts[pair]
        
        self.best_pair = best_pair
