# This is a naive implementation of BPE (Byte Pair Encoding)
# to make sure I understand the algorithm well before
# implementing a more efficient one that can properly
# handle large corpora.

import regex as re
import heapq
from functools import total_ordering
from collections import defaultdict
import multiprocessing as mp

# The original BPE implementation of Sennrich et al.[2016] pre-tokenizes by simply splitting on whitespace(i.e.,s.split(" ")).
# In contrast, we’ll use a regex-based pre-tokenizer (used by GPT-2;Radford et al.,2019)
# See: https://github.com/openai/tiktoken/pull/234/files 
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PAT = re.compile(PAT)
BYTE_TABLE = tuple(bytes([i]) for i in range(256))

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

    pretokenized_chunks = None
    with mp.Pool() as pool:
        pretokenized_chunks = pool.starmap(pretokenize, map(lambda x: (x, pretoken_regex), corpus_segments))

    pretokenized_cache = {}
    for chunk in pretokenized_chunks:
        merge_pretokenized_counters_in_place(pretokenized_cache, chunk)
    
    merge_pairs(vocab, pretokenized_cache, num_merges, merges)

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

def pretokenize(corpus: str, pretoken_regex: str = None) -> dict[tuple[bytes], int]:
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
    pretokens = COMPILED_PAT.finditer(pretoken_regex, corpus) if pretoken_regex is None else re.finditer(pretoken_regex, corpus)
    cache: dict[tuple[bytes], int] = {}
    for match in pretokens:
        token = match.group(0)
        encoded_token = token.encode("utf-8")
        token_key = tuple(BYTE_TABLE[b] for b in encoded_token)
        cache[token_key] = cache.get(token_key, 0) + 1

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

def merge_pairs(vocab: list[bytes], pretokenized_cache: dict[tuple[bytes], int], num_merges: int, merges: list[tuple[bytes, bytes]]) -> None:
    pair_index: TokenPairIndex = None
    for merge_step in range(num_merges):
        best_pair, pair_index = find_best_pair(pretokenized_cache, pair_index)

        if best_pair is None:
            return

        vocab.append(b"".join(best_pair))

        # merge the best pair in the pretokenized cache
        # the pretokenized cache is updated with the merged pair
        merge_token_pair(best_pair, pretokenized_cache, pair_index)

        # Keep track of the merges that occurred since
        # since it's required by the assignment.
        merges.append(best_pair)

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
    
    for token_key, index_to_replace in pair_index.get_words_with_pair(pair):
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
        self.pair_counts: PairCounts = None
        self.pair_to_words: dict[tuple[bytes, bytes], dict[tuple[bytes], int]] = {}
        self.word_to_pairs: dict[tuple[bytes], set[tuple[bytes, bytes]]] = {}
        self.token_cache = token_cache
        self.best_pair: tuple[bytes, bytes] = None
        self._build_index();
    
    def get_cached_best_pair(self):
        return self.pair_counts.get_cached_best_pair()
    
    def compute_best_pair(self):
        return self.pair_counts.pop_best_pair()
    
    def get_pair_count(self, pair: tuple[bytes, bytes]):
        return self.pair_counts.get_pair_count(pair)

    def increment_pair_count(self, pair: tuple[bytes, bytes], delta: int):
        self.pair_counts.increment_pair_count(pair, delta)
    
    def set_pair_count(self, pair: tuple[bytes, bytes], count: int):
        self.pair_counts.set_pair_count(pair, count)
    
    def remove_pair(self, pair: tuple[bytes, bytes]):
        self.pair_counts.remove_pair(pair)
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
        pair_counts = {}
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
        
        self.pair_counts = PairCounts(pair_counts)

class PairCounts:
    def __init__(self, initial_pair_counts: dict[tuple[bytes, bytes], int]):
        self.pair_counts = initial_pair_counts
        
        # What we want is a max heap. Max heap APIs are available in heapq starting Python 3.14
        # But we're still on 3.11. So we're going to use min heap to achieve max heap semantics
        # by using negative count and reverse ordering for the token pairs
        self.heap = [(-count, (ReverseSort(pair[0]), ReverseSort(pair[1])), pair) for pair, count in self.pair_counts.items()]

        # Max heap to efficiently get the most frequent pair without having to do a full scan
        # For efficiency reasons, the heap is not immediately update when token pairs are removed
        # or when their counts are updated, since that would require an O(n) scan
        # Instead, stale entries will be removed from the heap as items are popped
        # This means that heap will at times have more entries than the pair_counts dict,
        # that's the trade-off I'm willing to make.
        # I also considered using a sorted container such as a self-balanced sorted tree, but
        # I don't think there's any in the standard lib, and I didn't want to import an external
        # library like https://github.com/grantjenks/python-sortedcontainers
        heapq.heapify(self.heap)
    
    def get_pair_count(self, pair: tuple[bytes, bytes]):
        return self.pair_counts.get(pair, 0)

    def increment_pair_count(self, pair: tuple[bytes, bytes], delta: int):
        current_count = self.pair_counts.get(pair, 0)
        self.pair_counts[pair] = current_count + delta
        self._add_to_heap(pair, current_count + delta)
    
    def set_pair_count(self, pair: tuple[bytes, bytes], count: int):
        self.pair_counts[pair] = count
        # We add a new entry in the count without deleting the existing one
        # We'll remove stale entries when items get popped
        self._add_to_heap(pair, count)
    
    def remove_pair(self, pair: tuple[bytes, bytes]):
        del self.pair_counts[pair]
        # We don't remove the pair from the heap at this point
        # since that would be an expensive scan
    
    def get_cached_best_pair(self):
        """
        Gets the most frequent pair. This method is a minor
        optimization that should only be called once,
        after the instance is constructed. After that,
        you should always call pop_best_pair instead.
        """
        if not self.heap:
            return None
        
        return self.heap[0][2]
    
    def pop_best_pair(self):
        if not self.heap:
            assert False
            return None
        
        count, _, pair = heapq.heappop(self.heap)
        actual_count = self.pair_counts.get(pair, -1)
        while count != -actual_count:
            if not self.heap:
                assert False
                return None
            # we found stale entry, discard and continue search
            count, _, pair = heapq.heappop(self.heap)
            actual_count = self.pair_counts.get(pair, -1)
        
        assert count == -actual_count
        return pair
    
    def _add_to_heap(self, pair: tuple[bytes, bytes], count: int):
        entry = (-count, (ReverseSort(pair[0]), ReverseSort(pair[1])), pair)
        heapq.heappush(self.heap, entry)

    def __getitem__(self, key: tuple[bytes, bytes]):
        return self.pair_counts[key]
    

# See: https://docs.python.org/3/library/functools.html#functools.total_ordering
@total_ordering
class ReverseSort:
    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        return self.value > other.value

    def __eq__(self, other):
        return self.value == other.value