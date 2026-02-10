import heapq
from functools import total_ordering
from collections import defaultdict
import timeit
import datetime
from .train_bpe_core import train_bpe_core

def test_train_bpe():
    print("SCENARIO: simple corpus with no special tokens")
    sample_text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
    """

    vocab_size= 256 + 1 + 6 # 256 byte values, 1 special token, 6 tokens from BPE merges

    vocab, merges = train_bpe_core(
        corpus=sample_text,
        vocab_size=vocab_size,
        special_tokens=['<|endoftext|>'],
        pretoken_regex=r"\w+")

    assert vocab[0] == b"<|endoftext|>"
    assert vocab[257] == b"st"
    assert vocab[258] == b"est"
    assert vocab[259] == b"ow"
    assert vocab[260] == b"low"
    assert vocab[261] == b"west"
    assert vocab[262] == b"ne"
    assert len(vocab) == vocab_size
    assert len(merges) == vocab_size - 256 - 1
    assert merges[0] == (b"s", b"t")
    assert merges[1] == (b"e", b"st")
    assert merges[2] == (b"o", b"w")
    assert merges[3] == (b"l", b"ow")
    assert merges[4] == (b"w", b"est")
    assert merges[5] == (b"n", b"e")

    print("Test passed!")

def test_train_bpe_special_tokens():
    print("SCENARIO: simple corpus with special token")
    sample_text = """low low low low low
<|endoftext|>
lower lower widest<|endoftext|>widest widest
newest newest newest<|endoftext|> newest newest newest<|endoftext\>
    """

    vocab_size= 256 + 1 + 6 # 256 byte values, 1 special token, 6 tokens from BPE merges

    vocab, merges = train_bpe_core(
        corpus=sample_text,
        vocab_size=vocab_size,
        special_tokens=['<|endoftext|>'],
        pretoken_regex=r"\w+")

    # for i,v in vocab.items():
    #     print(i, v)

    assert len(vocab) == vocab_size
    assert vocab[0] == b"<|endoftext|>"
    assert vocab[257] == b"st"
    assert vocab[258] == b"est"
    assert vocab[259] == b"ow"
    assert vocab[260] == b"low"
    assert vocab[261] == b"west"
    assert vocab[262] == b"ne"
    
    assert len(merges) == vocab_size - 256 - 1
    assert merges[0] == (b"s", b"t")
    assert merges[1] == (b"e", b"st")
    assert merges[2] == (b"o", b"w")
    assert merges[3] == (b"l", b"ow")
    assert merges[4] == (b"w", b"est")
    assert merges[5] == (b"n", b"e")

    print("Test passed!")

def test_train_bpe_repeating_pairs():
    print("SCENARIO: simple corpus with repeated pairs")

    sample_text ="""
fining training raining
paining training training
gaining gaining
"""

    vocab_size = 256 + 1 + 2

    vocab, merges = train_bpe_core(
        corpus=sample_text,
        vocab_size=vocab_size,
        special_tokens=['<|endoftext|>'],
        pretoken_regex=r"\w+")

    # Initial pretokens
    # - (f, i, n, i, n, g): 1
    # - (p, a, i, n, i, n, g): 1
    # - (t, r, a, i, n, i, n, g): 3
    # - (r, a, in, i, n, g): 1
    # - (g, a, i, n, i, n, g): 2
    #
    # pair_counts
    # - (f, i): 1
    # - (i, n): 16
    # - (n, i): 8
    # - (n, g): 8
    # - (p, a): 1
    # - (a, i): 7
    # - (t, r): 3
    # - (r, a): 4
    # - (g, a): 2
    #
    # best pair: (i, n)
    #
    # merge 1
    # - (f, in, in, g): 1
    # - (p, a, in, in, g): 1
    # - (t, r, a, in, in, g): 3
    # - (r, a, in, in, g): 1
    # - (g, a, in, in, g): 2
    #
    # pair_counts after merge 1
    # - (f, in): 1
    # - (in, in): 8
    # - (in, g): 8
    # - (p, a): 1
    # - (a, in): 7
    # - (t, r): 3
    # - (r, a): 4
    # - (g, a): 2
    #
    # best pair: (in, in) due to tie-breaker between (in, in) and (in, g): in > g

    assert len(vocab) == vocab_size
    assert vocab[0] == b"<|endoftext|>"
    assert vocab[257] == b"in"
    assert vocab[258] == b"inin"

    assert len(merges) == 2
    merges[0] =(b'i', b'n')
    merges[1] == (b'in', b'in')

    print("Test passed!")

def test_train_bpe_repeating_char():
    print("SCENARIO: simple corpus with repeated character")

    sample_text ="""
ooo oo oooo
ooo ooo oooo
oo ooo
"""

    vocab_size = 256 + 1 + 3

    vocab, merges = train_bpe_core(
        corpus=sample_text,
        vocab_size=vocab_size,
        special_tokens=['<|endoftext|>'],
        pretoken_regex=r"\w+")
    
    # Initial pretokens
    # - (o, o, o): 4 => the first o, o and the second o, o are considered 2 distinct pairs, so this has 8 o, o pairs
    # - (o, o): 2
    # - (o, o, o, o): 2
    # 
    # pair_counts
    # - (o, o):  16
    #
    # best pair: (o, o)
    #
    # merge 1:
    # - (oo, o): 4
    # - (oo): 2
    # - (oo, oo): 2
    #
    # pair_counts
    # - (oo, o): 4
    # - (oo, oo): 2
    #
    # best pair (oo, o)
    #
    # merge 2:
    #
    # - (ooo): 4
    # - (oo): 2
    # - (oo, oo): 2
    #
    # pair_counts
    #
    # - (oo, oo): 2
    #
    # best pair: (oo, oo)
    #
    # merge 3:
    #
    # - (ooo): 4
    # - (oo): 2
    # - (oooo): 2

    assert len(vocab) == vocab_size
    assert vocab[0] == b"<|endoftext|>"
    assert vocab[257] == b"oo"
    assert vocab[258] == b"ooo"
    assert vocab[259] == b"oooo"

    assert len(merges) == 3
    assert merges[0] == (b'o', b'o')
    assert merges[1] == (b'oo', b'o')
    assert merges[2] == (b'oo', b'oo')

    print("Test passed!")

def test_max_heap_token_pairs_ordering():
    print("SCENARIO: max heap ordering for token pair counts")
    # Since we're still running on Python 3.11
    # We don't have access to max heap APIs of the heapq module which are available in 3.14+
    # We can use min heap heap with negative counts to get the same behaviour
    # I want a heap that returns the "best pair", i.e. the pair with the max count
    # For pairs with the same count, we select the lexigrophically largest pair
    # To get this order in the min heap, we ca use negative counts, and map
    # each byte sequence to an object that reverses the sorting order

    # See: https://docs.python.org/3/library/functools.html#functools.total_ordering
    @total_ordering
    class ReverseSort:
        def __init__(self, value):
            self.value = value

        def __lt__(self, other):
            return self.value > other.value

        def __eq__(self, other):
            return self.value == other.value
    

    pair_counts = {
        (b'a', b'b'): 10,
        (b'a', b'c'): 2,
        (b'ab', b'cd'): 10,
        (b'ab', b'ce'): 10,
        (b'd', b'e'): 2,
        (b'ab', b'c'): 1,
        (b'a', b'bc'): 1
    }

    heap = [(-count, (ReverseSort(pair[0]), ReverseSort(pair[1])), pair) for pair, count in pair_counts.items()]
    
    sorted = []
    while heap:
        entry = heapq.heappop(heap)
        sorted.append(entry[2])
    
    assert sorted[0] == (b'ab', b'ce')
    assert sorted[1] == (b'ab', b'cd')
    assert sorted[2] == (b'a', b'b')
    assert sorted[3] == (b'd', b'e')
    assert sorted[4] == (b'a', b'c')
    assert sorted[5] == (b'ab', b'c')
    assert sorted[6] == (b'a', b'bc')
    [(10, (b'a', b'b')), (2, (b'a', b'c')), (10, (b'ab', b'cd')), (10, (b'ab', b'ce')), (2, (b'd', b'e')), (1, (b'ab', b'c')), (1, (b'a', b'bc'))]

    print("Test passed!")

if __name__ == "__main__": 
    test_train_bpe()
    test_train_bpe_special_tokens()
    test_train_bpe_repeating_pairs()
    test_train_bpe_repeating_char()
    test_max_heap_token_pairs_ordering()

