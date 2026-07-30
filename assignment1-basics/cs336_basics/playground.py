import heapq
import math
from functools import total_ordering
from collections import defaultdict
import os
import timeit
import datetime
from .train_bpe_core import train_bpe_core_str
from .tokenizer import Tokenizer
from .bpe_common import dump_bpe_merges, load_bpe_merges, dump_bpe_vocab, load_bpe_vocab
from .nn_modules import Linear, Embedding
from .resource_accounting import get_gpt2_xl_config, create_transformer_params_counter
import multiprocessing as mp
import random

def test_train_bpe():
    print("SCENARIO: simple corpus with no special tokens")
    sample_text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
    """

    vocab_size= 256 + 1 + 6 # 256 byte values, 1 special token, 6 tokens from BPE merges

    vocab, merges = train_bpe_core_str(
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
newest newest newest<|endoftext|> newest newest newest<|endoftext|>
    """

    vocab_size= 256 + 1 + 6 # 256 byte values, 1 special token, 6 tokens from BPE merges

    vocab, merges = train_bpe_core_str(
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

    vocab, merges = train_bpe_core_str(
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

    vocab, merges = train_bpe_core_str(
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
        
        def __repr__(self):
            return f"ReverseSort({self.value})"
        
        def __str__(self):
            return self.__repr__()
    

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
    heapq.heapify(heap)
    
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

def add_tuple(args):
    x, y = args
    return x + y

def add(x, y):
    return x + y

def test_multiprocessing_pool():
    print("SCENARIO: simple multiprocessing pool")
    args = [(1, 2), (4, 5), (30, 50)]

    with mp.Pool() as pool:
        results = pool.imap_unordered(add_tuple, args)

        assert 3 in results
        assert 9 in results
        assert 80 in results

        results = pool.starmap(add, args)
        assert 3 in results
        assert 9 in results
        assert 80 in results
    
    print("Test passed!")

def test_simple_tokenizer_encoding():
    print("SCENARIO: Simple tokenizer encoding")
    text = "the cat ate"
    vocab = {
        0: b' ',
        1: b'a',
        2: b'c',
        3: b'e',
        4: b'h',
        5: b't',
        6: b'th',
        7: b' c',
        8: b' a',
        9: b'the',
        10: b' at'
    }

    merges = [
        (b't', b'h'),
        (b' ', b'c'),
        (b' ', b'a'),
        (b'th', b'e'),
        (b' a', b't')
    ]

    tokenizer = Tokenizer(vocab, merges)
    encoded = tokenizer.encode(text)
    
    # text will be pretokenized into:
    # ['the', ' cat', ' ate']
    # and as sequence of bytes:
    # [
    #   (b't', b'h', b'e'),
    #   (b' ', b'c', b'a', b't'),
    #   (b' ', b'a', b't', b'e')
    # ]
    # each pretoken will be independently encoded based
    # merges list order.
    # For each pretoken, apply the merges in order until we can't merge no more,
    # then convert to token ID
    # 'the':
    # - (b't', b'h', b'e') -- merge t-h -> (b'th', b'e') -- merge th e -> (b'the') -- token IDs -> [9]
    # - (b' ', b'c', b'a', b't') -- merge ' '-'c' -> (b' c', b'a', b't') -- token IDs -> [7, 1, 5]
    # - (b' ', b'a', b't', b'e') -- merge ' '-a -> (b' a', b't', b'e') -- merge ' a'-t -> (b' at', b'e') -- token IDs -> [10, 3]
    # encoded: [9, 7, 1, 5, 10, 3]
    expected = [9, 7, 1, 5, 10, 3]
    assert len(encoded) == len(expected), f"expected {expected} but got {encoded}"
    for i in range(len(encoded)):
        assert encoded[i] == expected[i], f"encoded tokens differ at pos {i}, {encoded[i]} != {expected[i]}. Expected = {expected}, Got = {encoded}"
    
    print("Test passed!")

def test_simple_tokenizer_encoding_with_special_tokens():
    print("SCENARIO: Simple tokenizer encoding with special tokens")
    text = "the<|endoftext|> cat ate"
    vocab = {
        0: b'<|endoftext|>',
        1: b' ',
        2: b'a',
        3: b'c',
        4: b'e',
        5: b'h',
        6: b't',
        7: b'th',
        8: b' c',
        9: b' a',
        10: b'the',
        11: b' at'
    }

    merges = [
        (b't', b'h'),
        (b' ', b'c'),
        (b' ', b'a'),
        (b'th', b'e'),
        (b' a', b't')
    ]

    special_tokens = ['<|endoftext|>']

    tokenizer = Tokenizer(vocab, merges, special_tokens)
    encoded = tokenizer.encode(text)
    
    # Special tokens are prepended to vocab.
    # text will be pretokenized into:
    # ['the','<|endoftext|>', ' cat', ' ate']
    # and as sequence of bytes:
    # [
    #   (b't', b'h', b'e'),
    #   (b'<|endoftext|>'),  # special token not split
    #   (b' ', b'c', b'a', b't'),
    #   (b' ', b'a', b't', b'e')
    # ]
    # After merges we'll get
    # - (b'the')
    # - (b'<|endoftext|>')
    # - (b' c', b'a', b't')
    # - (b' at', b'e')
    # So after mapping to token IDs, we'll get
    # -> (b'the') -> [10]
    # -> (b'<|endoftext|>') -> [0]
    # -> (b' c', b'a', b't') -> [8, 2, 6]
    # -> (b' at', b'e') -> [11, 4]

    expected = [10, 0, 8, 2, 6, 11, 4]
    assert len(encoded) == len(expected), f"expected {expected} but got {encoded}"
    for i in range(len(encoded)):
        assert encoded[i] == expected[i], f"encoded tokens differ at pos {i}, {encoded[i]} != {expected[i]}. Expected = {expected}, Got = {encoded}"
    
    print("Test passed!")

def test_simple_tokenizer_encoding_with_overlapping_special_tokens():
    print("SCENARIO: Simple tokenizer encoding with overlapping special tokens")
    text = "the<|endoftext|><|endoftext|> cat<|endoftext|> ate"
    vocab = {
        0: b'<|endoftext|>',
        1: b'<|endoftext|><|endoftext|>',
        2: b' ',
        3: b'a',
        4: b'c',
        5: b'e',
        6: b'h',
        7: b't',
        8: b'th',
        9: b' c',
        10: b' a',
        11: b'the',
        12: b' at'
    }

    merges = [
        (b't', b'h'),
        (b' ', b'c'),
        (b' ', b'a'),
        (b'th', b'e'),
        (b' a', b't')
    ]

    special_tokens = ['<|endoftext|>', '<|endoftext|><|endoftext|>']

    tokenizer = Tokenizer(vocab, merges, special_tokens)
    encoded = tokenizer.encode(text)

    # -> (b'the') -> [11]
    # -> (b'<|endoftext|><|endoftext|>') -> [1]
    # -> (b' c', b'a', b't') -> [9, 3, 7]
    # -> (b'<|endoftext|>') -> [0]
    # -> (b' at', b'e') -> [12, 5]

    expected = [11, 1, 9, 3, 7, 0, 12, 5]
    assert len(encoded) == len(expected), f"expected {expected} but got {encoded}"
    for i in range(len(encoded)):
        assert encoded[i] == expected[i], f"encoded tokens differ at pos {i}, {encoded[i]} != {expected[i]}. Expected = {expected}, Got = {encoded}"
    
    print("Test passed!")

def test_simple_tokenizer_decoding():
    print("SCENARIO: Simple tokenizer decoding")

    text = "the cat ate"
    vocab = {
        0: b' ',
        1: b'a',
        2: b'c',
        3: b'e',
        4: b'h',
        5: b't',
        6: b'th',
        7: b' c',
        8: b' a',
        9: b'the',
        10: b' at'
    }

    merges = [
        (b't', b'h'),
        (b' ', b'c'),
        (b' ', b'a'),
        (b'th', b'e'),
        (b' a', b't')
    ]

    tokenizer = Tokenizer(vocab, merges)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text

    print("Test passed!")

def test_simple_tokenizer_decoding_with_special_tokens():
    print("SCENARIO: Simple tokenizer decoding with special tokens")

    text = "the<|endoftext|> cat ate"
    vocab = {
        0: b' ',
        1: b'a',
        2: b'c',
        3: b'e',
        4: b'h',
        5: b't',
        6: b'th',
        7: b' c',
        8: b' a',
        9: b'the',
        10: b' at',
        11: b'<|endoftext|>'
    }

    merges = [
        (b't', b'h'),
        (b' ', b'c'),
        (b' ', b'a'),
        (b'th', b'e'),
        (b' a', b't')
    ]

    tokenizer = Tokenizer(vocab, merges, special_tokens=['<|endoftext|>'])
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text

    print("Test passed!")

def test_tokenizer_encoding_multiple_merges_in_word():
    print("SCENARIO: Tokenizer encoding with the same pair is merged multiple times")

    vocab = {
        0: b' ',
        1: b'c',
        2: b'd',
        3: b'e',
        4: b'a',
        5: b't',
        6: b'i',
        7: b' i',
        8: b'at',
        9: b'ed',
        10: b' d',
        11: b'ic',
        12: b'ate',
        13: b'ated',
        14: b' ded',
        15: b'icated',
        16: b'icate',
        17: b' dedicated',
        18: b' dedicate'
    }

    merges = [
        (b' ', b'i'),
        (b'a', b't'),
        (b'e', b'd'),
        (b' ', b'd'),
        (b'i', b'c'),
        (b'at', b'e'),
        (b'at', b'ed'),
        (b' d', b'ed'),
        (b'ic', b'ated'),
        (b'ic', b'ate'),
        (b' ded', b'icated'),
        (b' ded', b'icate')
    ]

    tokenizer = Tokenizer(vocab, merges)

    text = 'i dedicated i dedicate'
    encoded = tokenizer.encode(text)

    # pretokens: [(i), (' ',d,e,d,i,c,a,t,e,d), (' ',i), (' ',d,e,d,i,c,a,t,e)]
    # merges (skipped intermiedate merges):
    # - [('i'), (' ded','icate', 'd'), (' i'), (' ded', 'icate')]
    # - [('i'), (' ded','icated'), (' i'), (' ded', 'icate')]
    # - [(i), (' dedicated'), (' i'), (' dedicate')]
    # token ids:
    # - [6, 17, 7, 18]
    expected = [6, 17, 7, 18]
    assert len(encoded) == len(expected), f"Expected {expected} but got {encoded}"
    for i in range(len(encoded)):
        assert encoded[i] == expected[i], f"Mismatch at position {i}, {encoded[i]} != {expected[i]}. Expected {expected}, Got: {encoded}"

    decoded = tokenizer.decode(encoded)
    assert decoded == text

    print("Test passed!")

def test_simple_tokenizer_encode_iterable():
    print("SCENARIO: Simple tokenizer iterable encoding")
    vocab = {
        0: b' ',
        1: b'a',
        2: b'c',
        3: b'e',
        4: b'h',
        5: b't',
        6: b'th',
        7: b' c',
        8: b' a',
        9: b'the',
        10: b' at'
    }

    merges = [
        (b't', b'h'),
        (b' ', b'c'),
        (b' ', b'a'),
        (b'th', b'e'),
        (b' a', b't')
    ]

    tokenizer = Tokenizer(vocab, merges)

    text_iterable = ["the ", "cat ", "ate"]

    encoded_iterable = tokenizer.encode_iterable(text_iterable, chunk_size=5)
    encoded_list = list(encoded_iterable)

    expected = [9, 7, 1, 5, 10, 3]
    assert len(encoded_list) == len(expected), f"Expected {expected} but got {encoded_list}"
    for i in range(len(expected)):
        assert encoded_list[i] == expected[i], f"Mismatch at {i}, {encoded_list[i]} != {expected[i]}. Expected: {expected}, Got: {encoded_list}"

    decoded = tokenizer.decode(encoded_list)
    
    expected_text = "the cat ate"
    assert decoded == expected_text

    print("Test passed!")

def test_simple_tokenizer_encode_iterable_with_special_tokens():
    print("SCENARIO: Simple tokenizer iterable encoding with special tokes")
    vocab = {
        0: b' ',
        1: b'a',
        2: b'c',
        3: b'e',
        4: b'h',
        5: b't',
        6: b'th',
        7: b' c',
        8: b' a',
        9: b'the',
        10: b' at',
        11: b'<|endoftext|>'
    }

    merges = [
        (b't', b'h'),
        (b' ', b'c'),
        (b' ', b'a'),
        (b'th', b'e'),
        (b' a', b't')
    ]

    tokenizer = Tokenizer(vocab, merges, ['<|endoftext|>'])

    text_iterable = ["the<|endo", "ftext|> cat ", "ate"]

    encoded_iterable = tokenizer.encode_iterable(text_iterable, chunk_size=5)
    encoded_list = list(encoded_iterable)

    expected = [9, 7, 1, 5, 10, 3]
    assert len(encoded_list) == len(expected), f"Expected {expected} but got {encoded_list}"
    for i in range(len(expected)):
        assert encoded_list[i] == expected[i], f"Mismatch at {i}, {encoded_list[i]} != {expected[i]}. Expected: {expected}, Got: {encoded_list}"

    decoded = tokenizer.decode(encoded_list)
    
    expected_text = "the cat ate"
    assert decoded == expected_text

    print("Test passed!")

def test_bpe_vocab_serializer_roundtrip():
    print("SCENARIO: BPE vocab serialization roundtrip")

    vocab = {
        0: b' ',
        1: b'foo',
        2: b'\\x10'
    }

    path = "output/temp-test-bpe-vocab.json"
    try:
        os.remove(path)
    except:
        pass

    dump_bpe_vocab(vocab, path)
    loaded = load_bpe_vocab(path)

    assert len(loaded) == len(vocab), f"Expected {vocab}, Got {loaded}"
    for k, v in vocab.items():
        assert v == loaded[k], f"Key mismatch {k}, {v} != {loaded[k]}. Expected {vocab}, Got {loaded}"
    
    print("Test passed!")

def test_bpe_merges_serializer_roundtrip():
    print("SCENARIO: BPE merges serialization roundtrip")

    merges = [
        (b'foo', b' bar'),
        (b'\x80', b' baz')
    ]

    path = "output/temp-test-bpe-merges.json"
    try:
        os.remove(path)
    except:
        pass

    dump_bpe_merges(merges, path)
    loaded = load_bpe_merges(path)

    assert len(loaded) == len(merges), f"Expected {merges}, Got {loaded}"
    for i in range(len(merges)):
        assert merges[i] == loaded[i], f"Mismatch at {i}, {merges[i]} != {loaded[i]}. Expected {merges}, Got {loaded}"
    
    print("Test passed!")

def test_linear_module_initialization():
    print("SCENARIO: Verify Linear module weights are initialized with expected distribution")
    
    for _ in range(10):
        d_in, d_out = random.randint(1000, 2000), random.randint(1000, 2000)
        l = Linear(d_in, d_out)
        target_std = math.sqrt(2 / (d_in + d_out))
        target_mean = 0
        target_min, target_max = -3 * target_std, 3 * target_std

        mean = l.weights.mean()
        std = l.weights.std()
        max = l.weights.max()
        min = l.weights.mean()

        std_diff = abs(std - target_std)
        mean_diff = abs(mean - target_mean)

        assert min >= target_min, f"Got min value {min} lower than target min {target_min} for Linear module with d_in={d_in}, d_out={d_out}"
        assert max <= target_max, f"Got max value {max} greater than target max {target_max} for Linear module with d_in={d_in}, d_out={d_out}"
        assert mean_diff < 0.1, f"Got mean {mean}, but expected {target_mean} (diff = {mean_diff}) for Linear module with d_in={d_in}, d_out={d_out}"
        assert std_diff < 0.1, f"Got std {std}, but expected {target_std} (diff = {std_diff}) for Linear module with d_in={d_in}, d_out={d_out}"

    print("Test passed!")

def test_embedding_module_initialization():
    print("SCENARIO: Verify Embedding module weights are initialized with expected distribution")

    for _ in range(10):
        d_vocab, d_model = random.randint(1000, 2000), random.randint(1000, 2000)
        e = Embedding(d_vocab, d_model)
        target_mean = 0
        target_std = 1
        target_max = 3
        target_min = -3

        mean, std, min, max = e.weights.mean(), e.weights.std(), e.weights.min(), e.weights.max()
        mean_diff = abs(mean - target_mean)
        std_diff = abs(std - target_std)

        assert min >= target_min, f"Got min value {min} lower than target min {target_min} for Embedding module with d_vocab={d_vocab}, d_model={d_model}"
        assert max <= target_max, f"Got max value {max} greater than target max {target_max} for Embedding module with d_vocab={d_vocab}, d_model={d_model}"
        assert mean_diff < 0.1, f"Got mean {mean}, but expected {target_mean} (diff = {mean_diff}) for Embedding module with d_vocab={d_vocab}, d_model={d_model}"
        assert std_diff < 0.1, f"Got std {std}, but expected {target_std} (diff = {std_diff}) for Linear module with d_vocab={d_vocab}, d_model={d_model}"
    
    print("Test passed!")

def test_resource_counter():
    print("SCENARIO: Verify resource counter for GPT2-XL trainable params")
    config = get_gpt2_xl_config()
    counter = create_transformer_params_counter()
    
    count = counter.get_resource_count(config)

    expected = 1640531200
    assert count == expected, f"Got {count}, but expected {expected} trainable params"

    print("Test passed!")
    

if __name__ == "__main__": 
    test_train_bpe()
    test_train_bpe_special_tokens()
    test_train_bpe_repeating_pairs()
    test_train_bpe_repeating_char()
    test_max_heap_token_pairs_ordering()
    test_multiprocessing_pool()
    test_simple_tokenizer_encoding()
    test_simple_tokenizer_encoding_with_special_tokens()
    test_simple_tokenizer_encoding_with_overlapping_special_tokens()
    test_simple_tokenizer_decoding()
    test_simple_tokenizer_encoding_with_special_tokens()
    test_tokenizer_encoding_multiple_merges_in_word()
    test_simple_tokenizer_encode_iterable()
    test_simple_tokenizer_decoding_with_special_tokens()
    test_bpe_vocab_serializer_roundtrip()
    test_bpe_merges_serializer_roundtrip()
    test_linear_module_initialization()
    test_embedding_module_initialization()
    test_resource_counter()

