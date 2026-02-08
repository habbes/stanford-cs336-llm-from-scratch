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
    

if __name__ == "__main__": 
    test_train_bpe()
    test_train_bpe_special_tokens()
    test_train_bpe_repeating_pairs()
    