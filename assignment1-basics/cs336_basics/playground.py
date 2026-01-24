from naive_bpe import naive_bpe

def test_naive_bpe():
    print("Testing scenario: simple corpus with no special tokens")
    sample_text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
    """

    vocab_size= 256 + 1 + 6 # 256 byte values, 1 special token, 6 tokens from BPE merges

    vocab, merges = naive_bpe(
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

    print("Test passed!")

if __name__ == "__main__": 
    test_naive_bpe()
    