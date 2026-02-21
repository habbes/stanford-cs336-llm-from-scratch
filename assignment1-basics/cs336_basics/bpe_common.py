import json

def dump_bpe_vocab(vocab: dict[int, bytes], path: str):
    """
    Serializes and writes the BPE vocab tokens data
    to the file at the specified path
    """
    vocab_decoded = { k: str(v) for k,v in vocab.items() }
    with open(path, 'w') as f:
        json.dump(vocab_decoded, f)

def load_bpe_vocab(path: str) -> dict[int, bytes]:
    """
    Loads and deserializes a BPE vocab dictionary
    from the specified path.
    """
    with open(path, 'r') as f:
        vocab_raw = json.load(f)
        # TODO using eval here is SECURITY VULNERABILITY, change serialization/deserialization to avoid it
        vocab_decoded = { int(id): eval(raw_token) for id, raw_token in vocab_raw.items() }
        return vocab_decoded

def dump_bpe_merges(merges: list[tuple[bytes, bytes]], path: str):
    """
    Serializes and writes the BPE merges to the file
    at the specified path.
    """
    merges_decoded = [(str(a), str(b)) for a, b in merges]
    with open(path, 'w') as f:
        json.dump(merges_decoded, f)

def load_bpe_merges(path: str) -> list[tuple[bytes, bytes]]:
    """
    Loads and deserializes a BPE merge list
    from the specified path.
    """
    with open(path, 'r') as f:
        merges_raw = json.load(f)
        # TODO using eval here is SECURITY VULNERABILITY, change serialization/deserialization to avoid it
        merges_decoded = [(eval(t1), eval(t2)) for t1, t2 in merges_raw]
        return merges_decoded