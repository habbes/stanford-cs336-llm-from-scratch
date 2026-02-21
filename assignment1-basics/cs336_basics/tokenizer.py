import regex as re
from typing import Optional
from .train_bpe_core import COMPILED_PRETOKEN_RE, BYTE_TABLE

def get_utf8_bytes_tuple(text: str) -> tuple[bytes]:
    encoded = text.encode("utf-8")
    return tuple(BYTE_TABLE[b] for b in encoded)

def pretokenize(text: str) -> list[tuple[bytes]]:
    pretokens = COMPILED_PRETOKEN_RE.finditer(text)
    result = [get_utf8_bytes_tuple(match.group(0)) for match in pretokens]
    return result

def try_merge(pretoken: tuple[bytes], merge: tuple[bytes, bytes]) -> Optional[tuple[bytes]]:
    t1, t2 = merge
    merged = t1 + t2

    temp_new_tokens: list[bytes] = None
    length = len(pretoken)

    # Find first occurence of pair to merge, then create new temp list
    i = 0
    while i < length - 1:
        if pretoken[i] == t1 and pretoken[i + 1] == t2:
            temp_new_tokens = [item for item in pretoken[:i]] # copy all previous bytes
            temp_new_tokens.append(merged)
            i += 2
            break
        i += 1
    
    if temp_new_tokens:
        # i is positioned right after the first merged pair
        while i < length:
            new_i = len(temp_new_tokens) - 1
            if temp_new_tokens[new_i] == t1 and pretoken[i] == t2:
                temp_new_tokens.append(merged)
            else:
                temp_new_tokens.append(pretoken[i])
            i += 1
        
        return tuple(temp_new_tokens)
    

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[bytes] = None):
        self.vocab = vocab
        self.token_ids = { token : id for id, token in vocab.items() }
        self.merges = merges
        self.special_tokens = special_tokens

    def encode(self, text: str) -> list[int]:
        pretokens = pretokenize(text)
        
        # Each pretoken is encoded independently, we don't perform
        # any merges across pretoken boundaries
        # TODO: this means we can process pretokens in parallel
        result = [token for pretoken in pretokens for token in self._encode_pretoken(pretoken)]
        return result

    def _encode_pretoken(self, pretoken: tuple[bytes]) -> list[int]:
        result = pretoken
        for merge in self.merges:
            merge_result = try_merge(result, merge)
            if merge_result is not None:
                result = merge_result
                # short-circuit if the result is only 1 token
                if len(result) == 1:
                    break
        
        encoded = [self.token_ids[token] for token in result]
        return encoded

    
    

