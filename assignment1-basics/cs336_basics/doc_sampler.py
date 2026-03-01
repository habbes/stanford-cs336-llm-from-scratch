

import random

def sample_docs(path: str, num_samples: int, sep: str = '<|endoftext|>', random: bool = False) -> str:
    return sample_docs_random(path, num_samples, sep) if random else sample_docs_sequential(path, num_samples, sep)


def sample_docs_sequential(path: str, num_samples: int, sep: str = '<|endoftext|>') -> str:
    """
    Extracts the specified number of sample documents from the text corpus at the specified
    path using sep as the separator between documents.
    """
    if num_samples <= 0:
        return ""

    chunks: list[str] = []
    sep_count = 0
    chunk_size = 1 << 16

    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            chunks.append(chunk)
            sep_count += chunk.count(sep)
            if sep_count >= num_samples:
                break

    corpus = "".join(chunks)

    end = 0
    for _ in range(num_samples):
        idx = corpus.find(sep, end)
        if idx == -1:
            return corpus
        end = idx + len(sep)

    return corpus[:end]


def sample_docs_random(path: str, num_samples: int, sep: str = '<|endoftext|>', seed: int | None = None) -> str:
    """
    Extracts a random subset of documents from the text corpus at path using sep as the
    separator between documents, and returns them concatenated with sep included.

    Uses streaming reservoir sampling to avoid loading the entire corpus in memory.
    """
    if num_samples <= 0:
        return ""

    rng = random.Random(seed)
    reservoir: list[str] = []
    seen_docs = 0

    chunk_size = 1 << 16
    buffer = ""

    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            buffer += chunk
            start = 0

            while True:
                idx = buffer.find(sep, start)
                if idx == -1:
                    break

                doc = buffer[start:idx]
                seen_docs += 1

                if len(reservoir) < num_samples:
                    reservoir.append(doc)
                else:
                    j = rng.randint(1, seen_docs)
                    if j <= num_samples:
                        reservoir[j - 1] = doc

                start = idx + len(sep)

            buffer = buffer[start:]

    # Handle final trailing document when the corpus does not end with sep.
    if buffer:
        seen_docs += 1
        if len(reservoir) < num_samples:
            reservoir.append(buffer)
        else:
            j = rng.randint(1, seen_docs)
            if j <= num_samples:
                reservoir[j - 1] = buffer

    if not reservoir:
        return ""

    rng.shuffle(reservoir)
    return sep.join(reservoir) + sep