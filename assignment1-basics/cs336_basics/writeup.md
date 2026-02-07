# Write-up and answers for Assignment 1

I'll use this doc to keep track of my answers to [Assignment 1 questions](../cs336_spring2025_assignment1_basics.pdf).


## 2. Byte-Pair Encoding (BPE) Tokenizer

TODO: I didn't start writing down my answers until 2.5, remember
to revisit questions in each section and write down the answers.

### 2.5. Experiment with BPE Tokenizer Training

#### Problem (`train_bpe`): BPE Tokenizer Training 

I implemented the BPE tokenizer, with support for special tokens.

The core implementation is in the [`train_bpe_core`](./train_bpe_core.py) function.

I updated the [`train_bpe`](./train_bpe.py) to call the core implementation.

I create the [`playground.py`](./playground.py) file for running sanity checks and debugging issues.

```bash
# cd to the parent of cs336_basics first (i.e. the assignment1-basics folder)
uv run python -m cs336_basics.playground
```

The implementation passed the correctness tests, but fails the performance test

```bash
# cd to assignment1-basics folder
uv pytest tests/test_train_bpe.py
```

```
tests/test_train_bpe.py::test_train_bpe_speed FAILED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED
```

The assignment instructions provide the following tips to address the perf issue

> Optimizing the merging step The naïve implementation of BPE training in the stylized example above
is slow because for every merge, it iterates over all byte pairs to identify the most frequent pair. However,
the only pair counts that change after each merge are those that overlap with the merged pair. Thus,
BPE training speed can be improved by indexing the counts of all pairs and incrementally updating these
counts, rather than explicitly iterating over each pair of bytes to count pair frequencies. You can get
significant speedups with this caching procedure, though we note that the merging part of BPE training is
not parallelizable in Python.

> You should use profiling tools like `cProfile` or `scalene` to identify the bottlenecks in your imple-
mentation, and focus on optimizing those.

