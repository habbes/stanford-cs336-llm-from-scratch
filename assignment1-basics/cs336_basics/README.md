This folder contains my code implement problem solutions and projects for [homework 1](../cs336_spring2025_assignment1_basics.pdf).

The detailed reports for this section can be found in `writeup-*` files in this folder, such as [writeup-2-bpe.md](./writeup-2-bpe.md) file.
The docs contain
answers to problems in the [assignment document](../cs336_spring2025_assignment1_basics.pdf), but it's
more of a log that I update based on my thought process, insights, experiments and learnigns as I go.
Generally, there's a separate writeup doc for each chapter of the assignment.

To run scripts and code in this folder, navigate to the [parent folder](../)
and run the scripts as python modules:

- The `playground.py` module has some sanity checks I use for debugging

```bash
uv run python -m cs336_basics.playground
```

- The `perf_tests.py` module contains profiler tests

```bash
uv run python -m cs336_basics.perf_tests corpus_en
```

- Inspect the profiler results of a perf tests:

Start a python interpreter:

```bash
uv run python
```

Use the [`pstats`](https://docs.python.org/3/library/profile.html#module-pstats) module to
inspect the stats:

```python
import pstats
from pstats import SortKey

path = "cs336_basics/profiler_results/corpus_en-2026-02-09_21-12-43"
results = pstats.Stats(path)
results.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)
```

- Run tokenizer on a specified data set using the [`run_train_bpe`](./run_train_bpe.py) script:

```bash
uv run python -m cs336_basics.run_train_bpe data/TinyStoriesV2-GPT4-valid.txt
```

```
Training tokenizer, corpus: data/TinyStoriesV2-GPT4-valid.txt, vocab size: 10000, special tokens: ['<|endoftext|>']
Completed tokenizer training in 1.768303s
Saved vocab JSON file at output/TinyStoriesV2-GPT4-valid-vocab.json
Saved merges JSON file at output/TinyStoriesV2-GPT4-valid-merges.json
```

By default, this sets a vocab size of 10,000 tokens and sets the following
list as special tokens: `['<|endoftext|>']`. It saves a serialized vocab
and merges as JSON files in the [`output`](../output) folder by default.

You can also specify vocab size and special tokens to the list:

```bash
uv run python -m cs336_basics.run_train_bpe data/TinyStoriesV2-GPT4-valid.txt -v 5000 -s '<|endoftext|>' -s '<|foo|>' 
```

```
Training tokenizer, corpus: data/TinyStoriesV2-GPT4-valid.txt, vocab size: 5000, special tokens: ['<|endoftext|>', '<|foo|>']
```

You can specify the output dir where the vocab and merges files will be saved.

```bash
uv run python -m cs336_basics.run_train_bpe data/TinyStoriesV2-GPT4-valid.txt -o path/output/dir
```

The [`resource_accounting.py`](./resource_accounting.py) module contains helpers for counting params, flops, etc. and other
resources related to training and running the model. It also computes and prints answers related to resource
accounting questions in the assignment:

```bash
uv run python -m cs336_basics.resource_accounting
```

## Run official tests

### BPE training tests

```bash
uv run pytest tests/test_train_bpe.py
```

### Test specfic test function in test suite:

```bash
uv run pytest tests/test_train_bpe.py::test_train_bpe
```

By default truncated diffs are displayed for failed tests. To show full diff, use the `-vv` option:

```bash
uv run pytest tests/test_train_bpe.py::test_train_bpe -vv
```

### Tokenizer tests

```bash
uv run pytest tests/test_tokenizer.py
```

### Linear module tests

```bash
uv run pytest -k test_linear
```

### Embedding module tests

```sh
uv run pytest -k test_embedding
```

### RMSNorm tests

```sh
uv run pytest -k test_rmsnorm
```

### SwiGLU tests

```sh
uv run pytest -k test_swiglu
```

### RoPE tests

```sh
uv run pytest -k test_rope
```

### Softmax tests

```sh
uv run pytest -k test_softmax_matches_pytorch
```

### Scaled dot product attention tests

```sh
uv run pytest -k test_4d_scaled_dot_product_attention
```

### MultiHeadSelfAttention tests

```sh
uv run pytest -k test_multihead_self_attention
```