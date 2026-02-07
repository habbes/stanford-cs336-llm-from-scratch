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

The assignment instructions provide the following tips to address the perf issues

> **Parallelizing pre-tokenization**. You will find that a major bottleneck is the pre-tokenization step. You
can speed up pre-tokenization by parallelizing your code with the built-in library multiprocessing. Con-
cretely, we recommend that in parallel implementations of pre-tokenization, you chunk the corpus while
ensuring your chunk boundaries occur at the beginning of a special token. You are free to use the starter
code at the following link verbatim to obtain chunk boundaries, which you can then use to distribute work
across your processes

> **Optimizing the merging step**. The naïve implementation of BPE training in the stylized example above
is slow because for every merge, it iterates over all byte pairs to identify the most frequent pair. However,
the only pair counts that change after each merge are those that overlap with the merged pair. Thus,
BPE training speed can be improved by indexing the counts of all pairs and incrementally updating these
counts, rather than explicitly iterating over each pair of bytes to count pair frequencies. You can get
significant speedups with this caching procedure, though we note that the merging part of BPE training is
not parallelizable in Python.

> You should use profiling tools like `cProfile` or `scalene` to identify the bottlenecks in your imple-
mentation, and focus on optimizing those.

*Profiling**

I create the [`perf_tests.py`](./perf_tests.py) based on the built-in [`cProfile`](https://docs.python.org/3/library/profile.html) module.

Run the script from the parent of the `cs336_basics` folder (i.e. `assignment1-basics`)

```sh
uv run python -m cs336_basics.perf_tests corpus_en
```

This runs the profiler against the `train_bpe` function on the [`corpus.en`](../tests/fixtures/corpus.en) text fixture corpus
and vocab size of 500 tokens. This is small enough that it only takes a few seconds to run even with an unoptimized tokenizer.

Here are the results on my Macbook M1 Pro with 32GB RAM:

```
uv run python -m cs336_basics.perf_tests corpus_en
Running profiler for command: train_bpe("tests/fixtures/corpus.en", 500, ['<|endoftext|>'])
Sat Feb  7 23:59:34 2026    cs336_basics/profiler_results/corpus_en-2026-02-07_23-59-30

         19489037 function calls (19488939 primitive calls) in 4.103 seconds

   Ordered by: cumulative time
   List reduced from 190 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    4.103    4.103 {built-in method builtins.exec}
        1    0.000    0.000    4.103    4.103 <string>:1(<module>)
        1    0.000    0.000    4.103    4.103 train_bpe.py:3(train_bpe)
        1    0.000    0.000    4.102    4.102 train_bpe_core.py:13(train_bpe_core)
        1    0.024    0.024    4.044    4.044 train_bpe_core.py:132(merge_pairs)
      243    1.936    0.008    2.486    0.010 train_bpe_core.py:196(merge_token_pair)
      243    1.500    0.006    1.533    0.006 train_bpe_core.py:172(find_best_pair)
 13622607    0.415    0.000    0.415    0.000 {built-in method builtins.len}
  5640689    0.169    0.000    0.169    0.000 {method 'append' of 'list' objects}
        1    0.038    0.038    0.057    0.057 train_bpe_core.py:87(pretokenize)



Finsihed profiling in 4.103456 seconds. Results saved to cs336_basics/profiler_results/corpus_en-2026-02-07_23-59-30
```

This shows a summary of the top 10 functions by cumulative time:

- `ncalls`: the number of calls.
- `tottime`: the total time spent in the given function (and excluding time made in calls to sub-functions)
- `percall`: the quotient of tottime divided by ncalls
- `cumtime`: the cumulative time spent in this and all subfunctions (from invocation till exit). This figure is accurate even for recursive functions.
- `percall` the quotient of cumtime divided by primitive calls
- `filename:lineno(function)`: provides the respective data of each function

The scripts also saves the profiler results to a file in the `cs336_basics/profiler_results` directory. The file name
is displayed at the end of the script output above. You can open an inspect the file using the [**`pstats`**](https://docs.python.org/3/library/profile.html#module-pstats) module:

```python
import pstats
from pstats import SortKey

file = 'cs336_basics/profiler_results/corpus_en-2026-02-07_23-59-30'
p = pstats.Stats(file)
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)
```

Now let's analyze the results we got. For this scenario, the total runtime was about 4.103 seconds, the majority of the time (98.6%)
was spent in the `merge_pairs` function. The initialization, splitting and pretokenization code don't seem to make much of a dent here.
So I'll not get into parallelization at this point. Let me focus first on optimizing the mergint step.