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
Maybe they'll dominate more on large corpora, but for now let me focus on the small `corpus.en` to get the speed test to pass.
So I'll not get into parallelization at this point. Let me focus first on optimizing the `merge_pairs` function. We see
from the `tottime` column, that the `merge_token_pair`and `find_best_pair` functions stand out as hotspots. I'm also surprised
how much time `len` takes.

**Analyzing current implementation**

To see how to apply optimizations, let's recap how the merge implementation works.

Let's take the following sample corpus

```
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

After initializing the vocab with special tokens and 256 byte symbols, we'll
perform the pretokenization process to speed up the frequency counts.
This will produce the following pretokenization cache

(l,o,w): 5
(l,o,w,e,r): 2
(w,i,d,e,s,t): 3
(n,e,w,e,s,t): 6

Then in the first merge round, we'll go through the pretokenization cache to find the most common pair using
the `find_best_pair` function. This function takes 36% of the runtime, quite hot.

The `find_best_pair` function builds a `pair_counts` dictionary that has an entry for each pair of the consecutive characters
from the pretokenization cache, and the value will be the frequency of the pair. It loops through all entries
in the pretokenization cache, then slides through all consecutive pairs in each entry, adds these pairs to the `pair_counts`
dictionary while increasing their counts based on the value in the pretokenization cache. This is an O(NxM) operation that's
performed for each merge iteration. The `pair_counts` dictionary will contain the following entries in the first merge iteration:

- (l,o): 7
- (o,w): 7
- (e,r): 2
- (w, e): 8
- (w,i): 3
- (d,e): 3
- (e,s): 9
- (s,t): 9
- (n,e): 6
- (e,w): 6

`find_best_pair` will return `(s, t)` as the best pair, and its frequency 9. Note that the `pair_counts` dictionary is not returned
by the function. So the next invocation will create a new `pair_counts` dictionary from scratch.

`merge_pair` will then append this new pair `(s,t)` to the `vocab` list.

Then it will call `merge_token_pair` to merge `(s,t)` to the pretokenized cache. `merge_token_pair`
is the single biggest hotspot at 47% of the runtime. Let's see what it's doing:

First, it creates a new token cache (empty dictionary). Then it loops through the current token
cache (pretokenized cache) and copies entries from the old cache to the new cache, expect for
entries that contain the pair to merge, these entries are replaced by a new entry where the consecutive
`b's'`, `b't'` are replaced by a single merged `b'st'` object. So the new token cache will look like

(l, o, w): 5
(l, o, w, e, r): 2
(w, i, d, e, st): 3
(n, e, w, e, st): 6

`merge_token_pair` returns the new token cache.

Back in the parent `merge_pair` function, the old token cache will be discarded and replaced by the new cache
which will be used in the next operation.

The pair will also be appended to the `merges` list, which keeps track of the order in which tokens were merged.

In the next iteration, we call `find_best_pair` again. It will create a new `pair_counts` dictionary from
scratch. And after the counting loop, will generate the following table:

- (l,o): 7
- (o,w): 7
- (e,r): 2
- (w,e): 8
- (w,i): 3
- (d,e): 3
- (e,st): 9
- (n,e): 6
- (e,w): 6

And return `(e,st)` as the next pair to merge.

Note that this `pair_counts` is almost identical to the previous one, expect that the `(s,t)` entry has
been removed, and the `(e,s)` entry has been replaced with `(e,st)`.

After this, the pair will be merged and so on and so forth until we've reached the desired vocab size, or
there's nothing more to merge.

From this algorithm, we can spot some optimization opportunities:

- `merge_token_pair` can update the token cache in place instead of creating a new one
- `find_best_pair` can update the previous `pair_counts` instead of creating a new one. It just needs to update the entries that included the beginning or end of the merged pair.
- Since we append to the `vocab` and `merges` lists in each iteration, and the number of iterations is based on vocab size, perhaps we could preallocate list capacity to avoid frequend `append` calls?