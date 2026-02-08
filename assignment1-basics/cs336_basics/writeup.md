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
- (i, d): 3
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
- The while loop in `merge_token_pair` calls `len(token_key)` in each nested iteration. Perhaps this could explain the 0.415s spent on `len` (10% of the runtime)

**Optimizing `merge_token_pair`: updating token cache in place**:

Here are the results after updating token cache in place

```
Running profiler for command: train_bpe("tests/fixtures/corpus.en", 500, ['<|endoftext|>'])
Sun Feb  8 02:29:51 2026    cs336_basics/profiler_results/corpus_en-2026-02-08_02-29-49

         7178661 function calls (7178563 primitive calls) in 2.322 seconds

   Ordered by: cumulative time
   List reduced from 191 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    2.322    2.322 {built-in method builtins.exec}
        1    0.000    0.000    2.322    2.322 <string>:1(<module>)
        1    0.000    0.000    2.322    2.322 train_bpe.py:3(train_bpe)
        1    0.000    0.000    2.321    2.321 train_bpe_core.py:13(train_bpe_core)
        1    0.007    0.007    2.264    2.264 train_bpe_core.py:132(merge_pairs)
      243    1.325    0.005    1.357    0.006 train_bpe_core.py:172(find_best_pair)
      243    0.739    0.003    0.900    0.004 train_bpe_core.py:196(merge_token_pair)
  6880891    0.191    0.000    0.191    0.000 {built-in method builtins.len}
        1    0.037    0.037    0.056    0.056 train_bpe_core.py:87(pretokenize)
   160785    0.013    0.000    0.013    0.000 train_bpe_core.py:105(<genexpr>)



Finsihed profiling in 2.325115 seconds. Results saved to cs336_basics/profiler_results/corpus_en-2026-02-08_02-29-49
```

Runtime reduced by 43% to 2.322 seconds. Great speed up. Incidentally, all the `test_train_bpe` unit tests pass now, including the speed test

```
tests/test_train_bpe.py::test_train_bpe_speed PASSED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED
```

But let continue with the optimization plans since they're all relatively straightforward. Before I move to the next item, let me see
if I can make small adjustments to `merge_token_pair` before moving to the next function.

Here's after making sure `len` is only called once per while loop (only in the `merge_token_pair` function), i.e. replace

```python
while i < len(token_key):
```
with

```python
token_len = len(token_key)
while i < token_len:
```

```
         2655859 function calls (2655761 primitive calls) in 1.998 seconds

   Ordered by: cumulative time
   List reduced from 191 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    1.998    1.998 {built-in method builtins.exec}
        1    0.000    0.000    1.998    1.998 <string>:1(<module>)
        1    0.000    0.000    1.998    1.998 train_bpe.py:3(train_bpe)
        1    0.000    0.000    1.997    1.997 train_bpe_core.py:13(train_bpe_core)
        1    0.007    0.007    1.939    1.939 train_bpe_core.py:132(merge_pairs)
      243    1.325    0.005    1.358    0.006 train_bpe_core.py:172(find_best_pair)
      243    0.536    0.002    0.573    0.002 train_bpe_core.py:196(merge_token_pair)
  2358089    0.068    0.000    0.068    0.000 {built-in method builtins.len}
        1    0.038    0.038    0.057    0.057 train_bpe_core.py:87(pretokenize)
   160785    0.013    0.000    0.013    0.000 train_bpe_core.py:105(<genexpr>)
```

14% speedup (from 2.3s to 1.998). I'm genuinely surprise how much gain we can get from such a small mundane change.

The new `merge_token_pair` runs one loop to collect entries to replace, and another loop to create replacement tokens,
add them to the cache and remove the old tokens. I can't replace the tokens in the same loop because I can't modify the
dictonary while iterating on it. I did try to compute the replacement tokens in the first loop such that the second
loop just does the replacement of already computed tokens, but that turned out to be slower than the current implementation
by around 150ms. I suspect that this is due to the fact that creating the replacement token in the first loop made
the loop more complex with more if statements, which adds overhead even to those entries that do not need to be replaced.

**Optimizing `find_best_pair`: Updating `pair_counts` in place.

Instead of rebuilding the `pair_counts` dictionary from scratch in each invocation of the `find_best_pair` function,
I only build the dictionary on the first invocation, then pass it around and update it in place when necessary.
So now `find_best_pair` will just take the already built dictionary and iterate through it to find the pair
with the max count. And I've moved the update logic to the `merge_token_pair` function, so the `pair_counts`
dictionary gets updated in the same loop that scans the token cache to find the entries that contain
the pair to merge. This means we don't need a separate scan to find the overlapping pairs that need to be updated.

Here are the results

```
Sun Feb  8 05:10:16 2026    cs336_basics/profiler_results/corpus_en-2026-02-08_05-10-15

         1571139 function calls (1571041 primitive calls) in 0.719 seconds

   Ordered by: cumulative time
   List reduced from 194 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.719    0.719 {built-in method builtins.exec}
        1    0.000    0.000    0.719    0.719 <string>:1(<module>)
        1    0.000    0.000    0.719    0.719 train_bpe.py:3(train_bpe)
        1    0.000    0.000    0.719    0.719 train_bpe_core.py:13(train_bpe_core)
        1    0.001    0.001    0.661    0.661 train_bpe_core.py:132(merge_pairs)
      243    0.543    0.002    0.591    0.002 train_bpe_core.py:233(merge_token_pair)
      243    0.000    0.000    0.069    0.000 train_bpe_core.py:164(find_best_pair)
      242    0.061    0.000    0.061    0.000 train_bpe_core.py:220(find_best_pair_from_pair_counts)
        1    0.037    0.037    0.056    0.056 train_bpe_core.py:87(pretokenize)
  1205443    0.035    0.000    0.035    0.000 {built-in method builtins.len}



Finsihed profiling in 0.719684 seconds. Results saved to cs336_basics/profiler_results/corpus_en-2026-02-08_05-10-15
```

Got a 64% speedup, from 1.998s to 0.719s. This is the biggest jump so far.

This version passed all the tests, but there's a part that confuses me because it seems like a bug to me.
When updating the `pair_counts` cache, sometimes I attempt to update an entry that's not in the dictionary.
It seems like a bug to me that there such cases. Here's the code that does the update:

```python
def update_pair_counts_with_merged_pair(
        pair_counts: dict[tuple[bytes, bytes], int],
        entry_to_update: tuple[bytes, bytes],
        merged_pair: bytes,
        count: int,
        index_to_replace: int):
    assert index_to_replace == 0 or index_to_replace == 1
    new_entry = (entry_to_update[0], merged_pair) if index_to_replace == 1 else (merged_pair, entry_to_update[1])
    pair_counts[new_entry] = pair_counts.get(new_entry, 0) + count

    replaced_new_count = pair_counts.get(entry_to_update, 0) - count
    # assert replaced_new_count >= 0
    if replaced_new_count == 0:
        del pair_counts[entry_to_update]
    else:
        pair_counts[entry_to_update] = replaced_new_count
```

Notice the `assert replaced_new_count >= 0`. I had placed that code to catch such "bugs", but it caused
tests to fail. Commenting out the assertion statement allowed tests to pass.

What bothers is that since the `entry_to_update` is inferred from scanning the `token_cache` to find
pairs that overlap with the merged pair, those pairs should also exist in the `pair_counts` dictionary
and should not be removed from `pair_counts` (i.e. count get to 0) while there are still pretokens in
token cache that contain the pair.

Here's the general logic in `merge_token_pair`:

```python
if token_key[i] == pair[0] and token_key[i + 1] == pair[1]:
    temp_new_token.append(merged_pair)

    # Update pair counts
    if i > 0:
        pair_to_update = (token_key[i - 1], pair[0])
        update_pair_counts_with_merged_pair(pair_counts, pair_to_update, merged_pair, count, index_to_replace=1)
    if i + 2 < token_len:
        pair_to_update = (pair[1], token_key[i + 2])
        update_pair_counts_with_merged_pair(pair_counts, pair_to_update, merged_pair, count, index_to_replace=0)
```

Perhaps there's a bug in this logic that causes it incorrectly create a `pair_to_update` that doesn't
match an existing sequence? Or maybe we removed that pair from the cache prematurely? Or maybe the
code that builds the pair count is buggy and does not add all valid pairs to the table? But if
the latter were the case, wouldn't it cause tests to fail? Well it's possible for tests to pass
if vocab size is small such that we stop iterating before we reach the step that would have
attempted to merge the missing pair.

I've debugged the failing test and found the pair missing from the `pair_counts` to be `(b'in', b'in')`.
The merge step is trying to merge the pair `(b'in', b'g')`, the former pair overlaps with the latter
in words like "def-in-in-g", "conta-in-in-g", etc. which is pretty common pattern in English. Also,
from debugging, it appears that the pair `b('in', 'in')` is  never added to the `pair_counts` dictionary.
But this subsequence `(b'in', b'in')` does exist in the token cache in entries like `(b' ', b'r', b'a', b'in', b'in', b'g')`.

Let's attempt to retrace the steps:
- The subsequence `(b'i', b'n')` gets merged to `b'in'` (confirmed from logs).
- Then the token cache needs to be updated such that `(b'i', b'n')` are replaced with `b'in'`.
- So `(b' ', b'r', b'a', b'i', b'n', b'i', b'n', b'g')` gets updated to `(b' ', b'r', b'a', b'in', b'in', b'g')`
    - In the first step we merge the first occurrence of `b'i',b'n'`: `(b' ', b'r', b'a', b'i', b'n', b'i', b'n', b'g')` gets updated to `(b' ', b'r', b'a', b'in', b'i', b'n', b'g')`
    - Then in `pair_counts` we add or update the pair `(b'a', b'in')` in `pair_counts`
    - Then we merge the second occurrence of `b'i',b'n'`, we have to update the updated `(b' ', b'r', b'a', b'in', b'i', b'n', b'g')` to `(b' ', b'r', b'a', b'in', b'in', b'g')`
    - Then we'll add `(b'in', `b'in')` to `pair_counts`. This is what's supposed to happen, but not what our code does.
    - When merging the second occurrence, the code actually still looks up the old version of the pretoken without the previous merge `(b' ', b'r', b'a', b'i', b'n', b'i', b'n', b'g')`. And so, it will add `(b'n', b'in')` to the `pair_counts` instead.
    - This is definitely a bug, but we've now added the pair `(b'n', b'in')` which doesn't exist in the pretoken cache, so even if we try to merge it, we won't be able to.
