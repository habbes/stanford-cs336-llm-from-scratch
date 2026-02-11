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

**Investigaging bug**

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

I've created a simple test to reproduce this issue in the `playground.py` using the sample text:

```
fining training raining
paining training training
gaining gaining
```

and sample with the same repeating character

```
ooo oo oooo
ooo ooo oooo
oo ooo
```

```bash
uv run python -m cs336_basics.playground
```

**Running profiler after fixing bug**

I've fixed the bug, all tests pass and I've restored the assertions I had removed. Now let
me run the profiler again to see if I still have the same performance:

```
uv run python -m cs336_basics.perf_tests corpus_en
Running profiler for command: train_bpe("tests/fixtures/corpus.en", 500, ['<|endoftext|>'])
Mon Feb  9 18:25:11 2026    cs336_basics/profiler_results/corpus_en-2026-02-09_18-25-10

         1548578 function calls (1548480 primitive calls) in 0.715 seconds

   Ordered by: cumulative time
   List reduced from 194 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.715    0.715 {built-in method builtins.exec}
        1    0.000    0.000    0.715    0.715 <string>:1(<module>)
        1    0.000    0.000    0.715    0.715 train_bpe.py:3(train_bpe)
        1    0.000    0.000    0.715    0.715 train_bpe_core.py:13(train_bpe_core)
        1    0.001    0.001    0.656    0.656 train_bpe_core.py:132(merge_pairs)
      243    0.540    0.002    0.587    0.002 train_bpe_core.py:234(merge_token_pair)
      243    0.000    0.000    0.069    0.000 train_bpe_core.py:164(find_best_pair)
      242    0.060    0.000    0.060    0.000 train_bpe_core.py:221(find_best_pair_from_pair_counts)
        1    0.038    0.038    0.057    0.057 train_bpe_core.py:87(pretokenize)
  1205443    0.035    0.000    0.035    0.000 {built-in method builtins.len}



Finsihed profiling in 0.715595 seconds. Results saved to cs336_basics/profiler_results/corpus_en-2026-02-09_18-25-10
```

Great, no regression! I like the speedup so far. But I don't like that `merge_token_pair` takes 0.5s (75% of the runtime).
I believe this is because it runs a nested loop in each merge iteration: it scans through each entry in the token
cache to find which entries contain the pair to merge. This is an O(n*m) loop where n is the number of pretoken entries
and m the average pretoken size. This will not scale well with larger data sets or larger vocab sizes (vocab size -> number of merge iterations).

Instead of scanning all pretokens, I can keep an inverted index hat maps byte pairs to pretoken entries that contain that pair.
Since I already compute the pair_counts dictionary, the work is half done. I could repurpose this such that instead of mapping to frequencies, it
counts to a structure that contains the set of mapped pretokens. This will increase memory use, but it should avoid costly loops.

**Optimization: build index of byte pairs to words to avoid repeated loops**

I implemented the optimization I suggested in the previous step, I created a dictionary that maps byte pair sequences (`(bytes, bytes)` pairs)
to the pretoken sequences that contain those pairs. So when a pairs needs to be merged, we just do an O(1) lookup to find the words/pretokens
that need to be updated. When the word is updated, we also need to update pair counts and tally the pairs in this new word. To make this fast,
I also created a dictionary that maps words/pretokens to the set of pairs in that word (I need to confirm whether maintain this set is actually cheaper
than re-computing the list of pairs in the word given that I often iterate through the set anyway). I encapsulate all this in a class called
`TokenPairIndex` to contain the complexity, but it's not as clean as it ought to be. There is also still room for optimizing the code changes
I've made, but I want to see whether the overall change has significant improvements before I try to optimize it further:

```
uv run python -m cs336_basics.perf_tests corpus_en
Running profiler for command: train_bpe("tests/fixtures/corpus.en", 500, ['<|endoftext|>'])
Mon Feb  9 21:12:44 2026    cs336_basics/profiler_results/corpus_en-2026-02-09_21-12-43

         1523484 function calls (1523386 primitive calls) in 0.389 seconds

   Ordered by: cumulative time
   List reduced from 206 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.389    0.389 {built-in method builtins.exec}
        1    0.000    0.000    0.389    0.389 <string>:1(<module>)
        1    0.000    0.000    0.389    0.389 train_bpe.py:3(train_bpe)
        1    0.001    0.001    0.389    0.389 train_bpe_core.py:13(train_bpe_core)
        1    0.000    0.000    0.330    0.330 train_bpe_core.py:132(merge_pairs)
      243    0.032    0.000    0.228    0.001 train_bpe_core.py:191(merge_token_pair)
      243    0.000    0.000    0.102    0.000 train_bpe_core.py:163(find_best_pair)
    15404    0.027    0.000    0.092    0.000 train_bpe_core.py:450(add_word_with_pairs)
    15404    0.019    0.000    0.072    0.000 train_bpe_core.py:396(remove_word)
   105587    0.046    0.000    0.064    0.000 train_bpe_core.py:416(add_word_link)



Finsihed profiling in 0.389956 seconds. Results saved to cs336_basics/profiler_results/corpus_en-2026-02-09_21-12-43
```

Okay, an impressive 45% speedup (from 0.715s to 0.389s).

Given that the implementation is still sloppy, I'm confident I can do better than this, maybe halve it at least.
But first, I want to sort the results by total time to see how much is spent in each hot function without accounting
for child calls.

```python
import pstats
from pstats import SortKey

path = "cs336_basics/profiler_results/corpus_en-2026-02-09_21-12-43"
results = pstats.Stats(path)

results.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)
```

```
Mon Feb  9 21:12:44 2026    cs336_basics/profiler_results/corpus_en-2026-02-09_21-12-43

         1523484 function calls (1523386 primitive calls) in 0.389 seconds

   Ordered by: internal time
   List reduced from 206 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      242    0.061    0.000    0.061    0.000 train_bpe_core.py:360(compute_best_pair)
   105587    0.046    0.000    0.064    0.000 train_bpe_core.py:416(add_word_link)
        1    0.037    0.037    0.056    0.056 train_bpe_core.py:87(pretokenize)
    91137    0.036    0.000    0.051    0.000 train_bpe_core.py:407(remove_word_link)
   566166    0.036    0.000    0.036    0.000 {method 'get' of 'dict' objects}
      243    0.032    0.000    0.228    0.001 train_bpe_core.py:191(merge_token_pair)
    15404    0.027    0.000    0.092    0.000 train_bpe_core.py:450(add_word_with_pairs)
   105972    0.019    0.000    0.026    0.000 train_bpe_core.py:436(word_contains_pair)
    15404    0.019    0.000    0.072    0.000 train_bpe_core.py:396(remove_word)
        1    0.015    0.015    0.041    0.041 train_bpe_core.py:456(_build_index)
   160785    0.013    0.000    0.013    0.000 train_bpe_core.py:105(<genexpr>)
    22561    0.012    0.000    0.026    0.000 train_bpe_core.py:321(update_pair_counts_with_merged_pair)
    85575    0.005    0.000    0.005    0.000 {method 'add' of 'set' objects}
    22561    0.005    0.000    0.007    0.000 train_bpe_core.py:378(increment_pair_count)
    91140    0.004    0.000    0.004    0.000 {method 'discard' of 'set' objects}
    22561    0.003    0.000    0.005    0.000 train_bpe_core.py:375(get_pair_count)
    56382    0.003    0.000    0.003    0.000 {method 'append' of 'list' objects}
    63438    0.002    0.000    0.002    0.000 {built-in method builtins.len}
    21643    0.002    0.000    0.002    0.000 train_bpe_core.py:382(set_pair_count)
    27758    0.002    0.000    0.002    0.000 {method 'group' of '_regex.Match' objects}
```

There are a lot of small functions that individually don't take too much time, but it adds up. `compute_best_pair` is
the top culprit. This loops over each pair to find the one with the highest count. Perhaps I could use max heap
to keep track of the most frequent pairs, but I'll need to add more complexity to update the heap when
pair counts change. I can also consolidate dictionaries with the same key into one dictionary in the
`TokenPairIndex` class. I could also experiment with removing the word to pair index and just compute the pairs
on demand. But before jumping into more micro-optimizations, maybe I should run the tokenizer on a larger
corpus to see which functions scale poorly.

Let's profile `train_bpe` on the [TinyStories validation data set](../data/TinyStoriesV2-GPT4-train.txt) with
a vocab size of 10,000.

```
uv run python -m cs336_basics.perf_tests tiny_stories_valid
Running profiler for command: train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 10000, ['<|endoftext|>'])
Tue Feb 10 18:09:59 2026    cs336_basics/profiler_results/tiny_stories_validation-2026-02-10_18-09-40

         51397283 function calls (51397185 primitive calls) in 19.186 seconds

   Ordered by: cumulative time
   List reduced from 206 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000   19.186   19.186 {built-in method builtins.exec}
        1    0.001    0.001   19.186   19.186 <string>:1(<module>)
        1    0.002    0.002   19.185   19.185 train_bpe.py:3(train_bpe)
        1    0.061    0.061   19.169   19.169 train_bpe_core.py:13(train_bpe_core)
    27631    6.567    0.000    9.900    0.000 train_bpe_core.py:87(pretokenize)
        1    0.009    0.009    8.565    8.565 train_bpe_core.py:132(merge_pairs)
     9743    0.002    0.000    7.746    0.001 train_bpe_core.py:163(find_best_pair)
     9742    7.628    0.001    7.629    0.001 train_bpe_core.py:360(compute_best_pair)
 27562412    2.256    0.000    2.256    0.000 train_bpe_core.py:105(<genexpr>)
     9743    0.140    0.000    0.808    0.000 train_bpe_core.py:191(merge_token_pair)



Finsihed profiling in 19.185427 seconds. Results saved to cs336_basics/profiler_results/tiny_stories_validation-2026-02-10_18-09-40
```

This has taken 19s. `compute_best_pair` still stands out as a top contributor. Let me sort
the functions by total time to see if there are any other functions that stand out.

```
results.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)
Tue Feb 10 18:09:59 2026    cs336_basics/profiler_results/tiny_stories_validation-2026-02-10_18-09-40

         51397283 function calls (51397185 primitive calls) in 19.186 seconds

   Ordered by: internal time
   List reduced from 206 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     9742    7.628    0.001    7.629    0.001 train_bpe_core.py:360(compute_best_pair)
    27631    6.567    0.000    9.900    0.000 train_bpe_core.py:87(pretokenize)
 27562412    2.256    0.000    2.256    0.000 train_bpe_core.py:105(<genexpr>)
    27631    0.441    0.000    0.619    0.000 train_bpe_core.py:114(merge_pretokenized_counters_in_place)
  5419001    0.345    0.000    0.345    0.000 {method 'group' of '_regex.Match' objects}
  5419002    0.325    0.000    0.325    0.000 {method 'encode' of 'str' objects}
  4296947    0.303    0.000    0.303    0.000 {method 'get' of 'dict' objects}
  5569753    0.190    0.000    0.190    0.000 {built-in method builtins.len}
     9743    0.140    0.000    0.808    0.000 train_bpe_core.py:191(merge_token_pair)
   295961    0.131    0.000    0.181    0.000 train_bpe_core.py:416(add_word_link)
   287269    0.124    0.000    0.174    0.000 train_bpe_core.py:407(remove_word_link)
    68766    0.084    0.000    0.274    0.000 train_bpe_core.py:450(add_word_with_pairs)
    68766    0.068    0.000    0.255    0.000 train_bpe_core.py:396(remove_word)
        1    0.061    0.061   19.169   19.169 train_bpe_core.py:13(train_bpe_core)
   296838    0.056    0.000    0.078    0.000 train_bpe_core.py:436(word_contains_pair)
    83487    0.049    0.000    0.103    0.000 train_bpe_core.py:321(update_pair_counts_with_merged_pair)
   166295    0.043    0.000    0.060    0.000 enum.py:1507(_get_value)
    55423    0.043    0.000    0.127    0.000 enum.py:1525(__and__)
        1    0.042    0.042    0.114    0.114 train_bpe_core.py:456(_build_index)
    27632    0.041    0.000    0.196    0.000 regex.py:449(_compile)
```

85% of the time is spent in 3 functions:

- `compute_best_pair`: likely because for each merge iteration, it has to loop through each pair to find the max
- `pretokenize`: pretokenization does a regex match on each corpus segment. The segments are pretokenized sequentially, as they get merged. But they could be pretokenized independently in parallel then merged later into a single cache. I should also compile the regex in advance and reuse the compiled version.
- `<genexpr>`: This likely the tuple comprehension in the following statement: `token_key = tuple(encoded_token[i:i+1] for i in range(len(encoded_token)))`. Not yet sure how to optimize this.

I'll first start with `compute_best_pair`. I think it's worthwhile trying the max heap approach.

**Optimization: Use max heap to efficiently compute best pair**

I've refactored the code and added a `TokenPairIndex` class that encapsulates a max heap
that can retrieve the most frequent pair. The assignment code is using Python 3.11,
but Python's public max heap APIs (like `heapq.heapify_max`) were only added in 3.14.
So I resorted to a workaround using a min heap with max heap semantics
(e.g. using negative count and reverse ordering of the token pair).

Pushing and popping from the heap are O(logn) operation, but if you want to update
an arbitrary entry, you'd need to scan the list in O(n) since it isn't actually sorted.
I considered using a sorted collection, such as a self-balancing search tree, but I
didn't find one in the standard library. I did find a [`sortedcontainers`](https://pypi.org/project/sortedcontainers) library,
but I didn't want to rely on external libraries for this assignment.

So how do we update the heap when arbitrary pairs are removed or have their counts updated?
Well I decided not to update the heap at that moment. I still keep track of the `pair_counts`
dictionary. When I pop something from the heap, I check whether the count from the heap
matches that in the `pair_counts` map, if they don't match, I consider it a stale entry,
discard it and pop again until I find an item whose count is consistent with `pair_count`.

Here are the results of the optimization:

```
uv run python -m cs336_basics.perf_tests tiny_stories_valid
Running profiler for command: train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 10000, ['<|endoftext|>'])
Wed Feb 11 20:08:47 2026    cs336_basics/profiler_results/tiny_stories_validation-2026-02-11_20-08-34

         54161414 function calls (54161316 primitive calls) in 12.592 seconds

   Ordered by: cumulative time
   List reduced from 221 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000   12.592   12.592 {built-in method builtins.exec}
        1    0.001    0.001   12.592   12.592 <string>:1(<module>)
        1    0.003    0.003   12.591   12.591 train_bpe.py:3(train_bpe)
        1    0.067    0.067   12.572   12.572 train_bpe_core.py:16(train_bpe_core)
    27631    6.671    0.000   10.043    0.000 train_bpe_core.py:90(pretokenize)
 27562412    2.270    0.000    2.270    0.000 train_bpe_core.py:108(<genexpr>)
        1    0.008    0.008    1.797    1.797 train_bpe_core.py:135(merge_pairs)
     9743    0.141    0.000    1.196    0.000 train_bpe_core.py:194(merge_token_pair)
    27631    0.445    0.000    0.642    0.000 train_bpe_core.py:117(merge_pretokenized_counters_in_place)
     9743    0.002    0.000    0.590    0.000 train_bpe_core.py:166(find_best_pair)



Finsihed profiling in 12.593223 seconds. Results saved to cs336_basics/profiler_results/tiny_stories_validation-2026-02-11_20-08-34
```

We get a 34% speedup (19.2 to 12.6 seconds). Not that flashy compared to some of the other speedups, but pretty good. `find_best_pair`
hardly appears in the top 10 anymore.