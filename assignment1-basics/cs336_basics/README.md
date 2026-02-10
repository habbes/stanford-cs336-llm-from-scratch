This folder contains my code implement problem solutions and projects for [homework 1](../cs336_spring2025_assignment1_basics.pdf).

The detailed report for this section can be found in this [writeup.md](./writeup.md) file. That doc contains
answers to problems in the [assignment document](../cs336_spring2025_assignment1_basics.pdf), but it's
more of a log that I update based on my thought process and experiments as I go.

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