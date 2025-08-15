# Progress tracking

I used this to keep track of my progress to remember where I left off, in case I take an extended break from this.

- Implemented `naive_bpe` to get a hang of the algo
  - To run sanity checks of `naive_bpe` implementation, `uv run naive_bpe.py` in `cs336_basics`
- Created `train_bpe` wrapper function and updated `tests/adapters.run_train_bpe` to call the function
- To run the `train_bpe` tests run `uv run pytest tests/test_train_bpe.py`
- Tests currently fail
- Next steps:
  - I fixed the issue with special tokens getting tokenized, but tests still fail.
    - I probably need to also ensure that I do not merge across text boundaries delimited by these special tokens.
  - Profile performance (recommended tools: cProfile, scalene)
  - Create more efficient version of `train_bpe` with multiprocessing
  - Bonus: experiment with speeding some parts using Rust
  - Bonus bonus: can create an optimized C# version?
