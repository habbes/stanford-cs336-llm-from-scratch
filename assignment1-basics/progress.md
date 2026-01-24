# Progress tracking

I used this to keep track of my progress to remember where I left off, in case I take an extended break from this.

- Implemented `naive_bpe` to get a hang of the algo
  - To run sanity checks of `naive_bpe` implementation, `uv run playground.py` in `cs336_basics`
- Created `train_bpe` wrapper function and updated `tests/adapters.run_train_bpe` to call the function
- To run the `train_bpe` tests run `uv run pytest tests/test_train_bpe.py`
- Tests currently fail
- Next steps:
  - I managed to get the basic `test_train_bpe` test to pass after fixing some bugs. Next I want to focus on getting `test_bpe_special_tokens` to pass.
    Since now I know the basic bpe logic is correct. The focus should likely be on corpus splitting and handling logic.
  - Profile performance (recommended tools: cProfile, scalene)
  - Create more efficient version of `train_bpe` with multiprocessing
  - Bonus: experiment with speeding some parts using Rust
  - Bonus bonus: can create an optimized C# version?
