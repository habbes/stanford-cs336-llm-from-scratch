# Progress tracking

I used this to keep track of my progress to remember where I left off, in case I take an extended break from this.

- Implemented `naive_bpe` to get a hang of the algo
  - To run sanity checks of `naive_bpe` implementation, `uv run naive_bpe.py` in `cs336_basics`
- Created `train_bpe` wrapper function and updated `tests/adapters.run_train_bpe` to call the function
- To run the `train_bpe` tests run `uv run pytest tests/test_train_bpe.py`
- Tests currently fail
- Next steps:
  - I implemented splitting of the corpus on special tokens so that each segment could be pretokenized and merged
     indepedently, but train_bpe and train_bpe_special_tokens tests are still failing.
     need to be the train_bpe test more closely to see what the expected result is and why it's failing.
  - Profile performance (recommended tools: cProfile, scalene)
  - Create more efficient version of `train_bpe` with multiprocessing
  - Bonus: experiment with speeding some parts using Rust
  - Bonus bonus: can create an optimized C# version?
