# Progress tracking

I used this to keep track of my progress to remember where I left off, in case I take an extended break from this.

- Implemented `naive_bpe` to get a hang of the algo
  - To run sanity checks of `naive_bpe` implementation, `uv run playground.py` in `cs336_basics`
- Created `train_bpe` wrapper function and updated `tests/adapters.run_train_bpe` to call the function
- To run the `train_bpe` tests run `uv run pytest tests/test_train_bpe.py` from the `assignment1-basics` directory
- Tests currently fail (Correctness tests pass, but speed/performance test fails)
- I got 2/3 tests to pass on the `test_train_bpe` suite:
  - I managed to get the basic `test_train_bpe` test to pass after fixing some bugs.
  - I also managed to the `test_bpe_special_tokens` test to pass after fixing some bugs.
      - The issue is that I was pretokenizing and creating merges for each segment independently after splitting on special tokens
      - What I should have done was pretokenize all the segments first, and combine their output, then perform BPE merges on the combined
        output to achieve the same result as if I was running BPE on an equivalent corpus that didn't contain special tokens.
      - This was a very simple mistake, but I missed each time I tried to make progress, partly because I only worked on this sporadically
        and would lose some context each time I came back to this (which is why I'm keeping track of this progress doc), but
        also because I was trying to rush through it and resorting to trial-and-error debugging.
      - But when I took my time to re-read the question and specification, testing the implementation systematically on a simple
        dataset, walking through the expected behaviour by hand, it was obvious to spot the bug and fix it.
      - The key take-away here is that I should take my time with this project. I should allocate enough time to each
      session to go through things carefully and systematically. I should also do this more regularly, not going
      more than a month without making progress.
    Since now I know the basic bpe logic is correct. The focus should likely be on corpus splitting and handling logic.
- Next steps
  - Profile performance (recommended tools: cProfile, scalene)
    - Learn [cProfile](https://docs.python.org/3/library/profile.html) and use to find bottlenecks
  - Create more efficient version of `train_bpe` with multiprocessing
  - Bonus: experiment with speeding some parts using Rust
  - Bonus bonus: can create an optimized C# version?
