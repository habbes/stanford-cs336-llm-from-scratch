import cProfile
import pstats
from pstats import SortKey
from datetime import datetime
from os import path, makedirs
from sys import argv

from .train_bpe import train_bpe


def measure_training(prefix: str, corpus: str, vocab_size: int, special_tokens: list[str]|None = None):
    special_tokens = ["<|endoftext|>"] if special_tokens is None else special_tokens
    command = f"train_bpe(\"{corpus}\", {vocab_size}, {special_tokens})"
    
    output_dir = path.join("cs336_basics", "profiler_results")
    makedirs(output_dir, exist_ok=True)
    output = path.join(output_dir, f"{prefix}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Running profiler for command: {command}")

    started = datetime.now()
    # See: https://docs.python.org/3/library/profile.html
    cProfile.run(command, output)
    duration = datetime.now() - started

    results = pstats.Stats(output)
    results.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)
    
    print()
    print(f"Finsihed profiling in {duration.total_seconds()} seconds. Results saved to {output}")


def measure_corpus_en():
    corpus = "tests/fixtures/corpus.en"
    measure_training("corpus_en", corpus, 500)

def measure_tiny_stories_valid():
    corpus = "data/TinyStoriesV2-GPT4-valid.txt"
    measure_training("tiny_stories_validation", corpus, 10000)

def print_help():
    print("USAGE:")
    print("uv run python -m cs336_basics.perf_tests <SCENARIO>")
    print()
    print("Available scenarios:")
    for scenario, details in scenarios.items():
        print(f"{scenario}: {details['desc']}")

if __name__ == "__main__":
    scenarios = {
        "corpus_en": { "fn": measure_corpus_en, "desc": "Small test fixture, vocab size = 500" },
        "tiny_stories_valid": { "fn": measure_tiny_stories_valid, "desc": "TinyStoriesV2-GPT4, vocab size = 10000"}
    }

    if len(argv) < 2:
        print("Please specify scenario to profile")
        print_help()
        exit(1)
    
    scenario = argv[1]
    if scenario not in scenarios:
        print(f"You specified an unknown scenario '{scenario}'.")
        print_help()

    details = scenarios[scenario]
    fn = details["fn"]
    fn()

    

