"""Basic usage example for the Human-Confabulated Hallucination Benchmark.

Loads the dataset, prints summary statistics, and shows one sample pair
from each domain.

Usage:
    python examples/basic_usage.py
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

DATA_PATH: Path = Path(__file__).parent.parent / "data" / "human_confabulations.csv"

SEPARATOR: str = "=" * 60


def load_benchmark(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the benchmark CSV and return a typed DataFrame."""
    return pd.read_csv(path)


def print_summary(df: pd.DataFrame) -> None:
    """Print dataset-level statistics."""
    print(SEPARATOR)
    print("Human-Confabulated Hallucination Benchmark")
    print(SEPARATOR)
    print(f"\nTotal pairs: {len(df)}")
    print(f"Domains: {df['domain'].nunique()}")

    print("\nDomain distribution:")
    for domain, count in df["domain"].value_counts().items():
        print(f"  {domain:<20s} {count:>3d} pairs")


def print_length_stats(df: pd.DataFrame) -> None:
    """Print word-count statistics for grounded vs. confabulated responses."""
    grounded_len = df["grounded_response"].str.split().str.len()
    fabricated_len = df["fabricated_response"].str.split().str.len()

    print(
        f"\nGrounded response length:     "
        f"mean={grounded_len.mean():.1f}, median={grounded_len.median():.1f} words"
    )
    print(
        f"Confabulated response length: "
        f"mean={fabricated_len.mean():.1f}, median={fabricated_len.median():.1f} words"
    )


def print_examples(df: pd.DataFrame, *, max_chars: int = 100) -> None:
    """Print one truncated example per domain."""
    print(f"\n{SEPARATOR}")
    print("One example per domain")
    print(SEPARATOR)

    for domain in sorted(df["domain"].unique()):
        row = df[df["domain"] == domain].iloc[0]
        print(f"\n--- {domain.upper()} ---")
        print(f"Q: {row['question'][:max_chars]}...")
        print(f"Grounded:     {row['grounded_response'][:max_chars]}...")
        print(f"Confabulated: {row['fabricated_response'][:max_chars]}...")


def main() -> None:
    """Entry point."""
    df = load_benchmark()
    print_summary(df)
    print_length_stats(df)
    print_examples(df)


if __name__ == "__main__":
    main()
