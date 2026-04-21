"""Reproduce the core detection experiment from the paper.

Tests whether cosine similarity between query and response embeddings
can distinguish grounded from confabulated responses.

Requirements:
    pip install sentence-transformers pandas numpy scikit-learn

Usage:
    python scripts/validate.py
    python scripts/validate.py --model all-mpnet-base-v2
    python scripts/validate.py --model all-MiniLM-L6-v2 --domain finance
    python scripts/validate.py --all-models
"""

from __future__ import annotations

import argparse
import functools
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH: Path = Path(__file__).parent.parent / "data" / "human_confabulations.csv"

MODELS: list[str] = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "thenlper/gte-small",
]

# Type alias for embedding matrices (n_samples, embedding_dim).
Embeddings = npt.NDArray[np.float32]


@dataclass(frozen=True)
class DetectionMetrics:
    """Results of a cosine-similarity detection experiment."""

    accuracy: float
    mean_delta: float
    paired_similarity: float
    n: int

    @property
    def wilson_ci(self) -> tuple[float, float]:
        """Wilson score 95 % confidence interval for detection accuracy."""
        z: float = 1.96
        denom = 1 + z**2 / self.n
        centre = (self.accuracy + z**2 / (2 * self.n)) / denom
        spread = (
            z
            * np.sqrt(
                (self.accuracy * (1 - self.accuracy) + z**2 / (4 * self.n))
                / self.n
            )
            / denom
        )
        return max(0.0, centre - spread), min(1.0, centre + spread)


def _pairwise_cosines(a: Embeddings, b: Embeddings) -> npt.NDArray[np.float64]:
    """Row-wise cosine similarity between two aligned embedding matrices."""
    # Normalise to unit vectors, then dot product per row.
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a_norm * b_norm, axis=1)


def compute_metrics(
    questions: Embeddings,
    grounded: Embeddings,
    fabricated: Embeddings,
) -> DetectionMetrics:
    """Compute detection accuracy, paired similarity, and mean delta.

    Detection succeeds for a pair when cos(question, grounded) exceeds
    cos(question, fabricated).  Random baseline is 50 %.
    """
    cos_qg = _pairwise_cosines(questions, grounded)
    cos_qf = _pairwise_cosines(questions, fabricated)
    cos_gf = _pairwise_cosines(grounded, fabricated)

    return DetectionMetrics(
        accuracy=float(np.mean(cos_qg > cos_qf)),
        mean_delta=float(np.mean(cos_qg - cos_qf)),
        paired_similarity=float(np.mean(cos_gf)),
        n=len(questions),
    )


@functools.lru_cache(maxsize=4)
def _load_model(model_name: str) -> SentenceTransformer:
    """Load and cache a sentence-transformer model."""
    return SentenceTransformer(model_name)


def encode_dataset(
    model: SentenceTransformer,
    df: pd.DataFrame,
) -> tuple[Embeddings, Embeddings, Embeddings]:
    """Encode questions, grounded responses, and fabricated responses."""
    questions = model.encode(df["question"].tolist(), show_progress_bar=False)
    grounded = model.encode(df["grounded_response"].tolist(), show_progress_bar=False)
    fabricated = model.encode(df["fabricated_response"].tolist(), show_progress_bar=False)
    return questions, grounded, fabricated


def _print_header(model_name: str, n_pairs: int, domain: str | None) -> None:
    """Print experiment header."""
    print(f"\nModel: {model_name}")
    print(f"Pairs: {n_pairs}")
    if domain:
        print(f"Domain: {domain}")
    print("-" * 66)


def _print_metrics(metrics: DetectionMetrics) -> None:
    """Print aggregate metrics with confidence interval."""
    lo, hi = metrics.wilson_ci
    print(
        f"Detection accuracy: {metrics.accuracy:.1%}  "
        f"[95% CI: {lo:.1%}, {hi:.1%}]"
    )
    print(f"Mean delta:         {metrics.mean_delta:.4f}")
    print(f"Paired similarity:  {metrics.paired_similarity:.4f}")


def _print_domain_table(
    df: pd.DataFrame,
    questions: Embeddings,
    grounded: Embeddings,
    fabricated: Embeddings,
) -> None:
    """Print per-domain detection breakdown."""
    print(
        f"\n{'Domain':<20s} {'n':>4s} {'Accuracy':>10s} "
        f"{'95% CI':>16s} {'Paired sim':>12s}"
    )
    print("-" * 66)

    for domain in sorted(df["domain"].unique()):
        idx: npt.NDArray[np.bool_] = (df["domain"] == domain).values
        m = compute_metrics(questions[idx], grounded[idx], fabricated[idx])
        lo, hi = m.wilson_ci
        print(
            f"{domain:<20s} {m.n:>4d} {m.accuracy:>10.1%} "
            f"[{lo:.1%}, {hi:.1%}] {m.paired_similarity:>10.4f}"
        )


def run_experiment(
    model_name: str,
    df: pd.DataFrame,
    *,
    domain: str | None = None,
) -> None:
    """Run the detection experiment for one embedding model.

    Args:
        model_name: HuggingFace model identifier.
        df: Benchmark dataframe with columns ``question``,
            ``grounded_response``, ``fabricated_response``, ``domain``.
        domain: If provided, restrict evaluation to this single domain.
    """
    if domain is not None:
        df = df[df["domain"] == domain].copy()
        if df.empty:
            print(f"No data for domain '{domain}'")
            return

    _print_header(model_name, len(df), domain)

    model = _load_model(model_name)
    questions, grounded, fabricated = encode_dataset(model, df)

    _print_metrics(compute_metrics(questions, grounded, fabricated))

    if domain is None:
        _print_domain_table(df, questions, grounded, fabricated)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Reproduce detection experiment on human confabulations",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help=f"Embedding model. Options: {', '.join(MODELS)}",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter to a specific domain (e.g., finance, medical)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run experiment across all four models",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the detection experiment."""
    args = _parse_args()
    df: pd.DataFrame = pd.read_csv(DATA_PATH)

    print("=" * 66)
    print("Human-Confabulated Hallucination Benchmark — Detection Experiment")
    print("=" * 66)

    models = MODELS if args.all_models else [args.model]
    for model_name in models:
        run_experiment(model_name, df, domain=args.domain)


if __name__ == "__main__":
    main()
