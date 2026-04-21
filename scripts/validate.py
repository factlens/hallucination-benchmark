"""
Reproduce the core detection experiment from the paper.

Tests whether cosine similarity between query and response embeddings
can distinguish grounded from confabulated responses.

Requirements:
    pip install sentence-transformers pandas numpy scikit-learn

Usage:
    python scripts/validate.py
    python scripts/validate.py --model all-mpnet-base-v2
    python scripts/validate.py --model all-MiniLM-L6-v2 --domain finance
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = Path(__file__).parent.parent / "data" / "human_confabulations.csv"

MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "thenlper/gte-small",
]


def compute_metrics(questions, grounded, fabricated):
    """Compute detection accuracy, paired similarity, and mean delta."""
    cos_qg = np.array([
        cosine_similarity([q], [g])[0][0]
        for q, g in zip(questions, grounded)
    ])
    cos_qf = np.array([
        cosine_similarity([q], [f])[0][0]
        for q, f in zip(questions, fabricated)
    ])
    cos_gf = np.array([
        cosine_similarity([g], [f])[0][0]
        for g, f in zip(grounded, fabricated)
    ])

    accuracy = np.mean(cos_qg > cos_qf)
    delta = np.mean(cos_qg - cos_qf)
    paired_sim = np.mean(cos_gf)

    return {
        "accuracy": accuracy,
        "mean_delta": delta,
        "paired_similarity": paired_sim,
        "n": len(questions),
    }


def wilson_ci(p, n, z=1.96):
    """Wilson score 95% confidence interval."""
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0, centre - spread), min(1, centre + spread)


def run_experiment(model_name, df, domain=None):
    """Run detection experiment for one model."""
    if domain:
        df = df[df["domain"] == domain].copy()
        if len(df) == 0:
            print(f"No data for domain '{domain}'")
            return

    print(f"\nModel: {model_name}")
    print(f"Pairs: {len(df)}")
    if domain:
        print(f"Domain: {domain}")
    print("-" * 50)

    model = SentenceTransformer(model_name)

    questions = model.encode(df["question"].tolist(), show_progress_bar=False)
    grounded = model.encode(df["grounded_response"].tolist(), show_progress_bar=False)
    fabricated = model.encode(df["fabricated_response"].tolist(), show_progress_bar=False)

    # Overall metrics
    metrics = compute_metrics(questions, grounded, fabricated)
    lo, hi = wilson_ci(metrics["accuracy"], metrics["n"])

    print(f"Detection accuracy: {metrics['accuracy']:.1%}  "
          f"[95% CI: {lo:.1%}, {hi:.1%}]")
    print(f"Mean delta:         {metrics['mean_delta']:.4f}")
    print(f"Paired similarity:  {metrics['paired_similarity']:.4f}")

    # Per-domain breakdown
    if not domain:
        print(f"\n{'Domain':<20s} {'n':>4s} {'Accuracy':>10s} {'95% CI':>16s} "
              f"{'Paired sim':>12s}")
        print("-" * 66)
        for d in sorted(df["domain"].unique()):
            mask = df["domain"] == d
            idx = mask.values
            m = compute_metrics(questions[idx], grounded[idx], fabricated[idx])
            lo, hi = wilson_ci(m["accuracy"], m["n"])
            print(f"{d:<20s} {m['n']:>4d} {m['accuracy']:>10.1%} "
                  f"[{lo:.1%}, {hi:.1%}] {m['paired_similarity']:>10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce detection experiment on human confabulations"
    )
    parser.add_argument(
        "--model", type=str, default="all-MiniLM-L6-v2",
        help=f"Embedding model to use. Options: {', '.join(MODELS)}"
    )
    parser.add_argument(
        "--domain", type=str, default=None,
        help="Filter to a specific domain (e.g., finance, medical)"
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Run experiment across all four models"
    )
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH, sep=",")
    print("=" * 60)
    print("Human-Confabulated Hallucination Benchmark")
    print("Detection Experiment")
    print("=" * 60)

    if args.all_models:
        for model_name in MODELS:
            run_experiment(model_name, df, args.domain)
    else:
        run_experiment(args.model, df, args.domain)


if __name__ == "__main__":
    main()
