"""Evaluate the full guard pipeline on a held-out slice of the corpus.

Scores end-to-end decisions, not just the classifier, so regex false positives
and decision-engine thresholds are included in the numbers.

    python evaluate.py
    python evaluate.py --data data/prompts.csv --limit 500
    python evaluate.py --backend transformer

Runs entirely offline: it uses ``LLMGuard.analyze``, so it never calls Gemini.
"""

import argparse
import json
import logging
import time
from collections import Counter
from typing import List, Sequence, Tuple

import pandas as pd

import config

logger = logging.getLogger(__name__)

# A benign prompt should be ALLOWed; an attack should be blocked or sanitized -
# either outcome means it did not reach the model unchanged.
DEFENDED = ("block", "sanitize")


def load_holdout(
    data_path: str, test_size: float, seed: int, limit: int = 0
) -> Tuple[List[str], List[str]]:
    """Load the same held-out split the persisted baseline model was scored on.

    Using the identical split and seed as training keeps the evaluation honest:
    none of these rows were fit on.
    """
    from sklearn.model_selection import train_test_split

    frame = pd.read_csv(data_path).dropna(subset=["prompt", "label"])
    frame = frame[frame["prompt"].astype(str).str.strip() != ""]
    if frame.empty:
        raise ValueError(f"No usable rows in {data_path}")

    texts = frame["prompt"].astype(str).tolist()
    labels = frame["label"].astype(str).tolist()

    _, x_test, _, y_test = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=seed
    )

    if limit and limit < len(x_test):
        x_test, y_test = x_test[:limit], y_test[:limit]

    return x_test, y_test


def evaluate(guard, prompts: Sequence[str], labels: Sequence[str]) -> dict:
    """Score end-to-end decisions against benign/malicious ground truth."""
    started = time.perf_counter()
    decisions = [guard.analyze(prompt)["decision"] for prompt in prompts]
    elapsed = time.perf_counter() - started

    true_positive = sum(
        1 for d, y in zip(decisions, labels) if y != "benign" and d in DEFENDED
    )
    false_negative = sum(1 for d, y in zip(decisions, labels) if y != "benign" and d == "allow")
    false_positive = sum(1 for d, y in zip(decisions, labels) if y == "benign" and d in DEFENDED)
    true_negative = sum(1 for d, y in zip(decisions, labels) if y == "benign" and d == "allow")

    attacks = true_positive + false_negative
    benign = true_negative + false_positive

    recall = true_positive / attacks if attacks else 0.0
    specificity = true_negative / benign if benign else 0.0
    precision = (
        true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "n": len(prompts),
        "attacks": attacks,
        "benign": benign,
        "accuracy": round((true_positive + true_negative) / len(prompts), 4),
        "attack_recall": round(recall, 4),
        "benign_specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "false_negatives": false_negative,
        "false_positives": false_positive,
        "decisions": dict(Counter(decisions)),
        "mean_latency_ms": round(elapsed / len(prompts) * 1000, 3),
        "total_seconds": round(elapsed, 2),
    }


def print_report(metrics: dict, backend: str) -> None:
    print("\n" + "=" * 64)
    print("Guard pipeline evaluation (end-to-end decisions)")
    print("=" * 64)
    print(f"backend              {backend}")
    print(f"held-out prompts     {metrics['n']} ({metrics['attacks']} attacks, {metrics['benign']} benign)")
    print()
    print(f"overall accuracy     {metrics['accuracy']:.4f}")
    print(f"attack recall        {metrics['attack_recall']:.4f}  (blocked or sanitized)")
    print(f"benign specificity   {metrics['benign_specificity']:.4f}  (allowed untouched)")
    print(f"precision            {metrics['precision']:.4f}")
    print(f"f1                   {metrics['f1']:.4f}")
    print()
    print(f"false negatives      {metrics['false_negatives']}  (attacks allowed through)")
    print(f"false positives      {metrics['false_positives']}  (benign prompts held up)")
    print(f"decisions            {metrics['decisions']}")
    print()
    print(f"mean latency         {metrics['mean_latency_ms']} ms/prompt")
    print(f"total                {metrics['total_seconds']} s")
    print("=" * 64 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the guard pipeline")
    parser.add_argument("--data", default=str(config.TRAINING_DATA_PATH), help="Labelled CSV")
    parser.add_argument("--backend", choices=["baseline", "transformer"], default=None)
    parser.add_argument("--test-size", type=float, default=config.TEST_SPLIT)
    parser.add_argument("--seed", type=int, default=42, help="Must match the training seed")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most N prompts")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a report")
    args = parser.parse_args()

    # Every blocked prompt logs a warning; over a whole corpus that is noise.
    logging.basicConfig(level=logging.ERROR)
    logging.disable(logging.WARNING)

    prompts, labels = load_holdout(args.data, args.test_size, args.seed, args.limit)

    from app import LLMGuard

    guard = LLMGuard(classifier_backend=args.backend, enable_llm=False)
    metrics = evaluate(guard, prompts, labels)

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print_report(metrics, guard.classifier_backend)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
