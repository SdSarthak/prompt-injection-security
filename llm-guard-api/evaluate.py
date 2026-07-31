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
import os
import time
from collections import Counter
from typing import List, Optional, Sequence, Tuple

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
    none of these rows were fit on. :func:`drop_leaked` verifies that claim
    against the model's recorded training set rather than assuming it.
    """
    from sklearn.model_selection import train_test_split

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"No dataset at {data_path}. Pass --data, or run "
            "`python train.py --download-only` to fetch the bundled corpus."
        )
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"--test-size must be strictly between 0 and 1 (got {test_size})")

    frame = pd.read_csv(data_path)
    missing = {"prompt", "label"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"{data_path} is missing required column(s): {sorted(missing)}; "
            f"found {sorted(frame.columns)}"
        )
    frame = frame.dropna(subset=["prompt", "label"])
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


def drop_leaked(
    guard, prompts: Sequence[str], labels: Sequence[str]
) -> Tuple[List[str], List[str], Optional[int]]:
    """Remove prompts the classifier was trained on.

    ``--data``, ``--test-size`` or ``--seed`` that do not match the ones the
    model was fitted with silently put training rows into the "held-out" set,
    and the resulting numbers look excellent for the wrong reason. The trained
    artifact records a digest of every training prompt, so the overlap can be
    measured instead of assumed.

    Returns ``(prompts, labels, n_removed)``; ``n_removed`` is None when the
    model carries no provenance and the question cannot be answered.
    """
    classifier = getattr(guard, "classifier", None)
    was_trained_on = getattr(classifier, "was_trained_on", None)
    if was_trained_on is None or was_trained_on("probe") is None:
        return list(prompts), list(labels), None

    kept = [(p, y) for p, y in zip(prompts, labels) if not was_trained_on(p)]
    removed = len(prompts) - len(kept)
    if removed:
        logger.warning("Excluded %d evaluation prompt(s) the model was trained on", removed)
    if not kept:
        raise ValueError(
            "Every evaluation prompt was in the model's training set. Retrain with "
            "`python train.py --backend baseline` or evaluate on a different corpus."
        )
    return [p for p, _ in kept], [y for _, y in kept], removed


def evaluate(guard, prompts: Sequence[str], labels: Sequence[str]) -> dict:
    """Score end-to-end decisions against benign/malicious ground truth."""
    if len(prompts) != len(labels):
        raise ValueError("prompts and labels must be the same length")
    if not prompts:
        raise ValueError("nothing to evaluate")

    started = time.perf_counter()
    decisions = [result["decision"] for result in guard.analyze_batch(prompts)]
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
    leaked = metrics.get("excluded_seen_in_training")
    if leaked is None:
        print("training overlap     unknown (model carries no provenance)")
    elif leaked:
        print(f"training overlap     {leaked} prompt(s) excluded - the model was fit on them")
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
    parser.add_argument(
        "--keep-seen",
        action="store_true",
        help="Score prompts the model was trained on instead of excluding them",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a report")
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must not be negative")

    # Every blocked prompt logs a warning; over a whole corpus that is noise.
    logging.basicConfig(level=logging.ERROR)
    logging.disable(logging.WARNING)

    try:
        prompts, labels = load_holdout(args.data, args.test_size, args.seed, args.limit)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    from app import LLMGuard

    guard = LLMGuard(classifier_backend=args.backend, enable_llm=False)

    leaked = None
    if not args.keep_seen:
        prompts, labels, leaked = drop_leaked(guard, prompts, labels)

    metrics = evaluate(guard, prompts, labels)
    metrics["excluded_seen_in_training"] = leaked

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print_report(metrics, guard.classifier_backend)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
