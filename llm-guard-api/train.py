"""Training and data preparation script for the intent classifier.

Two backends are supported:

* ``baseline``    - TF-IDF + logistic regression. Seconds on a CPU, no downloads.
* ``transformer`` - fine-tunes DeBERTa-v3-small. Needs torch; a GPU is strongly
                    recommended.

Typical use:

    python train.py                          # download data if needed, train baseline
    python train.py --download-only          # just fetch the dataset
    python train.py --backend transformer --epochs 3
"""

import argparse
import json
import logging
import os
from typing import Optional

import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LOCAL_CSV_PATH = config.TRAINING_DATA_PATH

# Column aliases seen across public prompt-injection datasets.
_TEXT_COLUMNS = ("prompt", "text", "input", "sentence", "content")
_LABEL_COLUMNS = ("label", "labels", "target", "class", "is_injection")

# Numeric label conventions: 0 = safe, 1 = injection.
_NUMERIC_LABEL_MAP = {0: "benign", 1: "malicious"}
_STRING_LABEL_MAP = {
    "safe": "benign",
    "benign": "benign",
    "legitimate": "benign",
    "clean": "benign",
    "injection": "malicious",
    "malicious": "malicious",
    "jailbreak": "malicious",
    "attack": "malicious",
    "unsafe": "malicious",
    "suspicious": "suspicious",
}


def _pick_column(frame: pd.DataFrame, candidates) -> Optional[str]:
    lowered = {str(col).lower(): col for col in frame.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _normalize_labels(series: pd.Series) -> pd.Series:
    """Map a heterogeneous label column onto benign/suspicious/malicious."""
    if pd.api.types.is_numeric_dtype(series):
        return series.map(_NUMERIC_LABEL_MAP)
    return series.astype(str).str.strip().str.lower().map(_STRING_LABEL_MAP)


def download_and_process_dataset(
    output_path: str = str(LOCAL_CSV_PATH), force_download: bool = False
) -> pd.DataFrame:
    """
    Download the prompt-injection dataset from Hugging Face and normalise it.

    The resulting CSV always has exactly two columns, ``prompt`` and ``label``,
    with labels drawn from ``config.INTENT_CLASSES``.

    Args:
        output_path: Where to write the normalised CSV.
        force_download: Re-download even when the CSV already exists.

    Returns:
        The processed DataFrame.
    """
    if os.path.exists(output_path) and not force_download:
        logger.info("Dataset already exists at %s. Skipping download.", output_path)
        return pd.read_csv(output_path)

    logger.info("Downloading dataset '%s' from Hugging Face...", config.HF_DATASET_NAME)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required to download training data. "
            "Install it with `pip install datasets`, or drop a CSV with "
            f"`prompt` and `label` columns at {output_path}."
        ) from exc

    dataset = load_dataset(config.HF_DATASET_NAME)
    frame = dataset["train"].to_pandas()
    logger.info("Raw dataset columns: %s", frame.columns.tolist())

    text_col = _pick_column(frame, _TEXT_COLUMNS)
    label_col = _pick_column(frame, _LABEL_COLUMNS)
    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not find text/label columns in {frame.columns.tolist()}. "
            f"Expected one of {_TEXT_COLUMNS} and one of {_LABEL_COLUMNS}."
        )

    processed = pd.DataFrame(
        {
            "prompt": frame[text_col].astype(str),
            "label": _normalize_labels(frame[label_col]),
        }
    )

    before = len(processed)
    processed = processed.dropna(subset=["prompt", "label"])
    processed = processed[processed["prompt"].str.strip() != ""]
    processed = processed.drop_duplicates(subset=["prompt"])
    logger.info("Kept %d of %d rows after cleaning", len(processed), before)

    if processed.empty:
        raise ValueError("No usable rows after normalisation - check the label mapping")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    processed.to_csv(output_path, index=False)

    logger.info("Dataset saved to %s", output_path)
    for label, count in processed["label"].value_counts().items():
        logger.info("  %-12s %d", label, count)

    return processed


def train_baseline_classifier(dataset_path: str, model_path: str) -> bool:
    """Fit and persist the TF-IDF + logistic regression classifier."""
    from guard.baseline_classifier import train_baseline

    logger.info("Training baseline classifier on %s", dataset_path)
    report = train_baseline(csv_path=dataset_path, model_path=model_path)

    logger.info("Training complete: %s", report.summary())
    print("\n" + report.report)
    print(f"Model written to {report.model_path}\n")

    metrics_path = os.path.join(os.path.dirname(report.model_path), "baseline_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_train": report.n_train,
                "n_test": report.n_test,
                "accuracy": report.accuracy,
                "precision": report.precision,
                "recall": report.recall,
                "f1": report.f1,
                "roc_auc": report.roc_auc,
            },
            handle,
            indent=2,
        )
    logger.info("Metrics written to %s", metrics_path)
    return True


def train_transformer_classifier(dataset_path: str, model_output_dir: str, epochs: int = 3) -> bool:
    """Fine-tune DeBERTa on the dataset."""
    try:
        from sklearn.model_selection import train_test_split

        from guard.intent_classifier import IntentClassifier
    except ImportError as exc:
        logger.error(
            "The transformer backend needs torch and transformers: %s. "
            "Install them with `pip install -r requirements-transformer.txt` "
            "or train the baseline instead.",
            exc,
        )
        return False

    frame = pd.read_csv(dataset_path)
    frame = frame[frame["label"].isin(config.INTENT_CLASSES)].dropna(subset=["prompt", "label"])
    if frame.empty:
        logger.error("No rows with a recognised label in %s", dataset_path)
        return False

    present = sorted(frame["label"].unique())
    if len(present) < 2:
        logger.error("Need at least two classes to train, found %s", present)
        return False
    if "suspicious" not in present:
        # The public dataset is binary. Say so rather than letting the head
        # silently learn a class it never sees.
        logger.warning(
            "Dataset has no 'suspicious' examples (%s). The fine-tuned model can only "
            "emit benign/malicious; the decision engine will therefore never see the "
            "SANITIZE-by-intent path from this backend.",
            present,
        )

    train_df, val_df = train_test_split(
        frame, test_size=config.TEST_SPLIT, stratify=frame["label"], random_state=42
    )
    logger.info("Training set: %d samples, validation set: %d", len(train_df), len(val_df))

    classifier = IntentClassifier(device="cuda" if os.environ.get("USE_GPU") else None)
    metrics = classifier.train(
        train_texts=train_df["prompt"].astype(str).tolist(),
        train_labels=train_df["label"].tolist(),
        val_texts=val_df["prompt"].astype(str).tolist(),
        val_labels=val_df["label"].tolist(),
        epochs=epochs,
        batch_size=16,
        learning_rate=2e-5,
        output_dir=model_output_dir,
    )

    logger.info("Training complete!")
    logger.info("Final validation accuracy: %.4f", metrics["val_accuracy"][-1])
    logger.info("Final validation F1:       %.4f", metrics["val_f1"][-1])
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the LLM Guard intent classifier")
    parser.add_argument(
        "--backend",
        choices=["baseline", "transformer"],
        default="baseline",
        help="Which classifier to train (default: baseline)",
    )
    parser.add_argument("--download-only", action="store_true", help="Only download and process data")
    parser.add_argument("--train-only", action="store_true", help="Only train (assumes data exists)")
    parser.add_argument("--all", action="store_true", help="Download data and train (default)")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs for the transformer backend")
    parser.add_argument("--data", default=str(LOCAL_CSV_PATH), help="Path to the training CSV")
    parser.add_argument("--output", default=None, help="Where to write the trained model")
    args = parser.parse_args()

    if not any([args.download_only, args.train_only, args.all]):
        args.all = True

    dataset_path = args.data

    if args.download_only or args.all:
        try:
            download_and_process_dataset(dataset_path, force_download=args.force_download)
        except Exception as exc:
            logger.error("Dataset preparation failed: %s", exc)
            if not os.path.exists(dataset_path):
                return 1
            logger.warning("Continuing with the existing dataset at %s", dataset_path)

    if args.download_only:
        return 0

    if not os.path.exists(dataset_path):
        logger.error("Dataset not found at %s. Run `python train.py --download-only` first.", dataset_path)
        return 1

    if args.backend == "baseline":
        output = args.output or config.BASELINE_MODEL_PATH
        ok = train_baseline_classifier(dataset_path, output)
    else:
        output = args.output or config.CLASSIFIER_MODEL_PATH
        ok = train_transformer_classifier(dataset_path, output, epochs=args.epochs)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
