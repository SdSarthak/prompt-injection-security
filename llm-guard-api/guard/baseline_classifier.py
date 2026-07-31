"""CPU-only baseline intent classifier (TF-IDF + logistic regression).

Why this exists
---------------
The transformer classifier is only useful once it has been fine-tuned. Loading
`microsoft/deberta-v3-small` with a freshly initialised 3-way head produces
essentially random predictions, which silently makes the whole guard useless.

This module provides a classifier that is trained from the bundled dataset in
seconds on a CPU, ships as a single joblib artifact, and gives calibrated
probabilities. It is the default backend so a clean checkout behaves sensibly
before anyone touches a GPU.

The bundled dataset is binary (benign / malicious). The guard's decision engine
expects three intents, so the malicious probability is mapped onto three bands:

    p >= malicious_threshold  -> "malicious"
    p >= suspicious_threshold -> "suspicious"
    otherwise                 -> "benign"
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import joblib

import config
from .results import ClassificationResult

logger = logging.getLogger(__name__)

BACKEND_NAME = "baseline"

# Artifact schema version. Bumped when the stored structure changes; an artifact
# written by a different version is retrained rather than trusted, because a
# model whose preprocessing no longer matches the guard's is silently wrong.
ARTIFACT_FORMAT = 2

# Labels the underlying binary model is trained on.
POSITIVE_LABEL = "malicious"
NEGATIVE_LABEL = "benign"


class ModelNotTrainedError(RuntimeError):
    """Raised when a baseline model artifact is required but missing."""


def _file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """Content hash of a dataset file, streamed so large corpora stay cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_digest(text: str) -> str:
    """Stable short digest of a prompt, used to detect train/test overlap.

    Whitespace-insensitive so that a row which differs only in formatting is
    still recognised as the same example.
    """
    normalised = " ".join(str(text).split()).lower()
    return hashlib.blake2b(normalised.encode("utf-8"), digest_size=8).hexdigest()


@dataclass
class TrainingReport:
    """Summary of a baseline training run."""

    n_train: int
    n_test: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    report: str
    model_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        auc = f"{self.roc_auc:.4f}" if self.roc_auc is not None else "n/a"
        return (
            f"train={self.n_train} test={self.n_test} "
            f"accuracy={self.accuracy:.4f} precision={self.precision:.4f} "
            f"recall={self.recall:.4f} f1={self.f1:.4f} roc_auc={auc}"
        )


def _build_pipeline(n_samples: int = 0):
    """Build the TF-IDF + logistic regression pipeline.

    Word n-grams catch phrasing ("ignore all previous instructions"); character
    n-grams catch obfuscation and leetspeak that word tokenisation destroys.

    Args:
        n_samples: Size of the training corpus. Small corpora (tests, quick
            experiments) need ``min_df=1``, otherwise pruning removes every
            term and fitting raises.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, FeatureUnion

    word_min_df = 2 if n_samples >= 200 else 1
    char_min_df = 3 if n_samples >= 200 else 1

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=word_min_df,
        max_features=60000,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=char_min_df,
        max_features=60000,
        sublinear_tf=True,
        lowercase=True,
    )

    return Pipeline(
        [
            ("features", FeatureUnion([("word", word_vec), ("char", char_vec)])),
            (
                "clf",
                LogisticRegression(
                    C=4.0,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )


class BaselineIntentClassifier:
    """TF-IDF + logistic regression intent classifier.

    Args:
        model_path: Path to a joblib artifact. If omitted, ``config.BASELINE_MODEL_PATH``.
        suspicious_threshold: Malicious probability at/above which a prompt is
            reported as "suspicious".
        malicious_threshold: Malicious probability at/above which a prompt is
            reported as "malicious".
        auto_train: Train from the bundled dataset when no artifact is found.
        load: Set False for an unfitted instance (used by the training entry
            points, which are about to overwrite the artifact anyway).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        suspicious_threshold: Optional[float] = None,
        malicious_threshold: Optional[float] = None,
        auto_train: bool = True,
        load: bool = True,
    ):
        self.model_path = str(model_path or config.BASELINE_MODEL_PATH)
        self.suspicious_threshold = (
            config.SUSPICIOUS_THRESHOLD if suspicious_threshold is None else suspicious_threshold
        )
        self.malicious_threshold = (
            config.MALICIOUS_THRESHOLD if malicious_threshold is None else malicious_threshold
        )
        for name, value in (
            ("suspicious_threshold", self.suspicious_threshold),
            ("malicious_threshold", self.malicious_threshold),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be a probability in [0, 1] (got {value!r})")
        if self.suspicious_threshold > self.malicious_threshold:
            raise ValueError(
                "suspicious_threshold must not exceed malicious_threshold "
                f"(got {self.suspicious_threshold} > {self.malicious_threshold})"
            )

        self.pipeline = None
        self.metadata: Dict[str, Any] = {}
        self._train_digests: Optional[Set[str]] = None
        if load:
            self._load_or_train(auto_train=auto_train)

    # ------------------------------------------------------------------ setup

    def _load_artifact(self, path: str) -> bool:
        """Load ``path`` into this instance. False when it is unusable."""
        payload = joblib.load(path)

        if isinstance(payload, dict):
            fmt = payload.get("format")
            if fmt != ARTIFACT_FORMAT:
                logger.warning(
                    "Baseline artifact at %s is format %s, expected %s; retraining",
                    path,
                    fmt,
                    ARTIFACT_FORMAT,
                )
                return False
            self.pipeline = payload["pipeline"]
            self.metadata = dict(payload.get("metadata") or {})
            self._train_digests = payload.get("train_digests")
            return True

        # Format 1 wrote a bare sklearn pipeline with no provenance at all.
        # Usable, but nothing can be verified about it, so say so.
        logger.warning(
            "Baseline artifact at %s predates provenance tracking; its training data and "
            "held-out split are unknown. Retrain with `python train.py --backend baseline`.",
            path,
        )
        self.pipeline = payload
        self.metadata = {}
        self._train_digests = None
        return True

    def _load_or_train(self, auto_train: bool) -> None:
        if os.path.exists(self.model_path):
            try:
                if self._load_artifact(self.model_path):
                    logger.info("Loaded baseline classifier from %s", self.model_path)
                    return
            except (OSError, EOFError, KeyError, ValueError, TypeError, AttributeError) as exc:
                # Truncated, half-written or version-mismatched artifact.
                logger.warning("Could not load %s (%s); retraining", self.model_path, exc)

        if not auto_train:
            raise ModelNotTrainedError(
                f"No usable baseline model at {self.model_path}. "
                "Run `python train.py --backend baseline`."
            )

        if not os.path.exists(config.TRAINING_DATA_PATH):
            raise ModelNotTrainedError(
                f"No baseline model at {self.model_path} and no dataset at "
                f"{config.TRAINING_DATA_PATH}. Run `python train.py --download-only` first."
            )

        logger.info("No baseline model found; training one from %s", config.TRAINING_DATA_PATH)
        report = self.train_from_csv(str(config.TRAINING_DATA_PATH), model_path=self.model_path)
        logger.info("Baseline trained: %s", report.summary())

    def _save(self, target: str) -> None:
        """Persist the fitted pipeline and its provenance atomically.

        A plain ``joblib.dump`` to the live path leaves a truncated file visible
        to any other process (or uvicorn worker) that loads it mid-write, and
        two workers auto-training at once interleave their writes. Writing to a
        temporary file in the same directory and renaming makes the swap atomic.
        """
        directory = Path(target).parent
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": ARTIFACT_FORMAT,
            "pipeline": self.pipeline,
            "metadata": self.metadata,
            "train_digests": self._train_digests,
        }
        handle, tmp_path = tempfile.mkstemp(
            prefix=Path(target).name + ".", suffix=".tmp", dir=str(directory)
        )
        os.close(handle)
        try:
            joblib.dump(payload, tmp_path)
            os.replace(tmp_path, target)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    # ------------------------------------------------------------- provenance

    def was_trained_on(self, prompt: str) -> Optional[bool]:
        """Whether ``prompt`` was in this model's training split.

        Returns None when the artifact carries no provenance (format 1 or a
        pipeline handed in directly), which callers must treat as "unknown"
        rather than "no".
        """
        if not self._train_digests:
            return None
        return prompt_digest(prompt) in self._train_digests

    def count_training_overlap(self, prompts: Sequence[str]) -> Optional[int]:
        """How many of ``prompts`` the model was fit on. None when unknown."""
        if not self._train_digests:
            return None
        return sum(1 for p in prompts if prompt_digest(p) in self._train_digests)

    # --------------------------------------------------------------- training

    def train(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        model_path: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> TrainingReport:
        """Fit the pipeline and persist it.

        Args:
            texts: Prompt strings.
            labels: "benign" or "malicious" per prompt. Any other label is
                folded into the nearest binary class ("suspicious" -> malicious).
            model_path: Where to write the joblib artifact. Defaults to ``self.model_path``.
            test_size: Held-out fraction used for the reported metrics.
            random_state: Seed for the split.

        Returns:
            A :class:`TrainingReport` with held-out metrics.
        """
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split

        texts = [str(t) for t in texts]
        labels = list(labels)
        binary = [0 if lab == NEGATIVE_LABEL else 1 for lab in labels]

        if len(texts) != len(binary):
            raise ValueError("texts and labels must be the same length")
        if not 0.0 < float(test_size) < 1.0:
            raise ValueError(f"test_size must be strictly between 0 and 1 (got {test_size!r})")
        if len(set(binary)) < 2:
            raise ValueError("training data must contain both benign and malicious examples")

        duplicates = len(texts) - len({prompt_digest(t) for t in texts})
        if duplicates:
            # Identical rows on both sides of the split make the held-out score
            # a memorisation test. Say so rather than reporting an inflated
            # number as if it were generalisation.
            logger.warning(
                "%d duplicate prompt(s) in the training corpus; held-out metrics will be "
                "optimistic because copies can appear on both sides of the split",
                duplicates,
            )

        x_train, x_test, y_train, y_test = train_test_split(
            texts, binary, test_size=test_size, stratify=binary, random_state=random_state
        )

        pipeline = _build_pipeline(n_samples=len(x_train))
        pipeline.fit(x_train, y_train)

        y_pred = pipeline.predict(x_test)
        try:
            y_prob = pipeline.predict_proba(x_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_prob))
        except (ValueError, AttributeError, IndexError) as exc:
            # Single-class test split, or an estimator without probabilities.
            logger.debug("ROC-AUC unavailable: %s", exc)
            roc_auc = None

        self.pipeline = pipeline
        target = str(model_path or self.model_path)
        self._train_digests = {prompt_digest(t) for t in x_train}
        self.metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_rows": len(texts),
            "n_train": len(x_train),
            "n_test": len(x_test),
            "test_size": float(test_size),
            "random_state": int(random_state),
            "duplicate_rows": duplicates,
            "class_balance": {
                NEGATIVE_LABEL: binary.count(0),
                POSITIVE_LABEL: binary.count(1),
            },
        }
        self._save(target)
        self.model_path = target

        return TrainingReport(
            n_train=len(x_train),
            n_test=len(x_test),
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            f1=float(f1_score(y_test, y_pred, zero_division=0)),
            roc_auc=roc_auc,
            report=classification_report(
                y_test, y_pred, target_names=[NEGATIVE_LABEL, POSITIVE_LABEL], zero_division=0
            ),
            model_path=target,
            metadata=dict(self.metadata),
        )

    def train_from_csv(
        self,
        csv_path: Optional[str] = None,
        model_path: Optional[str] = None,
        **kwargs,
    ) -> TrainingReport:
        """Train from a CSV with ``prompt`` and ``label`` columns."""
        import pandas as pd

        path = str(csv_path or config.TRAINING_DATA_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training data not found at {path}")

        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"{path} is empty") from exc
        except pd.errors.ParserError as exc:
            raise ValueError(f"{path} is not valid CSV: {exc}") from exc

        missing = {"prompt", "label"} - set(frame.columns)
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {sorted(missing)}; "
                f"found {sorted(frame.columns)}"
            )

        frame = frame.dropna(subset=["prompt", "label"])
        frame = frame[frame["prompt"].astype(str).str.strip() != ""]
        if frame.empty:
            raise ValueError(f"{path} has no rows with both a prompt and a label")

        report = self.train(
            frame["prompt"].astype(str).tolist(),
            frame["label"].astype(str).tolist(),
            model_path=model_path,
            **kwargs,
        )
        self.metadata["source"] = os.path.abspath(path)
        self.metadata["source_sha256"] = _file_sha256(path)
        report.metadata = dict(self.metadata)
        # Rewrite so the persisted artifact carries the source provenance too.
        self._save(report.model_path)
        return report

    # -------------------------------------------------------------- inference

    def _require_pipeline(self):
        if self.pipeline is None:
            raise ModelNotTrainedError("Baseline classifier has no fitted pipeline")
        return self.pipeline

    def _to_intent(self, malicious_prob: float) -> Tuple[str, float]:
        """Map a malicious probability onto an intent label and a confidence."""
        if malicious_prob >= self.malicious_threshold:
            return "malicious", malicious_prob
        if malicious_prob >= self.suspicious_threshold:
            # Confidence that it sits in the suspicious band, not that it is malicious.
            band = self.malicious_threshold - self.suspicious_threshold
            position = (malicious_prob - self.suspicious_threshold) / band if band > 0 else 1.0
            return "suspicious", 0.5 + 0.5 * position
        return "benign", 1.0 - malicious_prob

    def classify(self, prompt: str) -> ClassificationResult:
        """Classify a single prompt."""
        return self.batch_classify([prompt])[0]

    def batch_classify(self, prompts: List[str]) -> List[ClassificationResult]:
        """Classify prompts in one vectorised pass."""
        pipeline = self._require_pipeline()
        cleaned = [("" if p is None else str(p)) for p in prompts]
        if not cleaned:
            return []

        probabilities = pipeline.predict_proba(cleaned)[:, 1]

        results = []
        for malicious_prob in probabilities:
            malicious_prob = float(malicious_prob)
            intent, confidence = self._to_intent(malicious_prob)
            suspicious_score = (
                malicious_prob if self.suspicious_threshold <= malicious_prob < self.malicious_threshold else 0.0
            )
            results.append(
                ClassificationResult(
                    intent=intent,
                    confidence=round(confidence, 6),
                    class_scores={
                        "benign": round(1.0 - malicious_prob, 6),
                        "suspicious": round(suspicious_score, 6),
                        "malicious": round(malicious_prob, 6),
                    },
                    backend=BACKEND_NAME,
                )
            )
        return results

    def malicious_probability(self, prompt: str) -> float:
        """Raw probability that the prompt is an injection attempt."""
        return float(self._require_pipeline().predict_proba([str(prompt)])[0, 1])


def train_baseline(
    csv_path: Optional[str] = None,
    model_path: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingReport:
    """Train and persist a baseline classifier without loading an existing one."""
    classifier = BaselineIntentClassifier(model_path=model_path, load=False)
    return classifier.train_from_csv(
        csv_path, model_path=classifier.model_path, test_size=test_size, random_state=random_state
    )
