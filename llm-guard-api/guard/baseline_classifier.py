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

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib

import config
from .results import ClassificationResult

logger = logging.getLogger(__name__)

BACKEND_NAME = "baseline"

# Labels the underlying binary model is trained on.
POSITIVE_LABEL = "malicious"
NEGATIVE_LABEL = "benign"


class ModelNotTrainedError(RuntimeError):
    """Raised when a baseline model artifact is required but missing."""


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

    def summary(self) -> str:
        auc = f"{self.roc_auc:.4f}" if self.roc_auc is not None else "n/a"
        return (
            f"train={self.n_train} test={self.n_test} "
            f"accuracy={self.accuracy:.4f} precision={self.precision:.4f} "
            f"recall={self.recall:.4f} f1={self.f1:.4f} roc_auc={auc}"
        )


def _build_pipeline():
    """Build the TF-IDF + logistic regression pipeline.

    Word n-grams catch phrasing ("ignore all previous instructions"); character
    n-grams catch obfuscation and leetspeak that word tokenisation destroys.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, FeatureUnion

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=60000,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
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
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        suspicious_threshold: Optional[float] = None,
        malicious_threshold: Optional[float] = None,
        auto_train: bool = True,
    ):
        self.model_path = str(model_path or config.BASELINE_MODEL_PATH)
        self.suspicious_threshold = (
            config.SUSPICIOUS_THRESHOLD if suspicious_threshold is None else suspicious_threshold
        )
        self.malicious_threshold = (
            config.MALICIOUS_THRESHOLD if malicious_threshold is None else malicious_threshold
        )
        if self.suspicious_threshold > self.malicious_threshold:
            raise ValueError(
                "suspicious_threshold must not exceed malicious_threshold "
                f"(got {self.suspicious_threshold} > {self.malicious_threshold})"
            )

        self.pipeline = None
        self._load_or_train(auto_train=auto_train)

    # ------------------------------------------------------------------ setup

    def _load_or_train(self, auto_train: bool) -> None:
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                logger.info("Loaded baseline classifier from %s", self.model_path)
                return
            except Exception as exc:  # corrupt or version-mismatched artifact
                logger.warning("Could not load %s (%s); retraining", self.model_path, exc)

        if not auto_train:
            raise ModelNotTrainedError(
                f"No baseline model at {self.model_path}. Run `python train.py --backend baseline`."
            )

        if not os.path.exists(config.TRAINING_DATA_PATH):
            raise ModelNotTrainedError(
                f"No baseline model at {self.model_path} and no dataset at "
                f"{config.TRAINING_DATA_PATH}. Run `python train.py --download-only` first."
            )

        logger.info("No baseline model found; training one from %s", config.TRAINING_DATA_PATH)
        report = self.train_from_csv(str(config.TRAINING_DATA_PATH), model_path=self.model_path)
        logger.info("Baseline trained: %s", report.summary())

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

        texts = list(texts)
        binary = [0 if lab == NEGATIVE_LABEL else 1 for lab in labels]

        if len(texts) != len(binary):
            raise ValueError("texts and labels must be the same length")
        if len(set(binary)) < 2:
            raise ValueError("training data must contain both benign and malicious examples")

        x_train, x_test, y_train, y_test = train_test_split(
            texts, binary, test_size=test_size, stratify=binary, random_state=random_state
        )

        pipeline = _build_pipeline()
        pipeline.fit(x_train, y_train)

        y_pred = pipeline.predict(x_test)
        try:
            y_prob = pipeline.predict_proba(x_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            roc_auc = None

        self.pipeline = pipeline
        target = str(model_path or self.model_path)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, target)
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

        frame = pd.read_csv(path)
        missing = {"prompt", "label"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")

        frame = frame.dropna(subset=["prompt", "label"])
        frame = frame[frame["prompt"].astype(str).str.strip() != ""]

        return self.train(
            frame["prompt"].astype(str).tolist(),
            frame["label"].astype(str).tolist(),
            model_path=model_path,
            **kwargs,
        )

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
    classifier = BaselineIntentClassifier.__new__(BaselineIntentClassifier)
    classifier.model_path = str(model_path or config.BASELINE_MODEL_PATH)
    classifier.suspicious_threshold = config.SUSPICIOUS_THRESHOLD
    classifier.malicious_threshold = config.MALICIOUS_THRESHOLD
    classifier.pipeline = None
    return classifier.train_from_csv(
        csv_path, model_path=classifier.model_path, test_size=test_size, random_state=random_state
    )
