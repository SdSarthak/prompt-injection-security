"""LLM Guard package for prompt injection detection and mitigation."""

import logging
from typing import Optional

import config

from .results import ClassificationResult, RegexResult
from .regex_rules import RegexFilter
from .decision_engine import Decision, DecisionEngine, DecisionResult
from .sanitizer import PromptSanitizer, SanitizationLevel
from .baseline_classifier import BaselineIntentClassifier, ModelNotTrainedError

logger = logging.getLogger(__name__)

__all__ = [
    "RegexFilter",
    "RegexResult",
    "IntentClassifier",
    "BaselineIntentClassifier",
    "ClassificationResult",
    "ModelNotTrainedError",
    "DecisionEngine",
    "Decision",
    "DecisionResult",
    "PromptSanitizer",
    "SanitizationLevel",
    "build_classifier",
]


def __getattr__(name):
    """Import the transformer classifier lazily.

    ``IntentClassifier`` pulls in torch and transformers, which take several
    seconds and hundreds of megabytes. The baseline backend needs neither, so
    the import only happens when the name is actually touched.
    """
    if name == "IntentClassifier":
        from .intent_classifier import IntentClassifier

        return IntentClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_classifier(backend: Optional[str] = None, model_path: Optional[str] = None):
    """Return an intent classifier for the requested backend.

    Args:
        backend: "baseline" or "transformer". Defaults to ``config.CLASSIFIER_BACKEND``.
        model_path: Override the artifact path for the chosen backend.

    Returns:
        A classifier exposing ``classify`` and ``batch_classify``.

    The transformer backend falls back to the baseline when torch is missing or
    no fine-tuned checkpoint exists, because an untrained transformer head
    returns random predictions and would silently disable the guard.
    """
    backend = (backend or config.CLASSIFIER_BACKEND or "baseline").strip().lower()

    if backend == "baseline":
        return BaselineIntentClassifier(model_path=model_path)

    if backend == "transformer":
        resolved = model_path or config.get_trained_model_path()
        if not config.has_transformer_weights(resolved):
            logger.warning(
                "No fine-tuned checkpoint at %s; falling back to the baseline backend. "
                "Train one with `python train.py --backend transformer`.",
                resolved,
            )
            return BaselineIntentClassifier()
        try:
            from .intent_classifier import IntentClassifier

            return IntentClassifier(model_path=resolved)
        except ImportError as exc:
            logger.warning("torch/transformers unavailable (%s); using the baseline backend", exc)
            return BaselineIntentClassifier()

    raise ValueError(f"Unknown classifier backend {backend!r}; expected 'baseline' or 'transformer'")
