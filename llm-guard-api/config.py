"""Central configuration. Every value is overridable through the environment."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_list(name: str) -> list:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
GUARD_DIR = PROJECT_ROOT / "guard"
MODELS_DIR = GUARD_DIR / "models"

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Classifier backend: "baseline" (TF-IDF + logistic regression, CPU, no download)
# or "transformer" (fine-tuned DeBERTa, needs torch and a trained checkpoint).
CLASSIFIER_BACKEND = os.getenv("CLASSIFIER_BACKEND", "baseline").strip().lower()

# Model paths - supports both Colab (Google Drive) and local training.
# Colab notebook saves to: /content/drive/My Drive/llm-guard/intent_classifier
# Local training saves to: guard/models/intent_classifier
CLASSIFIER_MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH", str(MODELS_DIR / "intent_classifier"))
TOKENIZER_PATH = CLASSIFIER_MODEL_PATH  # tokenizer lives beside the weights
BASELINE_MODEL_PATH = os.getenv("BASELINE_MODEL_PATH", str(MODELS_DIR / "baseline_classifier.joblib"))

# Security settings
MAX_PROMPT_LENGTH = _env_int("MAX_PROMPT_LENGTH", 2000)
SANITIZATION_LEVEL = os.getenv("SANITIZATION_LEVEL", "medium").strip().lower()

# Classification thresholds. A prompt whose malicious probability lands between
# SUSPICIOUS_THRESHOLD and MALICIOUS_THRESHOLD is treated as "suspicious" and
# gets sanitized rather than blocked.
INTENT_CLASSIFIER_THRESHOLD = _env_float("INTENT_CLASSIFIER_THRESHOLD", 0.6)
SUSPICIOUS_THRESHOLD = _env_float("SUSPICIOUS_THRESHOLD", 0.4)
MALICIOUS_THRESHOLD = _env_float("MALICIOUS_THRESHOLD", 0.7)

# Decision engine weights and gates
REGEX_WEIGHT = _env_float("REGEX_WEIGHT", 0.4)
INTENT_WEIGHT = _env_float("INTENT_WEIGHT", 0.6)
DECISION_SUSPICIOUS_THRESHOLD = _env_float("DECISION_SUSPICIOUS_THRESHOLD", 0.5)
DECISION_MALICIOUS_THRESHOLD = _env_float("DECISION_MALICIOUS_THRESHOLD", 0.8)

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Training data
TRAINING_DATA_PATH = DATA_DIR / "prompts.csv"
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME", "xTRam1/safe-guard-prompt-injection")
TEST_SPLIT = _env_float("TEST_SPLIT", 0.2)
VALIDATION_SPLIT = _env_float("VALIDATION_SPLIT", 0.1)

# Intent classes
INTENT_CLASSES = ["benign", "suspicious", "malicious"]
INTENT_TO_ID = {"benign": 0, "suspicious": 1, "malicious": 2}
ID_TO_INTENT = {v: k for k, v in INTENT_TO_ID.items()}

# Model training metadata (written by the notebook / train.py)
MODEL_METADATA_PATH = MODELS_DIR / "config.json"
TRAINING_METRICS_PATH = MODELS_DIR / "training_metrics.json"

# API server
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = _env_int("API_PORT", 8000)
API_KEYS = _env_list("API_KEYS")

# Weight files a transformers checkpoint may use.
_WEIGHT_FILES = ("pytorch_model.bin", "model.safetensors")


def has_transformer_weights(path: str) -> bool:
    """True when `path` looks like a saved transformers checkpoint."""
    if not path or not os.path.isdir(path):
        return False
    return any(os.path.exists(os.path.join(path, name)) for name in _WEIGHT_FILES)


def get_trained_model_path() -> str:
    """
    Detect the fine-tuned transformer checkpoint. Checks, in order:
    1. CLASSIFIER_MODEL_PATH (env var or default)
    2. guard/models/intent_classifier
    3. ./intent_classifier

    Returns the first path that holds real weights, otherwise the default path
    (the caller falls back to a pre-trained model).
    """
    candidates = [
        CLASSIFIER_MODEL_PATH,
        str(MODELS_DIR / "intent_classifier"),
        "./intent_classifier",
    ]

    for path in candidates:
        if has_transformer_weights(path):
            return path

    return CLASSIFIER_MODEL_PATH
