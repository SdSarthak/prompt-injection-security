"""Shared pytest fixtures.

The project is a flat script layout rather than an installed package, so the
repo root has to be importable before `import config` / `import guard` work.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from guard import (  # noqa: E402
    BaselineIntentClassifier,
    DecisionEngine,
    PromptSanitizer,
    RegexFilter,
    SanitizationLevel,
)


@pytest.fixture(scope="session")
def regex_filter() -> RegexFilter:
    return RegexFilter()


@pytest.fixture(scope="session")
def decision_engine() -> DecisionEngine:
    return DecisionEngine()


@pytest.fixture(scope="session")
def sanitizer() -> PromptSanitizer:
    return PromptSanitizer(level=SanitizationLevel.MEDIUM)


@pytest.fixture(scope="session")
def classifier():
    """A trained baseline classifier, or skip if neither model nor data exist."""
    have_model = Path(config.BASELINE_MODEL_PATH).exists()
    have_data = Path(config.TRAINING_DATA_PATH).exists()
    if not (have_model or have_data):
        pytest.skip("no baseline model artifact and no training data available")
    return BaselineIntentClassifier()


@pytest.fixture(scope="session")
def guard(classifier):
    """An analysis-only guard: no Gemini client, no network."""
    from app import LLMGuard

    return LLMGuard(enable_llm=False)
