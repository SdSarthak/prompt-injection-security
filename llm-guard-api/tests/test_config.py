"""Tests for environment parsing and configuration invariants.

A misconfigured guard is a silently disabled guard: ``MALICIOUS_THRESHOLD=95``
(a percentage where a probability belongs) means nothing is ever blocked, and
nothing in the logs would say so.
"""

import importlib

import pytest

import config


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.9", 0.9),
        ("", 0.7),          # unset-ish
        ("abc", 0.7),       # unparseable
        ("95", 0.7),        # percentage instead of probability
        ("-0.5", 0.7),      # below range
        ("1.0001", 0.7),    # above range
        ("  0.55  ", 0.55),  # whitespace tolerated
    ],
)
def test_env_float_rejects_unusable_values(monkeypatch, raw, expected):
    monkeypatch.setenv("PROBE_FLOAT", raw)
    assert config._env_float("PROBE_FLOAT", 0.7) == pytest.approx(expected)


def test_env_float_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("PROBE_FLOAT", raising=False)
    assert config._env_float("PROBE_FLOAT", 0.42) == pytest.approx(0.42)


@pytest.mark.parametrize(
    "raw,expected",
    [("500", 500), ("0", 0), ("-1", 2000), ("2.5", 2000), ("lots", 2000)],
)
def test_env_int_rejects_unusable_values(monkeypatch, raw, expected):
    monkeypatch.setenv("PROBE_INT", raw)
    assert config._env_int("PROBE_INT", 2000) == expected


def test_env_list_splits_and_trims(monkeypatch):
    monkeypatch.setenv("PROBE_LIST", " a , b ,, c ")
    assert config._env_list("PROBE_LIST") == ["a", "b", "c"]
    monkeypatch.setenv("PROBE_LIST", "   ")
    assert config._env_list("PROBE_LIST") == []


def test_validate_is_quiet_on_the_shipped_defaults():
    assert config.validate() == []


def test_validate_reports_inverted_thresholds(monkeypatch):
    monkeypatch.setattr(config, "SUSPICIOUS_THRESHOLD", 0.9)
    monkeypatch.setattr(config, "MALICIOUS_THRESHOLD", 0.4)
    problems = config.validate()
    assert any("SUSPICIOUS_THRESHOLD" in p for p in problems)


def test_validate_reports_unusable_backend_and_level(monkeypatch):
    monkeypatch.setattr(config, "CLASSIFIER_BACKEND", "magic")
    monkeypatch.setattr(config, "SANITIZATION_LEVEL", "aggresive")
    problems = config.validate()
    assert any("CLASSIFIER_BACKEND" in p for p in problems)
    assert any("SANITIZATION_LEVEL" in p for p in problems)


def test_validate_reports_a_split_that_leaves_no_training_data(monkeypatch):
    monkeypatch.setattr(config, "TEST_SPLIT", 0.7)
    monkeypatch.setattr(config, "VALIDATION_SPLIT", 0.4)
    assert any("TEST_SPLIT" in p for p in config.validate())


def test_out_of_range_threshold_in_the_environment_falls_back(monkeypatch):
    """End to end: a bad override must not reach the decision engine."""
    monkeypatch.setenv("MALICIOUS_THRESHOLD", "95")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.MALICIOUS_THRESHOLD == pytest.approx(0.7)
    finally:
        monkeypatch.delenv("MALICIOUS_THRESHOLD", raising=False)
        importlib.reload(config)


def test_has_transformer_weights_tolerates_bad_paths():
    assert config.has_transformer_weights("") is False
    assert config.has_transformer_weights("./definitely/not/here") is False
