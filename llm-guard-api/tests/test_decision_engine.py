"""Tests for the rule-based decision engine."""

import pytest

from guard import Decision, DecisionEngine


@pytest.mark.parametrize(
    "regex_flag,regex_score,intent,intent_score,expected",
    [
        # Clean prompt, confident benign classification.
        (False, 0.0, "benign", 0.95, Decision.ALLOW),
        # Classifier is sure it is an attack.
        (False, 0.0, "malicious", 0.85, Decision.BLOCK),
        # Both layers agree it is an attack.
        (True, 0.9, "malicious", 0.9, Decision.BLOCK),
        # Definitive signature plus a non-benign classification.
        (True, 1.0, "suspicious", 0.55, Decision.BLOCK),
        # Suspicious intent alone gets neutralised, not refused.
        (True, 0.7, "suspicious", 0.8, Decision.SANITIZE),
        (False, 0.0, "suspicious", 0.6, Decision.SANITIZE),
        # Medium regex hit with a benign classification: sanitize, do not block.
        (True, 0.7, "benign", 0.9, Decision.SANITIZE),
        # Low-severity keyword only.
        (True, 0.3, "benign", 0.9, Decision.ALLOW),
        # Malicious but the classifier is unsure -> neutralise, do not refuse.
        (False, 0.0, "malicious", 0.5, Decision.SANITIZE),
        (False, 0.0, "malicious", 0.79, Decision.SANITIZE),
        # Below even the sanitize gate there is nothing to act on.
        (False, 0.0, "malicious", 0.2, Decision.ALLOW),
    ],
)
def test_decision_matrix(decision_engine, regex_flag, regex_score, intent, intent_score, expected):
    result = decision_engine.decide(
        regex_flag=regex_flag, regex_score=regex_score, intent=intent, intent_score=intent_score
    )
    assert result.decision is expected
    assert result.rule_matched
    assert result.reasoning
    assert 0.0 <= result.confidence <= 1.0


def test_combined_score_is_reported(decision_engine):
    """The blended signal used to be computed and thrown away."""
    result = decision_engine.decide(
        regex_flag=True, regex_score=1.0, intent="malicious", intent_score=1.0
    )
    expected = decision_engine.regex_weight * 1.0 + decision_engine.intent_weight * 1.0
    assert result.combined_score == pytest.approx(expected)


def test_benign_prompt_does_not_contribute_risk(decision_engine):
    """A confidently benign classification must not inflate the combined score."""
    result = decision_engine.decide(
        regex_flag=False, regex_score=0.0, intent="benign", intent_score=0.99
    )
    assert result.combined_score == pytest.approx(0.0)


def test_allow_confidence_tracks_benign_confidence(decision_engine):
    """A 95%-confident benign verdict should be a 95%-confident ALLOW."""
    result = decision_engine.decide(
        regex_flag=False, regex_score=0.0, intent="benign", intent_score=0.95
    )
    assert result.decision is Decision.ALLOW
    assert result.confidence == pytest.approx(0.95)


def test_thresholds_are_configurable():
    strict = DecisionEngine(malicious_threshold=0.5)
    lenient = DecisionEngine(malicious_threshold=0.99)
    kwargs = dict(regex_flag=False, regex_score=0.0, intent="malicious", intent_score=0.6)

    assert strict.decide(**kwargs).decision is Decision.BLOCK
    assert lenient.decide(**kwargs).decision is not Decision.BLOCK


def test_scores_outside_range_are_clamped(decision_engine):
    result = decision_engine.decide(
        regex_flag=True, regex_score=5.0, intent="malicious", intent_score=-2.0
    )
    assert 0.0 <= result.combined_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0


def test_malicious_is_never_treated_more_leniently_than_suspicious(decision_engine):
    """The severity ordering must be monotonic across intents."""
    rank = {Decision.ALLOW: 0, Decision.SANITIZE: 1, Decision.BLOCK: 2}
    for score in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        suspicious = decision_engine.decide(False, 0.0, "suspicious", score)
        malicious = decision_engine.decide(False, 0.0, "malicious", score)
        assert rank[malicious.decision] >= rank[suspicious.decision], (
            f"at score {score}: malicious -> {malicious.decision}, "
            f"suspicious -> {suspicious.decision}"
        )


def test_corroborated_block_is_not_less_confident_than_either_layer(decision_engine):
    """Two agreeing layers used to report the *weaker* of the two scores."""
    result = decision_engine.decide(
        regex_flag=True, regex_score=1.0, intent="malicious", intent_score=0.72
    )
    assert result.decision is Decision.BLOCK
    assert result.confidence >= 0.72


def test_safe_response_is_non_empty(decision_engine):
    message = decision_engine.get_safe_response()
    assert isinstance(message, str)
    assert len(message) > 20
