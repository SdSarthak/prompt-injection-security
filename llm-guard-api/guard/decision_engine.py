"""Decision engine for determining prompt handling: ALLOW, SANITIZE, or BLOCK."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

import config


class Decision(Enum):
    """Possible decisions for prompt handling."""
    ALLOW = "allow"
    SANITIZE = "sanitize"
    BLOCK = "block"


@dataclass
class DecisionResult:
    """Result of decision engine analysis."""
    decision: Decision
    confidence: float  # 0.0 to 1.0
    reasoning: str
    rule_matched: str  # Which rule triggered the decision
    combined_score: float = 0.0  # Weighted blend of the regex and intent signals


class DecisionEngine:
    """Simple, defensible logic for determining prompt handling."""

    def __init__(
        self,
        regex_weight: Optional[float] = None,
        intent_weight: Optional[float] = None,
        suspicious_threshold: Optional[float] = None,
        malicious_threshold: Optional[float] = None,
    ):
        """
        Initialize decision engine with configurable thresholds.

        Any argument left as None is read from `config`, so deployments can tune
        the false-positive/false-negative tradeoff through the environment
        without touching code.

        Args:
            regex_weight: Weight of regex patterns in the combined score
            intent_weight: Weight of the intent classifier in the combined score
            suspicious_threshold: Intent confidence above which a suspicious
                prompt is sanitized
            malicious_threshold: Intent confidence above which a malicious
                prompt is blocked
        """
        self.regex_weight = config.REGEX_WEIGHT if regex_weight is None else regex_weight
        self.intent_weight = config.INTENT_WEIGHT if intent_weight is None else intent_weight
        self.suspicious_threshold = (
            config.DECISION_SUSPICIOUS_THRESHOLD if suspicious_threshold is None else suspicious_threshold
        )
        self.malicious_threshold = (
            config.DECISION_MALICIOUS_THRESHOLD if malicious_threshold is None else malicious_threshold
        )

    def decide(
        self,
        regex_flag: bool,
        regex_score: float,
        intent: str,  # "benign", "suspicious", "malicious"
        intent_score: float,
    ) -> DecisionResult:
        """
        Make a decision based on regex filter and intent classifier outputs.
        
        Args:
            regex_flag: Whether regex patterns were matched
            regex_score: Severity score from regex (0.0-1.0)
            intent: Classified intent ("benign", "suspicious", "malicious")
            intent_score: Confidence score from classifier (0.0-1.0)
            
        Returns:
            DecisionResult with decision, confidence, and reasoning
        """
        # Weighted blend of both signals, reported on every decision so callers
        # can log/alert on borderline traffic without re-deriving it.
        regex_score = max(0.0, min(1.0, float(regex_score)))
        intent_score = max(0.0, min(1.0, float(intent_score)))
        risk_contribution = intent_score if intent != "benign" else 0.0
        combined_score = round(
            (self.regex_weight * regex_score) + (self.intent_weight * risk_contribution), 6
        )

        # High-severity cases: Block if regex + malicious intent.
        # Two independent layers agreeing cannot be *less* certain than either
        # alone, so the confidence is the stronger signal, not the weaker one.
        if regex_flag and regex_score >= 0.8 and intent == "malicious":
            return DecisionResult(
                decision=Decision.BLOCK,
                confidence=max(regex_score, intent_score),
                reasoning="High-risk injection pattern detected with malicious intent",
                rule_matched="regex_high + intent_malicious",
                combined_score=combined_score,
            )

        # Malicious intent alone
        if intent == "malicious" and intent_score >= self.malicious_threshold:
            return DecisionResult(
                decision=Decision.BLOCK,
                confidence=intent_score,
                reasoning="Classified as malicious prompt",
                rule_matched="intent_malicious",
                combined_score=combined_score,
            )

        # A definitive injection signature paired with any non-benign intent is
        # blocked outright: two independent layers agreeing is strong evidence.
        if regex_flag and regex_score >= 1.0 and intent != "benign":
            return DecisionResult(
                decision=Decision.BLOCK,
                confidence=max(regex_score * 0.9, intent_score),
                reasoning="Definitive injection signature corroborated by the classifier",
                rule_matched="regex_definitive + intent_non_benign",
                combined_score=combined_score,
            )

        # Suspicious cases: Sanitize if suspicious intent or medium-level regex flag
        if intent == "suspicious" and intent_score >= self.suspicious_threshold:
            return DecisionResult(
                decision=Decision.SANITIZE,
                confidence=intent_score,
                reasoning="Suspicious intent detected - will sanitize",
                rule_matched="intent_suspicious",
                combined_score=combined_score,
            )

        # A malicious classification that is not confident enough to block is
        # still a stronger signal than a suspicious one, and a suspicious
        # prompt at the same score gets sanitized. Without this rule the
        # ordering inverts and the more serious verdict is handled the more
        # leniently: "malicious" at 0.79 sailed straight through as ALLOW.
        if intent == "malicious" and intent_score >= self.suspicious_threshold:
            return DecisionResult(
                decision=Decision.SANITIZE,
                confidence=intent_score,
                reasoning=(
                    "Malicious intent below the block threshold - will sanitize"
                ),
                rule_matched="intent_malicious_low_confidence",
                combined_score=combined_score,
            )

        # Regex flag with medium severity
        if regex_flag and regex_score >= 0.5:
            return DecisionResult(
                decision=Decision.SANITIZE,
                confidence=regex_score,
                reasoning="Potential injection pattern detected - will sanitize",
                rule_matched="regex_medium",
                combined_score=combined_score,
            )

        # Default: Allow benign prompts
        return DecisionResult(
            decision=Decision.ALLOW,
            confidence=intent_score if intent == "benign" else round(1.0 - combined_score, 6),
            reasoning="Prompt classified as benign - no risk detected",
            rule_matched="default_allow",
            combined_score=combined_score,
        )

    def get_safe_response(self) -> str:
        """Get a safe fallback response for blocked prompts."""
        return (
            "I cannot process this request as it appears to contain instructions that conflict "
            "with my guidelines. Please rephrase your question clearly, and I'll be happy to help."
        )
