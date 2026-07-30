"""Shared result types for the guard layers.

These live in their own module so that lightweight consumers (the regex filter,
the baseline classifier, the API schemas) can import them without pulling in
torch/transformers.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ClassificationResult:
    """Result of intent classification."""

    intent: str  # "benign", "suspicious", "malicious"
    confidence: float  # 0.0 to 1.0
    class_scores: Dict[str, float] = field(default_factory=dict)
    backend: str = "unknown"  # which classifier produced this result


@dataclass
class RegexResult:
    """Result of regex pattern matching."""

    flag: bool
    matched_patterns: List[str]
    score: float  # 0.0 to 1.0
