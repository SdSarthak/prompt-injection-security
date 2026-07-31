"""Regex-based heuristic filter for detecting obvious prompt injection attempts."""

import re
from typing import Dict, List

from .normalize import detect_obfuscation, matching_forms, normalize_for_matching
from .results import RegexResult

__all__ = ["RegexFilter", "RegexResult"]


class RegexFilter:
    """Fast first-pass filter using regex patterns to catch obvious attacks."""

    # Filler allowed between the verb and its object, e.g.
    # "ignore ALL OF THE PREVIOUS instructions". Bounded and sentence-local so
    # the pattern cannot span unrelated clauses or backtrack pathologically.
    _GAP = r"[^.!?\n]{0,40}?"

    # Words that mark the thing being overridden.
    _TARGET = (
        r"(?:instructions?|prompts?|rules?|directions?|directives?|commands?|"
        r"guidelines?|constraints?|restrictions?|protocols?|safeguards?|"
        r"context|conversation|programming|training)"
    )
    _SCOPE = r"(?:previous|prior|above|preceding|earlier|initial|original|system|all|any|your|the)"

    # High-severity patterns: instruction override attempts
    INSTRUCTION_OVERRIDE_PATTERNS = [
        # "ignore all previous instructions", "disregard your earlier rules", ...
        rf"\b(?:ignore|disregard|forget|discard|overlook|erase|delete)\b{_GAP}\b{_SCOPE}\b{_GAP}\b{_TARGET}\b",
        # "forget everything above", "forget everything you were told"
        r"\bforget\s+(?:everything|all)\b[^.!?\n]{0,40}",
        rf"\boverride\s+(?:the\s+|your\s+|all\s+|any\s+)*(?:\w+\s+){{0,2}}{_TARGET}\b",
        r"\bbypass\s+(?:all\s+|any\s+|the\s+|your\s+)*(?:restrictions?|filters?|rules?|guardrails?|safety|security|safeguards?)\b",
        r"\bdisable\s+(?:all\s+|the\s+|your\s+)*(?:safety|content|filter|filters|moderation|guardrails?|restrictions?|safeguards?)\b",
        # Classic injection headers used to smuggle a second instruction block.
        r"^\s*(?:new|updated|revised|real|actual)\s+(?:instructions?|prompt|task)\s*[:\-]",
        r"\b(?:new|updated|revised)\s+(?:instructions?|system\s+prompt)\s*[:\-]",
        r"\bend\s+of\s+(?:prompt|instructions?|context)\b",
        r"###\s*(?:instruction|system|end)\b",
    ]

    # High-severity patterns: identity override / unrestricted personas
    ROLE_HIJACKING_PATTERNS = [
        r"\byou\s+are\s+(?:now\s+)?(?:chatgpt|gpt-?[0-9]|claude|llama|gemini|bard|dan)\b",
        r"\byou\s+are\s+(?:now\s+)?an?\s+(?:unrestricted|unfiltered|uncensored|amoral|unethical|jailbroken|evil)\b",
        r"\bno\s+(?:longer\s+)?(?:bound|restricted|limited|constrained)\s+by\b",
        r"\b(?:without|with\s+no|free\s+from)\s+(?:any\s+)?(?:restrictions?|limitations?|filters?|rules?|ethical\s+guidelines?|safety\s+guidelines?)\b",
        r"\byou\s+have\s+no\s+(?:restrictions?|limits?|filters?|rules?|guidelines?|ethics)\b",
        r"\bdo\s+anything\s+now\b",
    ]

    # Medium-severity patterns: generic role-play framing.
    # Deliberately softer than ROLE_HIJACKING: "act as a translator" is a normal
    # request, so this only nudges the risk score instead of dominating it.
    ROLE_PLAY_PATTERNS = [
        r"\bpretend\s+(?:that\s+)?(?:you(?:'re|\s+are)|to\s+be)\b",
        r"\bact\s+as\s+(?:an?|the)?\s*\w+",
        r"\byou\s+will\s+(?:act|roleplay|role-play|behave|respond)\s+as\b",
        r"\bassume\s+(?:the\s+)?(?:role|persona|identity)\s+of\b",
        r"\brespond\s+only\s+as\b",
        r"\bsimulate\s+(?:being\s+)?an?\s+\w+",
    ]

    # Medium-severity patterns: system prompt disclosure
    PROMPT_DISCLOSURE_PATTERNS = [
        r"\b(?:system|initial|original|hidden|secret)\s+prompt\b",
        r"\bwhat\s+(?:are|is|was|were)\s+your\s+(?:system\s+|initial\s+|original\s+)?(?:prompt|instructions?|rules?|guidelines?)\b",
        r"\b(?:show|print|output|display|repeat|reveal|list|tell)\s+(?:me\s+)?(?:the\s+|your\s+|all\s+)*(?:system\s+|initial\s+|original\s+|hidden\s+)?(?:prompt|instructions?|directives?)\b",
        r"\brepeat\s+(?:the\s+|everything\s+)?(?:words?\s+)?above\b",
        r"\breturn\s+the\s+(?:original\s+)?system\s+prompt\b",
        r"\bwhat\s+were\s+you\s+told\b",
        # "tell me your underlying/hidden/internal instructions"
        r"\byour\s+(?:underlying|internal|hidden|secret|original|initial|base)\s+"
        r"(?:instructions?|prompts?|rules?|guidelines?|directives?|configuration)\b",
        r"\byour\s+training\s+data\b",
    ]

    # Medium-severity patterns: policy bypass
    POLICY_BYPASS_PATTERNS = [
        r"\bjailbreak\b|\bjailbroken\b",
        r"\bdeveloper\s+mode\b",
        r"\bgod\s+mode\b",
        r"\b(?:unrestricted|unfiltered|uncensored|debug|admin|sudo)\s+mode\b",
        r"\bremove\s+(?:all\s+)?(?:restrictions?|filters?|limits?|guardrails?)\b",
        r"\bmemories?\s+(?:disabled|removed|cleared)\b",
        r"\bhypothetically\s*,?\s+if\s+you\s+(?:had|could|were)\b",
        r"\bfor\s+(?:educational|research)\s+purposes\s+only\b",
    ]

    # Medium-severity patterns: dangerous code patterns
    DANGEROUS_CODE_PATTERNS = [
        r"\b(rm|del)\s+-rf\s+/\b",  # Destructive shell commands
        r"DROP\s+TABLE\b",  # SQL injection
        r"DELETE\s+FROM\b",  # SQL deletion
        r"UNION\s+SELECT\b",  # SQL injection technique
        r"exec\s*\(",  # Code execution
        r"eval\s*\(",  # Code evaluation
    ]

    # Low-severity patterns: suspicious keywords (context-dependent)
    SUSPICIOUS_KEYWORDS = [
        r"\b(payload|shellcode|exploit|vulnerability)\b",
        r"\b(private|secret|confidential|classified)\b",
        r"\b(backdoor|trojan|malware)\b",
    ]

    # Evasion techniques that are, on their own, evidence of intent. A user does
    # not insert a zero-width joiner mid-word or mix Cyrillic into an English
    # word by accident, so these carry a score even with no pattern hit.
    # Letter spacing and single-script look-alikes are reported but not scored:
    # "S P A C E D" text and non-Latin prose are both legitimate.
    OBFUSCATION_SEVERITY = {
        "invisible_characters": 0.7,
        "mixed_script_word": 0.7,
    }

    # Category -> (source patterns attribute, label used in output, severity)
    CATEGORIES = (
        ("high_override", "INSTRUCTION_OVERRIDE_PATTERNS", "instruction_override", 1.0),
        ("high_role", "ROLE_HIJACKING_PATTERNS", "role_hijacking", 1.0),
        ("medium_code", "DANGEROUS_CODE_PATTERNS", "dangerous_code", 0.8),
        ("medium_disclosure", "PROMPT_DISCLOSURE_PATTERNS", "prompt_disclosure", 0.7),
        ("medium_bypass", "POLICY_BYPASS_PATTERNS", "policy_bypass", 0.7),
        ("medium_roleplay", "ROLE_PLAY_PATTERNS", "role_play", 0.5),
        ("low_keywords", "SUSPICIOUS_KEYWORDS", "suspicious_keyword", 0.3),
    )

    def __init__(self):
        """Initialize compiled regex patterns with flags."""
        self.patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile all regex patterns with appropriate flags."""
        return {
            key: [re.compile(p, re.IGNORECASE) for p in getattr(self, attr)]
            for key, attr, _label, _severity in self.CATEGORIES
        }

    def check(self, prompt: str, deobfuscate: bool = True) -> RegexResult:
        """
        Check prompt against regex patterns.

        The patterns are matched against normalised, de-obfuscated views of the
        prompt rather than the raw bytes: a zero-width space or a Cyrillic "о"
        inside "ignore" would otherwise silence every high-severity rule while
        leaving the attack perfectly legible to the model.

        Args:
            prompt: User input prompt to check
            deobfuscate: When False, only the normalised text is matched and the
                de-obfuscated variants are skipped. Callers use this to ask
                "is this signature literally present?", which is what decides
                whether a text-substitution sanitizer could have removed it.

        Returns:
            RegexResult with flag, matched patterns, and risk score.
            The risk score is the highest severity among all matches, so a
            single high-severity hit is never diluted by low-severity noise.
        """
        if not prompt:
            return RegexResult(flag=False, matched_patterns=[], score=0.0)

        matched_patterns: List[str] = []
        seen = set()
        risk_score = 0.0

        forms = matching_forms(prompt) if deobfuscate else [normalize_for_matching(prompt)]

        for key, _attr, label, severity in self.CATEGORIES:
            for pattern in self.patterns[key]:
                for form in forms:
                    match = pattern.search(form)
                    if not match:
                        continue
                    # Overlapping patterns in the same category often capture the
                    # same span; report each distinct hit once.
                    entry = f"{label}: {match.group(0).strip()}"
                    if entry not in seen:
                        seen.add(entry)
                        matched_patterns.append(entry)
                    risk_score = max(risk_score, severity)
                    # One hit per pattern is enough; the remaining forms are
                    # rewrites of the same text.
                    break

        for marker in detect_obfuscation(prompt):
            entry = f"obfuscation: {marker}"
            if entry not in seen:
                seen.add(entry)
                matched_patterns.append(entry)
            risk_score = max(risk_score, self.OBFUSCATION_SEVERITY.get(marker, 0.0))

        return RegexResult(
            flag=len(matched_patterns) > 0,
            matched_patterns=matched_patterns,
            score=risk_score,
        )
