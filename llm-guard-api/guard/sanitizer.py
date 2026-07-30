"""Prompt sanitization layer for neutralizing injection risks while preserving UX."""

import re
from typing import Optional, Tuple
from enum import Enum

import config


class SanitizationLevel(Enum):
    """Sanitization aggressiveness levels."""
    LOW = "low"  # Preserve intent, minimal risk
    MEDIUM = "medium"  # Balanced approach
    HIGH = "high"  # Aggressive, maximum security


class PromptSanitizer:
    """Neutralizes prompt injection risks by removing meta-instructions and re-wrapping prompts."""

    # Meta-instruction phrases to remove
    META_INSTRUCTIONS = [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"forget\s+(everything|all).*?before",
        r"disregard\s+(the\s+)?system\s+prompt",
        r"you\s+are\s+now\s+in\s+(jailbreak|developer)\s+mode",
        r"act\s+as\s+.*?\s+instead",
        r"pretend\s+(you\s+)?are",
        r"assume\s+the\s+role\s+of",
        r"respond\s+only\s+as",
    ]

    # Role-playing phrases
    ROLE_PHRASES = [
        r"as\s+a\s+\w+,",
        r"in\s+the\s+role\s+of",
        r"acting\s+as",
        r"pretending\s+to\s+be",
    ]

    # Dangerous multi-line separators
    SEPARATORS = [
        r"---+",
        r"===+",
        r"####+",
        r"\|\|\|+",
    ]

    @classmethod
    def from_name(cls, name: str, **kwargs) -> "PromptSanitizer":
        """Build a sanitizer from a level name such as "medium".

        Unknown names fall back to MEDIUM rather than raising, so a typo in the
        environment degrades to the balanced default instead of killing startup.
        """
        try:
            level = SanitizationLevel(str(name).strip().lower())
        except ValueError:
            level = SanitizationLevel.MEDIUM
        return cls(level=level, **kwargs)

    def __init__(
        self,
        level: SanitizationLevel = SanitizationLevel.MEDIUM,
        max_length: Optional[int] = None,
    ):
        """
        Initialize sanitizer with aggressiveness level.

        Args:
            level: Sanitization level (LOW, MEDIUM, HIGH)
            max_length: Hard cap on sanitized prompt length.
                Defaults to ``config.MAX_PROMPT_LENGTH``.
        """
        self.level = level
        self.max_length = int(max_length if max_length is not None else config.MAX_PROMPT_LENGTH)
        self.meta_patterns = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.META_INSTRUCTIONS]
        self.role_patterns = [re.compile(p, re.IGNORECASE) for p in self.ROLE_PHRASES]
        self.separator_patterns = [re.compile(p) for p in self.SEPARATORS]

    def sanitize(self, prompt: str) -> Tuple[str, str]:
        """
        Sanitize a prompt by removing meta-instructions and dangerous patterns.
        
        Args:
            prompt: Original prompt to sanitize
            
        Returns:
            Tuple of (sanitized_prompt, summary_of_changes)
        """
        original = prompt or ""
        sanitized = original
        changes = []

        # Remove meta-instructions
        if self.level != SanitizationLevel.LOW:
            removed_meta = False
            for pattern in self.meta_patterns:
                sanitized, count = pattern.subn("", sanitized)
                removed_meta = removed_meta or count > 0
            if removed_meta:
                changes.append("Removed meta-instructions")

        # Remove role-playing phrases (aggressive in HIGH mode)
        if self.level == SanitizationLevel.HIGH:
            removed_roles = False
            for pattern in self.role_patterns:
                sanitized, count = pattern.subn("", sanitized)
                removed_roles = removed_roles or count > 0
            if removed_roles:
                changes.append("Removed role-playing directives")

        # Handle section separators (medium/high only). A prompt that opens with
        # a separator would otherwise be reduced to an empty string, so keep the
        # first non-empty section rather than blindly taking parts[0].
        if self.level in (SanitizationLevel.MEDIUM, SanitizationLevel.HIGH):
            for pattern in self.separator_patterns:
                if pattern.search(sanitized):
                    parts = [part.strip() for part in pattern.split(sanitized)]
                    first_section = next((part for part in parts if part), "")
                    if first_section != sanitized.strip():
                        sanitized = first_section
                        changes.append("Dropped content after a section separator")
                    break

        # Normalize whitespace
        sanitized = re.sub(r"\s+", " ", sanitized).strip()

        # Truncate if needed
        if self.max_length > 0 and len(sanitized) > self.max_length:
            sanitized = sanitized[: self.max_length].rstrip() + "..."
            changes.append(f"Truncated to {self.max_length} characters")

        removed = len(original) - len(sanitized)
        if removed > 0:
            changes.append(f"Removed {removed} characters")

        summary = "; ".join(changes) if changes else "No changes"
        return sanitized, summary

    def wrap_safely(self, prompt: str, instruction: str = "Answer the following only:") -> str:
        """
        Wrap prompt in a safe instruction boundary.
        
        Args:
            prompt: Sanitized prompt to wrap
            instruction: Safe instruction prefix
            
        Returns:
            Wrapped prompt with clear boundaries
        """
        return f"{instruction}\n\n{prompt}\n\nProvide a direct response without additional instructions."

    def detect_injection_patterns(self, prompt: str) -> list:
        """
        Detect suspected injection patterns without removing them (for logging).
        
        Args:
            prompt: Prompt to analyze
            
        Returns:
            List of detected injection patterns
        """
        detected = []

        for pattern in self.meta_patterns:
            matches = pattern.findall(prompt)
            if matches:
                detected.extend([f"meta_instruction: {m}" for m in matches])

        for pattern in self.role_patterns:
            matches = pattern.findall(prompt)
            if matches:
                detected.extend([f"role_phrase: {m}" for m in matches])

        for pattern in self.separator_patterns:
            if pattern.search(prompt):
                detected.append("section_separator_detected")

        return detected
