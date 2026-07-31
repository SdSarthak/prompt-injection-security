"""Text normalisation used by the detection layers.

Why this exists
---------------
Every pattern in :mod:`guard.regex_rules` matches literal ASCII words. An
attacker only has to break the spelling for the whole high-severity tier to go
quiet:

    "Ignore all previous instructions"      -> instruction_override, severity 1.0
    "Ig<U+200B>nore all previous instructions"  -> nothing
    "Ｉｇｎｏｒｅ all previous instructions"       -> nothing
    "Ignоre all previоus instructiоns" (Cyrillic о)  -> nothing
    "I g n o r e  a l l  p r e v i o u s"  -> nothing
    "1gn0re a11 prev10us 1nstruct10ns"     -> nothing

None of those rewrites change what the downstream model reads, so the guard has
to see through them. This module produces the *matching* view of a prompt: a
form that is only ever fed to detectors, never forwarded to the LLM. Losing
information (accents, case, punctuation) is therefore free.

Two products:

``matching_forms(text)``
    One to four candidate strings to run patterns against. The plain normalised
    form is always first; the expensive de-obfuscated variants are only built
    when the text actually carries obfuscation markers, so ordinary prompts pay
    a single NFKC pass.

``detect_obfuscation(text)``
    The markers themselves. Zero-width joiners and mixed-script words inside a
    single token are, in a prompt, essentially never accidental, so their mere
    presence is a signal in its own right.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Set

__all__ = [
    "normalize_for_matching",
    "matching_forms",
    "detect_obfuscation",
    "strip_invisible",
    "collapse_letter_spacing",
    "NORMALIZER_VERSION",
]

# Bumped whenever the normalisation changes in a way that invalidates a model
# trained against the previous behaviour.
NORMALIZER_VERSION = "normalize_v1"

# Latin look-alikes from other alphabets. Restricted to glyphs that really are
# confusable at normal font sizes - a wider table would start folding genuine
# foreign-language text into nonsense.
_CONFUSABLES = {
    # Cyrillic
    "а": "a", "в": "b", "е": "e", "ѕ": "s", "і": "i", "ї": "i", "ј": "j",
    "к": "k", "м": "m", "н": "h", "о": "o", "р": "p", "с": "c", "т": "t",
    "у": "y", "х": "x", "ѵ": "v", "ԁ": "d", "ɡ": "g",
    "А": "A", "В": "B", "Е": "E", "З": "3", "І": "I", "Ј": "J", "К": "K",
    "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y",
    "Ѕ": "S", "Х": "X",
    # Greek
    "α": "a", "ο": "o", "ρ": "p", "ε": "e", "ι": "i", "κ": "k", "ν": "v",
    "τ": "t", "υ": "u", "χ": "x", "ϲ": "c", "ϳ": "j", "ѐ": "e",
    "Α": "A", "Β": "B", "Ε": "E", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X", "Ζ": "Z",
}
_CONFUSABLE_TABLE = {ord(k): v for k, v in _CONFUSABLES.items()}

# Digit/symbol substitutions. ``1`` and ``0`` are genuinely ambiguous, so the
# caller gets both readings rather than one guess.
_LEET_COMMON = {"3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g",
                "@": "a", "$": "s", "!": "i", "0": "o"}
_LEET_VARIANTS = ({"1": "i"}, {"1": "l"})

# Separators an attacker inserts between letters: "i.g.n.o.r.e", "ig-nore".
_SEPARATORS = " .·•*_-+~/\\|"

_SCRIPT_RANGES = (
    ("latin", (0x0041, 0x024F)),
    ("greek", (0x0370, 0x03FF)),
    ("cyrillic", (0x0400, 0x04FF)),
)

# >= 3 single letters split by exactly one separator each: "I g n o r e".
# English has only two single-letter words, so three in a row is not prose.
_LETTER_SPACING_RE = re.compile(
    r"(?<![^\W\d_])(?:[^\W\d_][" + re.escape(_SEPARATORS) + r"]){2,}[^\W\d_](?![^\W\d_])"
)
# A separator sitting between two letters inside one word: "ig-nore".
_INTRAWORD_SEP_RE = re.compile(r"(?<=[^\W\d_])[.·•*_\-+~](?=[^\W\d_])")
# A digit wedged *between* letters: "prev10us". Deliberately not "7am" or "mp3",
# which are ordinary text and would make every prompt pay for leet expansion.
_DIGIT_IN_WORD_RE = re.compile(r"(?<=[^\W\d_])[0-9!@$]+(?=[^\W\d_])")

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _script_of(char: str) -> str:
    code = ord(char)
    for name, (low, high) in _SCRIPT_RANGES:
        if low <= code <= high:
            return name
    return "other"


# ASCII control characters, minus the three that carry meaning in a prompt.
_ASCII_CONTROL_TABLE = {
    code: None for code in list(range(0, 32)) + [127] if code not in (9, 10, 13)
}


def strip_invisible(text: str) -> str:
    """Drop format characters (zero-width, bidi overrides, soft hyphens).

    These are invisible to a human reviewer and to the model's tokenizer alike,
    but they split words for a regex engine, which is exactly why they are used.
    Control characters other than tab, newline and carriage return go too.
    """
    if text.isascii():
        # Fast path: ASCII has no Cf characters, so a translate table suffices
        # and the per-character unicodedata lookup is skipped entirely.
        return text.translate(_ASCII_CONTROL_TABLE)
    return "".join(
        ch
        for ch in text
        if unicodedata.category(ch) != "Cf"
        and not (unicodedata.category(ch) == "Cc" and ch not in "\t\n\r")
    )


def _strip_marks(text: str) -> str:
    """Remove combining marks so "Ignóre" reads as "ignore"."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _fold(text: str) -> str:
    """Everything in :func:`normalize_for_matching` except whitespace collapse.

    Whitespace has to survive until after letter-spacing is undone: the run
    "I g n o r e  a l l" only keeps its word boundaries while the double spaces
    are still there.
    """
    if not text:
        return ""
    text = str(text)
    if text.isascii():
        # The overwhelming majority of traffic. NFKC, confusable folding and
        # mark stripping are all no-ops on ASCII, so skip three full passes.
        return text.translate(_ASCII_CONTROL_TABLE)
    text = unicodedata.normalize("NFKC", text)
    text = strip_invisible(text)
    text = text.translate(_CONFUSABLE_TABLE)
    return _strip_marks(text)


def normalize_for_matching(text: str) -> str:
    """Return the canonical form detectors should match against.

    Compatibility-normalises (fullwidth and mathematical alphabets collapse to
    ASCII), removes invisible characters, folds Latin look-alikes, strips
    accents and collapses whitespace. Lossy by design: the result is never
    shown to a user or sent to the LLM.
    """
    return _WHITESPACE_RE.sub(" ", _fold(text)).strip()


def collapse_letter_spacing(text: str) -> str:
    """Join runs of separated single letters: "I g n o r e" -> "Ignore"."""
    return _LETTER_SPACING_RE.sub(
        lambda m: "".join(ch for ch in m.group(0) if ch not in _SEPARATORS), text
    )


def _strip_intraword_separators(text: str) -> str:
    """Delete punctuation wedged between two letters: "ig-nore" -> "ignore"."""
    return _INTRAWORD_SEP_RE.sub("", text)


def _apply_leet(text: str, extra: dict) -> str:
    table = dict(_LEET_COMMON)
    table.update(extra)
    return text.translate({ord(k): v for k, v in table.items()})


def _has_letter_spacing(text: str) -> bool:
    return _LETTER_SPACING_RE.search(text) is not None


def _has_intraword_separator(text: str) -> bool:
    return _INTRAWORD_SEP_RE.search(text) is not None


def _has_digit_obfuscation(text: str) -> bool:
    return _DIGIT_IN_WORD_RE.search(text) is not None


def _mixed_script_words(text: str) -> bool:
    """True when a single word mixes alphabets (the homoglyph signature).

    Whole sentences in Cyrillic or Greek are ordinary; a *word* that is part
    Latin and part Cyrillic is not something a keyboard produces by accident.
    """
    for word in _WORD_RE.findall(text):
        if len(word) < 3:
            continue
        scripts = {_script_of(ch) for ch in word} - {"other"}
        if len(scripts) > 1:
            return True
    return False


def detect_obfuscation(text: str) -> List[str]:
    """List the evasion techniques present in ``text``.

    Returns a sorted list of marker names; empty for ordinary prompts. Only
    ``invisible_characters`` and ``mixed_script_word`` are near-zero-false-
    positive signals; the rest are informational, which is why
    :class:`~guard.regex_rules.RegexFilter` scores only those two.
    """
    if not text:
        return []
    text = str(text)
    markers: Set[str] = set()

    if text.isascii():
        # No invisible, confusable or mixed-script characters are possible.
        return ["letter_spacing"] if _has_letter_spacing(text) else []

    if any(unicodedata.category(ch) == "Cf" for ch in text):
        markers.add("invisible_characters")
    if _mixed_script_words(text):
        markers.add("mixed_script_word")
    else:
        # A word spelled entirely in look-alikes is single-script, so the mixed
        # test misses it. Only claim it when the whole text folds to ASCII -
        # otherwise this fires on every Russian or Greek sentence.
        folded = text.translate(_CONFUSABLE_TABLE)
        if folded != text and folded.isascii():
            markers.add("confusable_characters")
    if _has_letter_spacing(text):
        markers.add("letter_spacing")

    return sorted(markers)


def matching_forms(text: str, max_forms: int = 4) -> List[str]:
    """Candidate strings to run detection patterns against.

    The normalised form always comes first. De-obfuscated variants are appended
    only when the corresponding marker is present, so the common case costs one
    NFKC pass and nothing else.

    Args:
        text: Raw prompt.
        max_forms: Hard cap on how many strings are returned, bounding the
            regex work per prompt.
    """
    folded = _fold(text)
    base = _WHITESPACE_RE.sub(" ", folded).strip()
    if not base:
        return [""]

    forms = [base]

    def _add(candidate: str) -> None:
        candidate = _WHITESPACE_RE.sub(" ", candidate).strip()
        if candidate and candidate not in forms and len(forms) < max_forms:
            forms.append(candidate)

    # Undo letter spacing on the un-collapsed text: "I g n o r e  a l l" needs
    # its double spaces to know where one word ends and the next begins.
    deobfuscated = folded
    if _has_letter_spacing(deobfuscated):
        deobfuscated = collapse_letter_spacing(deobfuscated)
    if _has_intraword_separator(deobfuscated):
        deobfuscated = _strip_intraword_separators(deobfuscated)
    _add(deobfuscated)

    if _has_digit_obfuscation(deobfuscated):
        for extra in _LEET_VARIANTS:
            _add(_apply_leet(deobfuscated, extra))

    return forms
