"""Conservative transcript normalization for metric calculation.

Scope of Milestone 0 normalization
----------------------------------

This module intentionally implements only transformations that are safe for any
Assamese text and do not require linguistic evidence:

1. Unicode normalization (``NFC`` by default; ``NFD``/``NFKC``/``NFKD``/``NONE``
   selectable). This matters for code-switched Latin text (``e`` + U+0301 vs
   ``é``) and keeps decomposed input comparable to composed input.
2. Collapsing of any whitespace run to a single space (U+0020). This happens
   *before* character removal so that tabs/newlines can never merge two words.
3. Removal of Unicode control characters (category ``Cc``) and format
   characters (category ``Cf``) *except* ZWNJ (U+200C) and ZWJ (U+200D), which
   are meaningful in Indic scripts and are preserved.
4. Stripping leading/trailing whitespace.

It deliberately does **not**:

* remove or rewrite punctuation (the Assamese danda ``।`` is kept),
* case-fold anything (Assamese has no case, and code-switching is analysed later),
* remove Assamese letters, dependent vowel signs, hasanta/virama, nukta,
  anusvara, visarga, digits or any other script character,
* apply Assamese-specific linguistic normalization (spelling variants, numeral
  conversion, transliteration, ...) — no such rules are applied without
  evidence.

The verbatim transcript is never modified in place: callers keep the raw string
(e.g. :class:`~assamese_asr.data.schema.ManifestRecord.transcript`) and derive
the normalized form with :func:`normalize_transcript`.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

ALLOWED_NORMALIZATION_FORMS: Final[frozenset[str]] = frozenset(
    {"NFC", "NFD", "NFKC", "NFKD", "NONE"}
)

DEFAULT_NORMALIZATION_FORM: Final = "NFC"

# ZWNJ/ZWJ: format characters that carry meaning in Indic scripts.
PRESERVED_FORMAT_CHARACTERS: Final[frozenset[str]] = frozenset({"\u200c", "\u200d"})

_DISCARDED_CATEGORIES: Final[frozenset[str]] = frozenset({"Cc", "Cf"})
_WHITESPACE_RUN: Final = re.compile(r"\s+", flags=re.UNICODE)


def validate_normalization_form(form: str) -> None:
    """Raise :class:`ValueError` if ``form`` is not a supported normalization form."""
    if form not in ALLOWED_NORMALIZATION_FORMS:
        allowed = ", ".join(sorted(ALLOWED_NORMALIZATION_FORMS))
        raise ValueError(f"unsupported unicode normalization form: {form!r} (allowed: {allowed})")


def _is_discarded(character: str) -> bool:
    return (
        unicodedata.category(character) in _DISCARDED_CATEGORIES
        and character not in PRESERVED_FORMAT_CHARACTERS
    )


def normalize_transcript(text: str, *, form: str = DEFAULT_NORMALIZATION_FORM) -> str:
    """Return the conservative normalized form of ``text`` used for metrics.

    The raw string passed in is not modified.
    """
    if not isinstance(text, str):
        raise TypeError(f"transcript must be a string, got {type(text).__name__}")
    validate_normalization_form(form)

    working = text if form == "NONE" else unicodedata.normalize(form, text)
    # Whitespace first: tabs/newlines are control characters, and dropping them
    # before collapsing would silently merge two words into one.
    working = _WHITESPACE_RUN.sub(" ", working)
    working = "".join(character for character in working if not _is_discarded(character))
    return working.strip()
