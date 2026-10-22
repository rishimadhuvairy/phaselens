# scripts/spectrum_generator and scripts/peak_finder are extensionless.
# This file uses a .py suffix so it can be run as a module.
"""Orthographic canonical spellings for symmetry labels.

The map changes spelling only. It does not merge distinct modes.

- ``A'1`` and ``A′1`` (U+2032 between A and 1) become ``A1'``.
- U+2032 becomes an ASCII apostrophe. U+2033 becomes two apostrophes,
  so ``E″`` stays distinct from ``E'``.
- ``E^1_2g`` and a standalone ``E2g`` become gold ``E1_2g``.
  ``E^2_2g`` is the shear mode and is left alone.
- ``A1g+B1g`` becomes ``A1g + B1g``. ``A1g`` and ``B1g`` are not combined.
- Subscript digits become ASCII digits. ``artifact`` and ``F2g`` are unchanged.
"""

from __future__ import annotations

import re

# Explicit variants -> canonical spelling. Longer tokens are listed first
# and applied as substrings so a display string can carry the same symbol.
ORTHOGRAPHIC_MAP: tuple[tuple[str, str], ...] = (
    ("A\u20321", "A1'"),
    ("A'1", "A1'"),
    ("E\u2033", "E''"),
    ("E\u2032", "E'"),
    ("E^1_2g", "E1_2g"),
    ("A1g+B1g", "A1g + B1g"),
)

_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_INFIX_PRIME = re.compile(r"([A-Za-z])('+)(\d+)")
_E2G_TOKEN = re.compile(r"(?<![A-Za-z0-9_])E2g(?![A-Za-z0-9_])", re.IGNORECASE)
_COMBINATION = re.compile(r"A1g\s*\+\s*B1g", re.IGNORECASE)
_CARET_E1 = re.compile(r"E\^1_2g", re.IGNORECASE)


def canonical_label(text: str) -> str:
    """Return the canonical spelling. Idempotent. Does not casefold."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    for old, new in ORTHOGRAPHIC_MAP:
        text = text.replace(old, new)
    text = text.replace("\u2033", "''")
    text = text.replace("\u2032", "'")
    text = text.translate(_SUBSCRIPT_DIGITS)
    text = _INFIX_PRIME.sub(lambda match: match.group(1) + match.group(3) + match.group(2), text)
    text = _CARET_E1.sub("E1_2g", text)
    text = _E2G_TOKEN.sub("E1_2g", text)
    text = _COMBINATION.sub("A1g + B1g", text)
    return text
