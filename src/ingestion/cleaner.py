"""
cleaner.py
----------
Responsible for cleaning raw Wikipedia article text before
it is passed to the sentence splitter and chunker.

Removes markup artifacts, normalizes whitespace and unicode,
and strips content that adds noise to retrieval (section headers,
citation brackets, empty parentheses).
"""

import re
import unicodedata


# ---------------------------------------------------------------------------
# Compiled patterns (compiled once at import time for performance)
# ---------------------------------------------------------------------------

_SECTION_HEADER = re.compile(r"\n==+[^=]+=+\n")   # == Section title ==
_BRACKETS       = re.compile(r"\[.*?\]")            # [1], [edit], [citation]
_PARENS_EMPTY   = re.compile(r"\(\s*\)")            # leftover empty ()
_MULTI_NEWLINE  = re.compile(r"\n{2,}")             # 2+ consecutive newlines
_MULTI_SPACE    = re.compile(r" {2,}")              # 2+ consecutive spaces
_HTML_TAG       = re.compile(r"<[^>]+>")            # residual HTML tags


# ---------------------------------------------------------------------------
# Unicode normalization map
# ---------------------------------------------------------------------------

_UNICODE_MAP = str.maketrans({
    "\u2019": "'",    # right single quotation mark → apostrophe
    "\u2018": "'",    # left  single quotation mark → apostrophe
    "\u201c": '"',    # left  double quotation mark
    "\u201d": '"',    # right double quotation mark
    "\u2013": "-",    # en dash
    "\u2014": "-",    # em dash
    "\u00a0": " ",    # non-breaking space
    "\u2026": "...",  # ellipsis
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Full cleaning pipeline for a raw Wikipedia article text.

    Steps
    -----
    1. Normalize unicode characters
    2. Remove HTML tags
    3. Remove section header markers (== Title ==)
    4. Remove citation / edit brackets ([1], [edit])
    5. Remove empty parentheses
    6. Collapse multiple newlines and spaces
    7. Strip leading/trailing whitespace

    Parameters
    ----------
    text : str
        Raw article text as returned by the wikipedia-api library.

    Returns
    -------
    str
        Clean, normalized text ready for sentence splitting.
    """
    if not text:
        return ""

    text = _normalize_unicode(text)
    text = _HTML_TAG.sub(" ", text)
    text = _SECTION_HEADER.sub(" ", text)
    text = _BRACKETS.sub("", text)
    text = _PARENS_EMPTY.sub("", text)
    text = _MULTI_NEWLINE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def is_valid_text(text: str, min_length: int = 100) -> bool:
    """
    Return True if the cleaned text meets the minimum length threshold.

    Parameters
    ----------
    text       : str  — cleaned article text
    min_length : int  — minimum character count (default 100)
    """
    return bool(text) and len(text) >= min_length


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_unicode(text: str) -> str:
    """
    Apply unicode normalization (NFC) and replace known problem characters.
    """
    text = unicodedata.normalize("NFC", text)
    return text.translate(_UNICODE_MAP)