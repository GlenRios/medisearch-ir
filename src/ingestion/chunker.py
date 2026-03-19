"""
chunker.py
----------
Splits a list of sentences into overlapping chunks using a
sliding window strategy.

Each chunk is a group of consecutive sentences. Adjacent chunks
share `overlap` sentences so that context at boundaries is not
lost when the text is later used for retrieval.

Example (size=5, overlap=1)
---------------------------
sentences : [s0, s1, s2, s3, s4, s5, s6, s7, s8]
chunk 0   : [s0, s1, s2, s3, s4]
chunk 1   : [s4, s5, s6, s7, s8]
chunk 2   : [s8]                  ← short, may be filtered downstream
"""

from dataclasses import dataclass
from typing import Generator


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RawChunk:
    """
    A raw text chunk before lemmatization.

    Attributes
    ----------
    chunk_index : int   position of this chunk within the parent article
    text        : str   raw joined text of the window sentences
    sentences   : list  the individual sentences that form this chunk
    """
    chunk_index: int
    text:        str
    sentences:   list[str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sliding_window(sentences: list[str],
                   size: int = 5,
                   overlap: int = 1) -> Generator[RawChunk, None, None]:
    """
    Yield consecutive RawChunk objects with the specified sentence overlap.

    Parameters
    ----------
    sentences : list[str]
        Ordered list of sentences from a single article.
    size : int
        Number of sentences per chunk (default 5).
    overlap : int
        Number of sentences shared between consecutive chunks (default 1).
        Must be strictly less than `size`.

    Yields
    ------
    RawChunk
        One chunk per sliding window position.

    Raises
    ------
    ValueError
        If overlap >= size (would cause an infinite loop or empty steps).
    """
    if overlap >= size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than size ({size})."
        )

    if not sentences:
        return

    step = size - overlap

    for idx, i in enumerate(range(0, len(sentences), step)):
        window = sentences[i : i + size]
        if not window:
            continue
        yield RawChunk(
            chunk_index = idx,
            text        = " ".join(window),
            sentences   = window,
        )


def split_into_sentences(doc) -> list[str]:
    """
    Extract non-empty sentence strings from a spaCy Doc object.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        A spaCy document with sentence boundaries detected (requires
        'senter' or 'parser' component in the pipeline).

    Returns
    -------
    list[str]
        Ordered list of sentence strings, stripped of whitespace.
    """
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]