"""
query_processor.py
------------------
Preprocesses a raw user query before it is sent to either retriever.

The query must go through the same cleaning and lemmatization steps
as the corpus documents in M1 — otherwise BM25 token matching breaks
(document tokens are lemmatized, so query tokens must be too).

For FAISS the raw cleaned text is used directly (the embedding model
handles tokenization internally), but it still benefits from basic
cleaning to remove noise.
"""

import logging
from dataclasses import dataclass

from src.ingestion.cleaner    import clean_text
from src.ingestion.lemmatizer import load_model, lemmatize

logger = logging.getLogger(__name__)

SPACY_MODEL = "en_core_web_sm"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ProcessedQuery:
    """
    Container for a preprocessed user query.

    Attributes
    ----------
    raw        : str        original query as typed by the user
    clean      : str        cleaned query text (used for embedding)
    tokens     : list[str]  lemmatized tokens (used for BM25)
    """
    raw:    str
    clean:  str
    tokens: list[str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_query(query: str,
                  model_name: str = SPACY_MODEL) -> ProcessedQuery:
    """
    Clean and lemmatize a raw user query.

    Parameters
    ----------
    query      : str   raw query string from the user
    model_name : str   spaCy model to use (must match M1)

    Returns
    -------
    ProcessedQuery
        Contains the original, cleaned, and tokenized versions of the query.

    Raises
    ------
    ValueError
        If the query is empty or produces no tokens after lemmatization.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    nlp    = load_model(model_name)
    clean  = clean_text(query.strip())
    tokens = lemmatize(clean, nlp)

    if not tokens:
        raise ValueError(
            f"Query '{query}' produced no tokens after lemmatization. "
            "Try a longer or more descriptive query."
        )

    logger.debug(f"Query processed — clean: '{clean}' | tokens: {tokens}")
    return ProcessedQuery(raw=query, clean=clean, tokens=tokens)