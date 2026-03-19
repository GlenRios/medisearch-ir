"""
lemmatizer.py
-------------
Handles all spaCy model loading and token-level processing.

Responsibilities
----------------
- Load and cache the spaCy model (singleton pattern)
- Split text into sentences (used by chunker)
- Lemmatize a chunk of text and return a filtered token list

The token list produced here is what BM25 will use as its
vocabulary. Quality of lemmatization directly affects retrieval.

Filtering rules (applied in lemmatize())
-----------------------------------------
- Stopwords removed      (spaCy's built-in English stop list)
- Punctuation removed
- Whitespace tokens removed
- Numeric tokens removed
- Tokens shorter than 3 characters removed
"""

import logging
import spacy
from spacy.language import Language


logger = logging.getLogger(__name__)

# Module-level cache: model is loaded once per process
_NLP_CACHE: dict[str, Language] = {}


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def load_model(model_name: str = "en_core_web_sm") -> Language:
    """
    Load a spaCy model by name and cache it for the lifetime of the process.

    Parameters
    ----------
    model_name : str
        Name of the spaCy model to load (default: 'en_core_web_sm').

    Returns
    -------
    spacy.language.Language
        The loaded spaCy pipeline.

    Raises
    ------
    SystemExit
        If the model is not installed, with instructions to download it.
    """
    if model_name in _NLP_CACHE:
        return _NLP_CACHE[model_name]

    logger.info(f"Loading spaCy model '{model_name}'...")
    try:
        nlp = spacy.load(model_name)
    except OSError:
        raise SystemExit(
            f"\n[ERROR] spaCy model '{model_name}' is not installed.\n"
            f"Fix: python -m spacy download {model_name}\n"
        )

    nlp.max_length = 2_000_000   # handle long Wikipedia articles
    _NLP_CACHE[model_name] = nlp
    logger.info(f"Model '{model_name}' loaded successfully.")
    return nlp


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def get_sentences(text: str,
                  nlp: Language) -> list[str]:
    """
    Run spaCy sentence segmentation on a text and return sentence strings.

    Tries available components in this order:
    1. 'senter'      — present in en_core_web_md / lg
    2. 'parser'      — dependency parser also sets sentence boundaries
    3. 'sentencizer' — rule-based fallback, added temporarily if needed
                       (this is the case for en_core_web_sm)

    Parameters
    ----------
    text : str      — cleaned article text
    nlp  : Language — loaded spaCy model

    Returns
    -------
    list[str]
        Ordered sentence strings, whitespace-stripped.
    """
    available = nlp.pipe_names

    if "senter" in available:
        with nlp.select_pipes(enable=["tok2vec", "senter"]):
            doc = nlp(text)

    elif "parser" in available:
        with nlp.select_pipes(enable=["tok2vec", "parser"]):
            doc = nlp(text)

    else:
        if "sentencizer" not in available:
            nlp.add_pipe("sentencizer", last=True)
        doc = nlp(text)

    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


# ---------------------------------------------------------------------------
# Lemmatization
# ---------------------------------------------------------------------------

def lemmatize(text: str, nlp: Language) -> list[str]:
    """
    Tokenize, lemmatize, and filter tokens from a chunk of text.

    Uses 'tok2vec', 'tagger', 'attribute_ruler', and 'lemmatizer'
    components. The 'ner' and 'parser' components are disabled
    to keep processing fast.

    Parameters
    ----------
    text : str       — raw chunk text (joined sentences)
    nlp  : Language  — loaded spaCy model

    Returns
    -------
    list[str]
        Lemmatized, lowercased, filtered tokens ready for BM25 indexing.
    """
    with nlp.select_pipes(
        enable=["tok2vec", "tagger", "attribute_ruler", "lemmatizer"]
    ):
        doc = nlp(text)

    return _filter_tokens(doc)


def _filter_tokens(doc: spacy.tokens.Doc) -> list[str]:
    """
    Apply filtering rules to a spaCy Doc and return cleaned lemmas.

    Excluded tokens
    ---------------
    - is_stop      : English stopwords (the, is, are, ...)
    - is_punct     : punctuation marks
    - is_space     : whitespace tokens
    - like_num     : numeric expressions (42, 3.14, ...)
    - len < 3      : very short tokens (prepositions, abbreviations)
    """
    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and not token.like_num
        and len(token.lemma_) >= 3
    ]