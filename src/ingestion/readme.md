# Module 1 — Ingestion & Preprocessing

## Overview

Module 1 is responsible for transforming the raw Wikipedia corpus produced by M0 into a structured collection of text chunks ready for indexing. Each chunk carries both its **raw text** (used for embedding generation in M2) and its **lemmatized token list** (used for BM25 indexing in M2).

---

## Location

```
src/ingestion/
├── __init__.py
├── __main__.py
├── cleaner.py
├── chunker.py
├── lemmatizer.py
└── pipeline.py
```

---

## Pipeline

```
data/raw/corpus.json
        │
        ▼
┌───────────────────┐
│   cleaner.py      │  Remove markup, normalize unicode, collapse whitespace
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  lemmatizer.py    │  Split cleaned text into sentences (spaCy senter/sentencizer)
│  get_sentences()  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   chunker.py      │  Group sentences into overlapping windows → RawChunk list
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  lemmatizer.py    │  Lemmatize + filter each chunk → token list
│  lemmatize()      │
└────────┬──────────┘
         │
         ▼
data/processed/chunks.json
data/processed/ingestion_stats.json
```

---

## Files

### `cleaner.py`

Cleans raw Wikipedia article text before it is passed to the sentence splitter.

**Key function:**

```python
clean_text(text: str) -> str
```

Applies the following steps in order:

| Step | Description |
|---|---|
| Unicode normalization | NFC normalization + replacement of curly quotes, dashes, non-breaking spaces |
| HTML tag removal | Strips residual `<tag>` elements |
| Section header removal | Removes `== Section title ==` markers |
| Bracket removal | Removes `[1]`, `[edit]`, `[citation needed]` |
| Empty parentheses removal | Removes `()` left after bracket stripping |
| Whitespace normalization | Collapses multiple newlines and spaces into single spaces |

```python
is_valid_text(text: str, min_length: int = 100) -> bool
```

Returns `True` if the cleaned text meets the minimum character threshold. Articles shorter than this are skipped entirely.

---

### `chunker.py`

Splits a list of sentences into overlapping chunks using a sliding window strategy.

**Key function:**

```python
sliding_window(
    sentences: list[str],
    size: int = 5,
    overlap: int = 1
) -> Generator[RawChunk, None, None]
```

**How the sliding window works:**

```
sentences : [s0, s1, s2, s3, s4, s5, s6, s7, s8]
size=5, overlap=1  →  step = size - overlap = 4

chunk 0 : [s0, s1, s2, s3, s4]
chunk 1 : [s4, s5, s6, s7, s8]
chunk 2 : [s8]                  ← may be filtered (too short)
```

Overlapping sentences preserve context at chunk boundaries, which improves retrieval quality for queries whose answer spans two consecutive chunks.

**`RawChunk` dataclass:**

| Field | Type | Description |
|---|---|---|
| `chunk_index` | int | Position of this chunk within the article |
| `text` | str | Joined raw text of the window |
| `sentences` | list[str] | Individual sentences forming the chunk |

---

### `lemmatizer.py`

Handles all spaCy model interactions: loading, sentence segmentation, and lemmatization.

**Model loading:**

```python
load_model(model_name: str = "en_core_web_sm") -> Language
```

Loads and caches the spaCy model for the lifetime of the process. Raises a descriptive `SystemExit` with installation instructions if the model is not found.

**Sentence splitting:**

```python
get_sentences(text: str, nlp: Language) -> list[str]
```

Detects sentence boundaries using the best available spaCy component in this order:

1. `senter` — statistical sentence recognizer (available in `md` / `lg` models)
2. `parser` — dependency parser (also sets sentence boundaries)
3. `sentencizer` — rule-based fallback, added automatically when needed (default for `en_core_web_sm`)

**Lemmatization:**

```python
lemmatize(text: str, nlp: Language) -> list[str]
```

Returns a filtered list of lowercased lemmas from the chunk text.

**Token filtering rules:**

| Rule | Excluded example |
|---|---|
| `is_stop` | the, is, are, of, in... |
| `is_punct` | `.` `,` `;` `(` `)` |
| `is_space` | `\n` `\t` ` ` |
| `like_num` | `42`, `3.14`, `2024` |
| `len < 3` | `to`, `be`, `an` |

Example:

```
input  : "The kidneys are responsible for long-term blood pressure regulation."
output : ["kidney", "responsible", "long-term", "blood", "pressure", "regulation"]
```

---

### `pipeline.py`

Orchestrates all the steps above and handles I/O.

**Entry point:**

```python
run() -> None
```

**Configuration constants:**

| Constant | Default | Description |
|---|---|---|
| `SPACY_MODEL` | `en_core_web_sm` | spaCy model to use |
| `CHUNK_SIZE` | `5` | Sentences per chunk |
| `CHUNK_OVERLAP` | `1` | Shared sentences between consecutive chunks |
| `MIN_CHUNK_TOKENS` | `30` | Minimum lemmatized tokens to keep a chunk |

**`Chunk` dataclass (output schema):**

| Field | Type | Description |
|---|---|---|
| `chunk_id` | str | Unique identifier, e.g. `wiki_00001_c003` |
| `doc_id` | str | Parent article ID |
| `title` | str | Parent article title |
| `url` | str | Wikipedia article URL |
| `category` | str | Wikipedia category |
| `chunk_index` | int | Position within the article |
| `text` | str | Raw chunk text → used for embeddings in M2 |
| `tokens` | list[str] | Lemmatized tokens → used for BM25 in M2 |
| `n_tokens` | int | Number of lemmatized tokens |

---

## Output files

### `data/processed/chunks.json`

Array of chunk objects. Example entry:

```json
{
  "chunk_id":    "wiki_00001_c003",
  "doc_id":      "wiki_00001",
  "title":       "Hypertension",
  "url":         "https://en.wikipedia.org/wiki/Hypertension",
  "category":    "Cardiology",
  "chunk_index": 3,
  "text":        "The kidneys are responsible for long-term blood pressure regulation...",
  "tokens":      ["kidney", "responsible", "long-term", "blood", "pressure", "regulation"],
  "n_tokens":    47
}
```

### `data/processed/ingestion_stats.json`

Processing summary. Example:

```json
{
  "articles_loaded":        300,
  "articles_processed":     294,
  "articles_skipped":       6,
  "total_chunks":           3812,
  "avg_tokens_per_chunk":   54,
  "chunk_size_sentences":   5,
  "chunk_overlap":          1,
  "min_chunk_tokens":       30,
  "chunks_per_category": {
    "Cardiology":   310,
    "Neurology":    287,
    "Oncology":     341
  }
}
```

---

## How to run

Make sure M0 has been executed and `data/raw/corpus.json` exists.

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model (only needed once)
python -m spacy download en_core_web_sm

# Run M1
python -m src.ingestion
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `spacy` | 3.7.4 | NLP pipeline: tokenization, POS tagging, lemmatization, sentence segmentation |
| `tqdm` | 4.66.4 | Progress bar during article processing |

---

## Design decisions

**Why spaCy for lemmatization instead of NLTK?**
spaCy's lemmatizer is context-aware — it uses POS tags to select the correct lemma (e.g. `better` → `good` as adjective, `better` → `well` as adverb). NLTK's `WordNetLemmatizer` requires the POS tag to be passed manually, making it more error-prone without an additional tagger.

**Why a sliding window with overlap instead of fixed chunks?**
Fixed chunks can split a sentence mid-thought, causing the relevant context to land in two different chunks. The 1-sentence overlap ensures that queries whose answer spans a boundary are still captured in at least one chunk.

**Why discard chunks shorter than 30 tokens?**
Very short chunks (stubs, captions, list items) carry little retrievable information and add noise to both BM25 and the embedding space. The 30-token threshold empirically removes these without discarding meaningful content.

---

## Related modules

| Module | Depends on M1? | Description |
|---|---|---|
| M0 | No — M1 depends on M0 | Produces `data/raw/corpus.json` |
| M2 | Yes | Reads `chunks.json` to build BM25 and FAISS indexes |
| M3 | Indirectly | Uses the indexes built from M1 output |