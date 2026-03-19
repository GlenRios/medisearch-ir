
# MediSearch-IR

> Hybrid information retrieval system combining BM25 and dense vector search over a medical Wikipedia corpus.

MediSearch-IR is an academic information retrieval project that implements a complete hybrid search pipeline over a domain-specific corpus of medical Wikipedia articles. The system combines sparse lexical matching via BM25 and semantic similarity via dense embeddings, merges their rankings through Reciprocal Rank Fusion (RRF), and applies a cross-encoder reranker as a final precision pass. Results are visualized through a Streamlit web interface that shows all three systems side by side.

---

## Table of contents

- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Modules](#modules)
- [Installation](#installation)
- [Usage](#usage)
- [Tech stack](#tech-stack)
- [Documentation](#documentation)
- [References](#references)

---

## Architecture

```
Wikipedia API
      │
      ▼
┌─────────────┐
│  M0 · Corpus│  ~300 medical articles via wikipedia-api
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  M1 · Ingest│  clean → sentence split → chunk → lemmatize (spaCy)
└──────┬──────┘
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
┌─────────────────┐                  ┌──────────────────────┐
│ M2 · BM25 Index │                  │ M2 · FAISS HNSW Index│
│ rank_bm25       │                  │ all-MiniLM-L6-v2     │
└────────┬────────┘                  └──────────┬───────────┘
         │                                       │
         └───────────────┬───────────────────────┘
                         │
                         ▼
               ┌──────────────────┐
               │  M3 · Retrieval  │  query → BM25 top-20 + FAISS top-20
               └────────┬─────────┘
                         │
                         ▼
               ┌──────────────────┐
               │  M4 · Fusion     │  RRF → cross-encoder reranker
               └────────┬─────────┘
                         │
                         ▼
               ┌──────────────────┐
               │  Streamlit app   │  interactive search interface
               └──────────────────┘
```

---

## Project structure

```
medisearch-ir/
│
├── data/
│   ├── raw/
│   │   ├── corpus.json              # downloaded Wikipedia articles
│   │   └── corpus_stats.json
│   └── processed/
│       ├── chunks.json              # lemmatized chunks (M1 output)
│       └── ingestion_stats.json
│
├── indexes/
│   ├── bm25.pkl                     # serialized BM25 index
│   ├── faiss.index                  # HNSW vector index
│   ├── faiss_ids.json               # chunk_id list (FAISS position map)
│   ├── chunk_map.json               # chunk_id → metadata lookup
│   └── index_stats.json
│
├── src/
│   ├── module0_corpus.py            # M0: corpus downloader
│   ├── ingestion/                   # M1: text preprocessing
│   │   ├── cleaner.py
│   │   ├── chunker.py
│   │   ├── lemmatizer.py
│   │   └── pipeline.py
│   ├── indexing/                    # M2: index construction
│   │   ├── embedder.py
│   │   ├── bm25_indexer.py
│   │   ├── faiss_indexer.py
│   │   └── pipeline.py
│   ├── retrieval/                   # M3: query execution
│   │   ├── query_processor.py
│   │   ├── bm25_retriever.py
│   │   ├── faiss_retriever.py
│   │   └── pipeline.py
│   └── fusion/                      # M4: RRF + reranker
│       ├── rrf.py
│       ├── reranker.py
│       └── pipeline.py
│
├── app/
│   ├── app.py                       # Streamlit entry point
│   ├── state.py                     # pipeline caching + search logic
│   └── components/
│       ├── sidebar.py
│       ├── search_bar.py
│       ├── result_card.py
│       ├── columns.py
│       └── comparison.py
│
├── docs/
│   ├── M1_INGESTION.md
│   ├── M2_INDEXING.md
│   ├── M3_RETRIEVAL.md
│   └── M4_FUSION.md
│
├── notebooks/
│   └── experiments.ipynb
│
├── tests/
│   └── test_retrieval.py
│
├── requirements.txt
└── README.md
```

---

## Modules

### M0 — Corpus downloader

Downloads ~300 Wikipedia articles from the Medicine & Health domain using the `wikipedia-api` library. Articles are fetched from 15 categories (Cardiology, Neurology, Oncology, Pharmacology, etc.) and saved as structured JSON with metadata.

```bash
python src/module0_corpus.py
```

Output: `data/raw/corpus.json`

---

### M1 — Ingestion & preprocessing

Cleans raw article text, splits it into sentences with spaCy, groups sentences into overlapping chunks using a sliding window (size=5, overlap=1), and lemmatizes each chunk for BM25 indexing.

```bash
python -m src.ingestion
```

Output: `data/processed/chunks.json`

Each chunk carries two representations:
- `text` — raw text used for embedding generation
- `tokens` — lemmatized tokens used for BM25

---

### M2 — Indexing

Builds both search indexes from the processed chunks.

```bash
python -m src.indexing
```

| Index | Type | Library | Output |
|---|---|---|---|
| Lexical | BM25Okapi (k1=1.5, b=0.75) | `rank-bm25` | `indexes/bm25.pkl` |
| Semantic | HNSW (M=32, ef=200) | `faiss-cpu` | `indexes/faiss.index` |

Embeddings are generated with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, L2-normalized). The model runs fully locally — no API key required.

---

### M3 — Retrieval

Preprocesses the query with the same spaCy pipeline as M1 and executes it against both indexes in parallel, returning top-20 results from each.

```bash
python -m src.retrieval   # interactive CLI
```

The query goes through two parallel paths:
- BM25 receives lemmatized tokens for term matching
- FAISS receives the cleaned text for embedding and cosine similarity search

---

### M4 — Fusion & re-ranking

Combines the two rankings from M3 using Reciprocal Rank Fusion (RRF) and then applies a cross-encoder reranker for a final precision pass.

```bash
python -m src.fusion   # interactive CLI
```

**Step 1 — RRF:**

```
RRF_score(d) = 1/(60 + rank_BM25(d)) + 1/(60 + rank_FAISS(d))
```

Requires only rank positions — not raw scores — making it robust to the scale difference between BM25 and cosine similarity.

**Step 2 — Cross-encoder:**

The top-20 RRF candidates are re-scored by `cross-encoder/ms-marco-MiniLM-L-6-v2`, which evaluates each `(query, passage)` pair jointly for higher accuracy. Final output: top-10 results.

---

### App — Streamlit interface

Interactive web search interface showing results from all three systems.

```bash
streamlit run app/app.py
```

Features:
- Search bar with example queries
- Three-column layout: BM25 · FAISS · Hybrid results
- Expandable result cards with score, category, and source link
- Unified ranking comparison table
- Sidebar with corpus and index statistics

---

## Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/medisearch-ir.git
cd medisearch-ir

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm
```

---

## Usage

Run the modules in order — each one depends on the output of the previous.

```bash
# Step 1 — Download corpus (~300 Wikipedia articles)
python src/module0_corpus.py

# Step 2 — Preprocess and chunk articles
python -m src.ingestion

# Step 3 — Build BM25 and FAISS indexes
#          (downloads ~150 MB of models on first run)
python -m src.indexing

# Step 4 — Launch the web interface
streamlit run app/app.py
```

To test retrieval and fusion from the terminal before launching the UI:

```bash
python -m src.retrieval   # BM25 + FAISS side by side
python -m src.fusion      # full hybrid pipeline with final ranking
```

---

## Tech stack

| Component | Library | Version |
|---|---|---|
| Corpus acquisition | `wikipedia-api` | 0.7.1 |
| NLP preprocessing | `spacy` | 3.7.4 |
| Lexical indexing | `rank-bm25` | 0.2.2 |
| Dense embeddings | `sentence-transformers` | 3.0.1 |
| Vector indexing | `faiss-cpu` | 1.8.0 |
| Cross-encoder reranker | `sentence-transformers` | 3.0.1 |
| Web interface | `streamlit` | 1.36.0 |
| Numerical ops | `numpy` | 1.26.4 |

---

## Documentation

Detailed documentation for each module is available in `docs/`:

| File | Description |
|---|---|
| `docs/M1_INGESTION.md` | Text cleaning, chunking strategy, spaCy lemmatization |
| `docs/M2_INDEXING.md` | BM25 parameters, HNSW configuration, embedding model |
| `docs/M3_RETRIEVAL.md` | Query preprocessing, retriever design, output schema |
| `docs/M4_FUSION.md` | RRF algorithm, cross-encoder architecture, design decisions |

---
