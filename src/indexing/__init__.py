"""
src/indexing/
-------------
M2 — Indexing package.

Modules
-------
embedder.py      — sentence-transformers model + embedding generation
bm25_indexer.py  — BM25 index build, save, load
faiss_indexer.py — FAISS index build, save, load, search
pipeline.py      — orchestrates the full M2 pipeline

Usage
-----
    python -m src.indexing
"""