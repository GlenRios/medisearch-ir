"""
src/ingestion/
--------------
M1 — Ingestion & Preprocessing package.

Public interface
----------------
    from ingestion.pipeline import run
    run()

Modules
-------
cleaner.py    — text cleaning (regex, unicode normalization)
chunker.py    — sliding window sentence chunking
lemmatizer.py — spaCy model loading, sentence splitting, lemmatization
pipeline.py   — orchestrates the full M1 pipeline
"""