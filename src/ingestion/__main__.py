"""
Allows running the ingestion package directly:
    python -m src.ingestion
"""

from .pipeline import run

if __name__ == "__main__":
    run()