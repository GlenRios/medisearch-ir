"""
Allows running the indexing package directly:
    python -m src.indexing
"""

from .pipeline import run

if __name__ == "__main__":
    run()