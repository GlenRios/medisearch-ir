"""
Allows running the retrieval package as an interactive CLI:
    python -m src.retrieval
"""

from .pipeline import run

if __name__ == "__main__":
    run()