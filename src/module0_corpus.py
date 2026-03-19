"""
Module 0 — Corpus Downloader
=============================
Downloads ~300 Wikipedia articles on Medicine & Health (English)
and saves them as a structured JSON corpus ready for M1 ingestion.

Output
------
data/raw/corpus.json       — full article texts + metadata
data/raw/corpus_stats.json — download summary

Requirements
------------
    pip install wikipedia-api tqdm
"""

import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import wikipediaapi
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LANGUAGE        = "en"
TARGET_ARTICLES = 300
OUTPUT_DIR      = Path("data/raw")
CORPUS_FILE     = OUTPUT_DIR / "corpus.json"
STATS_FILE      = OUTPUT_DIR / "corpus_stats.json"
REQUEST_DELAY   = 0.5   # seconds between requests (be polite to the API)
MIN_TEXT_LENGTH = 500   # characters — skip stubs shorter than this

# Wikipedia categories to crawl (Medicine & Health domain)
# Each category contributes a slice of articles.
CATEGORIES = [
    "Diseases and disorders",
    "Human anatomy",
    "Pharmacology",
    "Surgery",
    "Medical specialties",
    "Symptoms",
    "Vaccines",
    "Epidemiology",
    "Cardiology",
    "Neurology",
    "Oncology",
    "Infectious diseases",
    "Nutrition",
    "Mental health",
    "Public health",
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Article:
    doc_id:   str
    title:    str
    url:      str
    language: str
    category: str
    text:     str
    summary:  str
    n_chars:  int
    n_words:  int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_wiki_client() -> wikipediaapi.Wikipedia:
    return wikipediaapi.Wikipedia(
        language=LANGUAGE,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent="ir-hybrid-project/1.0 (academic research)"
    )


def fetch_article(wiki: wikipediaapi.Wikipedia,
                  title: str,
                  category: str,
                  seen: set) -> Optional[Article]:
    """
    Fetch a single article by title.
    Returns None if the article should be skipped.
    """
    if title in seen:
        return None

    try:
        page = wiki.page(title)
    except Exception as exc:
        logging.warning(f"Error fetching '{title}': {exc}")
        return None

    if not page.exists():
        return None

    text = page.text.strip()
    if len(text) < MIN_TEXT_LENGTH:
        return None                         # skip stubs

    seen.add(title)

    return Article(
        doc_id   = f"wiki_{len(seen):05d}",
        title    = page.title,
        url      = page.fullurl,
        language = LANGUAGE,
        category = category,
        text     = text,
        summary  = page.summary[:1000],     # first 1000 chars as summary
        n_chars  = len(text),
        n_words  = len(text.split()),
    )


def collect_titles_from_category(wiki: wikipediaapi.Wikipedia,
                                 category_name: str,
                                 max_titles: int = 40) -> list[str]:
    """
    Returns up to max_titles article titles from a Wikipedia category.
    Only top-level articles (namespace 0) are included — no subcategories.
    """
    cat_page = wiki.page(f"Category:{category_name}")
    if not cat_page.exists():
        logging.warning(f"Category not found: {category_name}")
        return []

    titles = []
    for title, member in cat_page.categorymembers.items():
        if member.ns == wikipediaapi.Namespace.MAIN:   # articles only
            titles.append(title)
        if len(titles) >= max_titles:
            break

    return titles


# ---------------------------------------------------------------------------
# Main downloader
# ---------------------------------------------------------------------------

def download_corpus() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    wiki   = build_wiki_client()
    seen   = set()           # track titles already downloaded
    corpus = []              # list of Article dicts

    # Distribute target evenly across categories
    per_category = max(1, TARGET_ARTICLES // len(CATEGORIES))

    logger.info(f"Target: {TARGET_ARTICLES} articles across "
                f"{len(CATEGORIES)} categories (~{per_category} each)")

    for category in CATEGORIES:
        if len(corpus) >= TARGET_ARTICLES:
            break

        logger.info(f"Crawling category: '{category}'")
        titles = collect_titles_from_category(wiki, category, max_titles=per_category * 2)

        if not titles:
            logger.warning(f"  No titles found in '{category}', skipping.")
            continue

        category_count = 0
        for title in tqdm(titles, desc=f"  {category[:30]}", leave=False):
            if len(corpus) >= TARGET_ARTICLES:
                break
            if category_count >= per_category:
                break

            article = fetch_article(wiki, title, category, seen)
            if article:
                corpus.append(asdict(article))
                category_count += 1

            time.sleep(REQUEST_DELAY)

        logger.info(f"  Collected {category_count} articles from '{category}'. "
                    f"Total so far: {len(corpus)}")

    # -------------------------------------------------------------------
    # Save corpus
    # -------------------------------------------------------------------
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------------
    # Save stats
    # -------------------------------------------------------------------
    if corpus:
        total_words = sum(a["n_words"] for a in corpus)
        total_chars = sum(a["n_chars"] for a in corpus)
        avg_words   = total_words // len(corpus)
        cat_counts  = {}
        for a in corpus:
            cat_counts[a["category"]] = cat_counts.get(a["category"], 0) + 1

        stats = {
            "total_articles": len(corpus),
            "total_words":    total_words,
            "total_chars":    total_chars,
            "avg_words_per_article": avg_words,
            "articles_per_category": cat_counts,
        }
    else:
        stats = {"total_articles": 0}

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("=" * 50)
    logger.info(f"Done. {len(corpus)} articles saved to {CORPUS_FILE}")
    logger.info(f"Stats saved to {STATS_FILE}")
    if corpus:
        logger.info(f"Total words: {stats['total_words']:,} | "
                    f"Avg words/article: {stats['avg_words_per_article']:,}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    download_corpus()