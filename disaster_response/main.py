"""Main driver script for the disaster response pipeline.

This script orchestrates the processing of tweets from an input TSV file,
generates headlines, retrieves contextual information, classifies each
tweet as a rumour or non-rumour, and writes the results to JSON files.

Usage::

    python -m disaster_response.main --input Dataset.tsv

The script expects a TSV file with columns ``user_id``, ``tweet``, ``location``,
``timestamp`` and ``followers``. It outputs two files in the current
working directory: ``rumors.json`` containing an array of rumour
classifications, and ``synthesized_data.json`` containing structured
information for non-rumour tweets.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

from .config import load_config, CONFIG
from .data_ingestion import load_dataset, Tweet
from .headline_generation import generate_headline
from .vector_db import VectorDB
from .search_enrichment import get_contexts
from .classification import classify_tweet


logger = logging.getLogger(__name__)


def process_tweets(
    tweets: List[Tweet],
    vector_db: VectorDB,
    progress_callback: Optional[Callable[[float], None]] = None
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process each tweet through the pipeline and return rumors and synthesized data.

    Parameters
    ----------
    tweets: List[Tweet]
        List of tweets to process.
    vector_db: VectorDB
        The vector database for contextual search and storage.
    progress_callback: Optional[Callable[[float], None]]
        Optional callback to update progress (e.g., for UI).
    """
    rumors: List[Dict[str, Any]] = []
    synthesized: List[Dict[str, Any]] = []
    total = len(tweets)
    for i, tweet in enumerate(tweets):
        # Generate a search-friendly headline
        headline = generate_headline(tweet.tweet, tweet.timestamp)
        logger.info("Generated headline: %s", headline)
        # Retrieve contexts from the vector database or via web search
        contexts = get_contexts(headline, vector_db, top_k=3, search_if_insufficient=True)
        # Classify the tweet and extract structured info
        classification = classify_tweet(tweet, contexts)
        label = classification.get("label")
        if label == "rumor":
            # Only store the needed fields for rumours
            rumors.append({
                "tweet": tweet.tweet,
                "label": "rumor",
                "reasoning": classification.get("reasoning"),
                "confidence": classification.get("confidence"),
            })
        elif label == "non-rumor":
            # Remove label for synthesized data; include all structured fields
            structured = {k: v for k, v in classification.items() if k != "label"}
            structured["tweet"] = tweet.tweet
            synthesized.append(structured)
        else:
            logger.warning("Unknown classification label %s for tweet: %s", label, tweet.tweet)
        
        if progress_callback:
            progress_callback((i + 1) / total)
    
    return rumors, synthesized


def main() -> None:
    parser = argparse.ArgumentParser(description="Disaster response pipeline")
    parser.add_argument("--input", required=True, help="Path to the TSV dataset file")
    parser.add_argument("--env", default=".env", help="Path to the .env file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Load config (overrides global CONFIG) to pick up env file path
    global CONFIG  # type: ignore
    CONFIG = load_config(args.env)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    tweets = load_dataset(args.input)
    logger.info("Loaded %d tweets from %s", len(tweets), args.input)
    vector_db = VectorDB()
    rumors, synthesized = process_tweets(tweets, vector_db)
    with open("rumors.json", "w", encoding="utf-8") as f:
        json.dump(rumors, f, indent=2, ensure_ascii=False)
    with open("synthesized_data.json", "w", encoding="utf-8") as f:
        json.dump(synthesized, f, indent=2, ensure_ascii=False)
    logger.info("Processing complete. Outputs written to rumors.json and synthesized_data.json.")


if __name__ == "__main__":
    main()