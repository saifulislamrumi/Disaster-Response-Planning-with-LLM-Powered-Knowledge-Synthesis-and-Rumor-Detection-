from __future__ import annotations

import re
import pandas as pd
import streamlit as st
import os
import logging
from typing import List, Dict, Any, Optional, Set
import json
import hashlib
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np

try:
    import requests  # type: ignore
except ImportError:
    requests = None  # type: ignore
try:
    from newspaper import Article  # type: ignore
except ImportError:
    Article = None  # type: ignore

from .config import CONFIG
from .vector_db import VectorDB


logger = logging.getLogger(__name__)

# Global cache to track what we've already searched for
_search_cache: Set[str] = set()
_topic_coverage: Dict[str, int] = defaultdict(int)

# Load BERT model for semantic similarity
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')


def search_news(headline: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """Query the Serper.dev API for news articles related to a headline."""
    api_key = api_key or CONFIG.serper_api_key
    if not api_key:
        logger.warning("Serper API key is not set; skipping web search.")
        return []
    base_url = base_url or CONFIG.serper_base_url
    endpoint = f"{base_url}/search"
    payload = {
        "q": headline,
        "type": "news",
        "num": 5,
    }
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    if requests is None:
        logger.warning("Requests library is unavailable; cannot perform web search.")
        return []
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("news", [])
    except Exception as exc:
        logger.error("Error querying Serper API: %s", exc)
        return []


def parse_article(url: str, cache_dir: Optional[str] = None) -> Optional[str]:
    """Download and parse an article using Newspaper3k."""
    cache_dir = cache_dir or CONFIG.article_cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = hashlib.sha256(url.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{safe_name}.txt")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    if Article is None:
        logger.warning("newspaper3k library is unavailable; cannot parse articles.")
        return None
    article = Article(url)
    try:
        article.download()
        article.parse()
        text = article.text.strip()
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except Exception as exc:
        logger.error("Failed to parse article %s: %s", url, exc)
        return None


def preprocess_text(text: str) -> str:
    """Preprocess text by lowercasing, removing punctuation, and splitting into words."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text


def calculate_context_relevance_score(headline: str, contexts: List[Dict[str, Any]]) -> float:
    """Calculate how relevant existing contexts are to the given headline using Sentence-BERT embeddings."""
    if not contexts:
        return 0.0
    
    headline_embedding = semantic_model.encode([headline])
    context_texts = [context_result.get("metadata", {}).get("text", "") for context_result in contexts]
    context_embeddings = semantic_model.encode(context_texts)
    
    # Calculate cosine similarities between headline and contexts
    cosine_similarities = np.dot(headline_embedding, context_embeddings.T) / (np.linalg.norm(headline_embedding) * np.linalg.norm(context_embeddings, axis=1))
    
    # Average similarity score
    avg_score = np.mean(cosine_similarities) if len(cosine_similarities) > 0 else 0.0
    
    return min(avg_score, 1.0)  # Cap at 1.0


def extract_key_topics(headline: str) -> List[str]:
    """Extract key topics/keywords from headline for topic tracking."""
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'a', 'an'}
    words = headline.lower().split()
    keywords = [word.strip('.,!?;:') for word in words if word.strip('.,!?;:') not in stop_words and len(word) > 2]
    return keywords[:3]  # Return top 3 keywords


def should_search_for_headline(headline: str, vector_db: VectorDB, top_k: int = 5) -> tuple[bool, str, List[str]]:
    """Intelligent decision making for whether to search for new content."""
    global _search_cache, _topic_coverage
    
    # Check if we've already searched for this exact headline
    headline_hash = hashlib.md5(headline.encode()).hexdigest()
    if headline_hash in _search_cache:
        # Get existing contexts without searching
        results = vector_db.search(headline, top_k=top_k)
        contexts = [res["metadata"]["text"] for res in results if "text" in res["metadata"]]
        return False, f"Already searched for similar headline", contexts
    
    # Get existing contexts from vector DB
    results = vector_db.search(headline, top_k=top_k)
    contexts = [res["metadata"]["text"] for res in results if "text" in res["metadata"]]
    
    # Decision criteria
    min_contexts = 2
    min_relevance_score = 0.4
    
    # Check 1: Do we have enough contexts?
    if len(contexts) < min_contexts:
        return True, f"Insufficient contexts: {len(contexts)}/{min_contexts} found", contexts
    
    # Check 2: Are existing contexts relevant enough?
    relevance_score = calculate_context_relevance_score(headline, results)
    if relevance_score < min_relevance_score:
        return True, f"Low relevance score: {relevance_score:.2f} < {min_relevance_score}", contexts
    
    # Check 3: Topic diversity - are we covering new ground?
    headline_topics = extract_key_topics(headline)
    topic_coverage_score = sum(_topic_coverage.get(topic, 0) for topic in headline_topics)
    
    # If this headline covers topics we haven't seen much, search for it
    if topic_coverage_score < 2:  # Low coverage threshold
        # Update topic coverage
        for topic in headline_topics:
            _topic_coverage[topic] += 1
        return True, f"New topic area detected: {headline_topics}", contexts
    
    # If we get here, existing contexts should be sufficient
    return False, f"Sufficient context found: {len(contexts)} contexts with {relevance_score:.2f} relevance", contexts


def get_contexts(
    headline: str,
    vector_db: VectorDB,
    top_k: int = 3,
    search_if_insufficient: bool = True,
    min_required_contexts: int = 2,
    relevance_threshold: float = 0.4
) -> List[str]:
    """Retrieve or fetch contextual documents for a headline with intelligent search decisions."""
    
    # Check if there are enough relevant contexts in the database
    results = vector_db.search(headline, top_k=top_k)
    contexts = [res["metadata"]["text"] for res in results if "text" in res["metadata"]]
    
    # Placeholder for updating average relevance score for all tweets
    relevance_placeholder = st.empty()

    # List to store relevance scores
    relevance_scores = []
    
    # Check relevance of existing contexts using semantic similarity
    relevance_score = calculate_context_relevance_score(headline, results)
    
    # Calculate the relevance score for each context and update the placeholder
    for i, context in enumerate(contexts):
        context_relevance_score = calculate_context_relevance_score(headline, [results[i]])
        relevance_scores.append(context_relevance_score)

    #Calculate the average relevance score for the top-k contexts
    #avg_relevance_score = np.mean(relevance_scores)
    
    #Overwrite the placeholder with the average relevance score
    #relevance_placeholder.text(f"Average Relevance Score for Top {top_k} Contexts: {avg_relevance_score:.2f}")
    
    # If relevance is low, perform web search
    if relevance_score < relevance_threshold and search_if_insufficient:
        #st.warning(f"Relevance score is too low ({relevance_score:.2f}), performing a web search.")
        articles = search_news(headline)
        texts = []
        metas = []
        
        for art in articles:
            url = art.get("link") or art.get("url")
            if not url:
                continue
            text = parse_article(url)
            if text:
                texts.append(text)
                meta = {
                    "title": art.get("title"),
                    "url": url,
                    "source": art.get("source"),
                    "published_date": art.get("date"),
                    "text": text,
                    "search_headline": headline,
                }
                metas.append(meta)
        
        if texts:
            vector_db.add_documents(texts, metas)
            results = vector_db.search(headline, top_k=top_k)
            contexts = [res["metadata"]["text"] for res in results if "text" in res["metadata"]]
    
    return contexts[:top_k]


def get_search_statistics() -> Dict[str, Any]:
    """Get statistics about search usage for monitoring."""
    return {
        "total_searches_performed": len(_search_cache),
        "unique_topics_covered": len(_topic_coverage),
        "topic_distribution": dict(_topic_coverage),
        "search_cache_size": len(_search_cache)
    }


def reset_search_cache():
    """Reset the search cache - useful for testing or starting fresh."""
    global _search_cache, _topic_coverage
    _search_cache.clear()
    _topic_coverage.clear()
    logger.info("Search cache reset")


# Enhanced context retrieval with smart batching
def get_contexts_batch_optimized(
    headlines: List[str],
    vector_db: VectorDB,
    top_k: int = 3,
    batch_search_threshold: int = 5
) -> Dict[str, List[str]]:
    """
    Batch-optimized context retrieval for multiple headlines.
    
    Groups similar headlines together and performs batch searches
    to minimize API calls while ensuring coverage.
    """
    headline_contexts = {}
    search_needed = []
    
    # First pass: check existing contexts for all headlines
    for headline in headlines:
        should_search, reason, existing_contexts = should_search_for_headline(headline, vector_db, top_k)
        
        if should_search:
            search_needed.append(headline)
        else:
            headline_contexts[headline] = existing_contexts[:top_k]
    
    # Batch search for headlines that need fresh content
    if search_needed:
        st.info(f"🔍 Batch searching for {len(search_needed)} headlines that need fresh context")
        
        for i, headline in enumerate(search_needed):
            if i % batch_search_threshold == 0:
                st.progress((i + 1) / len(search_needed))
            
            # Get contexts with search
            contexts = get_contexts(headline, vector_db, top_k)
            headline_contexts[headline] = contexts
    
    return headline_contexts
