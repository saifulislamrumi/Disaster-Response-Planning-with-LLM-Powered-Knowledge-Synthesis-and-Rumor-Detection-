"""Top-level package for the Disaster Response project.

This package implements a pipeline for processing social media posts related
to disasters, enriching them with contextual information, classifying
them as rumours or factual reports, and synthesising structured data
for response planning. A Streamlit dashboard is provided for interactive
visualisation of the synthesised information. See the README for
instructions on running the full pipeline.
"""

__all__ = [
    "config",
    "data_ingestion",
    "headline_generation",
    "search_enrichment",
    "vector_db",
    "classification",
    "evaluation",
    "dashboard",
]