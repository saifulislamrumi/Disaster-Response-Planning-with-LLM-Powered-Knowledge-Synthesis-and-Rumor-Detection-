"""Configuration and environment variable handling.

This module centralises configuration for the disaster response pipeline. It
loads secrets from a ``.env`` file if present and exposes them via a
``Config`` dataclass. You should create a ``.env`` file at the project
root with the necessary values (see ``.env.example``) or set the
corresponding environment variables in your shell.

The country scope for mapping is set to ``Bangladesh`` by default as
specified by the user. If you want to change the default country or add
additional configuration options, extend the ``Config`` dataclass
accordingly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
try:
    # The python-dotenv package is optional and used to load environment
    # variables from a .env file. If it is not available, define a
    # no-op fallback so that ``load_config`` still functions.
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - fallback for minimal environments
    def load_dotenv(path: str | None = None) -> None:
        """Fallback implementation if python-dotenv is unavailable.

        When python-dotenv cannot be imported, this fallback simply does
        nothing, meaning that values must be provided via actual
        environment variables. Logging could be added here if desired.
        """
        return None  # No-op


@dataclass
class Config:
    """Simple configuration holder loaded from environment variables."""

    # Serper.dev API key used for web search. Set this in ``.env``.
    serper_api_key: str | None = None
    # Base URL for the Serper API. You can override this if needed.
    serper_base_url: str = "https://serper.dev/api"
    # OpenAI / Ollama model name for both headline generation and classification.
    llm_model_name: str = "llama3.1:latest"
    # Default country for geocoding and map visualisations. Set to "Bangladesh".
    default_country: str = "Bangladesh"
    # Vector database index name / path. FAISS index will be stored here.
    faiss_index_path: str = "./faiss_index.bin"
    # Directory to store temporary downloaded articles.
    article_cache_dir: str = "./articles"


def load_config(env_path: str = ".env") -> Config:
    """Load configuration values from environment variables and return a Config object.

    The function will attempt to read a ``.env`` file if it exists using the
    ``python-dotenv`` package. If no ``.env`` file is found, it falls back
    to reading variables from the current environment. Missing optional
    variables will default to ``None`` or the values defined in
    :class:`Config`.

    Parameters
    ----------
    env_path: str
        Optional path to a ``.env`` file. Defaults to ``.env`` in the
        current working directory.

    Returns
    -------
    Config
        A configuration object with attributes populated from the
        environment.
    """
    # Load variables from .env file if present
    if os.path.exists(env_path):
        load_dotenv(env_path)

    return Config(
        serper_api_key=os.getenv("SERPER_API_KEY"),
        serper_base_url=os.getenv("SERPER_BASE_URL", "https://serper.dev/api"),
        llm_model_name=os.getenv("LLM_MODEL_NAME", "llama3.1:latest"),
        default_country=os.getenv("DEFAULT_COUNTRY", "Bangladesh"),
        faiss_index_path=os.getenv("FAISS_INDEX_PATH", "./faiss_index.bin"),
        article_cache_dir=os.getenv("ARTICLE_CACHE_DIR", "./articles"),
    )


CONFIG: Config = load_config()
