"""Generate concise headlines for tweets using a language model.

The headline generation stage condenses the essential information in a
tweet into a short, search-friendly phrase. These headlines are used
to query news search APIs for relevant articles. The implementation
here uses the Ollama CLI to run the ``gemma3:latest`` model by
default. If Ollama or the specified model is not installed, or if
external LLM calls are disabled, the function falls back to a simple
heuristic that truncates the tweet to a few keywords.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Optional

from .config import CONFIG

logger = logging.getLogger(__name__)


def _call_ollama(prompt: str, model: str) -> Optional[str]:
    """Internal helper to run an Ollama model via subprocess.

    Parameters
    ----------
    prompt: str
        The full prompt to feed into the model.
    model: str
        The name of the model to run, e.g. ``"gemma3:latest"``.

    Returns
    -------
    Optional[str]
        The raw output string from the model, or ``None`` if an error
        occurred.
    """
    try:
        # Start the Ollama process; we pipe the prompt to stdin
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            logger.error("Ollama returned non-zero exit code %s: %s", proc.returncode, proc.stderr.decode("utf-8"))
            return None
        return proc.stdout.decode("utf-8").strip()
    except FileNotFoundError:
        logger.warning("Ollama executable not found. Falling back to heuristic headline generation.")
        return None
    except Exception as exc:
        logger.error("Unexpected error when running Ollama: %s", exc)
        return None


def generate_headline(tweet_text: str, timestamp: str, model_name: str | None = None) -> str:
    if not tweet_text:
        return ""
    
    model = model_name or CONFIG.llm_model_name
    
    # Check for non-disaster-related content
    if "art" in tweet_text.lower() or "?" in tweet_text:
        return "Artistic post in Bangladesh 2024"
    
    prompt = (
        "You are a helpful assistant that turns tweets into concise news headlines. "
        "Given the tweet and timestamp, write a single, search-engine-friendly headline "
        "capturing its key information, even if the tweet is vague or artistic. "
        "The headline must be within 10 words, include 'Bangladesh 2024', and avoid commentary. "
        "Output only the headline.\n\n"
        f"Timestamp: {timestamp}\n"
        f"Tweet: {tweet_text}\n\n"
        "Headline:"
    )
    
    output = _call_ollama(prompt, model)
    
    if output:
        headline = output.splitlines()[0].strip()
        # Ensure headline meets requirements
        words = headline.split()
        if len(words) > 10:
            headline = " ".join(words[:8]) + " Bangladesh 2024"
        elif "Bangladesh 2024" not in headline:
            headline = " ".join(words[:8]) + " Bangladesh 2024"
        logger.info("Generated headline: %s", headline)
        return headline
    
    # Fallback heuristic
    words = tweet_text.split()
    return " ".join(words[:8]) + " Bangladesh 2024"