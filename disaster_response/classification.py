"""Classify tweets into rumors or non-rumors and extract structured data.

This module defines the logic for LLM Task 2. Given a tweet and a set
of contextual documents, the language model classifies the tweet as a rumor
or non-rumor. For non-rumors, it extracts structured information such as
need type, severity, and location.

The classification and extraction are performed using LangChain's RunnableSequence
with the Ollama model (llama3.1). A streamlined prompt with structured
output parsing ensures consistent JSON output. A robust fallback heuristic
is used when LLM calls fail or produce invalid output.
"""

from __future__ import annotations

import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import OllamaLLM
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from .data_ingestion import Tweet
from .config import CONFIG


logger = logging.getLogger(__name__)

def _get_output_parser() -> StructuredOutputParser:
    """Define the expected JSON schema for LLM output."""
    return StructuredOutputParser.from_response_schemas([
        ResponseSchema(name="label", description="One of 'rumor' or 'non-rumor'", type="string"),
        ResponseSchema(name="reasoning", description="Brief explanation of classification", type="string"),
        ResponseSchema(name="confidence", description="Confidence score (0 to 1)", type="float"),
        ResponseSchema(name="severity", description="For non-rumor: 'low', 'medium', 'high', 'critical', or 'unknown'", type="string"),
        ResponseSchema(name="need_type", description="For non-rumor: exactly one of 'food', 'shelter', 'medical', 'evacuation', 'water', or 'unknown'", type="string"),
        ResponseSchema(name="location", description="Place mentioned or 'unknown'", type="string"),
        ResponseSchema(name="tweet", description="Original tweet text", type="string"),
    ])

def _map_invalid_output(raw_output: Dict[str, Any], tweet: str) -> Optional[Dict[str, Any]]:
    """Attempt to map invalid LLM output to the expected schema."""
    if not isinstance(raw_output, dict):
        return None
    
    # Initialize default output
    mapped_output = {
        "label": "non-rumor",
        "reasoning": "Mapped from invalid LLM output.",
        "confidence": 0.7,
        "severity": "unknown",
        "need_type": "unknown",
        "location": raw_output.get("location", "unknown"),
        "tweet": tweet
    }

    # Handle invalid labels (e.g., 'non-emergency')
    if raw_output.get("label") == "non-emergency":
        mapped_output["label"] = "non-rumor"
        mapped_output["reasoning"] = raw_output.get("reasoning", "Mapped: Invalid label 'non-emergency' corrected to 'non-rumor'.")
        mapped_output["confidence"] = raw_output.get("confidence", 0.7)
        mapped_output["severity"] = raw_output.get("severity", "unknown")
        mapped_output["need_type"] = raw_output.get("need_type", "unknown")
        if mapped_output["need_type"] == "none":
            mapped_output["need_type"] = "unknown"
        mapped_output["location"] = raw_output.get("location", "unknown")
        return mapped_output

    # Map summary-like outputs (e.g., 'title', 'text', 'sections')
    text_content = (raw_output.get("text", "") + " " + 
                    raw_output.get("title", "") + " " + 
                    " ".join(section.get("text", "") for section in raw_output.get("sections", []))).lower()
    if "title" in raw_output or "text" in raw_output or "sections" in raw_output:
        mapped_output["reasoning"] = f"Mapped from LLM summary: {text_content[:50]}..."
        if any(word in text_content for word in ["rescue", "evacuation", "operations"]):
            mapped_output["need_type"] = "evacuation"
            mapped_output["severity"] = "critical"
        elif any(word in text_content for word in ["medical", "hospital", "injured", "medicine", "health"]):
            mapped_output["need_type"] = "medical"
            mapped_output["severity"] = "critical"
        elif any(word in text_content for word in ["water", "drinking", "thirsty", "sanitation"]):
            mapped_output["need_type"] = "water"
            mapped_output["severity"] = "high"
        elif any(word in text_content for word in ["food", "hungry", "starving", "meal"]):
            mapped_output["need_type"] = "food"
            mapped_output["severity"] = "high"
        elif any(word in text_content for word in ["shelter", "housing", "homeless", "refuge"]):
            mapped_output["need_type"] = "shelter"
            mapped_output["severity"] = "medium"
        if any(word in text_content for word in ["flood", "earthquake", "hurricane", "disaster", "crisis", "storm"]):
            mapped_output["severity"] = "high"
        if any(word in text_content for word in ["severe", "critical", "catastrophic"]):
            mapped_output["severity"] = "critical"

    # Check for rumor indicators
    if any(word in text_content for word in ["alien", "conspiracy", "vamp", "hoax", "fake", "misinformation", "rape", "murder"]):
        mapped_output["label"] = "rumor"
        mapped_output["reasoning"] = "Mapped: Detected rumor-related keywords."
        mapped_output["need_type"] = "unknown"
        mapped_output["severity"] = "unknown"

    return mapped_output

def _call_llm(tweet: str, contexts: List[str], model: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """Invoke Ollama model via LangChain with a streamlined prompt and structured output.

    Parameters
    ----------
    tweet: str
        The tweet text to analyze.
    contexts: List[str]
        List of contextual documents.
    model: str
        The model name to run.
    max_retries: int
        Maximum number of retries for LLM calls.

    Returns
    -------
    Optional[Dict[str, Any]]
        The parsed JSON output, or None if all attempts fail.
    """
    try:
        llm = OllamaLLM(model=model, temperature=0.0)  # Deterministic output
        output_parser = _get_output_parser()
        context_str = contexts[0] if contexts else "No context provided."  # Single context
        schema_instructions = output_parser.get_format_instructions()

        # Main prompt
        prompt_template = PromptTemplate(
            input_variables=["tweet", "context", "schema_instructions"],
            template="""
**CRITICAL**: Output ONLY a JSON object with EXACTLY the fields: label, reasoning, confidence, severity, need_type, location, tweet. No markdown, backticks, summaries, titles, sections, or extra text. Do NOT summarize the context or generate narrative outputs; focus solely on classifying the tweet and extracting required fields.

**Task**: Classify the tweet as 'rumor' or 'non-rumor' (disaster-related). For non-rumors, extract structured information.

**Steps**
1. **Classify**:
   - 'rumor': Unverifiable claims (e.g., "aliens", "vamp activities", "minorities are being raped").
   - 'non-rumor': Disaster-related or prayer tweets (e.g., "rescue operations", "pray for Feni"). Prayer tweets are 'non-rumor' with need_type 'unknown'.
2. **Extract (non-rumor only)**:
   - Severity: 'low', 'medium', 'high', 'critical', or 'unknown' based on impact.
   - Need Type: EXACTLY ONE of 'food', 'shelter', 'medical', 'evacuation', 'water', or 'unknown'. Prioritize: medical > evacuation > water > food > shelter. Use 'unknown' for prayer tweets.
   - Location: Place mentioned or 'unknown'.
3. **Constraints**:
   - Do NOT use labels like 'non-emergency' or need_type like 'none'.
   - Base classification primarily on the tweet, using context only to verify disaster details.

**Schema**
{schema_instructions}

**Examples**
{{
    "label": "rumor",
    "reasoning": "Unverified claim of alien involvement.",
    "confidence": 0.95,
    "severity": "unknown",
    "need_type": "unknown",
    "location": "unknown",
    "tweet": "Aliens caused the Feni flood!"
}}
{{
    "label": "non-rumor",
    "reasoning": "Mentions flooding and medical aid, aligns with context.",
    "confidence": 0.95,
    "severity": "critical",
    "need_type": "medical",
    "location": "Feni",
    "tweet": "Feni flooding: Need food, shelter, and medical aid urgently."
}}
{{
    "label": "non-rumor",
    "reasoning": "Describes rescue operations, indicates evacuation need.",
    "confidence": 0.90,
    "severity": "critical",
    "need_type": "evacuation",
    "location": "Feni",
    "tweet": "The Bangladesh Army is carrying out rescue operations in the flood-affected areas."
}}
{{
    "label": "rumor",
    "reasoning": "Vague and unverifiable claim about 'vamp activities'.",
    "confidence": 0.85,
    "severity": "unknown",
    "need_type": "unknown",
    "location": "Feni",
    "tweet": "Neighbours are happy now? Why always create this vamp types activities!"
}}
{{
    "label": "non-rumor",
    "reasoning": "Prayer tweet expressing concern for disaster, no specific needs mentioned.",
    "confidence": 0.80,
    "severity": "unknown",
    "need_type": "unknown",
    "location": "Feni and Noakhali",
    "tweet": "All Eyes On Feni and Noakhali This is terrible. May Allah have mercy on us. Pray for Feni & Noakhali"
}}

**Input**
Tweet: {tweet}
Context: {context}

**Output**
{{
    "label": "...",
    "reasoning": "...",
    "confidence": 0.0,
    "severity": "...",
    "need_type": "...",
    "location": "...",
    "tweet": "{tweet}"
}}
"""
        )

        # Fallback prompt
        fallback_prompt = PromptTemplate(
            input_variables=["tweet", "schema_instructions"],
            template="""
**CRITICAL**: Output ONLY a JSON object with EXACTLY the fields: label, reasoning, confidence, severity, need_type, location, tweet. No markdown, backticks, or extra text. Do NOT use labels like 'non-emergency' or need_type like 'none'.

**Task**: Classify the tweet as 'rumor' or 'non-rumor' (disaster-related).

**Steps**
1. 'rumor' if unverifiable (e.g., "aliens", "vamp", "minorities are being raped").
2. 'non-rumor' if disaster-related or prayer. Prayer tweets have need_type 'unknown'.
3. For non-rumors, extract:
   - Severity: 'low', 'medium', 'high', 'critical', or 'unknown'.
   - Need Type: EXACTLY ONE of 'food', 'shelter', 'medical', 'evacuation', 'water', or 'unknown'. Prioritize: medical > evacuation > water > food > shelter.
   - Location: Place mentioned or 'unknown'.

**Schema**
{schema_instructions}

**Input**
Tweet: {tweet}

**Output**
{{
    "label": "...",
    "reasoning": "...",
    "confidence": 0.0,
    "severity": "...",
    "need_type": "...",
    "location": "...",
    "tweet": "{tweet}"
}}
"""
        )

        chain = RunnableSequence(prompt_template | llm | output_parser)
        fallback_chain = RunnableSequence(fallback_prompt | llm | output_parser)

        for attempt in range(max_retries):
            try:
                chain_to_use = chain if attempt == 0 else fallback_chain
                inputs = {"tweet": tweet, "context": context_str, "schema_instructions": schema_instructions} if attempt == 0 else {"tweet": tweet, "schema_instructions": schema_instructions}
                response = chain_to_use.invoke(inputs)
                logger.debug("Parsed model output (attempt %d): %s", attempt + 1, response)
                return response
            except Exception as exc:
                logger.warning("LLM call failed on attempt %d: %s", attempt + 1, exc)
                if isinstance(exc, ValueError) and hasattr(exc, 'args') and isinstance(exc.args[0], dict):
                    raw_output = exc.args[0]
                    logger.debug("Raw LLM output (attempt %d): %s", attempt + 1, raw_output)
                    mapped_output = _map_invalid_output(raw_output, tweet)
                    if mapped_output:
                        logger.debug("Mapped invalid output to: %s", mapped_output)
                        return mapped_output
                    if isinstance(raw_output, dict) and 'text' in raw_output:
                        try:
                            json_str = raw_output['text']
                            parsed_json = json.loads(json_str)
                            if isinstance(parsed_json, dict) and "label" in parsed_json:
                                logger.debug("Successfully parsed text field as JSON: %s", parsed_json)
                                return parsed_json
                        except json.JSONDecodeError:
                            logger.debug("Failed to parse text field as JSON: %s", json_str)
        logger.error("All %d LLM call attempts failed", max_retries)
        return None
    except Exception as exc:
        logger.error("Error initializing LangChain with Ollama: %s", exc)
        return None

def classify_tweet(tweet: Tweet, contexts: List[str], model_name: str | None = None) -> Dict[str, Any]:
    """Classify a tweet as rumor or non-rumor and extract structured information.

    Parameters
    ----------
    tweet: Tweet
        The tweet to classify.
    contexts: List[str]
        A list of contextual documents retrieved from the knowledge base.
    model_name: str | None
        Override the default model name. If None, uses CONFIG.llm_model_name.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing classification and structured data.
    """
    model = model_name or CONFIG.llm_model_name or "llama3.1"  # Default to llama3.1
    output = _call_llm(tweet.tweet, contexts, model)
    if output and isinstance(output, dict):
        try:
            # Validate and correct output
            if output["label"] not in ["rumor", "non-rumor"]:
                logger.warning("Invalid label in JSON: %s, correcting to 'non-rumor'", output.get("label"))
                output["label"] = "non-rumor"
                output["reasoning"] = f"Corrected invalid label '{output.get('label')}' to 'non-rumor'."
            if output["label"] == "non-rumor":
                required_fields = ["severity", "need_type", "location", "reasoning", "confidence", "tweet"]
                missing = [field for field in required_fields if field not in output]
                if missing:
                    logger.warning("Missing required fields for non-rumor: %s", missing)
                    raise ValueError(f"Missing fields: {missing}")
                if output["severity"] not in ["low", "medium", "high", "critical", "unknown"]:
                    logger.warning("Invalid severity: %s, setting to 'unknown'", output["severity"])
                    output["severity"] = "unknown"
                if output["need_type"] not in ["food", "shelter", "medical", "evacuation", "water", "unknown"]:
                    logger.warning("Invalid need_type: %s, setting to 'unknown'", output["need_type"])
                    output["need_type"] = "unknown"
            return output
        except Exception as exc:
            logger.warning("Validation error: %s, Output: %s", exc, output)
            mapped_output = _map_invalid_output(output, tweet.tweet)
            if mapped_output:
                logger.debug("Mapped invalid output during validation: %s", mapped_output)
                return mapped_output
    
    # Enhanced fallback heuristic
    text_lower = tweet.tweet.lower()
    is_prayer = "pray" in text_lower
    is_rumor = not is_prayer and any(
        word in text_lower
        for word in ["rumor", "hoax", "fake", "unverified", "misinformation", "alien", "conspiracy", 
                     "looting", "vamp", "scam", "exaggerated", "unrealistic", "myth", "gossip", "rape", "murder"]
    )
    if is_rumor:
        return {
            "label": "rumor",
            "reasoning": "Heuristic detected rumor-related keywords or unverifiable claims in the tweet.",
            "confidence": 0.75,
            "location": tweet.location or "unknown",
            "tweet": tweet.tweet,
        }
    # Non-rumor (disaster-related or prayer tweet)
    severity = "unknown"
    need_type = "unknown"
    location = tweet.location or "unknown"
    # Prioritize need_type: medical > evacuation > water > food > shelter
    if any(word in text_lower for word in ["medical", "hospital", "injured", "medicine", "health"]):
        need_type = "medical"
    elif any(word in text_lower for word in ["evacuation", "rescue", "evacuate", "safety", "operations"]):
        need_type = "evacuation"
    elif any(word in text_lower for word in ["water", "drinking", "thirsty", "sanitation"]):
        need_type = "water"
    elif any(word in text_lower for word in ["food", "hungry", "starving", "meal"]):
        need_type = "food"
    elif any(word in text_lower for word in ["shelter", "housing", "homeless", "refuge"]):
        need_type = "shelter"
    if any(word in text_lower for word in ["flood", "earthquake", "hurricane", "disaster", "crisis", "storm"]):
        severity = "high"
    elif any(word in text_lower for word in ["rain", "damage", "affected"]):
        severity = "medium"
    # Handle prayer tweets
    if is_prayer:
        need_type = "unknown"
        severity = "unknown"
        reasoning = "Heuristic: Prayer tweet expressing concern for disaster, no specific needs mentioned."
    else:
        reasoning = "Heuristic: Tweet may describe a disaster event, no rumor keywords detected."
    return {
        "label": "non-rumor",
        "reasoning": reasoning,
        "confidence": 0.65,
        "severity": severity,
        "need_type": need_type,
        "location": location,
        "tweet": tweet.tweet,
    }
