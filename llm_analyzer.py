import json
import os
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def get_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip()


def normalize_scores(scores: Dict[str, Any]) -> Dict[str, float]:
    normalized = {
        "positive": float(scores.get("positive", 0) or 0),
        "negative": float(scores.get("negative", 0) or 0),
        "neutral": float(scores.get("neutral", 0) or 0),
    }

    total = sum(normalized.values())
    if total <= 0:
        normalized = {"positive": 34.0, "negative": 33.0, "neutral": 33.0}
        total = 100.0

    if abs(total - 100.0) > 1e-6:
        normalized = {
            key: round((value / total) * 100.0, 2)
            for key, value in normalized.items()
        }

    rounded_total = round(sum(normalized.values()), 2)
    if rounded_total != 100.0:
        difference = round(100.0 - rounded_total, 2)
        normalized["neutral"] = round(normalized["neutral"] + difference, 2)

    return normalized


def _extract_response_text(response: Any) -> str:
    if response is None:
        return ""

    if hasattr(response, "text") and response.text:
        return str(response.text)

    if hasattr(response, "candidates"):
        for candidate in response.candidates:
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                parts = candidate.content.parts
                for part in parts:
                    if hasattr(part, "text") and part.text:
                        return str(part.text)

    if isinstance(response, dict):
        if "text" in response:
            return str(response["text"])
        if "content" in response and isinstance(response["content"], str):
            return response["content"]

    return ""


def _build_llm_prompt(text: str) -> str:
    return f"""Analyze the sentiment of the following text exactly as provided.
Return only valid JSON in this structure with no markdown fences and no extra text:
{{
  "sentiment": "Positive",
  "scores": {{"positive": 85, "negative": 5, "neutral": 10}},
  "explanation": "Short explanation",
  "emotional_tone": "Happy",
  "contains_negation": false,
  "contains_sarcasm": false,
  "is_mixed_sentiment": false
}}
Important rules:
- sentiment must be one of: "Positive", "Negative", or "Neutral"
- scores values are estimated percentages that must add up to exactly 100
- scores keys must be exactly: positive, negative, neutral
- explanation should be brief and in plain English
- emotional_tone should be a short mood descriptor such as Happy, Angry, Calm, Sad, or Neutral
- contains_negation, contains_sarcasm, and is_mixed_sentiment are booleans
- Do not include any explanation outside the JSON object
Text to analyze:
{text}
"""



def analyze_text(
    text: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:

    if not text or not str(text).strip():
        raise ValueError("Input text cannot be empty.")

    resolved_api_key = (api_key or get_api_key()).strip()
    resolved_model_name = (model_name or DEFAULT_MODEL_NAME).strip()

    if not resolved_api_key:
        return {
            "error": "Gemini API key is missing. Add it in the sidebar or set the GEMINI_API_KEY environment variable.",
            "sentiment": "Unavailable",
            "scores": {"positive": 0.0, "negative": 0.0, "neutral": 0.0},
            "explanation": "Gemini analysis requires an API key.",
            "emotional_tone": "Unavailable",
            "contains_negation": False,
            "contains_sarcasm": False,
            "is_mixed_sentiment": False,
            "latency_ms": 0,
            "api_available": False,
        }

    start_time = time.perf_counter()

    try:
        from google import genai

        # Initialize Gemini client
        client = genai.Client(api_key=resolved_api_key)

        # Send sentiment analysis request
        response = client.models.generate_content(
            model=resolved_model_name,
            contents=_build_llm_prompt(str(text)),
            config={
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        )

        # Extract response text
        content = _extract_response_text(response)

        if not content:
            raise ValueError("Gemini returned an empty response.")

        # Clean JSON response
        clean_content = content.strip()

        if clean_content.startswith("```"):
            clean_content = (
                clean_content
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        # Parse JSON
        payload = json.loads(clean_content)

        # Validate sentiment
        sentiment = str(payload.get("sentiment", "Neutral")).strip()

        if sentiment not in {"Positive", "Negative", "Neutral"}:
            raise ValueError("Invalid sentiment label returned by Gemini.")

        # Normalize sentiment scores
        scores = normalize_scores(payload.get("scores", {}))

        explanation = str(
            payload.get(
                "explanation",
                "Gemini provided a contextual sentiment assessment."
            )
            or "Gemini provided a contextual sentiment assessment."
        )

        emotional_tone = str(
            payload.get("emotional_tone", "Neutral") or "Neutral"
        )

        contains_negation = bool(
            payload.get("contains_negation", False)
        )

        contains_sarcasm = bool(
            payload.get("contains_sarcasm", False)
        )

        is_mixed_sentiment = bool(
            payload.get("is_mixed_sentiment", False)
        )

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000, 2
        )

        return {
            "sentiment": sentiment,
            "scores": {
                "positive": round(scores["positive"], 2),
                "negative": round(scores["negative"], 2),
                "neutral": round(scores["neutral"], 2),
            },
            "explanation": explanation,
            "emotional_tone": emotional_tone,
            "contains_negation": contains_negation,
            "contains_sarcasm": contains_sarcasm,
            "is_mixed_sentiment": is_mixed_sentiment,
            "latency_ms": latency_ms,
            "api_available": True,
        }

    except Exception as exc:

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000, 2
        )

        return {
            "error": f"Gemini analysis failed: {exc}",
            "sentiment": "Unavailable",
            "scores": {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0
            },
            "explanation": (
                "Gemini could not analyse the text due to an API "
                "or network problem."
            ),
            "emotional_tone": "Unavailable",
            "contains_negation": False,
            "contains_sarcasm": False,
            "is_mixed_sentiment": False,
            "latency_ms": latency_ms,
            "api_available": False,
        }
