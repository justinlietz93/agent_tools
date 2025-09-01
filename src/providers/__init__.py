"""
Aggregator and exports for provider package.
"""

import logging
import json
import re
from typing import Dict, Any, Tuple, Union

# Re-export exception types
from .exceptions import (
    ProviderError,
    ApiCallError,
    ApiResponseError,
    JsonParsingError,
    JsonProcessingError,
)

__all__ = [
    "ProviderError",
    "ApiCallError",
    "ApiResponseError",
    "JsonParsingError",
    "JsonProcessingError",
    "call_with_retry",
    "anthropic_client",
    "deepseek_client",
    "gemini_client",
    "openai_client",
    "openrouter_client",
    "ollama_client",
    "xai_client",
    "model_config",
    "decorators",
]

logger = logging.getLogger(__name__)


def _safe_format(template: str, context: Dict[str, Any]) -> str:
    """Best-effort placeholder replacement without raising on missing keys."""
    try:
        formatted = template
        for k, v in context.items():
            formatted = formatted.replace(f"{{{k}}}", str(v))
        return formatted
    except Exception as e:
        logger.debug(f"Safe format failed: {e}")
        return template


def _clean_json_markers(s: str) -> str:
    """Strip common code fences from LLM JSON replies."""
    s = s.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _attempt_json_repair(s: str) -> str:
    """
    Attempt to repair common JSON formatting issues:
    - Trim to the first JSON object/array envelope
    - Remove code fences and trailing commas
    - Balance braces/brackets and quotes
    """
    # Trim to JSON envelope
    first_obj = s.find("{")
    first_arr = s.find("[")
    idx_candidates = [i for i in [first_obj, first_arr] if i != -1]
    if idx_candidates:
        start = min(idx_candidates)
        s = s[start:]

    # Clean code fences if any slipped through
    s = _clean_json_markers(s)

    # Remove trailing commas before a closing brace/bracket
    s = re.sub(r",\s*(\}|\])", r"\1", s)

    # Balance braces/brackets
    open_braces = s.count("{")
    close_braces = s.count("}")
    if open_braces > close_braces:
        s += "}" * (open_braces - close_braces)

    open_brackets = s.count("[")
    close_brackets = s.count("]")
    if open_brackets > close_brackets:
        s += "]" * (open_brackets - close_brackets)

    # Balance quotes in a naive but effective way
    s_wo_escaped = re.sub(r'\\"', "", s)
    if s_wo_escaped.count('"') % 2 == 1:
        s += '"'

    return s


def call_with_retry(
    prompt_template: str,
    context: Dict[str, Any],
    config: Dict[str, Any],
    is_structured: bool = True,
) -> Tuple[Union[str, Dict[str, Any]], str]:
    """
    Provider-agnostic call with retry that delegates to the configured provider.
    Returns a tuple of (response, model_used_label).
    """
    api_cfg = (config or {}).get("api", {}) or {}
    primary = api_cfg.get("primary_provider") or api_cfg.get("provider") or "gemini"
    provider = str(primary).lower()

    if provider == "gemini":
        from .gemini_client import call_gemini_with_retry
        resp, model = call_gemini_with_retry(
            prompt_template=prompt_template,
            context=context,
            config=config,
            is_structured=is_structured,
        )
        return resp, model

    if provider == "openai":
        from .openai_client import call_openai_with_retry
        resp, model = call_openai_with_retry(
            prompt_template=prompt_template,
            context=context,
            config=config,
            is_structured=is_structured,
        )
        return resp, model

    if provider == "openrouter":
        from .openrouter_client import run_openrouter_client
        from .model_config import get_openrouter_config

        formatted = _safe_format(prompt_template, context)
        if is_structured:
            formatted += "\n\nRespond strictly in valid JSON format. Do not include code fences."

        sys_msg = api_cfg.get("openrouter", {}).get(
            "system_message",
            "You are a helpful research assistant.",
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": formatted},
        ]

        # Execute via OpenRouter
        try:
            text = run_openrouter_client(messages=messages)
        except Exception as e:
            raise ApiCallError(f"OpenRouter call failed: {e}") from e

        model_name = get_openrouter_config().get("model", "unknown")
        label = f"openrouter:{model_name}"

        if is_structured:
            cleaned = _clean_json_markers(text)
            try:
                data = json.loads(cleaned)
                return data, label
            except json.JSONDecodeError as e1:
                # Attempt a best-effort repair for truncated or slightly malformed JSON
                repaired = _attempt_json_repair(cleaned)
                try:
                    data = json.loads(repaired)
                    logger.info("Repaired malformed JSON from OpenRouter response")
                    return data, label
                except Exception as e2:
                    raise JsonParsingError(
                        f"OpenRouter returned non-JSON when JSON expected: {e1}. Raw: {text[:800]}"
                    ) from e2
        else:
            return text, label

    if provider == "ollama":
        from .ollama_client import run_ollama_client

        formatted = _safe_format(prompt_template, context)
        # Do not inject any system prompt here; upstream components supply system prompts.
        messages = [
            {"role": "user", "content": formatted},
        ]

        try:
            text = run_ollama_client(messages=messages)
        except Exception as e:
            raise ApiCallError(f"Ollama call failed: {e}") from e

        label = "ollama"
        if is_structured:
            cleaned = _clean_json_markers(text)
            try:
                data = json.loads(cleaned)
                return data, label
            except json.JSONDecodeError as e1:
                repaired = _attempt_json_repair(cleaned)
                try:
                    data = json.loads(repaired)
                    logger.info("Repaired malformed JSON from Ollama response")
                    return data, label
                except Exception as e2:
                    raise JsonParsingError(
                        f"Ollama returned non-JSON when JSON expected: {e1}. Raw: {text[:800]}"
                    ) from e2
        else:
            return text, label

    raise ApiCallError(f"Unknown or unsupported provider: {provider}")