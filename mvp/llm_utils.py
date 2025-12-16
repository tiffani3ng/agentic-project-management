"""Shared helpers for OpenAI-backed agents with graceful fallbacks."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI

_FORCE_OPENAI_FALLBACK = False


def set_force_openai_fallback(force: bool) -> None:
    """Toggle a global switch that bypasses OpenAI calls even if API keys are configured."""
    global _FORCE_OPENAI_FALLBACK
    _FORCE_OPENAI_FALLBACK = force


def call_openai_json(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Returns parsed JSON from an OpenAI chat completion."""

    if _FORCE_OPENAI_FALLBACK:
        raise RuntimeError("OpenAI usage disabled via CLI flag; forcing fallback.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot call OpenAI API.")

    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    content = completion.choices[0].message.content or "{}"
    return json.loads(content)


def safe_openai_json(
    system_prompt: str,
    user_prompt: str,
    fallback: Dict[str, Any],
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Attempts an OpenAI JSON call, returning fallback on any failure."""

    try:
        return call_openai_json(system_prompt, user_prompt, model=model, temperature=temperature)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] OpenAI call failed, using fallback. Reason: {exc}")
        return fallback
