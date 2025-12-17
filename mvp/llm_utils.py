"""Shared helpers for OpenAI-backed agents with graceful fallbacks."""
from __future__ import annotations

import json
import os
import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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
    return _parse_json_response(content)


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
        context = _caller_context()
        context_note = f" [{context}]" if context else ""
        print(f"[WARN]{context_note} OpenAI call failed, using fallback. Reason: {exc}")
        return fallback


def _strip_code_fence(payload: str) -> str:
    stripped = payload.strip()
    if not stripped.startswith("```"):
        return stripped
    fence_close = stripped.rfind("```")
    if fence_close <= 0:
        return stripped
    body = stripped[len("```") : fence_close]
    if "\n" in body:
        _, remainder = body.split("\n", 1)
        body = remainder
    return body.strip()


def _slice_json_block(payload: str) -> str:
    """Return the substring spanning the outermost JSON object/array if found."""
    for open_char, close_char in [("{", "}"), ("[", "]")]:
        start = payload.find(open_char)
        end = payload.rfind(close_char)
        if start != -1 and end != -1 and start < end:
            return payload[start : end + 1]
    return payload


def _candidate_payloads(raw: str) -> Iterable[str]:
    yield raw
    stripped = raw.strip()
    if stripped != raw:
        yield stripped
    fence_stripped = _strip_code_fence(raw)
    if fence_stripped not in {raw, stripped}:
        yield fence_stripped
    sliced = _slice_json_block(fence_stripped)
    if sliced not in {raw, stripped, fence_stripped}:
        yield sliced


def _parse_json_response(payload: str) -> Dict[str, Any]:
    """Attempt multiple cleanup strategies before giving up on malformed JSON."""
    last_error: Optional[json.JSONDecodeError] = None
    for candidate in _candidate_payloads(payload):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return {}


def _caller_context() -> str:
    """Return first stack frame outside this module for logging context."""
    try:
        stack = inspect.stack()
        current_file = Path(__file__).resolve()
        for frame_info in stack[1:]:
            frame_path = Path(frame_info.filename).resolve()
            if frame_path == current_file:
                continue
            rel_path = frame_path
            try:
                rel_path = frame_path.relative_to(Path.cwd())
            except ValueError:
                rel_path = frame_path
            return f"{rel_path}:{frame_info.lineno} ({frame_info.function})"
    except Exception:  # noqa: BLE001
        return ""
    return ""
