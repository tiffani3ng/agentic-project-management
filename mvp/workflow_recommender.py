"""OpenAI-backed workflow recommender."""
from __future__ import annotations

import json
from typing import Dict, List

from .llm_utils import safe_openai_json
from .run_store import RunStore


class WorkflowRecommender:
    """Suggests process tweaks using OpenAI with rule-based fallback."""

    def __init__(self, assignments, ai_suggestions, bottlenecks, run_store: RunStore, run_id: str):
        self.assignments = assignments
        self.ai_suggestions = ai_suggestions
        self.bottlenecks = bottlenecks
        self.run_store = run_store
        self.run_id = run_id

    def _render_prompt(self) -> str:
        return json.dumps(
            {
                "context": {
                    "assignments": [getattr(a, "__dict__", a) for a in self.assignments],
                    "ai_suggestions": [getattr(s, "__dict__", s) for s in self.ai_suggestions],
                    "bottlenecks": [getattr(b, "__dict__", b) for b in self.bottlenecks],
                },
                "instructions": "Recommend concrete process changes: parallelize safe tasks, set WIP limits, resequence items blocked by bottlenecks, rebalance workload across timezones, and add AI co-pilots to repetitive writing/analysis steps. Return JSON list of recommendations.",
                "response_schema": {"recommendations": ["text" for _ in range(3)]},
            },
            indent=2,
        )

    def run(self) -> List[str]:
        system_prompt = (
            "You are a pragmatic workflow design expert. Use only the supplied "
            "assignments, AI suggestions, and bottlenecks to propose actionable next "
            "steps. Prefer specific moves (WIP limits, resequencing, pairing) over "
            "generic advice. Output JSON adhering to the response schema and nothing "
            "else."
        )
        user_prompt = self._render_prompt()

        fallback_recos: List[str] = []
        if self.bottlenecks:
            fallback_recos.append("Set WIP limits and add daily standups to relieve the top bottleneck stage.")
        if self.ai_suggestions:
            fallback_recos.append("Enable AI drafting for communication tasks with mandatory reviewer sign-off.")
        if self.assignments:
            fallback_recos.append("Rebalance workload by shifting low-skill tasks to underutilized team members.")

        result = safe_openai_json(
            system_prompt,
            user_prompt,
            fallback={"recommendations": fallback_recos or ["No recommendations generated."]},
        )
        recommendations = [str(r) for r in result.get("recommendations", fallback_recos)]

        self.run_store.log(
            self.run_id,
            "workflow_recommender",
            inputs={"assignments": len(self.assignments), "ai_suggestions": len(self.ai_suggestions)},
            outputs={"recommendations": recommendations},
        )
        return recommendations
