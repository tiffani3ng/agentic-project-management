"""OpenAI-backed AI opportunity scout."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

import pandas as pd

from .llm_utils import safe_openai_json
from .run_store import RunStore


@dataclass
class AISuggestion:
    task_id: str
    project_id: str
    recommended: bool
    reviewer_required: bool
    reason: str
    suggested_prompt: str


class AIOpportunityScout:
    """Flags low-risk, high-leverage AI support opportunities via OpenAI."""

    def __init__(self, tasks: pd.DataFrame, run_store: RunStore, run_id: str):
        self.tasks = tasks.copy()
        self.run_store = run_store
        self.run_id = run_id

    def _render_prompt(self) -> str:
        condensed_tasks = [
            {
                "id": row["id"],
                "project_id": row["project_id"],
                "name": row["name"],
                "skill_needed": row["skill_needed"],
            }
            for _, row in self.tasks.iterrows()
        ]
        return json.dumps(
            {
                "tasks": condensed_tasks,
                "policy": "Recommend AI assist for writing, summarization, drafting presentations, data QA summaries. Prohibit or require human review for finance, PII, payment, or fraud control. Always return JSON.",
                "response_schema": {
                    "suggestions": [
                        {
                            "task_id": "T0001",
                            "project_id": "P001",
                            "recommended": True,
                            "reviewer_required": True,
                            "reason": "why AI helps",
                            "suggested_prompt": "prompt text",
                        }
                    ]
                },
            },
            indent=2,
        )

    def run(self) -> List[AISuggestion]:
        system_prompt = "You are an AI opportunity scout. Mark where AI co-pilots can draft content or summarize, and require human review on risky domains."
        user_prompt = self._render_prompt()
        fallback_suggestions = []
        for _, task in self.tasks.iterrows():
            name = str(task["name"]).lower()
            risky = any(k in name for k in ["fraud", "payment", "chargeback", "finance", "checkout", "pii"])
            ai_friendly = any(k in name for k in ["email", "deck", "presentation", "summary", "documentation", "brief"])
            fallback_suggestions.append(
                {
                    "task_id": str(task["id"]),
                    "project_id": str(task["project_id"]),
                    "recommended": bool(ai_friendly and not risky),
                    "reviewer_required": bool(ai_friendly or risky),
                    "reason": "Rule-based fallback classification",
                    "suggested_prompt": "Draft the first version and highlight risks; route to human reviewer.",
                }
            )

        result = safe_openai_json(system_prompt, user_prompt, fallback={"suggestions": fallback_suggestions})
        suggestions: List[AISuggestion] = []
        for item in result.get("suggestions", []):
            suggestions.append(
                AISuggestion(
                    task_id=str(item.get("task_id")),
                    project_id=str(item.get("project_id")),
                    recommended=bool(item.get("recommended", False)),
                    reviewer_required=bool(item.get("reviewer_required", True)),
                    reason=str(item.get("reason", "")),
                    suggested_prompt=str(item.get("suggested_prompt", "")),
                )
            )

        self.run_store.log(
            self.run_id,
            "ai_opportunity_scout",
            inputs={"tasks": len(self.tasks)},
            outputs={"suggestions": [s.__dict__ for s in suggestions]},
        )
        return suggestions
