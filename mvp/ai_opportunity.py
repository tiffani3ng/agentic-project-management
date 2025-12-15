"""OpenAI-backed AI opportunity scout."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .llm_utils import safe_openai_json
from .run_store import RunStore


@dataclass
class AISuggestion:
    task_id: str
    project_id: str
    recommended: bool
    reviewer_required: bool
    reviewer: str
    department: str
    reason: str
    suggested_prompt: str
    safe_use_notes: str
    redaction_instructions: str
    prohibited_scope: List[str]


class AIOpportunityScout:
    """Flags low-risk, high-leverage AI support opportunities via OpenAI."""

    def __init__(
        self,
        tasks: pd.DataFrame,
        run_store: RunStore,
        run_id: str,
        employees: Optional[pd.DataFrame] = None,
    ):
        self.tasks = tasks.copy()
        self.run_store = run_store
        self.run_id = run_id
        self.employees = employees
        self._attach_department_metadata()

    def _attach_department_metadata(self) -> None:
        if self.employees is not None and "assignee" in self.tasks.columns:
            dept_map: Dict[str, str] = (
                self.employees.set_index("id").get("department", pd.Series(dtype=str)).to_dict()
            )
            role_map: Dict[str, str] = (
                self.employees.set_index("id").get("role", pd.Series(dtype=str)).to_dict()
            )
            self.tasks["department"] = self.tasks["assignee"].map(dept_map).fillna(
                self.tasks.get("department", "")
            )
            self.tasks["role"] = self.tasks["assignee"].map(role_map).fillna(self.tasks.get("role", ""))
        else:
            self.tasks["department"] = self.tasks.get("department", "")
            self.tasks["role"] = self.tasks.get("role", "")

    def _department_templates(self) -> Dict[str, str]:
        return {
            "Brand Marketing": "Draft campaign briefs, taglines, and stakeholder emails tailored to brand voice.",
            "Performance Marketing": "Prepare UTM-clean copy blocks, ad variants, and reporting summaries with spend/CPA placeholders.",
            "Merchandising & Buying": "Summarize product performance, vendor negotiation points, and SKU rationalization notes.",
            "Customer Support": "Draft macros, FAQ updates, and empathetic responses; keep PII masked in examples.",
            "Product": "Draft PRDs outlines, user stories, and release notes with explicit assumptions and open risks.",
            "Engineering": "Create design doc skeletons, API specs, and risk logs without executing deployments or access changes.",
        }

    def _department_reviewers(self) -> Dict[str, str]:
        return {
            "Brand Marketing": "Brand lead reviewer",
            "Performance Marketing": "Performance marketing manager",
            "Merchandising & Buying": "Merchandising lead reviewer",
            "Customer Support": "Support quality lead",
            "Product": "Product owner",
            "Engineering": "Tech lead",
        }

    def _guardrail_policy(self) -> Dict[str, object]:
        return {
            "pii_redaction": "Mask or remove PII (names, emails, phone, address, order IDs) in prompts and outputs.",
            "prohibited_autonomy": [
                "payments or refunds",
                "access control changes",
                "credential resets",
                "production deployment approvals",
            ],
            "review_requirement": "All AI drafts must be sent for human reviewer sign-off before distribution.",
        }

    def _render_prompt(self) -> str:
        department_templates = self._department_templates()
        reviewers = self._department_reviewers()
        policy = self._guardrail_policy()
        condensed_tasks = []
        for _, row in self.tasks.iterrows():
            department = str(row.get("department", "")).strip() or "General"
            condensed_tasks.append(
                {
                    "id": row["id"],
                    "project_id": row["project_id"],
                    "name": row["name"],
                    "skill_needed": row["skill_needed"],
                    "department": department,
                    "role": row.get("role", ""),
                    "preferred_prompt_template": department_templates.get(
                        department, "Draft a concise first version tailored to the department audience."
                    ),
                    "required_reviewer": reviewers.get(department, "Department lead"),
                }
            )
        return json.dumps(
            {
                "tasks": condensed_tasks,
                "policy": {
                    **policy,
                    "instructions": "Recommend AI assist for writing, summarization, drafting presentations, data QA summaries. Always include reviewer and guardrail notes per task department.",
                },
                "response_schema": {
                    "suggestions": [
                        {
                            "task_id": "T0001",
                            "project_id": "P001",
                            "department": "Product",
                            "recommended": True,
                            "reviewer_required": True,
                            "reviewer": "Product owner",
                            "reason": "why AI helps",
                            "suggested_prompt": "prompt text",
                            "safe_use_notes": "PII redaction, prohibited autonomy, reviewer sign-off.",
                            "redaction_instructions": "Mask customer identifiers and sensitive data.",
                            "prohibited_scope": ["payments or refunds", "access control changes"],
                        }
                    ]
                },
            },
            indent=2,
        )

    def run(self) -> List[AISuggestion]:
        system_prompt = (
            "You are an AI opportunity scout for product and operations teams. Identify "
            "tasks where AI drafting or summarization is a good fit, flag any risk "
            "areas requiring human review, and avoid hallucinating details. Return "
            "only JSON that matches the provided schema."
        )
        user_prompt = self._render_prompt()
        department_templates = self._department_templates()
        reviewers = self._department_reviewers()
        policy = self._guardrail_policy()
        fallback_suggestions = []
        for _, task in self.tasks.iterrows():
            name = str(task["name"]).lower()
            risky = any(k in name for k in ["fraud", "payment", "chargeback", "finance", "checkout", "pii"])
            ai_friendly = any(k in name for k in ["email", "deck", "presentation", "summary", "documentation", "brief"])
            department = str(task.get("department", "")).strip() or "General"
            reviewer = reviewers.get(department, "Department lead")
            redaction_instructions = policy["pii_redaction"]
            prohibited_scope = policy["prohibited_autonomy"]
            safe_use_notes = (
                f"{redaction_instructions} Avoid autonomy on {', '.join(prohibited_scope)}. Send draft to {reviewer} for sign-off."
            )
            fallback_suggestions.append(
                {
                    "task_id": str(task["id"]),
                    "project_id": str(task["project_id"]),
                    "department": department,
                    "recommended": bool(ai_friendly and not risky),
                    "reviewer_required": True,
                    "reviewer": reviewer,
                    "reason": "Rule-based fallback classification",
                    "suggested_prompt": f"{department_templates.get(department, 'Draft the first version tailored to the audience.')} Include a risk log and route to {reviewer} for approval. {redaction_instructions}",
                    "safe_use_notes": safe_use_notes,
                    "redaction_instructions": redaction_instructions,
                    "prohibited_scope": prohibited_scope,
                }
            )

        result = safe_openai_json(system_prompt, user_prompt, fallback={"suggestions": fallback_suggestions})
        suggestions: List[AISuggestion] = []
        default_safe_use = (
            f"{policy['pii_redaction']} Avoid autonomy on {', '.join(policy['prohibited_autonomy'])}. "
            f"{policy['review_requirement']}"
        )
        for item in result.get("suggestions", []):
            prohibited_scope_value = item.get("prohibited_scope")
            if isinstance(prohibited_scope_value, list):
                prohibited_scope = [str(scope) for scope in prohibited_scope_value]
            else:
                prohibited_scope = policy["prohibited_autonomy"]

            suggestions.append(
                AISuggestion(
                    task_id=str(item.get("task_id")),
                    project_id=str(item.get("project_id")),
                    department=str(item.get("department", "")),
                    recommended=bool(item.get("recommended", False)),
                    reviewer_required=bool(item.get("reviewer_required", True)),
                    reviewer=str(item.get("reviewer", "")),
                    reason=str(item.get("reason", "")),
                    suggested_prompt=str(item.get("suggested_prompt", "")),
                    safe_use_notes=str(item.get("safe_use_notes", "")) or default_safe_use,
                    redaction_instructions=str(item.get("redaction_instructions", ""))
                    or policy["pii_redaction"],
                    prohibited_scope=prohibited_scope,
                )
            )

        self.run_store.log(
            self.run_id,
            "ai_opportunity_scout",
            inputs={"tasks": len(self.tasks)},
            outputs={"suggestions": [s.__dict__ for s in suggestions]},
        )
        return suggestions
