"""OpenAI-backed AI opportunity scout."""
from __future__ import annotations

import json
from dataclasses import dataclass
import ast
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
        *,
        test_mode: bool = False,
    ):
        self.tasks = tasks.copy()
        self.run_store = run_store
        self.run_id = run_id
        self.employees = employees
        self.test_mode = test_mode
        self._skill_department_map = self._build_skill_department_map()
        self._attach_department_metadata()


    def _build_skill_department_map(self) -> Dict[str, str]:
        """Construct a mapping from skill keywords to departments."""
        mapping: Dict[str, str] = {}
        if self.employees is not None and "skills" in self.employees.columns:
            for _, row in self.employees.iterrows():
                department = str(row.get("department", "")).strip()
                if not department:
                    continue
                skills_raw = row.get("skills", "")
                parsed_skills: List[str]
                if isinstance(skills_raw, (list, tuple, set)):
                    parsed_skills = [str(skill).strip() for skill in skills_raw if str(skill).strip()]
                else:
                    if skills_raw is None:
                        continue
                    try:
                        if pd.isna(skills_raw):
                            continue
                    except Exception:
                        # Non-scalar; treat as generic iterable in fallback below.
                        pass
                    text_value = str(skills_raw).strip()
                    if not text_value:
                        continue
                    try:
                        parsed_skills = json.loads(text_value)
                    except Exception:
                        try:
                            parsed_skills = ast.literal_eval(text_value)
                        except Exception:
                            parsed_skills = [s.strip() for s in text_value.split(",") if s.strip()]
                for skill in parsed_skills:
                    skill_key = str(skill).lower().strip()
                    if skill_key and skill_key not in mapping:
                        mapping[skill_key] = department

        if "skill_needed" in self.tasks.columns and "department" in self.tasks.columns:
            for _, row in self.tasks.iterrows():
                department = str(row.get("department", "")).strip()
                skill = str(row.get("skill_needed", "")).lower().strip()
                if department and skill and skill not in mapping:
                    mapping[skill] = department

        return mapping


    def _department_keyword_hints(self) -> Dict[str, List[str]]:
        return {
            "Engineering": [
                "api",
                "integration",
                "code",
                "shopify",
                "theme",
                "deployment",
                "frontend",
                "backend",
                "ci/cd",
                "pipeline",
            ],
            "Product": ["roadmap", "requirements", "prd", "user story"],
            "Product Design (UX/UI)": ["ux", "ui", "wireframe", "design", "prototype"],
            "Brand Marketing": ["campaign", "creative", "brand", "launch"],
            "Performance Marketing": ["paid", "utm", "acquisition", "ads"],
            "Customer Experience (CX)": ["cx", "support", "macro", "ticket"],
            "Warehouse & Fulfillment Ops": ["warehouse", "slotting", "fulfillment", "pick", "3pl"],
            "Finance": ["finance", "fp&a", "budget", "tax"],
            "Legal & Compliance": ["legal", "compliance", "policy", "contract"],
            "Sales & Partnerships": ["partner", "sales", "deal"],
        }

    def _infer_department_for_row(self, row: pd.Series) -> str:
        skill = str(row.get("skill_needed", "")).lower().strip()
        if skill:
            for known_skill, dept in self._skill_department_map.items():
                if known_skill in skill or skill in known_skill:
                    return dept

        reference_text = f"{row.get('name', '')} {row.get('skill_needed', '')}".lower()
        for dept, keywords in self._department_keyword_hints().items():
            if any(keyword in reference_text for keyword in keywords):
                return dept
        return "General"

    def _attach_department_metadata(self) -> None:
        if self.employees is not None and "assignee" in self.tasks.columns:
            dept_map: Dict[str, str] = (
                self.employees.set_index("id").get("department", pd.Series(dtype=str)).to_dict()
            )
            role_map: Dict[str, str] = (
                self.employees.set_index("id").get("role", pd.Series(dtype=str)).to_dict()
            )
            self.tasks["department"] = self.tasks["assignee"].map(dept_map)
            self.tasks["role"] = self.tasks["assignee"].map(role_map)
        else:
            self.tasks["department"] = self.tasks.get("department", "")
            self.tasks["role"] = self.tasks.get("role", "")

        self.tasks["department"] = self.tasks["department"].fillna("")
        missing_mask = (self.tasks["department"].astype(str).str.strip() == "")
        if missing_mask.any():
            inferred_departments = self.tasks[missing_mask].apply(self._infer_department_for_row, axis=1)
            self.tasks.loc[missing_mask, "department"] = inferred_departments.values
        self.tasks["role"] = self.tasks["role"].fillna("")

    def _department_templates(self) -> Dict[str, str]:
        return {
            "Brand Marketing": "Draft campaign briefs, taglines, and stakeholder emails tailored to brand voice.",
            "Performance Marketing": "Prepare UTM-clean copy blocks, ad variants, and reporting summaries with spend/CPA placeholders.",
            "Merchandising & Buying": "Summarize product performance, vendor negotiation points, and SKU rationalization notes.",
            "Customer Support": "Draft macros, FAQ updates, and empathetic responses; keep PII masked in examples.",
            "Product": "Draft PRDs outlines, user stories, and release notes with explicit assumptions and open risks.",
            "Engineering": "Create design doc skeletons, API specs, and risk logs without executing deployments or access changes.",
            "Warehouse & Fulfillment Ops": "Summarize logistics updates, SOP drafts, and inventory notes.",
            "Finance": "Draft budget reviews, variance write-ups, and revenue memos while redacting PII.",
            "Legal & Compliance": "Draft policy summaries, risk memos, and compliance checklists without committing to approvals.",
            "Sales & Partnerships": "Draft outreach emails, partner recap notes, and meeting summaries with risk logs.",
        }

    def _department_reviewers(self) -> Dict[str, str]:
        reviewers = {
            "Brand Marketing": "Brand lead reviewer",
            "Performance Marketing": "Performance marketing manager",
            "Merchandising & Buying": "Merchandising lead reviewer",
            "Customer Support": "Support quality lead",
            "Product": "Product owner",
            "Engineering": "Tech lead",
            "Warehouse & Fulfillment Ops": "Fulfillment operations lead",
            "Finance": "Finance controller",
            "Legal & Compliance": "Chief legal counsel",
            "Sales & Partnerships": "Partnerships lead",
        }
        if self.employees is not None:
            for dept in self.employees.get("department", pd.Series(dtype=str)).dropna().unique():
                dept = str(dept)
                reviewers.setdefault(dept, f"{dept} lead reviewer")
        return reviewers

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

    def _build_rule_based_suggestion(
        self,
        task: pd.Series,
        department_templates: Dict[str, str],
        reviewers: Dict[str, str],
        policy: Dict[str, object],
    ) -> Dict[str, object]:
        name = str(task.get("name", "")).lower()
        risky = any(k in name for k in ["fraud", "payment", "chargeback", "finance", "checkout", "pii"])
        ai_friendly = any(k in name for k in ["email", "deck", "presentation", "summary", "documentation", "brief"])
        department = str(task.get("department", "")).strip() or "General"
        reviewer = reviewers.get(department, "Department lead")
        redaction_instructions = str(policy["pii_redaction"])
        prohibited_scope = list(policy["prohibited_autonomy"])
        safe_use_notes = (
            f"{redaction_instructions} Avoid autonomy on {', '.join(prohibited_scope)}. Send draft to {reviewer} for sign-off."
        )
        return {
            "task_id": str(task.get("id")),
            "project_id": str(task.get("project_id")),
            "department": department,
            "recommended": bool(ai_friendly and not risky),
            "reviewer_required": True,
            "reviewer": reviewer,
            "reason": "Rule-based fallback classification",
            "suggested_prompt": (
                f"{department_templates.get(department, 'Draft the first version tailored to the audience.')} "
                f"Include a risk log and route to {reviewer} for approval. {redaction_instructions}"
            ).strip(),
            "safe_use_notes": safe_use_notes,
            "redaction_instructions": redaction_instructions,
            "prohibited_scope": prohibited_scope,
        }

    def _build_fallback_suggestions(
        self,
        department_templates: Dict[str, str],
        reviewers: Dict[str, str],
        policy: Dict[str, object],
    ) -> Dict[str, Dict[str, object]]:
        fallback: Dict[str, Dict[str, object]] = {}
        for _, task in self.tasks.iterrows():
            suggestion = self._build_rule_based_suggestion(task, department_templates, reviewers, policy)
            fallback[suggestion["task_id"]] = suggestion
        return fallback

    def _iter_task_batches(self, tasks: pd.DataFrame, batch_size: int):
        if batch_size <= 0:
            batch_size = len(tasks) or 1
        for start in range(0, len(tasks), batch_size):
            yield tasks.iloc[start : start + batch_size]

    def _render_prompt(self, tasks_subset: pd.DataFrame) -> str:
        department_templates = self._department_templates()
        reviewers = self._department_reviewers()
        policy = self._guardrail_policy()
        condensed_tasks = []
        for _, row in tasks_subset.iterrows():
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
        llm_tasks = self.tasks
        if self.test_mode:
            llm_tasks = llm_tasks.head(3)
        department_templates = self._department_templates()
        reviewers = self._department_reviewers()
        policy = self._guardrail_policy()
        fallback_by_id = self._build_fallback_suggestions(department_templates, reviewers, policy)

        payload_items: List[Dict[str, object]] = []
        batch_size = len(llm_tasks) if self.test_mode else 20
        for batch in self._iter_task_batches(llm_tasks, batch_size):
            user_prompt = self._render_prompt(batch)
            fallback_payload = {
                "suggestions": [
                    dict(fallback_by_id[str(row["id"])])
                    for _, row in batch.iterrows()
                    if str(row["id"]) in fallback_by_id
                ]
            }
            result = safe_openai_json(system_prompt, user_prompt, fallback=fallback_payload)
            chunk_items = list(result.get("suggestions", []))
            batch_ids = [str(row["id"]) for _, row in batch.iterrows()]
            seen_chunk_ids = {str(item.get("task_id")) for item in chunk_items}
            for task_id in batch_ids:
                if task_id not in seen_chunk_ids and task_id in fallback_by_id:
                    chunk_items.append(dict(fallback_by_id[task_id]))
                    seen_chunk_ids.add(task_id)
            payload_items.extend(chunk_items)

        deduped_items: List[Dict[str, object]] = []
        seen_ids = set()
        for item in payload_items:
            task_id = str(item.get("task_id"))
            if not task_id or task_id in seen_ids:
                continue
            seen_ids.add(task_id)
            deduped_items.append(item)
        payload_items = deduped_items

        suggestions: List[AISuggestion] = []
        default_safe_use = (
            f"{policy['pii_redaction']} Avoid autonomy on {', '.join(policy['prohibited_autonomy'])}. "
            f"{policy['review_requirement']}"
        )
        for item in payload_items:
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
                    prohibited_scope=[str(scope) for scope in item.get("prohibited_scope", policy["prohibited_autonomy"])],
                )
            )

        self.run_store.log(
            self.run_id,
            "ai_opportunity_scout",
            inputs={"tasks": len(self.tasks), "test_mode": self.test_mode},
            outputs={"suggestions": [s.__dict__ for s in suggestions]},
        )
        return suggestions
