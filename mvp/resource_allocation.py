"""OpenAI-backed resource allocation agent with heuristic fallback."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .llm_utils import safe_openai_json
from .run_store import RunStore


@dataclass
class Assignment:
    task_id: str
    project_id: str
    assignee: str
    score: float
    rationale: str


class ResourceAllocationAgent:
    """
    Uses OpenAI to balance skill coverage, availability, and load.
    Falls back to a deterministic heuristic if the API is unavailable.
    """

    def __init__(self, employees: pd.DataFrame, availability: pd.DataFrame, run_store: RunStore, run_id: str):
        self.employees = employees.copy()
        self.availability = availability.copy()
        self.run_store = run_store
        self.run_id = run_id

    def _availability_score(self, employee_id: str, est_hours: float) -> float:
        avail_rows = self.availability[self.availability["employee_id"] == employee_id]
        if avail_rows.empty:
            return 0.2
        max_free = avail_rows["hours_free"].max()
        return min(max_free, est_hours) / (est_hours + 1e-6)

    def _heuristic_assign(self, tasks: pd.DataFrame) -> List[Assignment]:
        existing_workload = (
            tasks[tasks["assignee"].notna() & (tasks["assignee"] != "")]
            .groupby("assignee")["est_hours"]
            .sum()
            .to_dict()
        )
        incremental_load: Dict[str, float] = {}
        assignments: List[Assignment] = []

        open_tasks = tasks[tasks["status"].str.lower() != "completed"].copy().reset_index(drop=True)
        for _, task in open_tasks.iterrows():
            best_candidate: Optional[str] = None
            best_score = -1.0
            rationale = ""

            for _, employee in self.employees.iterrows():
                emp_id = employee["id"]
                skills = [s.lower() for s in employee["skills"]]
                needed = str(task["skill_needed"]).lower()
                max_hours = float(employee["max_hours"])
                current_load = existing_workload.get(emp_id, 0.0) + incremental_load.get(emp_id, 0.0)

                skill_score = 1.0 if needed in skills else (0.6 if any(needed in s or s in needed for s in skills) else 0.0)
                availability_score = self._availability_score(emp_id, float(task["est_hours"]))
                balance_score = max(0.0, (max_hours - current_load) / max_hours)
                penalty = max(0.0, (current_load + float(task["est_hours"]) - max_hours) / max_hours)
                score = (skill_score * 4) + (availability_score * 2) + balance_score - (penalty * 1.5)

                if score > best_score:
                    best_score = score
                    best_candidate = emp_id
                    rationale = (
                        f"skill_score={skill_score:.2f}, availability_score={availability_score:.2f}, "
                        f"balance_score={balance_score:.2f}, overtime_penalty={penalty*1.5:.2f}"
                    )

            if best_candidate:
                incremental_load[best_candidate] = incremental_load.get(best_candidate, 0.0) + float(task["est_hours"])
                assignments.append(
                    Assignment(
                        task_id=str(task["id"]),
                        project_id=str(task["project_id"]),
                        assignee=best_candidate,
                        score=best_score,
                        rationale=rationale,
                    )
                )

        return assignments

    def _render_prompt(self, tasks: pd.DataFrame) -> str:
        condensed_employees = [
            {
                "id": row["id"],
                "skills": row["skills"],
                "max_hours": row["max_hours"],
                "timezone": row["timezone"],
            }
            for _, row in self.employees.iterrows()
        ]
        condensed_tasks = [
            {
                "id": row["id"],
                "project_id": row["project_id"],
                "skill_needed": row["skill_needed"],
                "est_hours": row["est_hours"],
                "status": row["status"],
            }
            for _, row in tasks.iterrows()
        ]
        availability_summary = (
            self.availability.groupby("employee_id")["hours_free"].mean().reset_index().to_dict("records")
        )

        return json.dumps(
            {
                "employees": condensed_employees,
                "availability": availability_summary,
                "tasks": condensed_tasks,
                "instructions": "Assign only non-completed tasks. Maximize skill match, respect capacity and availability, avoid overtime. Include timezone overlap considerations and prefer balanced load across people.",
                "response_schema": {
                    "assignments": [
                        {
                            "task_id": "T0001",
                            "project_id": "P001",
                            "assignee": "E001",
                            "score": 0.95,
                            "rationale": "why this person fits",
                        }
                    ]
                },
            },
            indent=2,
        )

    def run(self, tasks: pd.DataFrame) -> List[Assignment]:
        system_prompt = "You are a resource allocation agent for cross-functional teams. Return strictly valid JSON."
        user_prompt = self._render_prompt(tasks)
        fallback = {"assignments": [a.__dict__ for a in self._heuristic_assign(tasks)]}
        result = safe_openai_json(system_prompt, user_prompt, fallback=fallback)

        assignments: List[Assignment] = []
        for item in result.get("assignments", []):
            assignments.append(
                Assignment(
                    task_id=str(item.get("task_id")),
                    project_id=str(item.get("project_id")),
                    assignee=str(item.get("assignee")),
                    score=float(item.get("score", 0.0)),
                    rationale=str(item.get("rationale", "")),
                )
            )

        self.run_store.log(
            self.run_id,
            "resource_allocation",
            inputs={"tasks": len(tasks)},
            outputs={"assignments": [a.__dict__ for a in assignments]},
        )
        return assignments
