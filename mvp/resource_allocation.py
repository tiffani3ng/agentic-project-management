"""OpenAI-backed resource allocation agent with heuristic fallback."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

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


@dataclass
class WorkloadRollup:
    employee_id: str
    baseline_hours: float
    newly_assigned_hours: float
    projected_hours: float
    max_hours: float
    utilization: float


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

    def _availability_score(
        self, employee_id: str, est_hours: float, task_start: Optional[pd.Timestamp], task_due: Optional[pd.Timestamp]
    ) -> float:
        avail_rows = self.availability[self.availability["employee_id"] == employee_id]
        if avail_rows.empty:
            return 0.2

        start_ts = task_start if isinstance(task_start, pd.Timestamp) else pd.to_datetime(task_start, errors="coerce")
        due_ts = task_due if isinstance(task_due, pd.Timestamp) else pd.to_datetime(task_due, errors="coerce")

        if pd.notna(start_ts) and pd.notna(due_ts):
            window_rows = avail_rows[
                (avail_rows["date"] >= start_ts.normalize()) & (avail_rows["date"] <= due_ts.normalize())
            ]
            if not window_rows.empty:
                window_hours = window_rows["hours_free"].clip(lower=0).sum()
                if window_hours <= 0:
                    return 0.0
                return min(window_hours, est_hours) / (est_hours + 1e-6)

        max_free = avail_rows["hours_free"].clip(lower=0).max()
        if max_free <= 0:
            return 0.0
        return min(max_free, est_hours) / (est_hours + 1e-6)

    def _timezone_overlap_score(
        self, task_start: Optional[pd.Timestamp], task_due: Optional[pd.Timestamp], employee_tz: str
    ) -> float:
        if pd.isna(task_start) or pd.isna(task_due):
            return 0.5

        try:
            tz = ZoneInfo(str(employee_tz))
        except Exception:
            return 0.4

        start_ts = task_start if isinstance(task_start, pd.Timestamp) else pd.to_datetime(task_start, errors="coerce")
        due_ts = task_due if isinstance(task_due, pd.Timestamp) else pd.to_datetime(task_due, errors="coerce")
        if pd.isna(start_ts) or pd.isna(due_ts):
            return 0.5

        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        if due_ts.tzinfo is None:
            due_ts = due_ts.tz_localize("UTC")
        else:
            due_ts = due_ts.tz_convert("UTC")

        start_local = start_ts.tz_convert(tz)
        due_local = due_ts.tz_convert(tz)
        total_duration_hours = (due_local - start_local).total_seconds() / 3600
        if total_duration_hours <= 0:
            return 0.5

        overlap_hours = 0.0
        current = start_local
        work_start = time(9, 0)
        work_end = time(17, 0)

        while current < due_local:
            day_start = datetime.combine(current.date(), work_start, tzinfo=tz)
            day_end = datetime.combine(current.date(), work_end, tzinfo=tz)
            window_start = max(current, day_start)
            window_end = min(due_local, day_end)
            if window_end > window_start:
                overlap_hours += (window_end - window_start).total_seconds() / 3600
            current = datetime.combine(current.date() + timedelta(days=1), time(0, 0), tzinfo=tz)

        return max(0.0, min(1.0, overlap_hours / total_duration_hours))

    @staticmethod
    def _is_unstarted(status: str) -> bool:
        normalized = str(status).strip().lower()
        unstarted_markers = {"", "not_started", "not started", "todo", "pending", "backlog"}
        return normalized in unstarted_markers

    def _existing_open_workload(self, tasks: pd.DataFrame) -> Dict[str, float]:
        open_tasks = tasks[
            tasks["assignee"].notna()
            & (tasks["assignee"] != "")
            & (tasks["status"].str.lower() != "completed")
        ]
        return open_tasks.groupby("assignee")["est_hours"].sum().to_dict()

    def _heuristic_assign(self, tasks: pd.DataFrame) -> List[Assignment]:
        existing_workload = self._existing_open_workload(tasks)
        incremental_load: Dict[str, float] = {}
        assignments: List[Assignment] = []

        open_tasks = tasks[tasks["status"].apply(self._is_unstarted)].copy().reset_index(drop=True)
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
                est_hours = float(task["est_hours"])

                skill_score = 1.0 if needed in skills else (0.6 if any(needed in s or s in needed for s in skills) else 0.0)
                availability_score = self._availability_score(emp_id, est_hours, task["start"], task["due"])
                balance_score = max(0.0, (max_hours - current_load) / max_hours)
                timezone_score = self._timezone_overlap_score(task["start"], task["due"], employee.get("timezone", "UTC"))
                overtime_risk = max(0.0, (current_load + est_hours - max_hours) / max_hours)
                score = (skill_score + availability_score + balance_score + timezone_score) - overtime_risk

                if score > best_score:
                    best_score = score
                    best_candidate = emp_id
                    rationale = (
                        f"skill_score={skill_score:.2f}, availability_score={availability_score:.2f}, "
                        f"balance_score={balance_score:.2f}, timezone_overlap={timezone_score:.2f}, "
                        f"overtime_risk={overtime_risk:.2f}"
                    )

            if best_candidate:
                incremental_load[best_candidate] = incremental_load.get(best_candidate, 0.0) + est_hours
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

    def _build_workload_rollups(
        self, assignments: List[Assignment], tasks: pd.DataFrame, baseline: Optional[Dict[str, float]] = None
    ) -> List[WorkloadRollup]:
        baseline_load = baseline if baseline is not None else self._existing_open_workload(tasks)
        tasks_by_id = {str(row["id"]): float(row["est_hours"]) for _, row in tasks.iterrows()}

        incremental: Dict[str, float] = {}
        for assignment in assignments:
            incremental[assignment.assignee] = incremental.get(assignment.assignee, 0.0) + tasks_by_id.get(
                assignment.task_id, 0.0
            )

        rollups: List[WorkloadRollup] = []
        for _, employee in self.employees.iterrows():
            emp_id = employee["id"]
            max_hours = float(employee["max_hours"])
            base_hours = float(baseline_load.get(emp_id, 0.0))
            new_hours = float(incremental.get(emp_id, 0.0))
            projected = base_hours + new_hours
            utilization = projected / max_hours if max_hours > 0 else 0.0
            rollups.append(
                WorkloadRollup(
                    employee_id=emp_id,
                    baseline_hours=base_hours,
                    newly_assigned_hours=new_hours,
                    projected_hours=projected,
                    max_hours=max_hours,
                    utilization=utilization,
                )
            )

        return rollups

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
                "instructions": "Assign only unstarted tasks. Maximize skill match, respect capacity and availability calendars (including zero-availability days), avoid overtime, and account for timezone overlap with task windows.",
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

    def run(self, tasks: pd.DataFrame) -> Dict[str, List]:
        system_prompt = (
            "You are a meticulous resource allocation co-pilot. Act like an operations "
            "manager who balances skill fit, capacity, timezone overlap, and fairness. "
            "Never invent employees or tasks. Return strictly valid JSON matching the "
            "requested schema and nothing else."
        )
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

        workloads = self._build_workload_rollups(assignments, tasks)
        self.run_store.log(
            self.run_id,
            "resource_allocation",
            inputs={"tasks": len(tasks)},
            outputs={"assignments": [a.__dict__ for a in assignments], "workloads": [w.__dict__ for w in workloads]},
        )
        return {"assignments": assignments, "workloads": workloads}
