"""OpenAI-backed bottleneck detector with metrics context."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from .llm_utils import safe_openai_json
from .run_store import RunStore


@dataclass
class Bottleneck:
    stage: str
    issue: str
    metric: float
    unit: str
    recommendation: str


@dataclass
class StageDelay:
    stage: str
    mean_service_hours: float
    mean_wait_hours: float
    p90_service_hours: float
    p90_wait_hours: float
    handoffs: int
    utilization_hours: float


class BottleneckDetector:
    """Computes baseline delays and enriches with OpenAI insights."""

    def __init__(self, events: pd.DataFrame, employees: pd.DataFrame, run_store: RunStore, run_id: str):
        self.events = events.copy()
        self.employees = employees.copy()
        self.run_store = run_store
        self.run_id = run_id

    def _role_lookup(self) -> Dict[str, str]:
        lookup = {}
        for _, row in self.employees.iterrows():
            role = str(row.get("role", "")).strip()
            dept = str(row.get("department", "")).strip()
            label = role if role else str(row.get("name", ""))
            if dept:
                label = f"{label} ({dept})" if label else dept
            lookup[str(row.get("id"))] = label or str(row.get("id"))
        return lookup

    def _percentile(self, values: List[float], q: float) -> float:
        if not values:
            return 0.0
        return float(pd.Series(values).quantile(q))

    def _compute_metrics(self) -> Dict[str, object]:
        role_lookup = self._role_lookup()

        stage_stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {
            "service_hours": [],
            "wait_hours": [],
            "handoffs": 0,
        })
        edge_stats: Dict[Tuple[str, str], Dict[str, List[float] | int]] = defaultdict(
            lambda: {"wait_hours": [], "count": 0}
        )
        utilization: Dict[str, float] = defaultdict(float)
        task_paths: List[Dict[str, object]] = []

        grouped = self.events.groupby("task_id")
        for task_id, evs in grouped:
            evs_sorted = evs.sort_values("timestamp").reset_index(drop=True)
            prev_end_time = None
            prev_role = None
            stages_for_task: List[Dict[str, object]] = []

            for idx, event in evs_sorted.iterrows():
                event_type = str(event.get("type", ""))
                timestamp = pd.to_datetime(event.get("timestamp"))

                if event_type not in {"start", "handoff"}:
                    if event_type == "end":
                        prev_end_time = timestamp
                    continue

                assignee_id = event.get("to_assignee") or event.get("from_assignee") or "Unassigned"
                stage_role = role_lookup.get(str(assignee_id), str(assignee_id))

                next_ts = (
                    pd.to_datetime(evs_sorted.iloc[idx + 1]["timestamp"])
                    if idx + 1 < len(evs_sorted)
                    else timestamp
                )

                wait_hours = 0.0 if prev_end_time is None else (timestamp - prev_end_time).total_seconds() / 3600.0
                service_hours = max((next_ts - timestamp).total_seconds() / 3600.0, 0.0)

                stage_stats[stage_role]["service_hours"].append(service_hours)
                stage_stats[stage_role]["wait_hours"].append(wait_hours)
                if event_type == "handoff":
                    stage_stats[stage_role]["handoffs"] += 1
                utilization[stage_role] += service_hours

                if prev_role:
                    edge_stats[(prev_role, stage_role)]["wait_hours"].append(wait_hours)
                    edge_stats[(prev_role, stage_role)]["count"] += 1

                stages_for_task.append(
                    {
                        "role": stage_role,
                        "start": timestamp,
                        "end": next_ts,
                        "wait_hours": wait_hours,
                        "service_hours": service_hours,
                    }
                )

                prev_end_time = next_ts
                prev_role = stage_role

            if stages_for_task:
                total_duration = (stages_for_task[-1]["end"] - stages_for_task[0]["start"]).total_seconds() / 3600.0
                task_paths.append(
                    {
                        "task_id": task_id,
                        "total_hours": total_duration,
                        "stages": [
                            {
                                "role": s["role"],
                                "service_hours": s["service_hours"],
                                "wait_hours": s["wait_hours"],
                            }
                            for s in stages_for_task
                        ],
                    }
                )

        stage_metrics: Dict[str, Dict[str, float]] = {}
        for stage, stats in stage_stats.items():
            service_hours_list = stats["service_hours"]
            wait_hours_list = stats["wait_hours"]
            stage_metrics[stage] = {
                "mean_service_hours": float(sum(service_hours_list) / max(len(service_hours_list), 1)),
                "p90_service_hours": self._percentile(service_hours_list, 0.9),
                "mean_wait_hours": float(sum(wait_hours_list) / max(len(wait_hours_list), 1)),
                "p90_wait_hours": self._percentile(wait_hours_list, 0.9),
                "handoffs": float(stats["handoffs"]),
                "instances": float(len(service_hours_list)),
                "utilization_hours": float(utilization.get(stage, 0.0)),
            }

        edges = []
        for (src, dst), stats in edge_stats.items():
            waits = stats["wait_hours"]
            edges.append(
                {
                    "from": src,
                    "to": dst,
                    "count": int(stats["count"]),
                    "mean_wait_hours": float(sum(waits) / max(len(waits), 1)),
                    "p90_wait_hours": self._percentile(waits, 0.9),
                }
            )

        critical_paths = sorted(task_paths, key=lambda t: t.get("total_hours", 0.0), reverse=True)[:3]

        return {
            "stage_metrics": stage_metrics,
            "edges": edges,
            "critical_paths": critical_paths,
        }

    def _render_prompt(self, metrics: Dict[str, object]) -> str:
        return json.dumps(
            {
                "stage_metrics": metrics.get("stage_metrics", {}),
                "edge_metrics": metrics.get("edges", []),
                "critical_paths": metrics.get("critical_paths", []),
                "instructions": "Identify up to 3 bottlenecks based on highest mean wait/processing time and excessive handoffs. Focus on roles and handoff edges with high p90 wait times or utilization. Provide remediation suggestions that target the affected roles or edges.",
                "response_schema": {
                    "bottlenecks": [
                        {
                            "stage": "stage name",
                            "issue": "description",
                            "metric": 12.5,
                            "unit": "hours",
                            "recommendation": "action",
                        }
                    ],
                },
            },
            indent=2,
        )

    def _render_bottleneck_map(self, metrics: Dict[str, object]) -> str:
        lines: List[str] = []
        stage_metrics: Dict[str, Dict[str, float]] = metrics.get("stage_metrics", {})  # type: ignore[assignment]
        edges: List[Dict[str, object]] = metrics.get("edges", [])  # type: ignore[assignment]
        lines.append("Process bottleneck map (roles as nodes, handoffs as edges):")
        for stage, stats in sorted(stage_metrics.items(), key=lambda kv: kv[1].get("p90_wait_hours", 0.0), reverse=True):
            lines.append(
                f"- {stage}: wait μ={stats.get('mean_wait_hours', 0.0):.1f}h p90={stats.get('p90_wait_hours', 0.0):.1f}h | service μ={stats.get('mean_service_hours', 0.0):.1f}h p90={stats.get('p90_service_hours', 0.0):.1f}h | utilization={stats.get('utilization_hours', 0.0):.1f}h"
            )
        if edges:
            lines.append("- Handoffs:")
            for edge in sorted(edges, key=lambda e: e.get("p90_wait_hours", 0.0), reverse=True):
                lines.append(
                    f"  {edge['from']} -> {edge['to']}: count={edge.get('count', 0)}, wait μ={edge.get('mean_wait_hours', 0.0):.1f}h p90={edge.get('p90_wait_hours', 0.0):.1f}h"
                )
        critical_paths: List[Dict[str, object]] = metrics.get("critical_paths", [])  # type: ignore[assignment]
        if critical_paths:
            lines.append("- Critical paths (longest tasks):")
            for path in critical_paths:
                stage_chain = " -> ".join([str(s.get("role")) for s in path.get("stages", [])])
                lines.append(
                    f"  Task {path.get('task_id')}: {path.get('total_hours', 0.0):.1f}h across {stage_chain}"
                )
        return "\n".join(lines)

    def run(self) -> Dict[str, List]:
        metrics = self._compute_metrics()
        stage_delays = [
            StageDelay(
                stage=s,
                mean_service_hours=v.get("mean_service_hours", 0.0),
                mean_wait_hours=v.get("mean_wait_hours", 0.0),
                p90_service_hours=v.get("p90_service_hours", 0.0),
                p90_wait_hours=v.get("p90_wait_hours", 0.0),
                handoffs=int(v.get("handoffs", 0)),
                utilization_hours=v.get("utilization_hours", 0.0),
            )
            for s, v in metrics.get("stage_metrics", {}).items()
        ]
        bottleneck_map = self._render_bottleneck_map(metrics)

        system_prompt = (
            "You are a sober workflow diagnostics expert. Use only the provided metrics to "
            "spot the top bottlenecks, quantify them, and suggest pragmatic fixes. Do not "
            "invent stages or metrics. Respond with JSON that matches the schema and "
            "nothing else."
        )
        user_prompt = self._render_prompt(metrics)

        fallback_bottlenecks = []
        stage_metric_values = metrics.get("stage_metrics", {})
        if stage_metric_values:
            worst_stage = max(stage_metric_values.items(), key=lambda kv: kv[1].get("p90_wait_hours", 0.0))[0]
            fallback_bottlenecks.append(
                {
                    "stage": worst_stage,
                    "issue": "Longest tail wait time",
                    "metric": stage_metric_values[worst_stage].get("p90_wait_hours", 0.0),
                    "unit": "hours",
                    "recommendation": "Add WIP limits and reduce handoffs at this stage.",
                }
            )

        result = safe_openai_json(
            system_prompt,
            user_prompt,
            fallback={"bottlenecks": fallback_bottlenecks},
            temperature=0.1,
        )

        bottlenecks: List[Bottleneck] = []
        for item in result.get("bottlenecks", []):
            bottlenecks.append(
                Bottleneck(
                    stage=str(item.get("stage")),
                    issue=str(item.get("issue")),
                    metric=float(item.get("metric", 0.0)),
                    unit=str(item.get("unit", "hours")),
                    recommendation=str(item.get("recommendation", "")),
                )
            )

        self.run_store.log(
            self.run_id,
            "bottleneck_detector",
            inputs={"events": len(self.events)},
            outputs={
                "bottlenecks": [b.__dict__ for b in bottlenecks],
                "metrics": metrics,
                "bottleneck_map": bottleneck_map,
            },
        )

        return {"bottlenecks": bottlenecks, "stage_delays": stage_delays, "process_graph": metrics, "bottleneck_map": bottleneck_map}
