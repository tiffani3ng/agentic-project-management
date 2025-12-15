"""OpenAI-backed bottleneck detector with metrics context."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

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
    mean_hours: float
    handoffs: int


class BottleneckDetector:
    """Computes baseline delays and enriches with OpenAI insights."""

    def __init__(self, events: pd.DataFrame, run_store: RunStore, run_id: str):
        self.events = events.copy()
        self.run_store = run_store
        self.run_id = run_id

    def _compute_metrics(self) -> Dict[str, Dict[str, float]]:
        stage_times: Dict[str, List[float]] = defaultdict(list)
        handoff_counts: Dict[str, int] = defaultdict(int)

        grouped = self.events.groupby("task_id")
        for _, evs in grouped:
            evs_sorted = evs.sort_values("timestamp")
            start_ts = evs_sorted[evs_sorted["type"] == "start"]["timestamp"]
            end_ts = evs_sorted[evs_sorted["type"] == "end"]["timestamp"]
            if start_ts.empty:
                continue
            start = pd.to_datetime(start_ts.iloc[0])
            end = pd.to_datetime(end_ts.iloc[0]) if not end_ts.empty else None
            stage = evs_sorted.iloc[0]["type"]
            if end is not None:
                hours = (end - start).total_seconds() / 3600.0
                stage_times[stage].append(hours)
            handoff_counts[stage] += int((evs_sorted["type"] == "handoff").sum())

        metrics = {}
        for stage, times in stage_times.items():
            metrics[stage] = {
                "mean_hours": float(sum(times) / max(len(times), 1)),
                "handoffs": float(handoff_counts.get(stage, 0)),
            }
        return metrics

    def _render_prompt(self, metrics: Dict[str, Dict[str, float]]) -> str:
        return json.dumps(
            {
                "stage_metrics": metrics,
                "instructions": "Identify up to 3 bottlenecks based on highest mean wait/processing time and excessive handoffs. Provide remediation suggestions.",
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

    def run(self) -> Dict[str, List]:
        metrics = self._compute_metrics()
        stage_delays = [StageDelay(stage=s, mean_hours=v.get("mean_hours", 0.0), handoffs=int(v.get("handoffs", 0))) for s, v in metrics.items()]

        system_prompt = (
            "You are a sober workflow diagnostics expert. Use only the provided metrics to "
            "spot the top bottlenecks, quantify them, and suggest pragmatic fixes. Do not "
            "invent stages or metrics. Respond with JSON that matches the schema and "
            "nothing else."
        )
        user_prompt = self._render_prompt(metrics)

        fallback_bottlenecks = []
        if metrics:
            worst_stage = max(metrics.items(), key=lambda kv: kv[1].get("mean_hours", 0.0))[0]
            fallback_bottlenecks.append(
                {
                    "stage": worst_stage,
                    "issue": "Longest mean duration",
                    "metric": metrics[worst_stage].get("mean_hours", 0.0),
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
            },
        )

        return {"bottlenecks": bottlenecks, "stage_delays": stage_delays}
