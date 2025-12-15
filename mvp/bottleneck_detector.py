"""OpenAI-backed bottleneck detector with metrics context."""
from __future__ import annotations

import json
import math
import os
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

_MPL_CACHE_DIR = Path("reports") / ".matplotlib_cache"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path
import matplotlib.colors as mcolors
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

    def __init__(
        self,
        events: pd.DataFrame,
        employees: pd.DataFrame,
        run_store: RunStore,
        run_id: str,
        reports_dir: Path,
    ):
        self.events = events.copy()
        self.employees = employees.copy()
        self.run_store = run_store
        self.run_id = run_id
        self.reports_dir = reports_dir

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

    def _stage_end_time(self, events: pd.DataFrame, idx: int, assignee_id: object, fallback: pd.Timestamp) -> pd.Timestamp:
        """Find when the current stage actually completes for wait time math."""
        assignee = str(assignee_id or "")
        if assignee:
            for next_idx in range(idx + 1, len(events)):
                next_event = events.iloc[next_idx]
                next_ts = pd.to_datetime(next_event.get("timestamp"))
                from_assignee = str(next_event.get("from_assignee") or "")
                if from_assignee == assignee:
                    return next_ts
        if idx + 1 < len(events):
            return pd.to_datetime(events.iloc[idx + 1].get("timestamp"))
        return fallback

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
                stage_end_time = self._stage_end_time(evs_sorted, idx, assignee_id, timestamp)

                wait_hours = 0.0 if prev_end_time is None else max(
                    (timestamp - prev_end_time).total_seconds() / 3600.0,
                    0.0,
                )
                service_hours = max((stage_end_time - timestamp).total_seconds() / 3600.0, 0.0)

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
                        "end": stage_end_time,
                        "wait_hours": wait_hours,
                        "service_hours": service_hours,
                    }
                )

                prev_end_time = stage_end_time
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

    def _load_image_template(self) -> Dict[str, object]:
        template_path = self.reports_dir / "bottleneck_map_template.json"
        if template_path.exists():
            try:
                return json.loads(template_path.read_text())
            except json.JSONDecodeError:
                pass
        return {
            "diagram": {
                "type": "chord",
                "title": "Bottleneck flow (wait hotspots & handoffs)",
                "max_nodes": 12,
                "edge_color": "#9db4c0",
                "background": "#ffffff",
                "node": {
                    "color_scale": ["#d0e2ff", "#a5c8ff", "#7baaf0", "#3c74d4", "#144272"],
                },
            },
            "table": {
                "max_rows": 8,
                "columns": [
                    {"id": "stage", "label": "Stage", "align": "left"},
                    {"id": "mean_wait_hours", "label": "Wait μ (h)", "format": "{:.1f}", "align": "right"},
                    {"id": "p90_wait_hours", "label": "Wait p90 (h)", "format": "{:.1f}", "align": "right"},
                    {"id": "mean_service_hours", "label": "Service μ (h)", "format": "{:.1f}", "align": "right"},
                    {"id": "utilization_hours", "label": "Utilization (h)", "format": "{:.1f}", "align": "right"},
                ],
                "header_fill": "#14213d",
                "header_font_color": "#ffffff",
                "row_fill": "#f1faee",
                "alt_row_fill": "#ffffff",
            },
        }

    def _resolve_node_color(self, value: float, scale: List[str], min_value: float, max_value: float) -> str:
        if not scale:
            return "#1f77b4"
        if max_value <= min_value:
            return scale[-1]
        ratio = (value - min_value) / (max_value - min_value)
        idx = min(int(ratio * (len(scale) - 1)), len(scale) - 1)
        return scale[idx]

    def _build_table_rows(
        self,
        stage_metrics: Dict[str, Dict[str, float]],
        template: Dict[str, object],
    ) -> Tuple[List[str], List[List[object]], List[str]]:
        table_cfg: Dict[str, object] = template.get("table", {})  # type: ignore[assignment]
        cols: List[Dict[str, object]] = table_cfg.get("columns") or []  # type: ignore[assignment]
        if not cols:
            cols = [
                {"id": "stage", "label": "Stage", "align": "left"},
                {"id": "mean_wait_hours", "label": "Wait μ (h)", "format": "{:.1f}", "align": "right"},
                {"id": "mean_service_hours", "label": "Service μ (h)", "format": "{:.1f}", "align": "right"},
                {"id": "handoffs", "label": "Handoffs", "format": "{:.0f}", "align": "right"},
            ]

        max_rows = int(table_cfg.get("max_rows", 10))
        sorted_rows = sorted(
            stage_metrics.items(),
            key=lambda kv: kv[1].get("mean_wait_hours", 0.0),
            reverse=True,
        )[:max_rows]

        header_values = [str(col.get("label", col.get("id", ""))) for col in cols]
        alignments = [str(col.get("align", "left")) for col in cols]
        rows: List[List[object]] = []
        for stage, stats in sorted_rows:
            row_values: List[object] = []
            for col in cols:
                key = str(col.get("id", ""))
                fmt = col.get("format")
                if key == "stage":
                    row_values.append(stage)
                    continue
                raw_value = stats.get(key, 0.0)
                if isinstance(raw_value, (int, float)) and fmt:
                    try:
                        row_values.append(fmt.format(raw_value))
                    except Exception:
                        row_values.append(raw_value)
                else:
                    row_values.append(raw_value)
            rows.append(row_values)

        return header_values, rows, alignments

    def _render_table(
        self,
        ax,
        headers: List[str],
        rows: List[List[object]],
        alignments: List[str],
        template: Dict[str, object],
    ) -> None:
        table_cfg: Dict[str, object] = template.get("table", {})  # type: ignore[assignment]
        table = ax.table(
            cellText=rows,
            colLabels=headers,
            cellLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)
        for (row_idx, col_idx), cell in table.get_celld().items():
            if row_idx == 0:
                cell.set_facecolor(table_cfg.get("header_fill", "#14213d"))
                cell.get_text().set_color(table_cfg.get("header_font_color", "#ffffff"))
            else:
                fill_color = table_cfg.get("row_fill", "#f8f9fa") if row_idx % 2 else table_cfg.get("alt_row_fill", "#ffffff")
                cell.set_facecolor(fill_color)
                desired_align = alignments[col_idx] if col_idx < len(alignments) else "left"
                cell._loc = "right" if desired_align == "right" else "left"

        ax.axis("off")

    def _generate_bottleneck_image(self, metrics: Dict[str, object]) -> str | None:
        stage_metrics: Dict[str, Dict[str, float]] = metrics.get("stage_metrics", {})  # type: ignore[assignment]
        edges: List[Dict[str, object]] = metrics.get("edges", [])  # type: ignore[assignment]
        if not stage_metrics:
            return None

        template = self._load_image_template()
        diagram_cfg: Dict[str, object] = template.get("diagram", {})  # type: ignore[assignment]
        max_nodes = int(diagram_cfg.get("max_nodes", 12))
        sorted_stages = sorted(
            stage_metrics.items(),
            key=lambda kv: kv[1].get("mean_wait_hours", 0.0),
            reverse=True,
        )[:max_nodes]
        if not sorted_stages:
            return None

        nodes = [stage for stage, _ in sorted_stages]
        filtered_edges = [
            edge
            for edge in edges
            if str(edge.get("from")) in nodes and str(edge.get("to")) in nodes
        ]
        if not filtered_edges:
            return None

        node_waits = [stage_metrics[stage].get("mean_wait_hours", 0.0) for stage in nodes]
        min_wait = min(node_waits) if node_waits else 0.0
        max_wait = max(node_waits) if node_waits else 1.0
        color_scale: List[str] = diagram_cfg.get("node", {}).get("color_scale", [])  # type: ignore[assignment]
        node_colors = [
            self._resolve_node_color(stage_metrics[stage].get("mean_wait_hours", 0.0), color_scale, min_wait, max_wait)
            for stage in nodes
        ]

        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.15)
        ax_flow = fig.add_subplot(gs[0])
        ax_flow.set_facecolor(diagram_cfg.get("background", "#ffffff"))
        ax_flow.axis("off")

        angles = {stage: 2 * math.pi * idx / len(nodes) for idx, stage in enumerate(nodes)}
        radius = 1.0
        for idx, stage in enumerate(nodes):
            angle = angles[stage]
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            ax_flow.scatter(x, y, s=300, color=node_colors[idx], edgecolors="#ffffff", linewidths=1.5, zorder=3)
            label = textwrap.fill(stage, width=25)
            align = "left" if x >= 0 else "right"
            ax_flow.text(
                x * 1.15,
                y * 1.15,
                label,
                ha=align,
                va="center",
                fontsize=9,
                color="#1f1f1f",
            )

        edge_values = [
            float(edge.get("count", 1)) * max(float(edge.get("mean_wait_hours", 0.0)), 0.1)
            for edge in filtered_edges
        ]
        max_value = max(edge_values) if edge_values else 1.0
        edge_color = diagram_cfg.get("edge_color", "#9db4c0")

        for edge, value in zip(filtered_edges, edge_values):
            src_stage = str(edge.get("from"))
            dst_stage = str(edge.get("to"))
            src_angle = angles.get(src_stage)
            dst_angle = angles.get(dst_stage)
            if src_angle is None or dst_angle is None:
                continue
            src_pos = (radius * math.cos(src_angle), radius * math.sin(src_angle))
            dst_pos = (radius * math.cos(dst_angle), radius * math.sin(dst_angle))
            rad = 0.2 if src_angle <= dst_angle else -0.2
            width = 0.5 + 4.0 * (value / max_value)
            arrow = patches.FancyArrowPatch(
                src_pos,
                dst_pos,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-",
                linewidth=width,
                color=edge_color,
                alpha=0.5,
                zorder=1,
            )
            ax_flow.add_patch(arrow)

        ax_flow.set_title(str(diagram_cfg.get("title", "Process bottleneck map")), fontsize=14, pad=20)

        headers, rows, alignments = self._build_table_rows(stage_metrics, template)
        ax_table = fig.add_subplot(gs[1])
        self._render_table(ax_table, headers, rows, alignments, template)

        image_path = self.reports_dir / "bottleneck_map.png"
        fig.savefig(image_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return str(image_path)

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
        bottleneck_image = self._generate_bottleneck_image(metrics)

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
            def stage_score(item: tuple[str, Dict[str, float]]) -> float:
                _, data = item
                wait = data.get("p90_wait_hours", 0.0)
                service = data.get("p90_service_hours", 0.0)
                utilization = data.get("utilization_hours", 0.0)
                return wait + service * 0.1 + utilization * 0.01

            worst_stage, stats = max(stage_metric_values.items(), key=stage_score)
            metric_value = stats.get("p90_wait_hours", 0.0)
            issue = "Longest tail wait time" if metric_value > 0 else "Sustained high service time/utilization"
            fallback_bottlenecks.append(
                {
                    "stage": worst_stage,
                    "issue": issue,
                    "metric": metric_value if metric_value > 0 else stats.get("p90_service_hours", 0.0),
                    "unit": "hours",
                    "recommendation": "Add WIP limits, parallelize reviews, and boost capacity at this stage.",
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
                "bottleneck_image": bottleneck_image,
            },
        )

        return {
            "bottlenecks": bottlenecks,
            "stage_delays": stage_delays,
            "process_graph": metrics,
            "bottleneck_map": bottleneck_map,
            "bottleneck_image": bottleneck_image,
        }
