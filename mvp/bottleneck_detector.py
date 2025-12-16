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
                "background": "#fefefe",
                "node": {
                    "color_scale": ["#d0e2ff", "#a5c8ff", "#7baaf0", "#3c74d4", "#144272"],
                },
                "palette": [
                    "#a1c9f4",
                    "#ffb482",
                    "#8de5a1",
                    "#ff9f9b",
                    "#d0bbff",
                    "#debb9b",
                    "#fab0e4",
                    "#cfcfcf",
                    "#fffea3",
                    "#b9f2f0",
                ],
                "font": {"family": "Cambria", "size": 13, "label_size": 11, "title_size": 17},
                "inner_radius": 0.9,
                "outer_radius": 1.1,
                "tick_length": 0.05,
                "gap_degrees": 2.0,
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

        palette: List[str] = diagram_cfg.get("palette", [])  # type: ignore[assignment]
        if not palette:
            palette = ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff", "#debb9b", "#fab0e4", "#cfcfcf"]

        font_cfg: Dict[str, object] = diagram_cfg.get("font", {})  # type: ignore[assignment]
        font_family = str(font_cfg.get("family", "Cambria"))
        title_size = float(font_cfg.get("title_size", font_cfg.get("size", 15)))
        label_size = float(font_cfg.get("label_size", font_cfg.get("size", 12)))

        def polar(angle: float, radius: float) -> Tuple[float, float]:
            return (radius * math.cos(angle), radius * math.sin(angle))

        def lighten(color: str, amount: float) -> Tuple[float, float, float]:
            r, g, b = mcolors.to_rgb(color)
            return tuple(min(1.0, c + (1.0 - c) * amount) for c in (r, g, b))  # type: ignore[return-value]

        def darken(color: str, factor: float) -> Tuple[float, float, float]:
            r, g, b = mcolors.to_rgb(color)
            return tuple(max(0.0, c * factor) for c in (r, g, b))  # type: ignore[return-value]

        matrix = [[0.0 for _ in nodes] for _ in nodes]
        for edge in filtered_edges:
            src = str(edge.get("from"))
            dst = str(edge.get("to"))
            if src == dst:
                continue
            src_idx = nodes.index(src)
            dst_idx = nodes.index(dst)
            base = float(edge.get("count", 1)) * max(float(edge.get("mean_wait_hours", 0.0)), 0.25)
            if base <= 0:
                continue
            matrix[src_idx][dst_idx] += base
            matrix[dst_idx][src_idx] += base

        row_totals = [sum(row) for row in matrix]
        total_flow = sum(row_totals)
        if total_flow <= 0:
            return None

        inner_radius = float(diagram_cfg.get("inner_radius", 0.9))
        outer_radius = float(diagram_cfg.get("outer_radius", 1.15))
        tick_length = float(diagram_cfg.get("tick_length", 0.05))
        gap_radians = math.radians(float(diagram_cfg.get("gap_degrees", 2.0)))
        total_gap = gap_radians * len(nodes)
        available_angle = max(2 * math.pi - total_gap, 0.1)

        fig = plt.figure(figsize=(9, 9))
        ax_flow = fig.add_subplot(111)
        ax_flow.set_facecolor(diagram_cfg.get("background", "#fefefe"))
        fig.patch.set_facecolor(diagram_cfg.get("background", "#fefefe"))
        ax_flow.axis("off")

        node_colors = [palette[idx % len(palette)] for idx in range(len(nodes))]
        arc_angles: Dict[str, Dict[str, float]] = {}
        current_angle = 0.0
        for idx, stage in enumerate(nodes):
            frac = row_totals[idx] / total_flow if total_flow else 0.0
            arc_span = frac * available_angle
            start_angle = current_angle
            end_angle = start_angle + arc_span
            arc_angles[stage] = {"start": start_angle, "end": end_angle, "extent": arc_span}
            current_angle = end_angle + gap_radians

        def ribbon_patch(source: Tuple[float, float], target: Tuple[float, float]) -> Path:
            src_start, src_end = source
            dst_start, dst_end = target
            control_r = inner_radius * 0.55
            verts = [
                polar(src_start, inner_radius),
                polar(src_start, control_r),
                polar(dst_start, control_r),
                polar(dst_start, inner_radius),
                polar(dst_end, inner_radius),
                polar(dst_end, control_r),
                polar(src_end, control_r),
                polar(src_end, inner_radius),
                polar(src_start, inner_radius),
            ]
            codes = [
                Path.MOVETO,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.LINETO,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.CLOSEPOLY,
            ]
            return Path(verts, codes)

        node_progress = {stage: arc_angles[stage]["start"] for stage in nodes}
        sub_arcs: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for i, stage in enumerate(nodes):
            extent = arc_angles[stage]["extent"]
            if extent <= 0 or row_totals[i] <= 0:
                continue
            for j, value in enumerate(matrix[i]):
                if value <= 0:
                    continue
                angle_start = node_progress[stage]
                angle_end = angle_start + (value / row_totals[i]) * extent
                sub_arcs[(i, j)] = (angle_start, angle_end)
                node_progress[stage] = angle_end

        drawn_pairs = set()
        for i in range(len(nodes)):
            for j in range(i, len(nodes)):
                if i == j:
                    continue
                if (i, j) not in sub_arcs or (j, i) not in sub_arcs:
                    continue
                pair_key = (i, j)
                if pair_key in drawn_pairs:
                    continue
                drawn_pairs.add(pair_key)
                path = ribbon_patch(sub_arcs[(i, j)], sub_arcs[(j, i)])
                color = node_colors[i]
                edge_color = darken(color, 0.7)
                patch = patches.PathPatch(
                    path,
                    facecolor=lighten(color, 0.35),
                    edgecolor=edge_color,
                    lw=0.8,
                    alpha=0.9,
                    zorder=2,
                )
                ax_flow.add_patch(patch)

        for idx, stage in enumerate(nodes):
            angles = arc_angles[stage]
            start = math.degrees(angles["start"])
            end = math.degrees(angles["end"])
            if end <= start:
                continue
            arc = patches.Wedge(
                center=(0, 0),
                r=outer_radius,
                theta1=start,
                theta2=end,
                width=outer_radius - inner_radius,
                facecolor=node_colors[idx],
                edgecolor=darken(node_colors[idx], 0.65),
                lw=1.0,
                zorder=3,
            )
            ax_flow.add_patch(arc)

            ticks = max(2, int(round(row_totals[idx] / total_flow * 12)))
            for t in range(ticks + 1):
                frac = t / ticks
                tick_angle = angles["start"] + frac * (angles["end"] - angles["start"])
                tick_start = polar(tick_angle, outer_radius)
                tick_end = polar(tick_angle, outer_radius + tick_length)
                ax_flow.plot(
                    [tick_start[0], tick_end[0]],
                    [tick_start[1], tick_end[1]],
                    color=darken(node_colors[idx], 0.7),
                    linewidth=0.6,
                    zorder=4,
                )

            mid_angle = (angles["start"] + angles["end"]) / 2
            label_radius = outer_radius + 0.15
            label_x, label_y = polar(mid_angle, label_radius)
            alignment = "left" if math.cos(mid_angle) >= 0 else "right"
            label = textwrap.fill(stage, width=18)
            ax_flow.text(
                label_x,
                label_y,
                label,
                ha=alignment,
                va="center",
                fontsize=label_size,
                fontname=font_family,
                color="#2f2f2f",
                zorder=5,
            )

        limit = outer_radius + 0.4
        ax_flow.set_xlim(-limit, limit)
        ax_flow.set_ylim(-limit, limit)
        title = str(diagram_cfg.get("title", "Bottleneck flow (wait hotspots & handoffs)"))
        ax_flow.set_title(
            title,
            fontsize=title_size,
            fontname=font_family,
            pad=30,
            color="#1b263b",
        )

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
