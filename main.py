"""CLI entrypoint for the multi-agent workflow optimization MVP."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from mvp.data_loader import load_employees, load_tasks
from mvp.orchestrator import Orchestrator


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a simple fixed-width table for console output."""
    if not rows:
        return "No rows to display."

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(str(cell))) for width, cell in zip(widths, row)]

    header_line = " | ".join(header.ljust(width) for header, width in zip(headers, widths))
    separator = "-+-".join("-" * width for width in widths)
    row_lines = [" | ".join(str(cell).ljust(width) for cell, width in zip(row, widths)) for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def _build_workload_table(assignments: List[Dict[str, object]], employees_df, tasks_df) -> str:
    """Summarize workload and capacity by employee."""
    tasks_by_id = tasks_df.set_index("id").to_dict("index")
    employees_by_id = employees_df.set_index("id").to_dict("index")

    hours_by_employee: Dict[str, float] = {emp_id: 0.0 for emp_id in employees_by_id.keys()}
    for assignment in assignments:
        task_id = assignment.get("task_id")
        est_hours = 0.0
        if task_id in tasks_by_id:
            est_hours = float(tasks_by_id[task_id].get("est_hours", 0.0))
        assignee = str(assignment.get("assignee"))
        hours_by_employee[assignee] = hours_by_employee.get(assignee, 0.0) + est_hours

    rows: List[List[str]] = []
    for emp_id, info in employees_by_id.items():
        assigned = hours_by_employee.get(emp_id, 0.0)
        capacity = float(info.get("max_hours", 0.0))
        load_pct = (assigned / capacity * 100) if capacity else 0.0
        status = "Over capacity" if assigned > capacity and capacity > 0 else "Within capacity"
        rows.append(
            [
                f"{info.get('name', '')} ({emp_id})",
                f"{assigned:.1f}",
                f"{capacity:.1f}",
                f"{load_pct:>5.1f}%",
                status,
            ]
        )

    rows.sort(key=lambda r: float(r[1]), reverse=True)
    headers = ["Employee", "Assigned (hrs)", "Capacity (hrs)", "Load", "Note"]
    return _format_table(headers, rows)


def _summarize_unstarted(assignments: List[Dict[str, object]], tasks_df, employees_df) -> str:
    """Render unstarted task assignments with rationales."""
    tasks_by_id = tasks_df.set_index("id").to_dict("index")
    employees_by_id = employees_df.set_index("id").to_dict("index")

    lines: List[str] = []
    for assignment in assignments:
        task_id = assignment.get("task_id")
        task = tasks_by_id.get(task_id, {})
        if str(task.get("status", "")).lower() != "not_started":
            continue

        assignee = employees_by_id.get(str(assignment.get("assignee")), {})
        assignee_label = f"{assignee.get('name', 'Unassigned')} ({assignment.get('assignee')})"
        task_name = task.get("name", task_id)
        est_hours = task.get("est_hours", 0.0)
        rationale = assignment.get("rationale", "")
        lines.append(
            f"- {task_name} [{task_id}] → {assignee_label} | est {est_hours}h | rationale: {rationale}"
        )

    return "\n".join(lines) if lines else "No unstarted tasks were routed in this run."


def _summarize_ai_flags(ai_flags: List[Dict[str, object]], tasks_df, employees_df) -> str:
    """Summarize AI-assist recommendations and reviewer needs."""
    tasks_by_id = tasks_df.set_index("id").to_dict("index")
    employees_by_id = employees_df.set_index("id").to_dict("index")

    lines: List[str] = []
    for flag in ai_flags:
        if not flag.get("recommended", False):
            continue
        task = tasks_by_id.get(flag.get("task_id"), {})
        task_name = task.get("name", flag.get("task_id"))
        reviewer_required = "Reviewer required" if flag.get("reviewer_required", True) else "Reviewer optional"
        owner = employees_by_id.get(task.get("assignee", ""), {})
        owner_label = owner.get("name", task.get("assignee", "Unassigned"))
        lines.append(
            "- "
            + f"{task_name} [{flag.get('task_id')}] (owner: {owner_label}) | "
            + f"{reviewer_required}; prompt: {flag.get('suggested_prompt', '').strip() or 'N/A'} "
            + f"– reason: {flag.get('reason', '')}"
        )

    return "\n".join(lines) if lines else "No AI-assist recommendations flagged."


def _summarize_bottlenecks(bottlenecks: List[Dict[str, object]], stage_delays: List[Dict[str, object]]) -> str:
    """Describe stage-level bottlenecks and handoff pressure."""
    bottlenecks_by_stage: Dict[str, List[Dict[str, object]]] = {}
    for b in bottlenecks:
        bottlenecks_by_stage.setdefault(str(b.get("stage")), []).append(b)

    lines: List[str] = []
    sorted_delays = sorted(stage_delays, key=lambda s: float(s.get("mean_hours", 0.0)), reverse=True)
    for delay in sorted_delays:
        stage = str(delay.get("stage"))
        mean_hours = float(delay.get("mean_hours", 0.0))
        handoffs = int(delay.get("handoffs", 0))
        stage_lines = [f"Stage {stage}: ~{mean_hours:.1f}h avg, {handoffs} handoffs"]
        for b in bottlenecks_by_stage.get(stage, []):
            stage_lines.append(
                f"  • Issue: {b.get('issue')} ({b.get('metric')} {b.get('unit')}); Recommendation: {b.get('recommendation')}"
            )
        if len(stage_lines) == 1:
            stage_lines.append("  • No bottleneck flagged; monitor handoffs for delays.")
        lines.append("\n".join(stage_lines))

    return "\n".join(lines) if lines else "No bottleneck metrics available."


def _render_console_report(report: Dict[str, object], employees_df, tasks_df) -> str:
    """Compose the full CLI-friendly report."""
    workload_table = _build_workload_table(report.get("assignments", []), employees_df, tasks_df)
    unstarted_block = _summarize_unstarted(report.get("assignments", []), tasks_df, employees_df)
    ai_block = _summarize_ai_flags(report.get("ai_opportunities", []), tasks_df, employees_df)
    bottlenecks_block = _summarize_bottlenecks(report.get("bottlenecks", []), report.get("stage_delays", []))

    sections = [
        "=== Human-readable run summary ===",
        f"Run ID: {report.get('run_id', 'N/A')}",
        "",
        "-- Per-person workload (assigned vs capacity) --",
        workload_table,
        "",
        "-- Unstarted task routing (with rationales) --",
        unstarted_block,
        "",
        "-- AI-assist flags (with reviewer guardrails) --",
        ai_block,
        "",
        "-- Bottleneck map (stage delays and handoffs) --",
        bottlenecks_block,
    ]
    return "\n".join(sections)


def main() -> None:
    data_dir = Path("data")
    reports_dir = Path("reports")
    orchestrator = Orchestrator(data_dir=data_dir, reports_dir=reports_dir)
    report = orchestrator.run()

    employees_df = load_employees(data_dir / "employees.csv")
    tasks_df = load_tasks(data_dir / "tasks.csv")

    console_report = _render_console_report(report, employees_df, tasks_df)
    print(console_report)
    print("\nRaw JSON payload (also written to reports/):")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
