"""CLI entrypoint for the multi-agent workflow optimization MVP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from mvp.data_loader import load_employees, load_tasks
from mvp.orchestrator import Orchestrator
from mvp.llm_utils import safe_openai_json


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a padded Markdown table for CLI readability."""
    if not rows:
        return "No rows to display."

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(str(cell))) for width, cell in zip(widths, row)]

    def _format_row(values: Sequence[object]) -> str:
        padded = [str(value).ljust(width) for value, width in zip(values, widths)]
        return "| " + " | ".join(padded) + " |"

    header_line = _format_row(headers)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    row_lines = [_format_row(row) for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def _build_workload_table(workloads: List[Dict[str, object]], employees_df) -> str:
    """Summarize baseline/new/projected hours vs capacity by employee."""
    employees_by_id = employees_df.set_index("id").to_dict("index")
    rows: List[List[str]] = []
    for rollup in workloads:
        emp_id = str(rollup.get("employee_id", ""))
        employee = employees_by_id.get(emp_id, {})
        baseline = float(rollup.get("baseline_hours", 0.0))
        new_hours = float(rollup.get("newly_assigned_hours", 0.0))
        projected = float(rollup.get("projected_hours", baseline + new_hours))
        capacity = float(rollup.get("max_hours", employee.get("max_hours", 0.0)))
        load_pct = (projected / capacity * 100) if capacity else 0.0
        status = "Over capacity" if capacity and projected > capacity else "Within capacity"
        rows.append(
            [
                f"{employee.get('name', 'Unknown')} ({emp_id})",
                f"{baseline:.1f}",
                f"{new_hours:.1f}",
                f"{projected:.1f}",
                f"{capacity:.1f}",
                f"{load_pct:.1f}%",
                status,
            ]
        )

    rows.sort(key=lambda r: float(r[3]), reverse=True)
    headers = [
        "Employee",
        "Baseline (hrs)",
        "New (hrs)",
        "Projected (hrs)",
        "Capacity (hrs)",
        "Load",
        "Note",
    ]
    return _format_table(headers, rows) if rows else "No workload data available."


def _is_unstarted_status(status: object) -> bool:
    normalized = str(status or "").strip().lower()
    return normalized in {"", "not_started", "not started", "todo", "pending", "backlog"}


def _summarize_unstarted(assignments: List[Dict[str, object]], tasks_df, employees_df) -> str:
    """Render unstarted task assignments with rationales."""
    tasks_by_id = tasks_df.set_index("id").to_dict("index")
    employees_by_id = employees_df.set_index("id").to_dict("index")

    rows: List[List[str]] = []
    for assignment in assignments:
        task_id = assignment.get("task_id")
        task = tasks_by_id.get(task_id, {})
        if not _is_unstarted_status(task.get("status")):
            continue

        assignee = employees_by_id.get(str(assignment.get("assignee")), {})
        assignee_label = f"{assignee.get('name', 'Unassigned')} ({assignment.get('assignee')})"
        task_name = task.get("name", task_id)
        est_hours = task.get("est_hours", 0.0)
        rationale = assignment.get("rationale", "")
        rows.append(
            [
                f"{task_name} [{task_id}]",
                assignee_label,
                f"{float(est_hours):.1f}h",
                rationale,
            ]
        )

    headers = ["Task", "Assignee", "Est. effort", "Rationale"]
    return _format_table(headers, rows) if rows else "No unstarted tasks were routed in this run."


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
        reviewer_required = "required" if flag.get("reviewer_required", True) else "optional"
        owner = employees_by_id.get(task.get("assignee", ""), {})
        owner_label = owner.get("name", task.get("assignee", "Unassigned"))
        reviewer = flag.get("reviewer", "Department lead")
        prompt = flag.get("suggested_prompt", "").strip() or "N/A"
        reason = flag.get("reason", "")
        entry_lines = [
            f"- **{task_name} [{flag.get('task_id')}]**",
            f"  - Owner: {owner_label}",
            f"  - Reviewer: {reviewer} ({reviewer_required})",
            f"  - Prompt: {prompt}",
            f"  - Reason: {reason}",
        ]
        lines.append("\n".join(entry_lines))

    return "\n".join(lines) if lines else "No AI-assist recommendations flagged."


def _summarize_bottlenecks(bottlenecks: List[Dict[str, object]], stage_delays: List[Dict[str, object]]) -> str:
    """Highlight flagged bottlenecks first, then list throughput metrics for all stages."""
    flagged_lines: List[str] = []
    for entry in bottlenecks:
        stage = str(entry.get("stage"))
        issue = entry.get("issue", "Issue not specified")
        metric = str(entry.get("metric") or "").strip()
        unit = str(entry.get("unit") or "").strip()
        if metric and unit:
            metric_value = f"{metric} {unit}"
        elif metric:
            metric_value = metric
        elif unit:
            metric_value = unit
        else:
            metric_value = "N/A"
        recommendation = entry.get("recommendation", "Recommendation pending.")
        flagged_lines.append(f"- **{stage}**: {issue} ({metric_value}); Recommendation: {recommendation}")

    if not flagged_lines:
        flagged_lines = ["No bottlenecks flagged; continue monitoring throughput."]

    sorted_delays = sorted(stage_delays, key=lambda s: float(s.get("mean_service_hours", 0.0)), reverse=True)
    delay_rows: List[List[str]] = []
    for delay in sorted_delays:
        stage = str(delay.get("stage"))
        mean_service = float(delay.get("mean_service_hours", 0.0))
        mean_wait = float(delay.get("mean_wait_hours", 0.0))
        handoffs = int(delay.get("handoffs", 0))
        delay_rows.append(
            [
                stage,
                f"{mean_service:.1f}h",
                f"{mean_wait:.1f}h",
                str(handoffs),
            ]
        )

    delay_table = _format_table(
        ["Stage", "Mean service", "Mean wait", "Handoffs"],
        delay_rows,
    ) if delay_rows else "No stage-level delay metrics available."

    return "\n".join(
        [
            "### Flagged bottlenecks",
            "\n".join(flagged_lines),
            "",
            "### Stage service + wait by role",
            delay_table,
        ]
    )


def _build_summary_context(report: Dict[str, object], employees_df, tasks_df) -> Dict[str, object]:
    """Assemble highlights for the executive summary prompt."""
    employees_by_id = employees_df.set_index("id").to_dict("index")
    tasks_by_id = tasks_df.set_index("id").to_dict("index")

    workloads = report.get("workloads", [])
    total_projected = sum(float(w.get("projected_hours", 0.0)) for w in workloads)
    over_capacity = []
    for rollup in workloads:
        projected = float(rollup.get("projected_hours", 0.0))
        emp_id = str(rollup.get("employee_id"))
        employee = employees_by_id.get(emp_id, {})
        capacity = float(rollup.get("max_hours", employee.get("max_hours", 0.0)))
        if capacity and projected > capacity:
            over_capacity.append(
                {
                    "employee": employee.get("name", emp_id),
                    "projected_hours": round(projected, 1),
                    "capacity_hours": round(capacity, 1),
                }
            )

    unstarted_assignments = []
    for assignment in report.get("assignments", []):
        task = tasks_by_id.get(assignment.get("task_id"))
        if not task or not _is_unstarted_status(task.get("status")):
            continue
        employee = employees_by_id.get(str(assignment.get("assignee")), {})
        unstarted_assignments.append(
            {
                "task": task.get("name", assignment.get("task_id")),
                "assignee": employee.get("name", assignment.get("assignee")),
                "hours": float(task.get("est_hours", 0.0)),
                "rationale": assignment.get("rationale", ""),
            }
        )

    ai_flags = []
    for flag in report.get("ai_opportunities", []):
        if not flag.get("recommended", False):
            continue
        task = tasks_by_id.get(flag.get("task_id"), {})
        owner = employees_by_id.get(task.get("assignee", ""), {})
        ai_flags.append(
            {
                "task": task.get("name", flag.get("task_id")),
                "owner": owner.get("name", task.get("assignee", "Unassigned")),
                "reviewer": flag.get("reviewer", "Department lead"),
                "reason": flag.get("reason", ""),
            }
        )

    bottleneck_flags = []
    for entry in report.get("bottlenecks", []):
        bottleneck_flags.append(
            {
                "stage": entry.get("stage"),
                "issue": entry.get("issue"),
                "metric": entry.get("metric"),
                "unit": entry.get("unit"),
                "recommendation": entry.get("recommendation"),
            }
        )

    stage_delay_highlights = []
    for delay in report.get("stage_delays", []):
        stage_delay_highlights.append(
            {
                "stage": delay.get("stage"),
                "mean_service_hours": float(delay.get("mean_service_hours", 0.0)),
                "mean_wait_hours": float(delay.get("mean_wait_hours", 0.0)),
                "handoffs": int(delay.get("handoffs", 0)),
            }
        )

    recommendations = report.get("recommendations", [])

    return {
        "run_id": report.get("run_id"),
        "total_employees": len(workloads),
        "total_projected_hours": round(total_projected, 1),
        "over_capacity": over_capacity[:3],
        "unstarted_assignments": unstarted_assignments[:3],
        "ai_opportunities": ai_flags[:3],
        "bottleneck_flags": bottleneck_flags[:3],
        "stage_delay_highlights": stage_delay_highlights[:3],
        "workflow_recommendations": recommendations[:3],
    }


def _generate_executive_summary(report: Dict[str, object], employees_df, tasks_df) -> str:
    """Call OpenAI to produce a concise executive summary. Empty string if unavailable."""
    context = _build_summary_context(report, employees_df, tasks_df)
    system_prompt = (
        "You are an operations chief of staff summarizing a run from a multi-agent planning system."
        " Provide crisp highlights that balance workload, AI guardrails, and process risks."
    )
    user_prompt = (
        "Summarize the run context in 2-3 sentences for busy executives. Mention key risks or next steps."
        " Respond as JSON with a 'sentences' array of strings."
        f"\n\nRun context:\n```json\n{json.dumps(context, indent=2)}\n```"
    )
    fallback = {"sentences": []}
    result = safe_openai_json(system_prompt, user_prompt, fallback=fallback, temperature=0.35)
    sentences = [str(s).strip() for s in result.get("sentences", []) if str(s).strip()]
    if not sentences:
        return ""
    return " ".join(sentences[:3])


def _render_console_report(
    report: Dict[str, object],
    employees_df,
    tasks_df,
    executive_summary: str | None = None,
) -> str:
    """Compose the full report."""
    workload_table = _build_workload_table(report.get("workloads", []), employees_df)
    unstarted_block = _summarize_unstarted(report.get("assignments", []), tasks_df, employees_df)
    ai_block = _summarize_ai_flags(report.get("ai_opportunities", []), tasks_df, employees_df)
    bottlenecks_block = _summarize_bottlenecks(report.get("bottlenecks", []), report.get("stage_delays", []))

    sections = [
        "# Human-readable run summary",
        f"- **Run ID:** `{report.get('run_id', 'N/A')}`",
    ]
    if executive_summary:
        sections.extend(
            [
                "",
                "## AI-generated executive summary",
                executive_summary,
            ]
        )
    sections += [
        "",
        "## Per-person workload (assigned vs capacity)",
        workload_table,
        "",
        "## Unstarted task routing (with rationales)",
        unstarted_block,
        "",
        "## AI-assist flags (with reviewer guardrails)",
        ai_block,
        "",
        "## Bottleneck map (stage delays and handoffs)",
        bottlenecks_block,
    ]
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-agent workflow optimization CLI.")
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Limit LLM calls for AI opportunity scouting to the first 3 tasks and use fallback for the rest.",
    )
    args = parser.parse_args()

    data_dir = Path("data")
    reports_dir = Path("reports")
    orchestrator = Orchestrator(data_dir=data_dir, reports_dir=reports_dir, test_mode=args.test_mode)
    if args.test_mode:
        print("[INFO] Running AI opportunity scout in test mode (LLM limited to first 3 tasks).")
    report = orchestrator.run()
    bottleneck_image = report.get("bottleneck_image")
    if bottleneck_image:
        print(f"Bottleneck map image saved to {bottleneck_image}")

    employees_df = load_employees(data_dir / "employees.csv")
    tasks_df = load_tasks(data_dir / "tasks.csv")

    executive_summary = _generate_executive_summary(report, employees_df, tasks_df)
    console_report = _render_console_report(report, employees_df, tasks_df, executive_summary=executive_summary)
    summary_path = reports_dir / "human_readable_summary.md"
    summary_path.write_text(console_report)
    print(f"Human-readable summary saved to {summary_path}")
    # print("\nRaw JSON payload (also written to reports/):")
    # print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
