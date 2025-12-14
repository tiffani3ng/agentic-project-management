# Multi-agent workflow optimization MVP

This repository implements a lightweight, rule-based multi-agent system for project management and workflow optimization. It ingests CSV data about employees, availability, projects, tasks, and event logs, then produces a one-time run report that includes:

- **Resource allocation**: Assigns unstarted tasks to employees based on skill match, availability, and workload balance.
- **AI opportunity scouting**: Flags tasks that are safe for AI co-pilots (human-in-the-loop) and provides prompts.
- **Bottleneck detection**: Analyzes event logs to highlight slow stages and repeated handoffs.
- **Workflow recommendations**: Summarizes process tweaks to address bottlenecks and safely introduce AI support.

## Data
Sample CSVs live in `data/` and mirror the user-provided datasets for employees, availability, projects, tasks, and past events.

## Running the MVP

```bash
python main.py
```

The script writes JSON reports to `reports/`:

- `assignment_report.json`: Proposed task-to-person matches with scoring rationale.
- `ai_opportunities.json`: AI assist recommendations and reviewer requirements.
- `bottleneck_report.json`: Stage delays and detected bottlenecks from the event logs.
- `workflow_recommendations.json`: Actionable process change suggestions.
- `full_run.json`: Combined payload for traceability.

The current scope is a single static run; future work can add calendar sync, live task updates, and richer orchestration.
