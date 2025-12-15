# Agent Guidelines for Project Management MVP

## Scope
This file applies to the entire repository. All contributors and coding agents must follow these directions when modifying any files.

## Product Vision
We are building an MVP for a multi-agent project management tool that optimizes workflows for cross-functional teams. The system should:
- Produce a polished CLI report summarizing planned employee assignments for unstarted tasks.
- Produce a polished CLI report that highlights bottlenecks and recommends process changes, and flags tasks with "AI assist" recommendations plus suggested prompts/templates and reviewer requirements.
- Emphasize human-in-the-loop AI usage: safe for drafts (emails, notes, slide outlines, summaries), prohibited for high-risk autonomy (payments, access control, PII exposure). Require reviewer sign-off and logging for AI outputs.

## Core Agents and Responsibilities
- **Resource allocation agent:** Match tasks to people based on skill coverage, availability, and workload balance; respect constraints like max hours/week, timezone overlap, and vacation. Use a heuristic such as `score = skill_match + availability_fit + workload_balance – overtime_risk` (greedy or Hungarian per milestone window is acceptable).
- **AI opportunity scout:** Identify low-risk, high-leverage tasks for AI assist. Rank reviewable outputs (emails, briefs, presentations, research summaries). Provide guardrails: reviewer required, redact PII, maintain prompt templates per department, limit scope. Exclude high-risk autonomy (financial transactions, access control).
- **Bottleneck detector:** Use event logs to compute wait time (handoff start – prior end), service time (duration), and build a process graph (roles/functions as nodes, handoffs as edges). Flag edges with long wait times (e.g., 90th percentile), nodes with high utilization, and critical-path tasks. Recommend capacity adds, parallelization, AI assist for drafting/review, clearer entry criteria, or SLA tweaks.
- **Workflow recommender:** Suggest process tweaks such as parallelization, WIP limits, resequencing, redistributing load, and adding AI copilots to repetitive steps.
- **Orchestration:** Chain agents (e.g., LangGraph-like routing) and log runs/decisions for traceability.

## Current MVP Scope (CLI only)
- Data sources: CSVs for employees (skills, capacity, timezone), projects, tasks, availability calendars, and past event logs.
- Assignment engine: Run heuristic assignment and display per-person workload; prioritize unstarted tasks and maximize skill coverage while minimizing overload.
- AI suggestion pass: Tag tasks with "AI assist recommended" plus suggested prompts and a required reviewer.
- Bottleneck report: Compute wait/processing times from events and generate a bottleneck map description (textual for now) with top bottlenecks and recommendations.
- No live updates yet: outputs are static one-time reports; calendar sync is simulated via availability CSV.

## Future-facing Notes (do not implement yet unless explicitly asked)
- Calendar API sync (Google/Microsoft) for live availability.
- Dynamic task completion updates via REST endpoints with continuous re-evaluation.
- Project planner agent that decomposes project descriptions into tasks.
- Web dashboard (Next.js/React/Streamlit), Slack/Teams bot, and JSON APIs for automation.

## Data Examples (for reference only)
Example CSV columns and sample rows are provided below to keep the data model aligned. Do not hardcode these values; use them as schema guidance.

### Employees
```
id,name,department,role,skills,max_hours,timezone
E001,Liam Nguyen,Brand Marketing,Brand Marketing Manager,"[Brief-to-launch execution, Creative briefing, ...]",43,America/Denver
...
```

### Availability
```
employee_id,date,hours_free
E001,2025-12-15,4
...
```

### Projects
```
id,name,description,deadline,priority
P001,Returns Experience Overhaul (Self-Service + Portal),...,2025-08-17,5
...
```

### Tasks
```
id,project_id,name,skill_needed,est_hours,assignee,start,due,status
T0001,P001,...,3PL management,19,E015,2025-07-06T18:00:00,2025-07-10T12:30:00,completed
...
```

### Events
```
task_id,type,timestamp,from_assignee,to_assignee
T0001,start,2025-07-06T18:00:00,,E015
...
```

## Coding Conventions
- Prefer clear, readable Python; avoid unnecessary abstraction until needed.
- Keep business logic encapsulated and make agent decisions traceable (log rationale where possible).
- Avoid try/except around imports.
- When adding tests or scripts, favor deterministic behavior (no hidden randomness) for reproducibility.

## Reporting Expectations
- Bottleneck map may be textual for now; describe nodes/edges with delays and critical paths.
- For AI assist tasks, include suggested prompts/templates and name a required reviewer.
- Always call out assumptions and constraints (e.g., simulated availability, static run).

## Documentation for Contributors
- If you add new folders, include nested `AGENTS.md` if special rules apply.
- Keep instructions concise and actionable; update this file when the MVP scope changes.

