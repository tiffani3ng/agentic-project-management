# AI-Generated Report Template

Use this template to structure the final human-readable summary after an orchestrator run. It focuses on unstarted task routing, AI assist guardrails, and bottleneck visibility.

---

## Run Metadata
- **Run ID**: `<UUID from orchestrator>`
- **Timestamp**: `<generation time>`
- **Data snapshot**: `employees.csv | tasks.csv | availability.csv | events.csv`

## Per-Person Workload
Render as a table with the following columns:

| Employee (ID) | Assigned Hours | Capacity Hours | Load % | Note |
| --- | --- | --- | --- | --- |
| `Full name (E###)` | `12.0` | `40.0` | `30%` | `Within capacity / Over capacity` |

Include a one-line takeaway on overload or spare capacity.

## Unstarted Task Assignments
List only tasks with `status=not_started` that were assigned in this run. Render them as a fixed table (no bullet lists) so every row shows a single task and the rationale stays readable.

| Task (ID) | Skill Need | Est Hours | Assignee | Rationale |
| --- | --- | --- | --- | --- |
| `Task name [T####]` | `Skill / capability` | `12` | `Name (E###)` | `Why this person is a fit (respecting availability/skill)` |

## AI-Assist Flags (Human-in-the-loop)
Highlight tasks where AI drafting/summarization is recommended. Always state reviewer requirements.
- `<Task name> [T####] (owner: <current assignee>) | Reviewer required/optional | prompt: <suggested prompt> – reason: <why AI helps>`

## Bottleneck Map
Describe stage-level delays and handoff risk.
- `Stage <name>: ~<mean_hours>h avg, <handoffs> handoffs`
  - `Issue: <bottleneck issue> (<metric> <unit>); Recommendation: <process/AI assist tweak>`

## Additional Recommendations
- `<Workflow recommendation 1>`
- `<Workflow recommendation 2>`

> Guardrails: Keep AI usage to drafting/summarization; require reviewer sign-off, avoid PII/financial autonomy, and log prompts/responses for traceability.
