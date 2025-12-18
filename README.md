# Agentic Project Management for Cross-Functional Teams

**Turning routine project data into workflow intelligence**

> *An MVP agentic system that assigns work, detects hidden bottlenecks, and recommends safe AI augmentation using only tasks, availability, and event logs.*

**Author:** Tiffanie Ng
**Course:** IPHS 391 – Frontiers of AI (Fall 2025), Kenyon College
**Poster:** *An Agentic Project-Management Tool for Cross-Functional Teams* 

---

## 🚩 Motivation

Project management tools excel at tracking **what** needs to be done.
They are far worse at revealing **where work actually slows down**.

In cross-functional teams, delays often emerge from:

* overloaded specialists,
* approval ping-pong,
* vendor dependencies,
* interruptions and context switching,
* late-stage compliance gates,
* and misallocation of work to the wrong people.

These issues accumulate quietly in **handoffs and waiting time**, not execution itself.

This project explores a simple but powerful idea:

> **Once tasks, assignments, and timestamps exist, workflow bottlenecks become measurable—and optimizable.**

---

## 🧩 What This MVP Does

This repository contains a **minimal agentic project-management system** that consumes standard project artifacts and produces **manager-facing workflow insight**.

### Inputs

* Employee skills, capacity, and time zones
* Project and task definitions
* Availability calendars
* Event logs (task start / end / handoff)

### Outputs

* Capacity-aware task assignment plans
* Automated bottleneck detection from event logs
* Human-in-the-loop AI assist recommendations
* Consolidated, readable management reports

All outputs are generated via a **CLI-driven pipeline** and saved as structured JSON + Markdown reports.

---

## 🏗 System Architecture (MVP)

The system is orchestrated as a small **agent pipeline**:

1. **Data Loader**
   Loads employees, tasks, availability, and events from CSV/JSON.

2. **Resource Allocation Agent**
   Assigns *unstarted* tasks using heuristic scoring:

   * skill match
   * capacity risk
   * availability fit

3. **AI Opportunity Scout**
   Flags low-risk, reviewable tasks (drafting, summarization, planning) for AI assistance, enforcing human-in-the-loop constraints.

4. **Bottleneck Detector**
   Converts event logs into workflow metrics:

   * wait time after handoffs
   * handoff counts
   * late completions
   * approval loops
   * role-level congestion

5. **Workflow Recommender**
   Maps detected bottlenecks to actionable fixes:

   * WIP limits
   * reviewer consolidation
   * intake triage
   * on-call rotations
   * AI copilots for repetitive steps

6. **Report Emitter**
   Produces structured JSON artifacts and a consolidated, human-readable Markdown summary.

---

## 📊 Synthetic Dataset (for Evaluation)

The app itself is **domain-agnostic**, but the MVP is evaluated using a **synthetic e-commerce / retail organization** to enable controlled testing.

### Why synthetic data?

* Real workflow logs are inaccessible due to privacy and policy constraints.
* Synthetic data allows **explicit injection of known bottlenecks**.
* Enables benchmarking: *did the system recover the bottlenecks we encoded?*

### Data tables

Located in `/data`:

* `employees.csv`
* `availability.csv`
* `projects.csv`
* `tasks.csv`
* `events.csv`
* `synthetic_retail_dataset.json`

The generator emphasizes:

* realistic availability patterns,
* clustered work bursts,
* idle gaps and weekends,
* multi-step cross-role handoffs.

---

## ⛔ Bottlenecks Modeled

Six common workflow bottlenecks are intentionally embedded in the event logs:

1. **Analytics backlog / single-threaded specialists**
2. **Creative review ping-pong**
3. **Engineering interruptions & late completions**
4. **Ops delays due to vendor dependencies**
5. **Late-stage finance / legal gates**
6. **Misallocation from poor skill fit**

Each bottleneck has a **distinct event-level signature** (e.g., queue delays, repeated handoffs, late “end” events), enabling systematic detection.

---

## 📁 Repository Structure

```text
agentic-project-management/
├── data/                     # Synthetic datasets
├── mvp/                      # Core agent logic
│   ├── orchestrator.py
│   ├── resource_allocation.py
│   ├── ai_opportunity.py
│   ├── bottleneck_detector.py
│   ├── workflow_recommender.py
│   └── data_loader.py
├── reports/                  # Generated outputs
│   ├── human_readable_summary.md
│   ├── workload_report.json
│   ├── bottleneck_report.json
│   ├── assignment_report.json
│   ├── ai_opportunities.json
│   └── bottleneck_map.png
├── synthetic_data_generator.py
├── main.py
└── README.md
```

---

## 📄 Key Output

👉 **Start here:**
[`reports/human_readable_summary.md`](reports/human_readable_summary.md)

This consolidated report includes:

* workload & overload risk by employee,
* detected bottlenecks with explanations,
* AI-assist recommendations,
* process improvement suggestions.

---

## 🧪 Evaluation (MVP Scope)

This MVP is evaluated on **detectability and plausibility**, not production performance:

* **Bottleneck detectability:**
  Can the system surface the six injected bottlenecks from event signatures alone?

* **Assignment plausibility:**
  Do assignments respect capacity and reduce skill mismatch?

* **AI policy compliance:**
  Are AI recommendations limited to human-reviewable outputs?

The system successfully recovers all six bottleneck patterns and produces interpretable narratives linking delays to workflow structure.

---

## 🔮 Future Work

Planned extensions include:

* Real-time operation with continuous re-evaluation
* Calendar API integration (Google / Microsoft)
* Web dashboard (Streamlit or Next.js)
* Slack / Teams bot for updates and AI drafts
* Learning-based task assignment (beyond heuristics)
* Role-based access control and audit logs

---

## ⚖ Ethics & Responsible AI

* **No real employee data**
  All evaluation uses synthetic data designed for privacy and repeatability.

* **Human-in-the-loop by design**
  AI is restricted to drafting and summarization—no autonomous execution of high-stakes actions.

* **Auditability**
  Agent inputs, decisions, and recommendations are logged to support inspection, override, and iteration.

---

## 📚 References

Selected literature grounding this work in operations and workflow research:

* Abad et al., *EASE ’18* (task interruption)
* Karimi et al., *Systems* (bottleneck management)
* Bull, *Zenodo* (Theory of Constraints in workforce systems)

(Full citations included in the poster.)
