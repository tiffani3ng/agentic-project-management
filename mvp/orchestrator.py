"""LangGraph orchestrator chaining OpenAI agents."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from .ai_opportunity import AIOpportunityScout
from .bottleneck_detector import BottleneckDetector
from .data_loader import load_availability, load_employees, load_events, load_projects, load_tasks
from .resource_allocation import Assignment, ResourceAllocationAgent
from .run_store import RunStore
from .workflow_recommender import WorkflowRecommender


class WorkflowState(TypedDict, total=False):
    run_id: str
    employees: object
    availability: object
    projects: object
    tasks: object
    events: object
    assignments: List[Assignment]
    ai_opportunities: object
    bottlenecks: object
    stage_delays: object
    process_graph: object
    bottleneck_map: str
    recommendations: List[str]


class Orchestrator:
    def __init__(self, data_dir: Path, reports_dir: Path):
        self.data_dir = data_dir
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.run_store = RunStore(self.reports_dir / "agent_runs.db")

    def _load_data(self) -> Dict[str, object]:
        employees = load_employees(self.data_dir / "employees.csv")
        availability = load_availability(self.data_dir / "availability.csv")
        projects = load_projects(self.data_dir / "projects.csv")
        tasks = load_tasks(self.data_dir / "tasks.csv")
        events = load_events(self.data_dir / "events.csv")
        return {
            "employees": employees,
            "availability": availability,
            "projects": projects,
            "tasks": tasks,
            "events": events,
        }

    def _build_graph(self):
        graph = StateGraph(WorkflowState)

        def allocate(state: WorkflowState) -> WorkflowState:
            agent = ResourceAllocationAgent(
                state["employees"],
                state["availability"],
                run_store=self.run_store,
                run_id=state["run_id"],
            )
            assignments = agent.run(state["tasks"])
            return {"assignments": assignments}

        def scout(state: WorkflowState) -> WorkflowState:
            agent = AIOpportunityScout(
                state["tasks"],
                run_store=self.run_store,
                run_id=state["run_id"],
                employees=state.get("employees"),
            )
            suggestions = agent.run()
            return {"ai_opportunities": suggestions}

        def detect(state: WorkflowState) -> WorkflowState:
            agent = BottleneckDetector(
                state["events"],
                state["employees"],
                run_store=self.run_store,
                run_id=state["run_id"],
            )
            result = agent.run()
            return {
                "bottlenecks": result["bottlenecks"],
                "stage_delays": result["stage_delays"],
                "process_graph": result.get("process_graph", {}),
                "bottleneck_map": result.get("bottleneck_map", ""),
            }

        def recommend(state: WorkflowState) -> WorkflowState:
            agent = WorkflowRecommender(
                assignments=state.get("assignments", []),
                ai_suggestions=state.get("ai_opportunities", []),
                bottlenecks=state.get("bottlenecks", []),
                run_store=self.run_store,
                run_id=state["run_id"],
            )
            recs = agent.run()
            return {"recommendations": recs}

        graph.add_node("allocation", allocate)
        graph.add_node("ai_scout", scout)
        graph.add_node("bottlenecks", detect)
        graph.add_node("recommend", recommend)

        graph.set_entry_point("allocation")
        graph.add_edge("allocation", "ai_scout")
        graph.add_edge("ai_scout", "bottlenecks")
        graph.add_edge("bottlenecks", "recommend")
        graph.add_edge("recommend", END)

        return graph.compile()

    def run(self) -> Dict[str, object]:
        data = self._load_data()
        run_id = str(uuid.uuid4())

        app = self._build_graph()
        final_state = app.invoke(
            {
                "run_id": run_id,
                "employees": data["employees"],
                "availability": data["availability"],
                "projects": data["projects"],
                "tasks": data["tasks"],
                "events": data["events"],
            }
        )

        report = {
            "assignments": [asdict(a) for a in final_state.get("assignments", [])],
            "ai_opportunities": [asdict(s) for s in final_state.get("ai_opportunities", [])],
            "bottlenecks": [asdict(b) for b in final_state.get("bottlenecks", [])],
            "stage_delays": [asdict(d) for d in final_state.get("stage_delays", [])],
            "process_graph": final_state.get("process_graph", {}),
            "bottleneck_map": final_state.get("bottleneck_map", ""),
            "recommendations": final_state.get("recommendations", []),
            "run_id": run_id,
        }
        self._persist_reports(report)
        return report

    def _persist_reports(self, report: Dict[str, object]) -> None:
        assignment_path = self.reports_dir / "assignment_report.json"
        ai_path = self.reports_dir / "ai_opportunities.json"
        bottleneck_path = self.reports_dir / "bottleneck_report.json"
        recommendations_path = self.reports_dir / "workflow_recommendations.json"
        full_run_path = self.reports_dir / "full_run.json"

        assignment_path.write_text(json.dumps(report["assignments"], indent=2))
        ai_path.write_text(json.dumps(report["ai_opportunities"], indent=2))
        bottleneck_payload = {
            "bottlenecks": report["bottlenecks"],
            "stage_delays": report["stage_delays"],
            "process_graph": report.get("process_graph", {}),
            "bottleneck_map": report.get("bottleneck_map", ""),
        }
        bottleneck_path.write_text(json.dumps(bottleneck_payload, indent=2))
        recommendations_path.write_text(json.dumps(report["recommendations"], indent=2))
        full_run_path.write_text(json.dumps(report, indent=2))
