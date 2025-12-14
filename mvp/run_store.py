"""Simple SQLite run store for agent traceability."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


class RunStore:
    """Persists agent inputs/outputs to SQLite for auditing."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()

    def log(self, run_id: str, agent_name: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO agent_runs (run_id, agent_name, input_json, output_json) VALUES (?, ?, ?, ?)",
                (run_id, agent_name, json.dumps(inputs), json.dumps(outputs)),
            )
            conn.commit()

    def latest_for_run(self, run_id: str, agent_name: Optional[str] = None) -> list[Dict[str, Any]]:
        query = "SELECT run_id, agent_name, input_json, output_json, created_at FROM agent_runs WHERE run_id = ?"
        params: tuple[Any, ...] = (run_id,)
        if agent_name:
            query += " AND agent_name = ?"
            params = (run_id, agent_name)
        query += " ORDER BY created_at DESC"

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            results.append(
                {
                    "run_id": r[0],
                    "agent_name": r[1],
                    "input": json.loads(r[2]),
                    "output": json.loads(r[3]),
                    "created_at": r[4],
                }
            )
        return results
