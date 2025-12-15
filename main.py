"""CLI entrypoint for the multi-agent workflow optimization MVP."""
from __future__ import annotations

import json
from pathlib import Path

from mvp.orchestrator import Orchestrator


def main() -> None:
    data_dir = Path("data")
    reports_dir = Path("reports")
    orchestrator = Orchestrator(data_dir=data_dir, reports_dir=reports_dir)
    report = orchestrator.run()
    bottleneck_map = report.get("bottleneck_map")
    if bottleneck_map:
        print(bottleneck_map)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
