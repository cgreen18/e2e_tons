#!/usr/bin/env python3
"""Normalize the synthetic communicator-aware trace runs into one table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

BACKENDS = {"Unaware": "congestion_unaware", "Aware": "congestion_aware"}


def collect(runs_root: Path) -> list[dict]:
    rows: list[dict] = []
    for directory in sorted(runs_root.iterdir()):
        if not directory.is_dir():
            continue
        stats_path = directory / "statistics.json"
        if not stats_path.is_file():
            rows.append({"run_id": directory.name, "status": "missing-statistics"})
            continue
        stats = json.loads(stats_path.read_text())
        topology, mode, backend = directory.name.split(".")
        ranks = stats["ranks"]
        complete = sum(1 for rank in ranks if rank["complete"])
        per_collective: dict[str, dict[str, float]] = {}
        for rank in ranks:
            for entry in rank.get("collectives") or []:
                bucket = per_collective.setdefault(
                    entry["comm_type"], {"count": 0, "time_ns": 0, "ranks": 0}
                )
                bucket["count"] += entry.get("count", 0)
                bucket["time_ns"] += entry.get("time_ns", 0)
                bucket["ranks"] += 1
        row = {
            "run_id": directory.name,
            "topology": topology,
            "schedule_mode": mode,
            "backend": BACKENDS.get(backend, backend),
            "status": "ok" if stats.get("complete") else "INCOMPLETE",
            "ranks_complete": complete,
            "ranks_total": len(ranks),
            "simulation_end_ns": stats.get("simulation_end_ns"),
        }
        for name, bucket in sorted(per_collective.items()):
            row[f"{name.lower()}_ops"] = bucket["count"]
            row[f"{name.lower()}_ranks"] = bucket["ranks"]
            # Mean per-rank time for the ranks that ran this collective.
            row[f"{name.lower()}_mean_ns"] = round(
                bucket["time_ns"] / bucket["ranks"], 1
            )
            row[f"{name.lower()}_max_ns"] = None
        for name in per_collective:
            row[f"{name.lower()}_max_ns"] = max(
                entry["time_ns"]
                for rank in ranks
                for entry in (rank.get("collectives") or [])
                if entry["comm_type"] == name
            )
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path,
                        default=Path("generated/synthetic_128/runs"))
    parser.add_argument("--output-csv", type=Path,
                        default=Path("generated/synthetic_128/summary.csv"))
    parser.add_argument("--output-json", type=Path,
                        default=Path("generated/synthetic_128/summary.json"))
    args = parser.parse_args(argv)

    rows = collect(args.runs)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.write_text(json.dumps(rows, indent=2) + "\n")

    incomplete = [row for row in rows if row.get("status") != "ok"]
    for row in rows:
        print(
            f"{row['run_id']:28} {row.get('status'):11} "
            f"ranks={row.get('ranks_complete')}/{row.get('ranks_total')} "
            f"end_ns={row.get('simulation_end_ns')}"
        )
    if incomplete:
        print(f"\n{len(incomplete)} run(s) did not complete", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
