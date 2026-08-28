#!/usr/bin/env python3
"""
Recursively find topologies under files/paper_solutions/, submit SLURM jobs for
l3ss_tree build with coll_comm in [ag, ar], and optionally post-process outputs to CSV.
"""

import argparse
import csv
import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = "/home/green456/adorn"
PAPER_SOLUTIONS = os.path.join(REPO_ROOT, "files", "paper_solutions")
L3SS_TREE_BIN = os.path.join(REPO_ROOT, "l3ss_tree", "target", "release", "l3ss_tree")
SLURM_OUTPUTS = os.path.join(REPO_ROOT, "slurm", "outputs")
COLL_COMM_VALUES = ["ag", "ar"]

# SLURM args: size <= 4096 vs larger (each element is one #SBATCH line)
SLURM_LINES_SMALL = [
    "--account mithuna",
    "-p cpu",
    "--cpus-per-task=16",
    "-q standby",
    "-t 04:00:00",
]
SLURM_LINES_LARGE = [
    "--account mithuna",
    "-p cpu",
    "--cpus-per-task=32",
    "-t 1-00:00:00",
]
SIZE_THRESHOLD = 4096


def find_topologies(min_size: int | None, name_filter: str | None):
    """Yield (size, name, abspath) for each .map under files/paper_solutions/."""
    for root, _dirs, files in os.walk(PAPER_SOLUTIONS):
        for f in files:
            if not f.endswith(".map"):
                continue
            # Parent dir is like 128r or 8192r
            parent = os.path.basename(root)
            if not re.match(r"^\d+r$", parent):
                continue
            size = int(parent.rstrip("r"))
            if min_size is not None and size < min_size:
                continue
            name = os.path.splitext(f)[0]
            if name_filter is not None and not fnmatch.fnmatch(name, name_filter):
                continue
            abspath = os.path.join(root, f)
            yield size, name, abspath


def slurm_lines_for_size(size: int) -> list[str]:
    if size <= SIZE_THRESHOLD:
        return SLURM_LINES_SMALL
    return SLURM_LINES_LARGE


def submit_jobs(name_filter: str | None, min_size: int | None, dry_run: bool) -> None:
    os.makedirs(SLURM_OUTPUTS, exist_ok=True)
    for size, name, abspath in find_topologies(min_size, name_filter):
        for coll_comm in COLL_COMM_VALUES:
            outerr_name = f"coll_comm_{name}_{coll_comm}"
            xml_path = os.path.join(f"/home/green456/adorn/xml_{coll_comm}", f"{name}_{coll_comm}.xml")
            outerr_path = os.path.join("slurm", "outputs", f"{outerr_name}.outerr")
            cmd = f"{L3SS_TREE_BIN} --xml {xml_path} build {abspath} {coll_comm} rr 1"
            slurm_lines = slurm_lines_for_size(size)
            script_lines = [
                "#!/bin/bash",
                *["#SBATCH " + line for line in slurm_lines],
                f"#SBATCH -J {outerr_name}",
                f"#SBATCH -o {outerr_path}",
                f"#SBATCH -e {outerr_path}",
                "",
                f"cd {REPO_ROOT}",
                "mkdir -p slurm/outputs",
                cmd,
            ]
            script = "\n".join(script_lines)
            if dry_run:
                print(f"[dry-run] would submit: {name} {coll_comm}")
                print(script)
                print("---")
                continue
            tmp = f"/tmp/coll_comm_sweep_{os.getpid()}_{name}_{coll_comm}.sh"
            with open(tmp, "w") as f:
                f.write(script)
            try:
                subprocess.run(["sbatch", tmp], cwd=REPO_ROOT, check=True)
                print(f"Submitted {outerr_name}")
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass


def parse_output_file(path: str) -> dict | None:
    """Parse l3ss_tree output; return dict with utilization, num_trees, num_steps, time_schedule or None."""
    try:
        with open(path, "r") as f:
            text = f.read()
    except OSError:
        return None

    out = {}
    # Utilization 96.2121%
    m = re.search(r"Utilization\s+([\d.]+)\s*%", text)
    out["utilization"] = float(m.group(1)) if m else None
    # 1 * 128 trees. 22 steps.
    m = re.search(r"\d+\s*\*\s*(\d+)\s+trees\.\s*(\d+)\s+steps", text)
    out["num_trees"] = int(m.group(1)) if m else None
    out["num_steps"] = int(m.group(2)) if m else None
    # Scheduling took 0.129125271s
    m = re.search(r"Scheduling took ([\d.]+)s", text)
    out["time_schedule"] = float(m.group(1)) if m else None
    return out


def post_process(
    name_filter: str | None,
    min_size: int | None,
    csv_path: str,
) -> None:
    """Discover topologies, find coll_comm output files, parse, write CSV."""
    rows = []
    for size, name, _ in find_topologies(min_size, name_filter):
        for coll_comm in COLL_COMM_VALUES:
            outerr_name = f"coll_comm_{name}_{coll_comm}"
            outerr_path = os.path.join(SLURM_OUTPUTS, f"{outerr_name}.outerr")
            if not os.path.isfile(outerr_path):
                continue
            parsed = parse_output_file(outerr_path)
            if not parsed:
                continue
            rows.append({
                "size": size,
                "name": name,
                "coll_comm": coll_comm,
                "utilization": parsed.get("utilization"),
                "num_trees": parsed.get("num_trees"),
                "num_steps": parsed.get("num_steps"),
                "time_schedule": parsed.get("time_schedule"),
            })

    if not rows:
        print("No output files found to parse.", file=sys.stderr)
        return

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = ["size", "name", "coll_comm", "utilization", "num_trees", "num_steps", "time_schedule"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {csv_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Run l3ss_tree build (ag/ar) on paper_solutions topologies via SLURM, or post-process outputs to CSV."
    )
    ap.add_argument(
        "--filter",
        type=str,
        default=None,
        metavar="PATTERN",
        help="Only topologies whose name matches this fnmatch pattern (e.g. 'pdtt_*', '*_128r_*')",
    )
    ap.add_argument(
        "--min-size",
        type=int,
        default=None,
        metavar="N",
        help="Minimum topology size (inclusive); only run/post-process sizes >= N",
    )
    ap.add_argument(
        "--post-process",
        action="store_true",
        help="Do not submit jobs; parse existing slurm/outputs/coll_comm_*.outerr and write CSV",
    )
    ap.add_argument(
        "--csv",
        type=str,
        default=os.path.join(REPO_ROOT, "slurm", "coll_comm_sweep_results.csv"),
        metavar="PATH",
        help="Output CSV path for --post-process (default: slurm/coll_comm_sweep_results.csv)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print job scripts only, do not submit (ignored when --post-process)",
    )
    args = ap.parse_args()

    if args.post_process:
        post_process(args.filter, args.min_size, args.csv)
    else:
        submit_jobs(args.filter, args.min_size, args.dry_run)


if __name__ == "__main__":
    main()
