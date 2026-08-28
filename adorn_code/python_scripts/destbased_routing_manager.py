#!/usr/bin/env python3
"""
Destination-based routing manager: CSV state + next_action for Launch/Check rounds.

CLI:
  init | list-unfinished | mark-running | reconcile | reconcile-all
  mark-verified | summary
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "files" / "paper_solutions" / "destbased_routing_status.csv"
PAPER_SOLUTIONS = REPO_ROOT / "files" / "paper_solutions"
ROUTEPATH_DIR = REPO_ROOT / "topologies_and_routing" / "routepath_lists"
NRL_DIR = REPO_ROOT / "topologies_and_routing" / "nr_lists"
VC_DIR = REPO_ROOT / "topologies_and_routing" / "vc_mats"
LOG_DIR = REPO_ROOT / "slurm" / "outputs" / "destbased_manager"

MAX_CPUS = 128
DEFAULT_CPUS = 8
DEFAULT_CPU_BUDGET = 480
DEFAULT_C = 4

CSV_COLUMNS = [
    "topology",
    "routing paths file",
    "routing nrl file",
    "vc file",
    "routed throughput",
    "status",
    "family",
    "n_routers",
    "map_path",
    "phase_status",
    "cpus_last_tried",
    "cpus_next",
    "symmetry_tried",
    "symmetry_params",
    "script_last",
    "attempt_outcome",
    "next_action",
    "cpl_path",
    "log_path",
    "agent_note",
    "last_updated",
]

TERMINAL_PHASE = {"succeeded", "failed_terminal"}
NON_LAUNCH_ACTIONS = {"done", "none", "failed_terminal", "verify"}
DIM_RE = re.compile(r"(\d+)x(\d+)x(\d+)")
ROUTERS_RE = re.compile(r"_(\d+)r_")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def parse_family(topo: str) -> Optional[str]:
    if topo.startswith("pdtt_"):
        return "pdtt"
    if topo.startswith("ptt_") or topo.startswith("pttt_"):
        return None
    if topo.startswith("pt_"):
        return "pt"
    if topo.startswith("asc_"):
        return "asc"
    return None


def parse_n_routers(topo: str, map_path: Path) -> int:
    m = ROUTERS_RE.search(topo)
    if m:
        return int(m.group(1))
    parent = map_path.parent.name
    if parent.endswith("r") and parent[:-1].isdigit():
        return int(parent[:-1])
    raise ValueError(f"Cannot parse n_routers for {topo}")


def parse_dims(topo: str) -> Tuple[Tuple[int, int, int], Optional[Tuple[int, int, int]]]:
    dims = [(int(a), int(b), int(c)) for a, b, c in DIM_RE.findall(topo)]
    if not dims:
        raise ValueError(f"No AxBxC dims in topology name: {topo}")
    primary = dims[0]
    secondary = dims[1] if len(dims) > 1 else None
    return primary, secondary


def pdtt_mc_dims(xyz: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """PDTT mega-cube: max(X/2, 4) x max(Y/2, 4) x 4 (min MC dim is 4 on each axis)."""
    x, y, _z = xyz
    return (max(x // 2, 4), max(y // 2, 4), 4)


def asc_sym_params(topo: str) -> str:
    (x, y, z), secondary = parse_dims(topo)
    mx, my, mz = secondary if secondary else (4, 4, 4)
    return f"{x} {y} {z} {DEFAULT_C} {mx} {my} {mz} trans"


def pt_sym_params(topo: str) -> str:
    (x, y, z), _ = parse_dims(topo)
    return f"{x} {y} {z} {DEFAULT_C} 4 4 4 trans"


def pdtt_sym_params(topo: str) -> str:
    (x, y, z), _ = parse_dims(topo)
    mx, my, mz = pdtt_mc_dims((x, y, z))
    return f"{x} {y} {z} {DEFAULT_C} {mx} {my} {mz} refl-trans"


def find_best_cpl(topo: str) -> Optional[str]:
    """Best existing non-failocs DOR .paths for PDTT, or None if missing."""
    if not ROUTEPATH_DIR.is_dir():
        return None
    candidates = [
        p
        for p in ROUTEPATH_DIR.glob(f"{topo}_dor*.paths")
        if "failocs" not in p.name and p.is_file() and p.stat().st_size > 0
    ]
    if not candidates:
        return None

    (x, y, z), _ = parse_dims(topo)
    mx, my, mz = pdtt_mc_dims((x, y, z))
    prefer_mc = f"{mx}x{my}x{mz}mc"

    def score(p: Path) -> Tuple[int, int]:
        name = p.name
        s = 0
        if "new_mclb" in name:
            s += 200
        elif "_mclb_" in name or name.endswith("_mclb.paths"):
            s += 100
        if "refl-trans" in name:
            s += 50
        if prefer_mc in name:
            s += 80
        if "sym_" in name:
            s += 20
        if "1048576ps" in name:
            s -= 10
        return (s, p.stat().st_size)

    best = max(candidates, key=score)
    return str(best.relative_to(REPO_ROOT))


def expected_default_cpl(topo: str) -> str:
    (x, y, z), _ = parse_dims(topo)
    mx, my, mz = pdtt_mc_dims((x, y, z))
    st = "refl-trans"
    name = (
        f"{topo}_dor_{st}sym_{mx}x{my}x{mz}mc_new_mclb_{st}sym_{mx}x{my}x{mz}mc.paths"
    )
    return str((ROUTEPATH_DIR / name).relative_to(REPO_ROOT))


def expected_paths_for_row(row: Dict[str, str]) -> Tuple[str, str, str]:
    """Expected (paths, nrl, vc) relative paths for current symmetry mode."""
    topo = row["topology"]
    family = row["family"]
    sym_tried = row.get("symmetry_tried", "no") == "yes"
    action = row.get("next_action", "")

    use_sym = (sym_tried or action == "launch_sym") and action != "launch_nonsym"

    # PT nonsym: DOR dim-tiebreak via bash/modern/implement_pt.sh
    if family == "pt" and not use_sym:
        base = f"{topo}_dor_dim_tiebreak_destbased"
        paths = f"topologies_and_routing/routepath_lists/{base}.paths"
        nrl = f"topologies_and_routing/nr_lists/{base}.nrl2"
        vc = f"topologies_and_routing/vc_mats/{base}.vcmat2"
        return paths, nrl, vc

    if family in ("asc", "pt", "pdtt") and use_sym:
        params = (row.get("symmetry_params") or "").split()
        if family == "asc":
            if not params:
                params = asc_sym_params(topo).split()
            st, mx, my, mz = params[7], params[4], params[5], params[6]
        elif family == "pt":
            if not params:
                params = pt_sym_params(topo).split()
            st, mx, my, mz = params[7], params[4], params[5], params[6]
        else:
            if not params:
                params = pdtt_sym_params(topo).split()
            st, mx, my, mz = params[7], params[4], params[5], params[6]
        base = (
            f"{topo}_turns_allowed_cpl_safe_{st}sym_{mx}x{my}x{mz}mc_"
            f"symrouting_destbased_canonsrcs_new_mclb_destbased_"
            f"{st}sym_{mx}x{my}x{mz}mc"
        )
    else:
        base = f"{topo}_turns_allowed_cpl_safe_destbased_new_mclb_destbased"

    paths = f"topologies_and_routing/routepath_lists/{base}.paths"
    nrl = f"topologies_and_routing/nr_lists/{base}.nrl2"
    vc = f"topologies_and_routing/vc_mats/{base}_olb.vcmat2"
    return paths, nrl, vc


def artifacts_present(row: Dict[str, str]) -> bool:
    paths, nrl, vc = expected_paths_for_row(row)
    for rel in (paths, nrl, vc):
        p = REPO_ROOT / rel
        if not p.is_file() or p.stat().st_size == 0:
            return False
    return True


def compute_next_action(row: Dict[str, str]) -> str:
    """Deterministic next_action from row state (plan state machine)."""
    family = row["family"]
    phase = row.get("phase_status", "pending")
    outcome = (row.get("attempt_outcome") or "").strip()
    sym_tried = row.get("symmetry_tried", "no") == "yes"
    try:
        cpus_last = int(row["cpus_last_tried"]) if row.get("cpus_last_tried") else 0
    except ValueError:
        cpus_last = 0

    if phase == "succeeded" or row.get("next_action") == "done":
        return "done"
    if phase == "failed_terminal":
        return "failed_terminal"

    if phase == "needs_verify" or outcome == "success":
        if artifacts_present(row):
            return "verify"

    if family == "pdtt":
        # Small topologies: nonsym first (same policy as ASC/PT). Larger / after
        # OOM-at-max: use symmetry once a DOR CPL is available.
        try:
            n_r = int(row.get("n_routers") or 0)
        except ValueError:
            n_r = 0

        def _pdtt_sym_or_cpl() -> str:
            cpl = (row.get("cpl_path") or "").strip()
            cpl_ok = bool(cpl) and (REPO_ROOT / cpl).is_file()
            if not cpl_ok:
                return "generate_cpl"
            return "launch_sym"

        if not outcome:
            if sym_tried:
                return _pdtt_sym_or_cpl()
            if n_r <= 512:
                return "launch_nonsym"
            return _pdtt_sym_or_cpl()

        if outcome == "oom":
            if cpus_last and cpus_last < MAX_CPUS:
                return "bump_cpus"
            if not sym_tried:
                return _pdtt_sym_or_cpl()
            return "failed_terminal"

        if outcome == "timeout":
            if cpus_last and cpus_last < MAX_CPUS and not sym_tried:
                return "bump_cpus"
            if not sym_tried:
                return _pdtt_sym_or_cpl()
            if cpus_last >= MAX_CPUS and sym_tried:
                return "failed_terminal"
            return _pdtt_sym_or_cpl()

        if outcome in ("error", "verify_fail"):
            if sym_tried:
                return _pdtt_sym_or_cpl()
            return "launch_nonsym" if n_r <= 512 else _pdtt_sym_or_cpl()

        return "launch_nonsym" if (n_r <= 512 and not sym_tried) else _pdtt_sym_or_cpl()

    # ASC / PT
    if not outcome:
        if sym_tried:
            return "launch_sym"
        return "launch_nonsym"

    if outcome == "oom":
        if cpus_last and cpus_last < MAX_CPUS:
            return "bump_cpus"
        if not sym_tried:
            return "launch_sym"
        return "failed_terminal"

    if outcome == "timeout":
        if cpus_last and cpus_last < MAX_CPUS and not sym_tried:
            return "bump_cpus"
        if not sym_tried:
            return "launch_sym"
        if cpus_last >= MAX_CPUS and sym_tried:
            return "failed_terminal"
        return "launch_sym" if not sym_tried else "bump_cpus"

    if outcome in ("error", "verify_fail"):
        # Retry same class of action once via needs_retry semantics
        if sym_tried:
            return "launch_sym"
        return "launch_nonsym"

    if outcome == "success":
        return "verify"

    return "launch_nonsym"


def apply_next_action_side_effects(row: Dict[str, str]) -> None:
    """Update cpus_next / symmetry_params / cpl_path to match next_action."""
    action = row["next_action"]
    family = row["family"]
    topo = row["topology"]

    try:
        cpus_last = int(row["cpus_last_tried"]) if row.get("cpus_last_tried") else 0
    except ValueError:
        cpus_last = 0
    try:
        cpus_next = int(row["cpus_next"]) if row.get("cpus_next") else DEFAULT_CPUS
    except ValueError:
        cpus_next = DEFAULT_CPUS

    if action == "bump_cpus":
        row["cpus_next"] = str(min(max(cpus_last * 2, DEFAULT_CPUS), MAX_CPUS))
        # After bump, the concrete launch is nonsym or sym depending on prior try
        if row.get("symmetry_tried") == "yes":
            row["next_action"] = "launch_sym"
            action = "launch_sym"
        else:
            row["next_action"] = "launch_nonsym"
            action = "launch_nonsym"
    elif action in ("launch_nonsym", "launch_sym", "generate_cpl") and not row.get("cpus_next"):
        row["cpus_next"] = str(DEFAULT_CPUS)

    if action == "launch_sym":
        if family == "asc":
            row["symmetry_params"] = asc_sym_params(topo)
        elif family == "pt":
            row["symmetry_params"] = pt_sym_params(topo)
        elif family == "pdtt":
            row["symmetry_params"] = pdtt_sym_params(topo)

    if family == "pdtt":
        found = find_best_cpl(topo)
        if found:
            row["cpl_path"] = found
            if action == "generate_cpl":
                row["next_action"] = "launch_sym"
                action = "launch_sym"
                row["symmetry_params"] = pdtt_sym_params(topo)
        else:
            row["cpl_path"] = expected_default_cpl(topo)
            if action != "generate_cpl" and action != "done":
                row["next_action"] = "generate_cpl"

    # Prefer heavy PDTT for large topos
    if family == "pdtt" and int(row["n_routers"]) >= 2048:
        row["agent_note"] = (row.get("agent_note") or "").strip()
        if "heavy" not in (row.get("agent_note") or ""):
            note = row.get("agent_note") or ""
            row["agent_note"] = (note + "; prefer_heavy").strip("; ").strip()

    if not row.get("cpus_next"):
        row["cpus_next"] = str(cpus_next if cpus_next else DEFAULT_CPUS)


def empty_row(topo: str, map_path: Path) -> Dict[str, str]:
    family = parse_family(topo)
    if family is None:
        raise ValueError(f"Unsupported topology family: {topo}")
    n_routers = parse_n_routers(topo, map_path)
    rel_map = str(map_path.relative_to(REPO_ROOT))
    log_rel = str((LOG_DIR / f"{topo}.log").relative_to(REPO_ROOT))
    row = {c: "" for c in CSV_COLUMNS}
    row.update(
        {
            "topology": topo,
            "family": family,
            "n_routers": str(n_routers),
            "map_path": rel_map,
            "phase_status": "pending",
            "cpus_next": str(DEFAULT_CPUS),
            "symmetry_tried": "no",
            "attempt_outcome": "",
            "log_path": log_rel,
            "status": "seeded; awaiting first launch",
            "last_updated": now_iso(),
        }
    )
    if family == "pdtt":
        row["symmetry_params"] = pdtt_sym_params(topo)
        found = find_best_cpl(topo)
        if found:
            row["cpl_path"] = found
        else:
            row["cpl_path"] = expected_default_cpl(topo)
        if n_routers <= 512:
            row["next_action"] = "launch_nonsym"
            row["status"] = "seeded; next=launch_nonsym N_CPUS=8 (sym on OOM/timeout)"
        elif found:
            row["next_action"] = "launch_sym"
            row["status"] = f"seeded; next=launch_sym cpl={found}"
        else:
            row["next_action"] = "generate_cpl"
            row["status"] = "seeded; next=generate_cpl (no usable DOR CPL)"
        if n_routers >= 2048:
            row["agent_note"] = "prefer_heavy"
    else:
        row["next_action"] = "launch_nonsym"
        row["status"] = "seeded; next=launch_nonsym N_CPUS=8"
    return row


def discover_maps() -> List[Path]:
    maps: List[Path] = []
    for p in sorted(PAPER_SOLUTIONS.rglob("*.map")):
        name = p.stem
        if name.endswith("_demand"):
            continue
        fam = parse_family(name)
        if fam is None:
            continue
        maps.append(p)
    return maps


def read_csv(path: Path = CSV_PATH) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for raw in reader:
            row = {c: (raw.get(c) or "") for c in CSV_COLUMNS}
            rows.append(row)
        return rows


def write_csv(rows: List[Dict[str, str]], path: Path = CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in CSV_COLUMNS})


def find_row(rows: List[Dict[str, str]], topology: str) -> Dict[str, str]:
    for row in rows:
        if row["topology"] == topology:
            return row
    raise KeyError(f"Topology not in CSV: {topology}")


def is_unfinished(row: Dict[str, str]) -> bool:
    return row.get("phase_status") not in TERMINAL_PHASE


def is_launchable(row: Dict[str, str]) -> bool:
    if not is_unfinished(row):
        return False
    if row.get("phase_status") == "running":
        return False
    action = row.get("next_action", "")
    if action in NON_LAUNCH_ACTIONS:
        return False
    return action in {
        "launch_nonsym",
        "launch_sym",
        "bump_cpus",
        "generate_cpl",
        "needs_retry",
    }


def use_heavy_pdtt(row: Dict[str, str]) -> bool:
    if row["family"] != "pdtt":
        return False
    if int(row["n_routers"]) >= 2048:
        return True
    if row.get("attempt_outcome") == "oom":
        return True
    note = row.get("agent_note") or ""
    return "prefer_heavy" in note or "heavy" in note


def build_launch_cmd(row: Dict[str, str]) -> str:
    """Shell command a Launch agent should run (with N_CPUS / CPL_PATH)."""
    action = row["next_action"]
    # Resolve bump_cpus into concrete launch for command building
    if action == "bump_cpus":
        action = "launch_sym" if row.get("symmetry_tried") == "yes" else "launch_nonsym"
    if action == "needs_retry":
        action = "launch_sym" if row.get("symmetry_tried") == "yes" else "launch_nonsym"

    topo = row["topology"]
    cpus = row.get("cpus_next") or str(DEFAULT_CPUS)
    log = row.get("log_path") or str((LOG_DIR / f"{topo}.log").relative_to(REPO_ROOT))
    exit_side = log + ".exit"
    family = row["family"]

    def wrap(env_prefix: str, script_and_args: str) -> str:
        return (
            f"( {env_prefix}{script_and_args} ) > {log} 2>&1; "
            f"ec=$?; echo $ec > {exit_side}; exit $ec"
        )

    if action == "generate_cpl":
        params = (row.get("symmetry_params") or pdtt_sym_params(topo)).split()
        # topo X Y Z C MX MY MZ ST
        args = " ".join([topo] + params)
        return wrap(
            f"N_CPUS={cpus} ",
            f"bash bash/implement_dor_mclb.sh {args}",
        )

    if family == "asc" and action == "launch_nonsym":
        return wrap(
            f"N_CPUS={cpus} ",
            f"bash bash/modern/implement_asc_destbased.sh {topo} {row['n_routers']}",
        )

    if family == "asc" and action == "launch_sym":
        params = row.get("symmetry_params") or asc_sym_params(topo)
        return wrap(
            f"N_CPUS={cpus} ",
            f"bash bash/modern/implement_asc_sym_destbased.sh {topo} {params}",
        )

    if family == "pt" and action == "launch_nonsym":
        (x, y, z), _ = parse_dims(topo)
        return wrap(
            "",
            f"bash bash/modern/implement_pt.sh {x} {y} {z} {DEFAULT_C} {cpus}",
        )

    if family == "pt" and action == "launch_sym":
        params = row.get("symmetry_params") or pt_sym_params(topo)
        return wrap(
            f"N_CPUS={cpus} ",
            f"bash bash/modern/implement_pt_sym_destbased.sh {topo} {params}",
        )

    if family == "pdtt" and action == "launch_nonsym":
        (x, y, z), _ = parse_dims(topo)
        return wrap(
            f"N_CPUS={cpus} ",
            f"bash bash/modern/implement_pdtt_destbased.sh {topo} {x} {y} {z} {DEFAULT_C}",
        )

    if family == "pdtt" and action == "launch_sym":
        params = row.get("symmetry_params") or pdtt_sym_params(topo)
        cpl = row.get("cpl_path") or find_best_cpl(topo) or expected_default_cpl(topo)
        script = (
            "bash/modern/implement_pdtt_sym_heavy_destbased.sh"
            if use_heavy_pdtt(row)
            else "bash/modern/implement_pdtt_sym_destbased.sh"
        )
        env = f"N_CPUS={cpus} CPL_PATH={cpl} "
        if use_heavy_pdtt(row):
            env += f"N_CPUS_GUROBI={cpus} "
        return wrap(env, f"bash {script} {topo} {params}")

    return f"echo 'No launch command for next_action={action} topology={topo}' >&2; exit 2"


def script_basename_for_row(row: Dict[str, str]) -> str:
    action = row["next_action"]
    if action == "bump_cpus":
        action = "launch_sym" if row.get("symmetry_tried") == "yes" else "launch_nonsym"
    family = row["family"]
    if action == "generate_cpl":
        return "implement_dor_mclb.sh"
    if family == "asc" and action == "launch_nonsym":
        return "implement_asc_destbased.sh"
    if family == "asc" and action == "launch_sym":
        return "implement_asc_sym_destbased.sh"
    if family == "pt" and action == "launch_nonsym":
        return "implement_pt.sh"
    if family == "pt" and action == "launch_sym":
        return "implement_pt_sym_destbased.sh"
    if family == "pdtt" and action == "launch_nonsym":
        return "implement_pdtt_destbased.sh"
    if family == "pdtt":
        return (
            "implement_pdtt_sym_heavy_destbased.sh"
            if use_heavy_pdtt(row)
            else "implement_pdtt_sym_destbased.sh"
        )
    return ""


def cmd_init(_args: argparse.Namespace) -> int:
    ensure_log_dir()
    maps = discover_maps()
    rows = [empty_row(p.stem, p) for p in maps]
    rows.sort(key=lambda r: (int(r["n_routers"]), r["family"], r["topology"]))
    write_csv(rows)
    print(f"Wrote {len(rows)} rows to {CSV_PATH.relative_to(REPO_ROOT)}")
    print(f"Log dir: {LOG_DIR.relative_to(REPO_ROOT)}")
    by_fam: Dict[str, int] = {}
    by_action: Dict[str, int] = {}
    for r in rows:
        by_fam[r["family"]] = by_fam.get(r["family"], 0) + 1
        by_action[r["next_action"]] = by_action.get(r["next_action"], 0) + 1
    print("By family:", dict(sorted(by_fam.items())))
    print("By next_action:", dict(sorted(by_action.items())))
    return 0


def running_cpu_sum(rows: Iterable[Dict[str, str]]) -> int:
    total = 0
    for row in rows:
        if row.get("phase_status") != "running":
            continue
        try:
            total += int(row.get("cpus_last_tried") or row.get("cpus_next") or 0)
        except ValueError:
            pass
    return total


def cmd_list_unfinished(args: argparse.Namespace) -> int:
    """List all launchable unfinished rows. Do not gate on a local CPU budget —
    SLURM queues srun jobs; N_CPUS is passed through for cpus-per-task only.
    --cpu-budget is accepted for backward compatibility but ignored.
    """
    rows = read_csv()
    if not rows:
        print("CSV empty; run init first", file=sys.stderr)
        return 1

    # Normalize bump_cpus side effects for display/commands
    for row in rows:
        if row.get("next_action") == "bump_cpus":
            apply_next_action_side_effects(row)

    if args.cpu_budget is not None:
        print(
            "# note: --cpu-budget ignored; SLURM manages resources via N_CPUS/cpus-per-task",
            file=sys.stderr,
        )

    selected = [r for r in rows if is_launchable(r)]
    selected.sort(key=lambda r: (int(r.get("cpus_next") or DEFAULT_CPUS), int(r["n_routers"]), r["topology"]))

    # Print TSV for Launch agents
    header = [
        "topology",
        "family",
        "n_routers",
        "cpus_next",
        "next_action",
        "script",
        "cpl_path",
        "log_path",
        "launch_cmd",
    ]
    print("\t".join(header))
    for row in selected:
        fields = [
            row["topology"],
            row["family"],
            row["n_routers"],
            row.get("cpus_next") or "",
            row["next_action"],
            script_basename_for_row(row),
            row.get("cpl_path") or "",
            row.get("log_path") or "",
            build_launch_cmd(row),
        ]
        print("\t".join(fields))

    print(
        f"# selected={len(selected)} running_cpus={running_cpu_sum(rows)} "
        f"(informational; not used to throttle launches)",
        file=sys.stderr,
    )
    return 0


def cmd_mark_running(args: argparse.Namespace) -> int:
    rows = read_csv()
    row = find_row(rows, args.topology)
    cpus = str(args.cpus)
    row["phase_status"] = "running"
    row["cpus_last_tried"] = cpus
    row["cpus_next"] = cpus
    row["script_last"] = args.script
    if args.log:
        row["log_path"] = args.log
    if row["next_action"] == "launch_sym":
        row["symmetry_tried"] = "yes"
        if not row.get("symmetry_params"):
            if row["family"] == "asc":
                row["symmetry_params"] = asc_sym_params(row["topology"])
            elif row["family"] == "pt":
                row["symmetry_params"] = pt_sym_params(row["topology"])
            elif row["family"] == "pdtt":
                row["symmetry_params"] = pdtt_sym_params(row["topology"])
    row["status"] = f"{row['next_action']} N_CPUS={cpus} script={args.script} running"
    row["last_updated"] = now_iso()
    if args.note:
        row["agent_note"] = args.note
    write_csv(rows)
    print(f"marked running: {args.topology} cpus={cpus} script={args.script}")
    return 0


def classify_log_outcome(log_path: Path, exit_code: Optional[int]) -> str:
    text = ""
    if log_path.is_file():
        try:
            text = log_path.read_text(errors="replace")[-200_000:].lower()
        except OSError:
            text = ""

    oom_hints = (
        "out of memory",
        "oom",
        "killed",
        "cannot allocate memory",
        "slurmstepd: error: *** job",
        "exceeded memory",
        "memory limit",
    )
    timeout_hints = ("time limit", "walltime", "cancelled due to time", "dws/job timeout")

    if exit_code == 137 or any(h in text for h in oom_hints):
        return "oom"
    if exit_code in (124, 140) or any(h in text for h in timeout_hints):
        return "timeout"
    if exit_code == 0:
        return "success"
    if exit_code is not None and exit_code != 0:
        return "error"
    if any(h in text for h in oom_hints):
        return "oom"
    if text:
        return "error"
    return ""


def read_exit_code(log_path: Path) -> Optional[int]:
    side = Path(str(log_path) + ".exit")
    if not side.is_file():
        return None
    try:
        return int(side.read_text().strip().split()[0])
    except (ValueError, OSError, IndexError):
        return None


def cmd_reconcile(args: argparse.Namespace) -> int:
    rows = read_csv()
    targets = rows if args.all else [find_row(rows, args.topology)]
    for row in targets:
        if row.get("phase_status") in TERMINAL_PHASE and not args.force:
            continue
        if row.get("phase_status") not in ("running", "needs_retry", "needs_verify", "pending") and not args.force:
            if row.get("phase_status") not in ("",):
                # Still allow unfinished non-running if forced via reconcile-all unfinished
                if not is_unfinished(row):
                    continue

        log_rel = row.get("log_path") or ""
        log_path = REPO_ROOT / log_rel if log_rel else LOG_DIR / f"{row['topology']}.log"
        exit_code = read_exit_code(log_path)

        # Only reconcile running (or forced) rows that have an exit sidecar or artifacts
        if row.get("phase_status") == "running" and exit_code is None and not artifacts_present(row):
            # Still running
            continue

        if artifacts_present(row):
            paths, nrl, vc = expected_paths_for_row(row)
            row["routing paths file"] = paths
            row["routing nrl file"] = nrl
            row["vc file"] = vc
            row["attempt_outcome"] = "success"
            row["phase_status"] = "needs_verify"
            row["next_action"] = "verify"
            row["status"] = f"artifacts present; next=verify exit={exit_code}"
        else:
            outcome = classify_log_outcome(log_path, exit_code)
            if not outcome and row.get("phase_status") != "running":
                continue
            if not outcome:
                outcome = "error"
            row["attempt_outcome"] = outcome
            # Update next_action via state machine
            tmp = dict(row)
            tmp["next_action"] = ""  # recompute
            action = compute_next_action(tmp)
            row["next_action"] = action
            apply_next_action_side_effects(row)
            action = row["next_action"]
            if action == "failed_terminal":
                row["phase_status"] = "failed_terminal"
            elif action == "verify":
                row["phase_status"] = "needs_verify"
            else:
                row["phase_status"] = "needs_retry" if outcome in ("error", "verify_fail") else "pending"
            row["status"] = (
                f"{row.get('script_last') or 'job'} N_CPUS={row.get('cpus_last_tried')} "
                f"exit={exit_code} {outcome}; next={action}"
                + (f"→{row.get('cpus_next')}" if action in ("bump_cpus", "launch_nonsym", "launch_sym") else "")
            )
        row["last_updated"] = now_iso()
        print(f"{row['topology']}: phase={row['phase_status']} next={row['next_action']} outcome={row.get('attempt_outcome')}")

    write_csv(rows)
    return 0


def cmd_reconcile_all(args: argparse.Namespace) -> int:
    args.all = True
    args.topology = None
    args.force = getattr(args, "force", False)
    return cmd_reconcile(args)


def cmd_mark_verified(args: argparse.Namespace) -> int:
    rows = read_csv()
    row = find_row(rows, args.topology)
    if args.paths:
        row["routing paths file"] = args.paths
    if args.nrl:
        row["routing nrl file"] = args.nrl
    if args.vc:
        row["vc file"] = args.vc
    if args.throughput is not None:
        row["routed throughput"] = str(args.throughput)
    row["attempt_outcome"] = "success"
    row["phase_status"] = "succeeded"
    row["next_action"] = "done"
    row["status"] = (
        f"verified destbased ok; throughput={row.get('routed throughput') or 'n/a'}"
    )
    row["last_updated"] = now_iso()
    write_csv(rows)
    print(f"marked verified/succeeded: {args.topology}")
    return 0


def cmd_summary(_args: argparse.Namespace) -> int:
    rows = read_csv()
    if not rows:
        print("CSV empty")
        return 0
    by_phase: Dict[str, int] = {}
    by_action: Dict[str, int] = {}
    by_fam: Dict[str, int] = {}
    for r in rows:
        by_phase[r.get("phase_status") or ""] = by_phase.get(r.get("phase_status") or "", 0) + 1
        by_action[r.get("next_action") or ""] = by_action.get(r.get("next_action") or "", 0) + 1
        by_fam[r.get("family") or ""] = by_fam.get(r.get("family") or "", 0) + 1
    unfinished = sum(1 for r in rows if is_unfinished(r))
    running = sum(1 for r in rows if r.get("phase_status") == "running")
    print(f"total={len(rows)} unfinished={unfinished} running={running} running_cpus={running_cpu_sum(rows)}")
    print("phase_status:", dict(sorted(by_phase.items())))
    print("next_action:", dict(sorted(by_action.items())))
    print("family:", dict(sorted(by_fam.items())))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Destbased routing manager (CSV state machine)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="Seed CSV from paper_solutions maps")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser(
        "list-unfinished",
        help="Rows to launch this round (all unfinished; SLURM manages queue)",
    )
    sp.add_argument(
        "--cpu-budget",
        type=int,
        default=None,
        help="Deprecated/ignored: do not throttle locally; pass N_CPUS to scripts for SLURM",
    )
    sp.set_defaults(func=cmd_list_unfinished)

    sp = sub.add_parser("mark-running", help="Mark a topology as running")
    sp.add_argument("--topology", "-t", required=True)
    sp.add_argument("--cpus", "-c", type=int, required=True)
    sp.add_argument("--script", "-s", required=True)
    sp.add_argument("--log", default="")
    sp.add_argument("--note", default="")
    sp.set_defaults(func=cmd_mark_running)

    sp = sub.add_parser("reconcile", help="Reconcile one topology from log/artifacts")
    sp.add_argument("--topology", "-t", required=True)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_reconcile, all=False)

    sp = sub.add_parser("reconcile-all", help="Reconcile all non-terminal rows")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_reconcile_all)

    sp = sub.add_parser("mark-verified", help="Record verify+throughput success")
    sp.add_argument("--topology", "-t", required=True)
    sp.add_argument("--paths", default="")
    sp.add_argument("--nrl", default="")
    sp.add_argument("--vc", default="")
    sp.add_argument("--throughput", type=float, default=None)
    sp.set_defaults(func=cmd_mark_verified)

    sp = sub.add_parser("summary", help="Counts by phase_status / next_action")
    sp.set_defaults(func=cmd_summary)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    os.chdir(REPO_ROOT)
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.exit(0)
