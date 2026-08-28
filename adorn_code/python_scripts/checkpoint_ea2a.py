#!/usr/bin/env python3
"""
Launch SLURM jobs to checkpoint/restore decomposed all-to-all MCF:
  - checkpoint_master: run master only, write checkpoint
  - checkpoint_children: run child for each (canonical) source
  - checkpoint_restore: load all checkpoints, write XML

Iterates over .map files in files/paper_solutions/<N>r/ (N = node count).
For topologies > 512 nodes, parses X,Y,Z and MC dims from filename (ASC/PT/PDTT).
"""

import argparse
import os
import re
import subprocess
import sys

# Repo root: parent of python_scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PAPER_SOLUTIONS = os.path.join(REPO_ROOT, "files", "paper_solutions")
XML_OUT_DIR = "/scratch/negishi/green456/xml_a2a"

SLURM_ARGS_BASE = "--account mithuna -p cpu -q standby -t 04:00:00"  # --cpus-per-task set via --slurm_cpus
BASE_CMD = (
    "source setup.sh && python src/gen_coll_comm_a2a_tpuv4_sym.py "
    "--topology {topo_path} --algorithm decomp --method 2 --threads 64 --crossover 0"
)
SYMMETRIC_PREFIX = " --symmetric --mc_dims {mcx} {mcy} {mcz} --xyzc_dims {x} {y} {z} 4 --sym_type {sym_type}"

# Directory name like "512r" -> 512
DIR_NODES_RE = re.compile(r"^(\d+)r$")

# ASC (sym): asc_*_sym_<n>c_<n>r_6p_<X>x<Y>x<Z>_<MCX>x<MCY>x<MCZ>.map
ASC_SYM_RE = re.compile(
    r"^asc_.*_sym_\d+c_\d+r_6p_(\d+)x(\d+)x(\d+)_(\d+)x(\d+)x(\d+)\.map$", re.I
)
# PT: pt_<n>c_<n>r_6p_<X>x<Y>x<Z>.map  -> mc 4 4 4
PT_RE = re.compile(r"^pt_\d+c_\d+r_6p_(\d+)x(\d+)x(\d+)\.map$", re.I)
# PDTT: pdtt_<n>c_<n>r_6p_<X>x<Y>x<Z>.map
PDTT_RE = re.compile(r"^pdtt_\d+c_\d+r_6p_(\d+)x(\d+)x(\d+)\.map$", re.I)


def parse_nodes_from_dir(dirname):
    """Return node count from directory name (e.g. '512r' -> 512) or None."""
    m = DIR_NODES_RE.match(dirname)
    return int(m.group(1)) if m else None


def parse_dims_for_large_topo(basename, n_nodes):
    """
    Parse (x, y, z, cube=4), (mcx, mcy, mcz), sym_type from filename for n_nodes > 512.
    Returns (xyzc_dims, mc_dims, sym_type) or None if not parseable.
    """
    cube = 4
    # ASC sym: X Y Z _ MCX MCY MCZ
    m = ASC_SYM_RE.match(basename)
    if m:
        x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
        mcx, mcy, mcz = int(m.group(4)), int(m.group(5)), int(m.group(6))
        return ((x, y, z, cube), (mcx, mcy, mcz), "trans")

    # PT: X Y Z, mc = 4 4 4
    m = PT_RE.match(basename)
    if m:
        x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return ((x, y, z, cube), (4, 4, 4), "trans")

    # PDTT: X Y Z; kxkx2k (2*X==2*Y==Z) -> mc X/2 Y/2 4; kx2kx2k (2*X==Y==Z) -> mc X/2 4 4
    m = PDTT_RE.match(basename)
    if m:
        x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 2 * x == z and 2 * y == z:  # kxkx2k
            mc_dims = (x // 2, y // 2, 4)
        elif 2 * x == y and y == z:  # kx2kx2k
            mc_dims = (x // 2, 4, 4)
        else:
            # fallback kxkx2k when Z=2*X and Z=2*Y
            mc_dims = (x // 2, y // 2, 4)
        return ((x, y, z, cube), mc_dims, "refl-trans")

    return None


def get_canonical_sources_for_topo(xyzc_dims, mc_dims, sym_type):
    """Return list of canonical source indices using TPUv4_Symmetry (no graph)."""
    sys.path.insert(0, SCRIPT_DIR)
    from tpuv4_symmetry import TPUv4_Symmetry

    sym = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type=sym_type)
    return sym.get_canonical_nodes()


def discover_topologies(size_filter=None):
    """Yield (topo_path, n_nodes, dims_info).
    dims_info is None for <=512; else (xyzc_dims, mc_dims, sym_type) for >512.
    If size_filter is set (list of int), only look at directories files/paper_solutions/<N>r/ for each N.
    """
    if not os.path.isdir(PAPER_SOLUTIONS):
        return
    if size_filter is not None and len(size_filter) > 0:
        dirs_to_scan = [f"{s}r" for s in size_filter]
    else:
        dirs_to_scan = sorted(
            d for d in os.listdir(PAPER_SOLUTIONS)
            if os.path.isdir(os.path.join(PAPER_SOLUTIONS, d)) and parse_nodes_from_dir(d) is not None
        )
    for dirname in dirs_to_scan:
        dpath = os.path.join(PAPER_SOLUTIONS, dirname)
        if not os.path.isdir(dpath):
            if size_filter is not None and len(size_filter) > 0:
                print("No such directory:", dpath)
            continue
        n_nodes = parse_nodes_from_dir(dirname)
        if n_nodes is None:
            continue  # skip e.g. "old"
        for fname in sorted(os.listdir(dpath)):
            if not fname.endswith(".map"):
                continue
            topo_path = os.path.join(dpath, fname)
            if not os.path.isfile(topo_path):
                continue
            dims_info = None
            if n_nodes > 512:
                dims_info = parse_dims_for_large_topo(fname, n_nodes)
            yield (topo_path, n_nodes, dims_info)


def build_base_cmd(topo_path, dims_info, repo_root=REPO_ROOT):
    """Base command with optional symmetric args for >512. topo_path is made relative to repo_root."""
    rel_topo = os.path.relpath(topo_path, repo_root)
    cmd = BASE_CMD.format(topo_path=rel_topo)
    if dims_info is not None:
        (x, y, z, c), (mcx, mcy, mcz), sym_type = dims_info
        cmd += SYMMETRIC_PREFIX.format(x=x, y=y, z=z, mcx=mcx, mcy=mcy, mcz=mcz, sym_type=sym_type)
    return cmd


def run_sbatch(cmd, job_name, dry_run, slurm_cpus=64):
    """Launch sbatch with cmd as the job script body."""
    # Safe filename: no path separators
    outerr_name = job_name.replace("/", "_").replace("\\", "_")
    outerr_path = os.path.join("slurm", "outputs", f"{outerr_name}.outerr")
    script_lines = [
        "#!/bin/bash",
        "#SBATCH " + SLURM_ARGS_BASE.replace(" ", "\n#SBATCH "),
        f"#SBATCH --cpus-per-task={slurm_cpus}",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -o {outerr_path}",
        f"#SBATCH -e {outerr_path}",
        "",
        "cd " + REPO_ROOT,
        "mkdir -p slurm/outputs",
        cmd,
    ]
    script = "\n".join(script_lines)
    if dry_run:
        print("[dry-run] would run sbatch with:")
        print(script)
        print("---")
        return
    tmp = os.path.join("/tmp", "checkpoint_ea2a_slurm_%d.sh" % os.getpid())
    with open(tmp, "w") as f:
        f.write(script)
    try:
        subprocess.run(["sbatch", tmp], cwd=REPO_ROOT, check=True)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description="Launch checkpoint/restore jobs for decomposed A2A")
    ap.add_argument("--checkpoint_master", action="store_true", help="Checkpoint masters for all topologies")
    ap.add_argument("--checkpoint_children", action="store_true", help="Checkpoint children (all or canonical)")
    ap.add_argument("--checkpoint_restore", action="store_true", help="Restore all checkpoints and write XML")
    ap.add_argument("--dry_run", action="store_true", help="Print commands / job scripts, do not submit")
    ap.add_argument("--size_filter", type=int, nargs="*", default=None, metavar="N", help="Only look at directories files/paper_solutions/<N>r/ (e.g. --size_filter 768 1024 1536)")
    ap.add_argument("--name_filter", type=str, nargs="*", default=None, metavar="STR", help="Only run topologies whose name contains any of these strings (e.g. --name_filter asc pdtt)")
    ap.add_argument("--slurm_cpus", type=int, default=64, metavar="N", help="SLURM --cpus-per-task (default: 64)")
    args = ap.parse_args()

    if not any([args.checkpoint_master, args.checkpoint_children, args.checkpoint_restore]):
        ap.error("One of --checkpoint_master, --checkpoint_children, --checkpoint_restore required")

    size_filter = args.size_filter if args.size_filter else None  # empty list -> None
    topologies = list(discover_topologies(size_filter=size_filter))
    name_filter = args.name_filter if args.name_filter else None
    if name_filter:
        topologies = [
            t for t in topologies
            if any(n in os.path.splitext(os.path.basename(t[0]))[0] for n in name_filter)
        ]
    if not topologies:
        print("No topologies found under", PAPER_SOLUTIONS)
        sys.exit(1)

    for topo_path, n_nodes, dims_info in topologies:
        topo_basename = os.path.splitext(os.path.basename(topo_path))[0]
        if n_nodes > 512 and dims_info is None:
            print("Skip (no dims for >512):", topo_path)
            continue
        base_cmd = build_base_cmd(topo_path, dims_info)

        if args.checkpoint_master:
            cmd = base_cmd + " --checkpoint_master"
            job_name = f"ea2a_master_{topo_basename}"[:64]
            run_sbatch(cmd, job_name, args.dry_run, slurm_cpus=args.slurm_cpus)

        elif args.checkpoint_children:
            if n_nodes <= 512:
                children = list(range(n_nodes))
            else:
                if dims_info is None:
                    print("Skip (no dims):", topo_path)
                    continue
                xyzc_dims, mc_dims, sym_type = dims_info
                children = get_canonical_sources_for_topo(xyzc_dims, mc_dims, sym_type)
            for src in children:
                cmd = base_cmd + f" --checkpoint_child {src}"
                job_name = f"ea2a_child_{topo_basename}_{src}"[:64]
                run_sbatch(cmd, job_name, args.dry_run, slurm_cpus=args.slurm_cpus)

        elif args.checkpoint_restore:
            cmd = base_cmd + " --restore_all_checkpoints"
            xml_path = os.path.join(XML_OUT_DIR, f"{topo_basename}_decomp.xml")
            cmd = f"mkdir -p {XML_OUT_DIR} && " + cmd + f" --xml {xml_path}"
            job_name = f"ea2a_restore_{topo_basename}"[:64]
            run_sbatch(cmd, job_name, args.dry_run, slurm_cpus=args.slurm_cpus)

    print("Done.")


if __name__ == "__main__":
    main()
