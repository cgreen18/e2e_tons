#!/usr/bin/env python3
"""
Launch SLURM jobs to run full decomposed all-to-all optimization.

Recursively iterates over .map files in the specified topology base directory
and submits SLURM jobs for each topology.
"""

import argparse
import os
import re
import subprocess
import sys

# Repo root: parent of python_scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
XML_OUT_DIR = "/scratch/negishi/green456/xml_a2a"
SLURM_OUTPUT_DIR = os.path.join(REPO_ROOT, "slurm", "outputs")

SLURM_ARGS_STANDBY = "--account mithuna -p cpu -q standby --cpus-per-task=128 -t 04:00:00"
SLURM_ARGS = "--account mithuna -p cpu --cpus-per-task=128 -t 20:00:00"

ALG = "link"
APL_BASE = os.path.join(REPO_ROOT, "topologies_and_routing", "allpath_lists")

# Topology filename patterns: <topo-type>_<c>c_<n>r_6p_<X>x<Y>x<Z>*
# ASC (sym): asc_*_sym_<n>c_<n>r_6p_<X>x<Y>x<Z>_<MCX>x<MCY>x<MCZ>.map
ASC_SYM_RE = re.compile(
    r"^asc_.*_sym_\d+c_\d+r_6p_(\d+)x(\d+)x(\d+)_(\d+)x(\d+)x(\d+)\.map$", re.I
)
# PT: pt_<n>c_<n>r_6p_<X>x<Y>x<Z>.map -> mc 4 4 4
PT_RE = re.compile(r"^pt_\d+c_\d+r_6p_(\d+)x(\d+)x(\d+)\.map$", re.I)
# PDTT: pdtt_<n>c_<n>r_6p_<X>x<Y>x<Z>.map
PDTT_RE = re.compile(r"^pdtt_\d+c_\d+r_6p_(\d+)x(\d+)x(\d+)\.map$", re.I)


def parse_symmetric_dims(basename):
    """
    Parse (x, y, z), (mcx, mcy, mcz), sym_type from topology filename.
    Returns (x, y, z, mcx, mcy, mcz, sym_type) or None if not parseable.
    - ASC: MC dims in name (_<MCX>x<MCY>x<MCZ>), sym_type trans
    - PT: MC=4,4,4, sym_type trans
    - PDTT: kxkx2k (2*X==2*Y==Z) -> MC X/2,Y/2,4; kx2kx2k (2*X==Y==Z) -> MC X/2,4,4; sym_type refl-trans
    """
    # ASC sym: X Y Z _ MCX MCY MCZ
    m = ASC_SYM_RE.match(basename)
    if m:
        x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
        mcx, mcy, mcz = int(m.group(4)), int(m.group(5)), int(m.group(6))
        return (x, y, z, mcx, mcy, mcz, "trans")
    # PT: X Y Z, mc = 4 4 4
    m = PT_RE.match(basename)
    if m:
        x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return (x, y, z, 4, 4, 4, "trans")
    # PDTT: kxkx2k (2*X==2*Y==Z) -> mc X/2 Y/2 4; kx2kx2k (2*X==Y==Z) -> mc X/2 4 4
    m = PDTT_RE.match(basename)
    if m:
        x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 2 * x == z and 2 * y == z:  # kxkx2k
            mc_dims = (x // 2, y // 2, 4)
        elif 2 * x == y and y == z:  # kx2kx2k
            mc_dims = (x // 2, 4, 4)
        else:
            mc_dims = (x // 2, y // 2, 4)
        return (x, y, z, mc_dims[0], mc_dims[1], mc_dims[2], "refl-trans")
    return None


def resolve_apl_path(topo_name, apl_base=None):
    """
    Return path to the allpaths list file for this topology, or None if not found.
    - ASC topologies (name starts with asc_): match basename starting with <topo_name>_allallowed
    - Non-ASC: match basename starting with <topo_name>_dor
    (Filenames may have extra suffix, e.g. pdtt_..._dor_refl-transsym_8x8x4mc.rallpaths)
    Looks in apl_base if provided, else APL_BASE.
    """
    base_dir = os.path.abspath(apl_base) if apl_base else APL_BASE
    if not os.path.isdir(base_dir):
        return None
    is_asc = topo_name.lower().startswith("asc_")
    prefix = f"{topo_name}_allallowed" if is_asc else f"{topo_name}_dor"
    try:
        matches = []
        for fname in os.listdir(base_dir):
            base, _ = os.path.splitext(fname)
            if base.startswith(prefix):
                matches.append(os.path.join(base_dir, fname))
        if matches:
            return sorted(matches)[0]  # deterministic choice if multiple
    except OSError:
        pass
    return None


def discover_topologies(topology_base_dir, exclude_dirs=None):
    """
    Recursively find all .map files under topology_base_dir.
    Yields absolute paths to each .map file.
    If exclude_dirs is set (iterable of directory names), do not descend into
    those directories.
    """
    if not os.path.isdir(topology_base_dir):
        print(f"Error: {topology_base_dir} is not a directory")
        sys.exit(1)

    exclude = set(exclude_dirs) if exclude_dirs else set()

    for root, dirs, files in os.walk(topology_base_dir):
        dirs[:] = [d for d in dirs if d not in exclude]
        for fname in sorted(files):
            if fname.endswith(".map"):
                yield os.path.join(root, fname)


def run_sbatch(topo_path, job_name, dry_run, slurm_args, symmetric=False, apl_base=None):
    """Launch sbatch with the full command for the given topology."""
    # Extract topology name (basename without extension)
    topo_basename = os.path.basename(topo_path)
    topo_name = os.path.splitext(topo_basename)[0]
    xml_path = os.path.join(XML_OUT_DIR, f"{topo_name}_{ALG}.xml")
    output_file = os.path.join(SLURM_OUTPUT_DIR, f"{job_name}.outerr")

    # Resolve allpaths list: ASC -> <topo_name>_allallowed, non-ASC -> <topo_name>_dor
    apl_path = resolve_apl_path(topo_name, apl_base=apl_base)

    # Build the command
    cmd = (
        f"source setup.sh && "
        f"python -u src/gen_coll_comm_a2a_tpuv4_sym.py "
        f"--topology {topo_path} "
        f"--algorithm {ALG} "
        f"--xml {xml_path} "
        f" --method 2 "
        f" --crossover 0 "
        f" --threads 64 "
    )
    if apl_path is not None:
        cmd += f" -apl {apl_path}"
    if symmetric:
        dims = parse_symmetric_dims(topo_basename)
        if dims is not None:
            x, y, z, mcx, mcy, mcz, sym_type = dims
            cmd += f" --symmetric --mc_dims {mcx} {mcy} {mcz} --xyzc_dims {x} {y} {z} 4 --sym_type {sym_type}"
        else:
            print(f"Warning: --symmetric set but could not parse dims for {topo_basename}, skipping symmetric args")
    print(cmd)

    # Create SLURM script
    script_lines = [
        "#!/bin/bash",
        "#SBATCH " + slurm_args.replace(" ", "\n#SBATCH "),
        f"#SBATCH -J {job_name}",
        f"#SBATCH -o {output_file}",
        f"#SBATCH -e {output_file}",
        "",
        f"mkdir -p {XML_OUT_DIR}",
        f"cd {REPO_ROOT}",
        cmd,
    ]
    script = "\n".join(script_lines)

    if dry_run:
        print(f"[dry-run] would run sbatch for {topo_path}:")
        print(script)
        print("---")
        return

    # Write temporary script and submit
    tmp = os.path.join("/tmp", f"full_ea2a_slurm_{os.getpid()}_{topo_name}.sh")
    with open(tmp, "w") as f:
        f.write(script)
    try:
        subprocess.run(["sbatch", tmp], cwd=REPO_ROOT, check=True)
        print(f"Submitted job for: {topo_path}")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(
        description="Launch full decomposed A2A optimization jobs for all .map files"
    )
    ap.add_argument(
        "--topology_base_dir",
        required=True,
        help="Base directory to recursively search for .map files"
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands / job scripts, do not submit"
    )
    ap.add_argument(
        "--exclude_dirs",
        nargs="*",
        default=[],
        metavar="DIR",
        help="Directory names to exclude from the search (do not descend into these)"
    )
    ap.add_argument(
        "--only_dir",
        metavar="SUBDIR",
        help="Only look inside this subdirectory of topology_base_dir (e.g. --only_dir sub looks in <base>/sub)"
    )
    ap.add_argument(
        "--standby",
        type=int,
        default=1,
        choices=[0, 1],
        metavar="0|1",
        help="1 = use standby queue (default); 0 = use regular cpu queue"
    )
    ap.add_argument(
        "--symmetric",
        action="store_true",
        help="Add --symmetric --mc_dims ... --xyzc_dims ... --sym_type ... to the Python script (parsed from topology filename)"
    )
    ap.add_argument(
        "--apl_base",
        metavar="DIR",
        default=None,
        help="Directory for allpaths lists (default: topologies_and_routing/allpaths_lists). Used to pass -apl to the Python script (ASC: <topo>_allallowed, non-ASC: <topo>_dor)."
    )
    args = ap.parse_args()

    # Convert to absolute path
    topology_base_dir = os.path.abspath(args.topology_base_dir)
    search_dir = os.path.join(topology_base_dir, args.only_dir) if args.only_dir else topology_base_dir

    # Discover all topologies
    topologies = list(discover_topologies(search_dir, exclude_dirs=args.exclude_dirs))

    if not topologies:
        print(f"No .map files found under {search_dir}")
        sys.exit(1)

    print(f"Found {len(topologies)} topology file(s)")

    # Create SLURM output directory if it doesn't exist
    if not args.dry_run:
        os.makedirs(SLURM_OUTPUT_DIR, exist_ok=True)

    slurm_args = SLURM_ARGS_STANDBY if args.standby == 1 else SLURM_ARGS

    # Submit a job for each topology
    for topo_path in topologies:
        topo_name = os.path.splitext(os.path.basename(topo_path))[0]
        job_name = f"full_ea2a_{topo_name}"[:64]  # SLURM job name limit
        run_sbatch(topo_path, job_name, args.dry_run, slurm_args, symmetric=args.symmetric, apl_base=args.apl_base)

    print("Done.")


if __name__ == "__main__":
    main()
