#!/usr/bin/env python3
"""
Topology, routing, and deadlock freedom (VC allocation) synthesis using a
complete channel dependency graph (CDG).

Reference(s):
    MCF: Shahrokhi & Matula; MTZ: Miller, Tucker & Zemlin;
    CDG: Dally & Seitz; Complete CDG: Domke, Hoefler & Matsuoka.
"""

import argparse
import os
import sys

from ccdg_io import (
    build_output_paths,
    default_commodities,
    parse_commodities_file,
    print_results,
    read_matrix_file,
    solution_summary_payload,
    uniform_matrix,
    write_json,
    write_results,
    write_topology_image,
)
from ccdg_model import CCDGModel
from ccdg_optimizer import CCDGOptimizer


def define_all_arguments():
    ap = argparse.ArgumentParser(
        description="One-shot topology, routing, and deadlock-free VC synthesis via complete CDG."
    )
    ap.add_argument("--n_nodes", type=int, required=True)
    ap.add_argument("--radix", type=int, required=True)
    ap.add_argument("--n_vcs", type=int, required=True)
    ap.add_argument("--out", type=str, help="Output base file name (no extension)")
    ap.add_argument("--out-json", type=str, default=None, help="Optional JSON summary path")
    ap.add_argument("--write-model", action="store_true")
    ap.add_argument("--capacity", type=float, default=1.0)
    ap.add_argument("--demand", type=float, default=1.0)
    ap.add_argument("--capacity-matrix", type=str, default=None)
    ap.add_argument("--demand-matrix", type=str, default=None)
    ap.add_argument("--commodities-file", type=str, default=None)
    ap.add_argument("--continuous-topo-edges", action="store_true")
    ap.add_argument("--continuous-turn-edges", action="store_true")
    ap.add_argument("--k-paths", type=int, default=1)
    ap.add_argument("--integral-flow", action="store_true")
    ap.add_argument("--symmetric-links", action="store_true")
    ap.add_argument("--per-source-solves", action="store_true")
    ap.add_argument("--allow-vc-trans", action="store_true")
    ap.add_argument("--time_limit", type=float, default=None)
    ap.add_argument("--mip_gap", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--silent", action="store_true")
    return ap


def parse_problem_arguments(args):
    n = args.n_nodes
    cap_mat = uniform_matrix(n, args.capacity)
    dem_mat = uniform_matrix(n, args.demand)
    if args.capacity_matrix:
        cap_mat = read_matrix_file(args.capacity_matrix, n)
    if args.demand_matrix:
        dem_mat = read_matrix_file(args.demand_matrix, n)

    commodities = (
        parse_commodities_file(args.commodities_file, n)
        if args.commodities_file
        else default_commodities(n)
    )

    base_out_name = args.out or f"ccdg_{n}n_{args.radix}r_{args.n_vcs}vcs"
    file_args = build_output_paths(base_out_name)
    file_args["write_model"] = args.write_model

    problem_args = {
        "n_nodes": n,
        "radix": args.radix,
        "n_vcs": args.n_vcs,
        "cap_mat": cap_mat,
        "dem_mat": dem_mat,
        "commodities": commodities,
        "relax_topo_edges": args.continuous_topo_edges,
        "relax_turn_edges": args.continuous_turn_edges,
        "integral_flow": args.integral_flow,
        "per_source_solves": args.per_source_solves,
        "k_paths": args.k_paths,
        "allow_vc_trans": args.allow_vc_trans,
        "symmetric_links": args.symmetric_links,
    }
    solver_params = {
        "silent": args.silent,
        "threads": args.threads,
        "time_limit": args.time_limit,
        "mip_gap": args.mip_gap,
    }
    return problem_args, file_args, solver_params


def _make_ccdg_model(
    problem_args,
    *,
    relax_topo_edges=None,
    relax_turn_edges=None,
    hardened_topo_edges=None,
    hardened_turn_edges=None,
):
    if relax_topo_edges is None:
        relax_topo_edges = problem_args["relax_topo_edges"]
    if relax_turn_edges is None:
        relax_turn_edges = problem_args["relax_turn_edges"]
    return CCDGModel(
        n_nodes=problem_args["n_nodes"],
        radix=problem_args["radix"],
        n_vcs=problem_args["n_vcs"],
        capacity=problem_args["cap_mat"],
        demand=problem_args["dem_mat"],
        commodities=problem_args["commodities"],
        relax_topo_edges=relax_topo_edges,
        relax_turn_edges=relax_turn_edges,
        integral_flow=problem_args["integral_flow"],
        per_source_solves=problem_args["per_source_solves"],
        k_paths=problem_args["k_paths"],
        allow_vc_trans=problem_args["allow_vc_trans"],
        symmetric_links=problem_args["symmetric_links"],
        hardened_topo_edges=hardened_topo_edges,
        hardened_turn_edges=hardened_turn_edges,
    )


def _solve_model(ccdg_model, solver_params):
    optimizer = CCDGOptimizer(ccdg_model=ccdg_model, model_params=solver_params)
    optimizer.solve()
    optimizer.extract_resultant_values()
    return optimizer


def run_optimization(problem_args, file_args, solver_params):
    if problem_args["relax_topo_edges"] or problem_args["relax_turn_edges"]:
        return _run_phased_relaxed_optimization(problem_args, file_args, solver_params)

    ccdg_model = _make_ccdg_model(problem_args)
    ccdg_model.build_model_()

    if file_args["write_model"]:
        ccdg_model.write_model_(
            os.path.join("files/models", file_args["base_out_name"] + ".lp")
        )

    optimizer = _solve_model(ccdg_model, solver_params)
    optimizer.dump_var_vals_to_file_(
        os.path.join(".", file_args["base_out_name"] + "_var_vals.txt")
    )
    results = {"lambda": optimizer.lam}
    return (results["lambda"], *optimizer.harden_results())


def _run_phased_relaxed_optimization(problem_args, file_args, solver_params):
    """Relax → harden topology → resolve → harden turns → resolve → extract paths."""
    print("Phase 1: solve with relaxed topology/turn edges")
    phase1_model = _make_ccdg_model(problem_args)
    phase1_model.build_model_()
    if file_args["write_model"]:
        phase1_model.write_model_(
            os.path.join("files/models", file_args["base_out_name"] + "_phase1.lp")
        )
    phase1_opt = _solve_model(phase1_model, solver_params)

    print("Phase 1: harden topology edges")
    topology = phase1_opt.harden_topology(phase1_opt.topo_adj_mat_vals)
    hardened_topo_edges = CCDGOptimizer.topology_to_edges(topology)
    print(f"  hardened {len(hardened_topo_edges)} directed topology edges")
    topo_stats = write_topology_image(
        file_args["hardened_topo_png_path"],
        topology,
        title=(
            f"Phase 1 hardened topology ({problem_args['n_nodes']} nodes, "
            f"radix {problem_args['radix']}, {len(hardened_topo_edges)} edges)"
        ),
    )
    print(f"  out-degrees: {topo_stats['out_degrees']}")
    print(f"  in-degrees:  {topo_stats['in_degrees']}")
    print(f"  wrote hardened topology image to {file_args['hardened_topo_png_path']}")

    print("Phase 2: resolve with hardened topology")
    phase2_model = _make_ccdg_model(
        problem_args,
        relax_topo_edges=False,
        relax_turn_edges=problem_args["relax_turn_edges"],
        hardened_topo_edges=hardened_topo_edges,
    )
    phase2_model.build_model_()
    if file_args["write_model"]:
        phase2_model.write_model_(
            os.path.join("files/models", file_args["base_out_name"] + "_phase2.lp")
        )
    phase2_opt = _solve_model(phase2_model, solver_params)

    print("Phase 2: harden turn edges")
    hardened_turn_edges = phase2_opt.harden_turns(phase2_opt.turn_adj_mat_vals)
    print(f"  hardened {len(hardened_turn_edges)} CDG turns")

    print("Phase 3: resolve with hardened topology and turns (binary)")
    phase3_model = _make_ccdg_model(
        problem_args,
        relax_topo_edges=False,
        relax_turn_edges=False,
        hardened_topo_edges=hardened_topo_edges,
        hardened_turn_edges=hardened_turn_edges,
    )
    phase3_model.build_model_()
    if file_args["write_model"]:
        phase3_model.write_model_(
            os.path.join("files/models", file_args["base_out_name"] + ".lp")
        )
    phase3_opt = _solve_model(phase3_model, solver_params)
    phase3_opt.dump_var_vals_to_file_(
        os.path.join(".", file_args["base_out_name"] + "_var_vals.txt")
    )

    print("Phase 3: extract paths (unconstrained hardening)")
    return (phase3_opt.lam, *phase3_opt.harden_results(constrain_paths=False))


def main():
    args = define_all_arguments().parse_args()
    problem_args, file_args, solver_params = parse_problem_arguments(args)

    if problem_args["per_source_solves"]:
        print("UNIMPLEMENTED: per-source solves are not supported yet.")
        sys.exit(1)

    (
        final_mcf_val,
        final_topology,
        final_paths,
        final_link_load,
        final_routing_table,
        final_vc_table,
    ) = run_optimization(problem_args, file_args, solver_params)

    print_results(
        final_mcf_val,
        final_topology,
        final_paths,
        final_link_load,
        final_routing_table,
        final_vc_table,
        problem_args["cap_mat"],
        problem_args["dem_mat"],
    )
    write_results(
        file_args,
        problem_args,
        final_mcf_val,
        final_topology,
        final_paths,
        final_link_load,
        final_routing_table,
        final_vc_table,
    )

    for label, key in (
        ("topology (map)", "topo_out_path"),
        ("topology (GraphML)", "graphml_out_path"),
        ("pathlist", "paths_out_path"),
        ("pathlist (JSONL)", "paths_jsonl_out_path"),
        ("next-router list", "nr_out_path"),
        ("VC allocation", "vc_out_path"),
    ):
        print(f"Wrote {label} to {file_args[key]}")
    print(f"lambda = {final_mcf_val}")

    if args.out_json:
        write_json(
            args.out_json,
            solution_summary_payload(
                final_mcf_val,
                problem_args["n_nodes"],
                problem_args["radix"],
                problem_args["n_vcs"],
                problem_args["commodities"],
                file_args,
            ),
        )


if __name__ == "__main__":
    main()
