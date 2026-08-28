import os
import gc
import time
import argparse
import os
import sys
from collections import defaultdict
from copy import deepcopy

# Gurobi libs
import gurobipy as gp
from gurobipy import GRB

# locals
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "python_scripts"))

from tpuv4_symmetry import TPUv4_Symmetry

# Reuse your helpers from the script
# - ingest_map() should return (adj_mat, adj_list)
# - output_pathlist(path_list, base_file_name, out_prefix)
# - print_max_cload(flat_route_pathlist)
# - setup_solver_params(args)
#
# And your symmetry class:
# from tpuv4_symmetry import TPUv4_Symmetry

INF = 10**10


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def mem_checkpoint(tag, model=None):
    """
    Lightweight checkpoint logger.
    If you want deeper profiling, add tracemalloc / pympler here.
    """
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss_gb = proc.memory_info().rss / (1024 ** 3)
        print(f"[mem] {tag}: RSS={rss_gb:.2f} GB")
    except Exception:
        print(f"[mem] {tag}: (psutil unavailable)")

    if model is not None:
        try:
            print(f"[gurobi] NumVars={model.NumVars} NumConstrs={model.NumConstrs} NumNZs={model.NumNZs}")
        except Exception:
            pass

def validate_path(adj_list, path, sr=None, dr=None):
    if sr is not None and path[0] != sr:
        raise RuntimeError(f"Path start mismatch: expected {sr}, got {path[0]} path={path}")
    if dr is not None and path[-1] != dr:
        raise RuntimeError(f"Path end mismatch: expected {dr}, got {path[-1]} path={path}")

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        if v not in adj_list[u]:
            raise RuntimeError(f"Invalid hop {u}->{v} not in adj_list (path={path})")

def output_pathlist( path_list, base_file_name, paths_output_path_prefix):

    full_name = f'{base_file_name}.paths'

    full_out_path = os.path.join(paths_output_path_prefix, \
            full_name)

    with open(full_out_path, 'w+') as of:
        of.write(base_file_name + '\n')
        for path in path_list:
            of.write(f'{path}\n')

    print(f'Wrote to {full_out_path}')

def stream_raw_paths(filepath):
    with open(filepath, "rb", buffering=1024*1024) as inf:
        for bline in inf:
            bline = bline.strip()
            if not bline:
                # skip empty
                continue
            # split bytes into byte tokens, convert each to int
            row = [int(tok) for tok in bline.split()]
            yield row

def ingest_map(path_name):

    if True:
        print(f'Ingesting r map ({path_name})')

    adj_list = []
    adj_mat = []

    with open(path_name, 'r') as inf:
        for row in inf:
            r_conns = row.split(' ')
            if '\n' in r_conns:
                r_conns.remove('\n')

            # deal with approximate values (from MIP)
            try:
                r_conns = [int(elem) for elem in r_conns]
            except:
                r_conns = [int(float(elem)) for elem in r_conns]

            adj_mat.append(r_conns)
            adj_list.append( [r for r, c in enumerate(r_conns) if c > 0] )

    return adj_mat, adj_list

def print_max_cload(flat_route_pathlist):

    print(f"Finding max channel load")

    max_cload = 0
    cload_dict = defaultdict(int)

    maximally_loaded_edges = []

    for path in flat_route_pathlist:
        n_hops = len(path)-1
        for i in range(n_hops):
            a = path[i]
            b = path [i+1]
            edge = (a,b)
    
            cload_dict[ edge ] += 1

            if cload_dict[ edge ] > max_cload:
                max_cload = cload_dict[ edge ]
                maximally_loaded_edges = [ edge ]
            elif cload_dict[ edge ] == max_cload:
                maximally_loaded_edges.append(edge)

    n_to_print = min(5,len(maximally_loaded_edges))
    print(f"After everything, throughput is {1/max_cload} from max cload {max_cload}")
    print(f"\tFrom edges {maximally_loaded_edges[:n_to_print]} (truncated)")

def setup_solver_params(args):

    # solver params
    solver_params = {}

    if args.time_limit:
        solver_params.update({'TimeLimit':args.time_limit*60})
    if args.threads:
        solver_params.update({'Threads':args.threads})
    if args.concurrent_mip:
        solver_params.update({'ConcurrentMIP':args.concurrent_mip})
    if args.mip_focus:
        solver_params.update({'MIPFocus':args.mip_focus})
    if args.heuristic_ratio:
        solver_params.update({'Heuristics':args.heuristic_ratio})
    if args.symmetry_detection:
        solver_params.update({'Symmetry':args.symmetry_detection})
    if args.barrier_iter_limit:
        solver_params.update({'BarIterLimit':args.barrier_iter_limit})
    if args.iter_limit:
        solver_params.update({'IterationLimit':args.iter_limit})
    if args.cut_passes:
        solver_params.update({'CutPasses':args.cut_passes})
    if args.method:
        solver_params.update({'Method':args.method})
    if args.node_method:
        solver_params.update({'NodeMethod':args.node_method})
    if args.crossover:
        solver_params.update({'Crossover':args.crossover})
    if args.crossover_basis:
        solver_params.update({'CrossoverBasis':args.crossover_basis})
    if args.no_rel_heur_time:
        solver_params.update({'NoRelHeurTime':args.no_rel_heur_time})
    if args.presolve:
        solver_params.update({'Presolve':args.presolve})
    if args.presparsify:
        solver_params.update({'PreSparsify':args.presparsify})
    if args.cuts:
        solver_params.update({'Cuts':args.cuts})
    if args.scale_flag:
        solver_params.update({'ScaleFlag':args.scale_flag})
    if args.feas_tol:
        solver_params.update({'FeasibilityTol':args.feas_tol})
    if args.predual:
        solver_params.update({'PreDual':args.predual})
    if args.degen_moves:
        solver_params.update({'DegenMoves':args.degen_moves})

    return solver_params

# --------------------------------------------------------------------------------------
# Main solver class
# --------------------------------------------------------------------------------------

class MCLBSolver:
    def __init__(
        self,
        adj_list,
        allpaths_filepath,
        solver_params=None,
        symmetric=False,
        my_tpuv4_symmetry=None,
        robust=False,
        backup_paths_filepath=None,
        override_ingest_all=False,
        prob_load=False,
        foresight=False,
        write_model=False,
    ):
        self.adj_list = adj_list
        self.n_routers = len(adj_list)

        self.allpaths_filepath = allpaths_filepath
        self.backup_paths_filepath = backup_paths_filepath

        self.symmetric = symmetric
        self.my_tpuv4_symmetry = my_tpuv4_symmetry
        self.override_ingest_all = override_ingest_all

        self.robust = robust
        self.prob_load = prob_load
        self.foresight = foresight
        self.write_model = write_model

        self.solver_params = solver_params or {}

        if self.symmetric:
            if self.my_tpuv4_symmetry is None:
                raise ValueError("symmetric=True requires my_tpuv4_symmetry")
            # force foresight when symmetric
            self.foresight = True

        if self.symmetric:
            self.src_set = set(self.my_tpuv4_symmetry.get_canonical_nodes())
        else:
            self.src_set = set(range(self.n_routers))

        print(f"[init] n_routers={self.n_routers} symmetric={self.symmetric} foresight={self.foresight}")

    # -----------------------------------------
    # Input ingestion
    # -----------------------------------------

    def ingest_path_list(self, filepath, lb_flow=None, ub_flow=None, src_set=None, override_ingest_all=False):

        print(f'Ingesting path list {filepath}')

        line_num = 0
        allpath_dict = defaultdict(list)
        for path in stream_raw_paths(filepath):

            s = path[0]
            d = path[-1]

            skip = False
            above_upper = False
            if lb_flow:
                (lb_s, lb_d) = lb_flow
                if s < lb_s:
                    skip = True
                    # print(f's<lb_s')
                elif s==lb_s and d < lb_d:
                    skip = True
                    # print(f's==lb_s and d < lb_d')
            if ub_flow:
                (ub_s, ub_d) = ub_flow
                if s > ub_s:
                    skip = True
                    above_upper = True
                    # print(f's > ub_s')
                elif s==ub_s and d >= ub_d:
                    skip = True
                    above_upper = True
                    # print(f's==ub_s and d > ub_d')
            
            if src_set:
                if s not in src_set:
                    skip = True
            

            if above_upper:
                return allpath_dict

            if skip and not override_ingest_all:
                continue
            
            allpath_dict[(s,d)].append(path)

            line_num += 1
            if line_num % 1_000_000 == 0:
                print(f'read {line_num}')

        return allpath_dict

    def ingest_and_setup_input_paths(self, lb_flow=None, ub_flow=None):
        """
        1) ingest paths
        2) merge backup paths if robust
        3) prune known-path flows (1 path)
        4) build current_edge_state from knowns and optional prob_load
        """
        mem_checkpoint("Before ingest")

        this_path_dict = self.ingest_path_list(
            self.allpaths_filepath,
            lb_flow=lb_flow,
            ub_flow=ub_flow,
            src_set=self.src_set,
            override_ingest_all=self.override_ingest_all,
        )

        if self.robust:
            backup_path_dict = self.ingest_path_list(
                self.backup_paths_filepath,
                lb_flow=lb_flow,
                ub_flow=ub_flow,
                src_set=self.src_set,
                override_ingest_all=self.override_ingest_all,
            )
            for flow, paths in backup_path_dict.items():
                for path in paths:
                    if path not in this_path_dict[flow]:
                        this_path_dict[flow].append(path)

        print(f"[ingest] flows={len(this_path_dict)}")
        mem_checkpoint("After ingest")

        known_paths = {}
        flows_to_remove = set()
        current_edge_state = defaultdict(float)

        # Identify knowns and (optional) probabilistic load
        for flow, paths in this_path_dict.items():
            s, d = flow

            if (not self.override_ingest_all) and (s not in self.src_set):
                continue

            n_paths = len(paths)

            if n_paths == 1:
                known_paths[flow] = paths[0]
                flows_to_remove.add(flow)
            elif self.prob_load:
                # fractional load contribution for warm-starting the channel load state
                for path in paths:
                    for i in range(len(path) - 1):
                        u = path[i]
                        v = path[i + 1]
                        current_edge_state[(u, v)] += (1.0 / n_paths)

        # Apply known-path loads to current_edge_state
        for (s, d), path in known_paths.items():
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                current_edge_state[(u, v)] += 1.0

            # If symmetric+foresight, your original code also adds transformed loads for known paths.
            # Keep that behavior.
            if self.symmetric and self.foresight:
                for s_out in self.my_tpuv4_symmetry.get_all_noncanonical_equivalents(s):
                    if s_out == s:
                        continue
                    tform = self.my_tpuv4_symmetry.calc_transform_delta(s, s_out)
                    tpath = [self.my_tpuv4_symmetry.apply_transformation(n, tform) for n in path]
                    for i in range(len(tpath) - 1):
                        u = tpath[i]
                        v = tpath[i + 1]
                        current_edge_state[(u, v)] += 1.0

        # Remove known flows from decision set
        for flow in flows_to_remove:
            this_path_dict.pop(flow, None)

        print(f"[setup] decision_flows={len(this_path_dict)} known_flows={len(known_paths)}")
        mem_checkpoint("After known-path pruning")
        return this_path_dict, current_edge_state, known_paths

    # -----------------------------------------
    # Edge indexing for constraints
    # -----------------------------------------

    def build_edge_path_index(self, path_dict):
        """
        Build edge -> list of (sr, dr, pidx) signatures.
        Also build foresight_edge_paths when enabled.
        This is based on your existing approach :contentReference[oaicite:4]{index=4}
        """
        edge_paths = defaultdict(list)
        foresight_edge_paths = defaultdict(list)

        for (sr, dr), paths in path_dict.items():
            for p_idx, path in enumerate(paths):
                sig = (sr, dr, p_idx)

                for i in range(len(path) - 1):
                    u = path[i]
                    v = path[i + 1]
                    edge_paths[(u, v)].append(sig)

                if self.foresight and self.symmetric:
                    # replicate loads for all noncanonical equivalents of sr
                    for sr_out in self.my_tpuv4_symmetry.get_all_noncanonical_equivalents(sr):
                        if sr_out == sr:
                            continue
                        tform = self.my_tpuv4_symmetry.calc_transform_delta(sr, sr_out)
                        tpath = [self.my_tpuv4_symmetry.apply_transformation(n, tform) for n in path]
                        for i in range(len(tpath) - 1):
                            u = tpath[i]
                            v = tpath[i + 1]
                            foresight_edge_paths[(u, v)].append(sig)

        mem_checkpoint("After edge-path index build")
        return edge_paths, foresight_edge_paths

    # -----------------------------------------
    # Model build & solve
    # -----------------------------------------

    def find_mclb(self, path_dict, current_edge_state):
        """
        Solve the MCLB MILP and return chosen paths for the *decision flows* in path_dict.
        """
        print("================================================================================")
        print("[solve] Finding MCLB")
        print(f"[solve] routers={self.n_routers} decision_flows={len(path_dict)}")

        edge_paths, foresight_edge_paths = self.build_edge_path_index(path_dict)

        m = gp.Model("mclb")

        # Parameters
        for k, v in self.solver_params.items():
            m.setParam(k, v)

        # Your earlier runs show model sizes can explode (tens of millions of rows) :contentReference[oaicite:5]{index=5}
        # so keep model output controlled.
        m.setParam("OutputFlag", 1)
        # very important
        # m.setParam('CrossoverBasis',1) # is this helpful?
        m.setParam('DegenMoves',0)
        m.setParam('MIPFocus',1)
        m.setParam('Method',2)

        # likely important
        m.setParam('Cuts', 2)
        m.setParam('Threads', 32)
        m.setParam('Heuristics',0.5)
        m.setParam('Symmetry',2)

        n_ports = len(self.adj_list[0])  # radix
        n_links = n_ports * self.n_routers
        n_total_flows = (self.n_routers * self.n_routers) - self.n_routers
        min_cload = n_total_flows // n_links

        max_cload = m.addVar(lb=min_cload, ub=n_total_flows, vtype=GRB.INTEGER, name="max_cload")

        # Decision variables: choose exactly one path per flow
        path_chosen = {}
        for (sr, dr), paths in path_dict.items():
            if len(paths) <= 1:
                # should be pruned before we get here
                print(f"SINGLE PATH")
                continue
            vars_for_flow = []
            for p_idx in range(len(paths)):
                vars_for_flow.append(m.addVar(vtype=GRB.BINARY, name=f"choose_{sr}_{dr}_{p_idx}"))
            path_chosen[(sr, dr)] = vars_for_flow

        m.update()
        mem_checkpoint("After vars", model=m)

        # Constraints
        # 1) One path per flow
        for (sr, dr), vars_for_flow in path_chosen.items():
            m.addConstr(gp.quicksum(vars_for_flow) == 1, name=f"onepath_{sr}_{dr}")

        # 2) Edge load <= max_cload for every directed edge in topology
        for u in range(self.n_routers):
            for v in self.adj_list[u]:
                expr = current_edge_state.get((u, v), 0.0)

                # canonical contributions
                for (sr, dr, p_idx) in edge_paths.get((u, v), []):
                    expr += path_chosen[(sr, dr)][p_idx]

                # foresight contributions
                if self.foresight and self.symmetric:
                    for (sr, dr, p_idx) in foresight_edge_paths.get((u, v), []):
                        expr += path_chosen[(sr, dr)][p_idx]

                m.addConstr(expr <= max_cload, name=f"cload_{u}_{v}")

        mem_checkpoint("After constraints", model=m)

        # Objective: minimize maximum channel load
        m.setObjective(max_cload, GRB.MINIMIZE)


        # # clear memory
        # path_dict = None
        # current_edge_state = None

        mem_checkpoint("After clearing memory", model=m)


        if self.write_model:
            # optionally write model for debugging
            m.write("mclb_model.lp")

        mem_checkpoint("Before optimize", model=m)
        m.optimize()
        mem_checkpoint("After optimize", model=m)

        if m.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
            raise RuntimeError(f"Gurobi failed: status={m.Status}")


        # return path_chosen

        # Decode chosen paths
        chosen_paths = {}
        for (sr, dr), vars_for_flow in path_chosen.items():
            p_star = None
            for p_idx, var in enumerate(vars_for_flow):
                if var.X > 0.5:
                    p_star = p_idx
                    break
            if p_star is None:
                raise RuntimeError(f"No chosen path for flow {sr}->{dr}")
            chosen_paths[(sr, dr)] = path_dict[(sr, dr)][p_star]
        return chosen_paths


    # -----------------------------------------
    # Symmetry expansion (FIXED)
    # -----------------------------------------

    # def handle_symmetry(self, all_chosen_paths):
    #     """
    #     Expand canonical routes to all sources.

    #     FIX: use inverse transform for dr_prime, and validate endpoints.
    #     The old code used forward transform for dr_prime :contentReference[oaicite:6]{index=6}
    #     """
    #     if not self.symmetric:
    #         return all_chosen_paths

    #     canonical_srcs = set(self.my_tpuv4_symmetry.get_canonical_nodes())

    #     for sr in range(self.n_routers):
    #         if sr in canonical_srcs:
    #             continue

    #         for dr in range(self.n_routers):
    #             if sr == dr:
    #                 continue

    #             # sr (outside) -> sr_prime (inside)
    #             sr_prime, sr_tform = self.my_tpuv4_symmetry.get_canonical_equivalent(sr)

    #             # sr -> sr_prime
    #             inv_tform = self.my_tpuv4_symmetry.calc_transform_delta(sr, sr_prime)
    #             dr_prime = self.my_tpuv4_symmetry.apply_transformation(dr, inv_tform)

    #             base_path = all_chosen_paths[(sr_prime, dr_prime)]
    #             new_path = [self.my_tpuv4_symmetry.apply_transformation(n, sr_tform) for n in base_path]

    #             # # Strong correctness checks (missing previously)
    #             # validate_path(self.adj_list, new_path, sr=sr, dr=dr)

    #             all_chosen_paths[(sr, dr)] = new_path

    #     return all_chosen_paths


    def handle_symmetry(self, all_chosen_paths):

        src_set = set(self.my_tpuv4_symmetry.get_canonical_nodes() )

        for sr in range(self.n_routers):
            for dr in range(self.n_routers):
                # will skip if not symmetric
                if sr in src_set:
                    continue
                if sr==dr:
                    continue

                sr_prime, sr_tform =self.my_tpuv4_symmetry.get_canonical_equivalent(sr)
                dr_prime = self.my_tpuv4_symmetry.apply_transformation(dr, sr_tform)

                base_path = all_chosen_paths[(sr_prime,dr_prime)]

                new_path = []
                for n in base_path:
                    n_prime = self.my_tpuv4_symmetry.apply_transformation(n, sr_tform)
                    new_path.append(n_prime)
                for i in range(len(new_path) - 1):

                    if not (new_path[i+1] in self.adj_list[new_path[i]]):
                        print(f"ERROR: base_path {base_path} => new_path {new_path} has issues")
                        for n in base_path:
                            print(f"\tbase_path {n} : {self.adj_list[n]}")
                        for n in new_path:
                            print(f"\tnew_path {n} : {self.adj_list[n]}")
                    assert( new_path[i+1] in self.adj_list[new_path[i]])
                    # assert(r_map[new_path[i]][new_path[i+1]] == 1)

                all_chosen_paths[(sr,dr)] = new_path
        return all_chosen_paths

    # -----------------------------------------
    # Public entry
    # -----------------------------------------

    def determine_chosen_paths(self,path_dict,path_chosen):

        # Decode chosen paths
        chosen_paths = {}
        for (sr, dr), vars_for_flow in path_chosen.items():
            p_star = None
            for p_idx, var in enumerate(vars_for_flow):
                if var.X > 0.5:
                    p_star = p_idx
                    break
            if p_star is None:
                raise RuntimeError(f"No chosen path for flow {sr}->{dr}")
            chosen_paths[(sr, dr)] = path_dict[(sr, dr)][p_star]
        return chosen_paths

    def solve(self):
        """
        Single-shot solve for all decision flows in the ingested file.
        """
        path_dict, current_edge_state, known_paths = self.ingest_and_setup_input_paths(lb_flow=None, ub_flow=None)

        # known_paths = None
        # var_path_chosen = self.find_mclb(path_dict, current_edge_state)
        # path_dict, current_edge_state, known_paths = self.ingest_and_setup_input_paths(lb_flow=None, ub_flow=None)
        # chosen_paths = self.determine_chosen_paths(path_dict, var_path_chosen)


        chosen_paths = self.find_mclb(path_dict, current_edge_state)


        # merge
        all_chosen_paths = {}
        all_chosen_paths.update(chosen_paths)
        all_chosen_paths.update(known_paths)

        # symmetry fill
        if self.symmetric:
            all_chosen_paths = self.handle_symmetry(all_chosen_paths)

        return all_chosen_paths

    def create_flat_pathlist(self, all_chosen_paths):
        """
        Your original create_flat_pathlist() structure
        but with endpoint validation.
        """
        flat = []
        for sr in range(self.n_routers):
            for dr in range(self.n_routers):
                if sr == dr:
                    flat.append([sr])
                    continue
                path = all_chosen_paths[(sr, dr)]
                # validate_path(self.adj_list, path, sr=sr, dr=dr)
                flat.append(path)
        return flat

    def print_max_cload(self, flat_route_pathlist):
        """
        Your original load checker :contentReference[oaicite:8]{index=8}
        """
        cload = defaultdict(int)
        max_load = 0
        max_edges = []

        for path in flat_route_pathlist:
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                cload[(u, v)] += 1
                if cload[(u, v)] > max_load:
                    max_load = cload[(u, v)]
                    max_edges = [(u, v)]
                elif cload[(u, v)] == max_load:
                    max_edges.append((u, v))

        print(f"[result] max_cload={max_load} throughput={1.0 / max_load if max_load > 0 else 0.0}")
        print(f"[result] example max edges (up to 5): {max_edges[:5]}")

    def run(self):
        routes = self.solve()
        flat = self.create_flat_pathlist(routes)
        self.print_max_cload(flat)
        return routes



# --------------------------------------------------------------------------------------
# Script functions
# --------------------------------------------------------------------------------------


def build_argparser():
    parser = argparse.ArgumentParser(description="Route MCLB (Max Channel Load Balancing)")

    # Inputs
    parser.add_argument("--topology", type=str, required=True,
                        help=".map file (adjacency matrix format)")
    parser.add_argument("--allpath_list", "-apl", type=str, required=True,
                        help="precomputed rallpaths file")

    # Routing modes
    parser.add_argument("--destination_based", action="store_true",
                        help="UNIMPLEMENTED in your code path")
    parser.add_argument("--robust", action="store_true",
                        help="include backup paths for fault tolerance")
    parser.add_argument("--backup_paths_list", type=str,
                        help="rallpaths backup/escape paths")
    parser.add_argument("--any_link_failure", action="store_true",
                        help="WFR-related; currently not used here")

    # Partitioning (kept for compatibility; symmetric forces no partitions)
    parser.add_argument("--partition_size", type=int, default=INF,
                        help="number of flows per partition")
    parser.add_argument("--partition_metric", type=str, default="flows",
                        choices=["flows", "paths"],
                        help="partition metric (flows supported)")

    # Symmetry
    parser.add_argument("--symmetric", action="store_true",
                        help="vertex-symmetric topology; solve canonical sources only")
    parser.add_argument("--sym_type", type=str, default="trans",
                        choices=["trans", "refl-trans"],
                        help="symmetry type")
    parser.add_argument("--xyzc_dims", nargs="+", type=int,
                        help="x y z cube_dim (4 ints)")
    parser.add_argument("--mc_dims", nargs="+", type=int,
                        help="mc_x mc_y mc_z (3 ints)")
    parser.add_argument("--override_ingest_all", action="store_true",
                        help="ingest all paths even if symmetry/partitioning would restrict it")
    parser.add_argument("--prob_load", action="store_true",
                        help="UNIMPLEMENTED in your current run path")
    parser.add_argument("--foresight", action="store_true",
                        help="apply canonical path choice as load to all equivalents")

    # Multi-failure modes (kept for compatibility)
    parser.add_argument("--wfr", action="store_true",
                        help="UNIMPLEMENTED here")
    parser.add_argument("--single_ocs_failure", action="store_true",
                        help="UNIMPLEMENTED here")

    # Output
    parser.add_argument("--out_dir", type=str, default="topologies_and_routing/routepath_lists",
                        help="directory to write *.paths output")
    parser.add_argument("--out_prefix", type=str, default=None,
                        help="override base output name (otherwise derived from allpath_list)")

    # Solver params (same as your script)
    parser.add_argument("--time_limit", type=int, help="minutes")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--concurrent_mip", type=int)
    parser.add_argument("--heuristic_ratio", type=float)
    parser.add_argument("--mip_focus", type=int)
    parser.add_argument("--symmetry_detection", type=int)
    parser.add_argument("--barrier_iter_limit", type=int)
    parser.add_argument("--iter_limit", type=int)
    parser.add_argument("--cut_passes", type=int)
    parser.add_argument("--method", type=int)
    parser.add_argument("--node_method", type=int)
    parser.add_argument("--crossover", type=int)
    parser.add_argument("--crossover_basis", type=int)
    parser.add_argument("--no_rel_heur_time", type=int)
    parser.add_argument("--presolve", type=int)
    parser.add_argument("--presparsify", type=int)
    parser.add_argument("--cuts", type=int)
    parser.add_argument("--scale_flag", type=int)
    parser.add_argument("--feas_tol", type=float)
    parser.add_argument("--degen_moves", type=int)
    parser.add_argument("--predual", type=int)

    parser.add_argument("--write_model", action="store_true",
                        help="write model to LP (debug)")
    parser.add_argument("--read_presolved", type=str,
                        help="UNSUPPORTED here; kept for compatibility")

    return parser


def derive_base_file_name(args):
    # Your script style: base on rallpaths filename
    apl = args.allpath_list
    base = os.path.basename(apl).replace(".rallpaths", "_mclb")

    if args.destination_based:
        base += "_destbased"

    if args.partition_size != INF and args.partition_size != -1:
        base += f"_{args.partition_size}ps_{args.partition_metric}pm"

    if args.symmetric:
        # keep naming convention
        mc_x, mc_y, mc_z = args.mc_dims
        base += f"_{args.sym_type}sym_{mc_x}x{mc_y}x{mc_z}mc"
        if args.foresight:
            base += "_foresight"

    if args.override_ingest_all:
        base += "_allpathsinfo"

    return base


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.destination_based:
        print("ERROR: --destination_based is not implemented in this runner.")
        sys.exit(1)

    if args.partition_metric != "flows":
        print("ERROR: only --partition_metric flows is supported.")
        sys.exit(1)

    # Robust requires backup list
    if args.robust and not args.backup_paths_list:
        print("ERROR: --robust requires --backup_paths_list")
        sys.exit(1)

    # Symmetry requires dims
    my_tpuv4_symmetry = None
    if args.symmetric:
        if not args.xyzc_dims or not args.mc_dims:
            print("ERROR: --symmetric requires --xyzc_dims (4 ints) and --mc_dims (3 ints)")
            sys.exit(1)

        xyzc_dims = tuple(args.xyzc_dims)
        mc_dims = tuple(args.mc_dims)
        if len(xyzc_dims) != 4 or len(mc_dims) != 3:
            print("ERROR: bad dims: xyzc_dims must be 4 ints; mc_dims must be 3 ints")
            sys.exit(1)

    # Solver params
    solver_params = setup_solver_params(args)

    # Ingest map correctly (IMPORTANT: ingest_map returns adj_mat, adj_list)
    adj_mat, adj_list = ingest_map(args.topology)
    n_routers = len(adj_list)
    print(f"[topology] n_routers={n_routers} radix={len(adj_list[0]) if n_routers > 0 else 0}")

    # Symmetry construction + verify
    if args.symmetric:
        xyzc_dims = tuple(args.xyzc_dims)
        mc_dims = tuple(args.mc_dims)

        # Must import TPUv4_Symmetry as in your script:
        # from tpuv4_symmetry import TPUv4_Symmetry
        my_tpuv4_symmetry = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type=args.sym_type)
        my_tpuv4_symmetry.verify_symmetry_for_topology(adj_mat, verify_dist=True)

        # Symmetric case: you usually do NOT partition (kept consistent with your old logic)
        if args.partition_size != INF and args.partition_size != -1:
            print("ERROR: partitioning is not supported with --symmetric in this runner.")
            sys.exit(1)

    # Derive output name
    base_file_name = args.out_prefix if args.out_prefix else derive_base_file_name(args)

    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------
    # RUN SOLVER
    # ------------------------------
    # You should already have your refactored class in the same file, e.g.:
    #   solver = MCLBSolver(...)
    #
    # IMPORTANT: use adj_list for real edges, and adj_mat only for validation checks.
    #
    # The solver should return a dict:
    #   all_chosen_paths[(sr,dr)] = [sr, ..., dr]
    #
    # And then you flatten it to a full list of paths for sr=0..N-1, dr=0..N-1.

    solver = MCLBSolver(
        adj_list=adj_list,
        allpaths_filepath=args.allpath_list,
        solver_params=solver_params,
        symmetric=args.symmetric,
        my_tpuv4_symmetry=my_tpuv4_symmetry,
        robust=args.robust,
        backup_paths_filepath=args.backup_paths_list,
        override_ingest_all=args.override_ingest_all,
        prob_load=args.prob_load,
        foresight=(args.foresight or args.symmetric),
        write_model=args.write_model,
    )

    all_chosen_paths = solver.solve()

    flat_route_pathlist = solver.create_flat_pathlist(all_chosen_paths)

    # Output pathlist
    output_pathlist(flat_route_pathlist, base_file_name, args.out_dir)

    # Print quality stats
    solver.print_max_cload(flat_route_pathlist)


if __name__ == "__main__":
    main()
