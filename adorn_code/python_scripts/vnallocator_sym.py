# Copyright (c) 2024 Purdue University
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Authors: Conor Green

"""

Description:
    Implementation of Nue (https://dl.acm.org/doi/10.1145/2907294.2907313)

    Uses "multilevel k-way partitioning algorithm"
        (https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=d545a4e5a5e00935e59141f26041531ba1aa97a0)

"""

# std
import argparse
import ast
import os
from collections import deque

# pipd
import orjson

# local
from omnicdg import OmniCDG
from routing import Routing

# symmetry (optional)
try:
    from tpuv4_symmetry import TPUv4_Symmetry
except Exception:
    TPUv4_Symmetry = None


class VNAllocator:

    # class vars
    ############

    verbose = False
    slow  = False

    INF = 999

    vc_mat_output_path_prefix = './topologies_and_routing/vc_mats'


    def __init__(self):

        # object vars
        #############

        # # topology vars
        # -------------
        self.n_routers = -1
        self.r_map = None

        # path vars
        # ---------
        # held in routing

        # chosen path vars
        # ----------------
        self.chosen_paths_flat = None
        self.chosen_paths_twod = None

        # Routing object
        # --------------
        self.my_Routing = None

        # name
        # ----
        self.base_name = None
        
        # symmetry (optional)
        # -------------------
        self.symmetric = False
        self.sym_expand = False  # if True, expects canonical-source input and expands to all equivalent paths
        self.my_tpuv4_symmetry = None
        self.canonical_nodes = None
        self.canonical_node_set = None
        # cache: canonical_path_tuple -> vcs_along_path list
        self._canon_path_to_vcs_cache = {}

    
    # setups
    ####################################################################################################

    def setup_w_apl(self, apl_file_path):

        if self.my_Routing is None:
            self.my_Routing = Routing()
        self.my_Routing.setup_given_all_path_list(apl_file_path)

        # required values
        if self.n_routers == -1:
            self.n_routers = self.my_Routing.get_n_routers()
        if self.r_map is None:
            self.r_map = self.my_Routing.r_map
        self.base_name = self.my_Routing.get_base_name()

    def setup_w_rmap(self, rmap_file_path, binary_r_map=False):

        if self.my_Routing is None:
            self.my_Routing = Routing()
        self.my_Routing.setup_given_r_map(rmap_file_path, binary_r_map=False)

        # required values
        if self.n_routers == -1:
            self.n_routers = self.my_Routing.get_n_routers()
        self.base_name = self.my_Routing.get_base_name()
        self.r_map = self.my_Routing.r_map


    # def setup(self, rmap_file_path, apl_file_path, binary_r_map=False):

    #     self.my_Routing = Routing()
    #     self.my_Routing.setup_given_both(rmap_file_path, apl_file_path, binary_r_map=False)

    def setup_w_pl(self, pl_file_path):

        # if self.my_Routing is None:
        #     self.my_Routing = Routing()
        # self.my_Routing.setup_given_path_list( pl_file_path)

        # required values
        # if self.n_routers == -1:
        #     self.n_routers = self.my_Routing.get_n_routers()
        # if self.r_map is None:
        #     self.r_map = self.my_Routing.r_map
        
        self.chosen_paths_pathname = pl_file_path
        # self.chosen_paths_flat, self.chosen_paths_twod = self.my_Routing.get_chosen_paths()
        # self.base_name = self.my_Routing.get_base_name()

        self.base_name = pl_file_path.split('/')[-1].replace('.paths','')

    def ingest_allowed_turns(self, atv_file_path):
        self.allowed_turns_w_vcs_dict = self.ingest_an_allowed_turns_(atv_file_path)

    @classmethod
    def ingest_an_allowed_turns_(cls, path_name):
        atvcs_dict = {}

        with open(path_name, 'r') as inf:
            for line in inf.readlines():
                line_no_newline = line.strip('\n')
                line_w_curly = f'{{ {line_no_newline} }}'
                as_dict = ast.literal_eval(line_w_curly)
                atvcs_dict.update(as_dict)
        return atvcs_dict

    def stream_pathlist(self, path):

        with open(path, "r", buffering=1024*1024) as inf:
            next(inf, None)  # skip header
            for line in inf:
                line = line.strip()
                if line:
                    yield orjson.loads(line)  # produce one row at a time

    # getters and setters
    ####################################################################################################

    def set_xyzc_dims(self, dims_tuple):

        self.x_dim, self.y_dim, self.z_dim, self.cube_dim = dims_tuple
        self.dim_str_dict = {'x':self.x_dim, 'y':self.y_dim, 'z':self.z_dim}

    def get_base_name(self):
        bn = self.base_name
        if self.online_load_balancing:
            bn = f"{bn}_olb"
        return bn

    # symmetry
    ####################################################################################################
    def setup_symmetry(self, xyzc_dims, mc_dims=None, sym_type='trans', expand_equivalents=False):
        """
        Enables symmetry lifting via TPUv4_Symmetry.

        If expand_equivalents=True, the allocator expects the *input* pathlist to contain only canonical-source flows
        (source in canonical set) and it will write VC decisions for *all* symmetric-equivalent paths.
        """
        if TPUv4_Symmetry is None:
            raise RuntimeError('TPUv4_Symmetry could not be imported; ensure tpuv4_symmetry.py is on PYTHONPATH')

        self.symmetric = True
        self.sym_expand = bool(expand_equivalents)

        if mc_dims is None:
            # default: a single cube (this matches TPUv4_Symmetry defaults)
            (x_dim, y_dim, z_dim, cube_dim) = tuple(xyzc_dims)
            mc_dims = (cube_dim, cube_dim, cube_dim)

        self.my_tpuv4_symmetry = TPUv4_Symmetry(tuple(xyzc_dims), mc_dims=tuple(mc_dims), sym_type=sym_type)
        self.canonical_nodes = self.my_tpuv4_symmetry.get_canonical_nodes()
        self.canonical_node_set = set(self.canonical_nodes)

    def _canonicalize_path(self, path):
        """Return canonicalized path using the transform induced by mapping src -> canonical(src)."""
        if not self.symmetric or self.my_tpuv4_symmetry is None:
            return path, None, None

        src = path[0]
        src_prime, src_tform = self.my_tpuv4_symmetry.get_canonical_equivalent(src)

        if src_prime == src:
            return path, src_prime, src_tform

        canon_path = [self.my_tpuv4_symmetry.apply_transformation(n, src_tform) for n in path]
        # basic sanity
        if canon_path[0] != src_prime:
            # fall back to original if something is inconsistent
            return path, src_prime, src_tform
        return canon_path, src_prime, src_tform

    def _get_vcs_for_canonical_path(self, canon_path, vc_hop_load=None, vc_path_load=None, mult=1):
        """
        Compute (or reuse) vcs_along_path for a canonical path. Optionally updates load vectors by multiplier mult.
        """
        key = tuple(canon_path)
        if key in self._canon_path_to_vcs_cache:
            vcs_along_path = self._canon_path_to_vcs_cache[key]
            return vcs_along_path

        if self.online_load_balancing and vc_hop_load is not None:
            _, vcs_along_path = self.calc_vcs_of_path_balanced(canon_path, vc_hop_load, vc_path_load)
            if vcs_along_path is None:
                return None
        else:
            _, vcs_along_path = self.calc_vcs_of_path(canon_path)
            if vcs_along_path is None:
                return None

        self._canon_path_to_vcs_cache[key] = vcs_along_path

        # print(f"returning from _get_vcs_for_canonical_path")
        return vcs_along_path

    def _update_vc_loads_mult(self, path_vcs, vc_hop_load, vc_path_load, mult=1):
        seen_vcs_in_path = set()
        for vc in path_vcs:
            vc_hop_load[vc] += mult
            seen_vcs_in_path.add(vc)
        for vc in seen_vcs_in_path:
            vc_path_load[vc] += mult
        return vc_hop_load, vc_path_load

    # vc management along a path
    ####################################################################################################

    def update_vc_loads_for_path(self, path_vcs, vc_hop_load, vc_path_load):
        # path_vcs length = number_of_hops + 1; VC per channel hop.
        # Count hops per VC (each hop uses the VC of the outgoing channel).
        seen_vcs_in_path = set()
        for vc in path_vcs:
            vc_hop_load[vc] += 1
            seen_vcs_in_path.add(vc)
        for vc in seen_vcs_in_path:
            vc_path_load[vc] += 1

        return vc_hop_load, vc_path_load

    def calc_vcs_of_path_balanced(self, path, vc_hop_load, vc_path_load=None, seed=None):
        """
        Finds ONE feasible VC assignment for 'path', but orders VC exploration to balance global VC usage.

        vc_hop_load: list length max_n_vcs, global hop-count load per VC (mutated by caller AFTER acceptance).
        vc_path_load: optional list length max_n_vcs, global path-count load per VC (if you want).
        seed: optional; if provided, breaks ties pseudo-randomly but repeatably.

        Returns: (a_path_of_turns, path_vcs)
        """
        assert self.max_n_vcs is not None
        assert self.allowed_turns_w_vcs_dict is not None
        max_n_vcs = self.max_n_vcs
        allowed_turns_dict = self.allowed_turns_w_vcs_dict
        verbose = self.verbose

        path_len = len(path) - 1
        if path_len <= 1:
            return [], [0]

        if vc_path_load is None:
            vc_path_load = [0] * max_n_vcs

        # Small deterministic tie-break; optional light randomization if you want it.
        # (Avoid importing random unless you want randomness; deterministic is often fine.)
        def vc_cost(vc):
            # Weight hops more than path-count by default; tune as you like.
            return (vc_hop_load[vc] * 10) + vc_path_load[vc]

        if verbose:
            print('-' * 20)
            print(f'path={path}')

        dest = path[-1]
        queue = deque()

        # ----- Seed the queue with first turn(s), but ordered by global load -----
        # Your original code loops vc_1 then vc_2 in ascending order. :contentReference[oaicite:1]{index=1}
        # We instead build candidates and sort by load.
        first_turn_candidates = []
        for vc_1 in range(max_n_vcs):
            for vc_2 in range(max_n_vcs):
                channel_a_w_vc = (path[0], path[1], vc_1)
                channel_b_w_vc = (path[1], path[2], vc_2)
                turn_w_vc = (channel_a_w_vc, channel_b_w_vc)
                if allowed_turns_dict.get(turn_w_vc, False):
                    # Cost uses both VCs since both are now committed for the first 2 hops.
                    cost = vc_cost(vc_1) + vc_cost(vc_2)
                    first_turn_candidates.append((cost, vc_1, vc_2, turn_w_vc))

        if not first_turn_candidates:
            return None, None  # or raise; depends on your calling conventions

        first_turn_candidates.sort(key=lambda x: (x[0], x[1], x[2]))

        for _, _, _, turn_w_vc in first_turn_candidates:
            queue.append([turn_w_vc])

        a_path_of_turns = None

        q_iters = -1
        while queue:
            q_iters += 1
            # print(f"q_iters = {q_iters}")
            list_of_turns = queue.popleft()

            most_recent_turn = list_of_turns[-1]
            most_recent_node = most_recent_turn[1][1]
            before_most_recent_node = most_recent_turn[0][1]

            if verbose:
                print(f'list_of_turns={list_of_turns}, most_recent_turn={most_recent_turn}, most_recent_node={most_recent_node}')

            if most_recent_node == dest:
                a_path_of_turns = list_of_turns
                break

            cur_loc = path.index(most_recent_node)
            next_node = path[cur_loc + 1]
            cur_vc = most_recent_turn[1][2]

            # ----- Expand next_vc in order of least-loaded VC first -----
            next_vc_candidates = []
            for next_vc in range(max_n_vcs):
                channel_a_w_vc = (before_most_recent_node, most_recent_node, cur_vc)
                channel_b_w_vc = (most_recent_node, next_node, next_vc)
                turn_w_vc = (channel_a_w_vc, channel_b_w_vc)

                if allowed_turns_dict.get(turn_w_vc, False):
                    next_vc_candidates.append((vc_cost(next_vc), next_vc, turn_w_vc))

            next_vc_candidates.sort(key=lambda x: (x[0], x[1]))

            for _, _, turn_w_vc in next_vc_candidates:
                queue.append(list_of_turns + [turn_w_vc])

        # print(f"outside queue. a_path_of_turns = {a_path_of_turns}")
        if a_path_of_turns is None:
            return None, None

        # Deconstruct to VC list exactly like your original logic. :contentReference[oaicite:2]{index=2}
        path_vcs = []
        first_vc = a_path_of_turns[0][0][2]
        path_vcs.append(first_vc)
        for turn in a_path_of_turns:
            path_vcs.append(turn[1][2])

        # print(f"returning. path_vcs = {path_vcs}")

        return a_path_of_turns, path_vcs


    def calc_vcs_of_path(self, path):
        assert(self.max_n_vcs is not None)
        assert(self.allowed_turns_w_vcs_dict is not None)
        max_n_vcs = self.max_n_vcs
        allowed_turns_dict = self.allowed_turns_w_vcs_dict
        verbose = self.verbose

        path_len = len(path) - 1
        # any path of length 1 (e.g. [n0,n1]) or less valid
        if path_len <= 1:
            return [0]

        if verbose:
            print(f'-'*20)
            print(f'path={path}')

        dest = path[-1]
        queue = deque()
        # list of paths
        # list of list of turns
        # list of list (paths) of tuples (turns)
        vcs_of_paths_list = []
        # get first turns of path (if valid)
        for vc_1 in range(max_n_vcs):
            for vc_2 in range(max_n_vcs):
                channel_a_w_vc = (path[0],path[1], vc_1)
                channel_b_w_vc = (path[1],path[2], vc_2)
                turn_w_vc = (channel_a_w_vc,channel_b_w_vc)

                if allowed_turns_dict.get(turn_w_vc, False):
                    # queue holds paths of turns
                    new_q_obj = [turn_w_vc]
                    queue.append(new_q_obj)
                if verbose:
                    print(f'turn {turn_w_vc} allowed? {allowed_turns_dict.get(turn_w_vc, False)}')

        a_path_of_turns = None
        while queue:
            list_of_turns = queue.popleft()

            most_recent_turn = list_of_turns[-1]
            most_recent_node = most_recent_turn[1][1]
            before_most_recent_node = most_recent_turn[0][1]

            if verbose:
                print(f'list_of_turns={list_of_turns}, most_recent_turn={most_recent_turn}, most_recent_node={most_recent_node}')

            # arrived
            if most_recent_node == dest:
                a_path_of_turns = list_of_turns
                break

            # else continue searching
            cur_loc = path.index(most_recent_node)
            next_node = path[cur_loc + 1]
            cur_vc =  most_recent_turn[1][2]
            for next_vc in range(max_n_vcs):
                channel_a_w_vc = ( before_most_recent_node,most_recent_node, cur_vc)
                channel_b_w_vc = (most_recent_node, next_node, next_vc)        
                turn_w_vc = (channel_a_w_vc,channel_b_w_vc)

                if allowed_turns_dict.get(turn_w_vc, False):
                    new_list_of_turns = list_of_turns + [turn_w_vc]
                    queue.append(new_list_of_turns)
                
                if verbose:
                    print(f'turn {turn_w_vc} allowed? {allowed_turns_dict.get(turn_w_vc, False)}')



        if a_path_of_turns is None:
            # no feasible VC assignment found for this path under current allowed turns
            return None, None

        # deconstruct
        path_vcs = []
        # manually set first
        # print(f"a_path_of_turns = {a_path_of_turns}")
        first_vc = a_path_of_turns[0][0][2]
        path_vcs.append(first_vc)
        for turn in a_path_of_turns:
            turn_vc = turn[1][2]
            path_vcs.append(turn_vc)

        if verbose:
            input(f'a_path_of_turns = {a_path_of_turns} w/ vcs {path_vcs}')

        return a_path_of_turns, path_vcs

    # big worker(s)
    ####################################################################################################
    def alloc_allowed_turns_w_vcs(self):

        # assert(self.chosen_paths_flat is not None)
        assert(self.allowed_turns_w_vcs_dict is not None)
        # assert(self.n_routers != -1)
        assert(self.max_n_vcs != -1)
        # original_flat_path_list = self.chosen_paths_flat
        pathlist_filepath = self.chosen_paths_pathname
        allowed_turns_w_vcs_dict = self.allowed_turns_w_vcs_dict
        # n_routers = self.n_routers
        max_n_vcs = self.max_n_vcs

        online_load_balancing = self.online_load_balancing

        verbose = self.verbose

        print('\n' + '='*100)
        print('Alloc allowed turns w vcs' )
        print('-------------------------')



        # vc a function of flow and current router
        #   ie 3d (src,dest,cur)
        # vc_matrix = [ [ [ -1 for _ in range(n_routers)  ] for __ in range(n_routers) ] for ___ in range(n_routers)]

        # vc_list = []


        out_name_base = self.get_base_name()

        out_name = f'{out_name_base}.vcmat2'
        # if cla_out_name is not None:
        #     out_name = f'{cla_out_name}.vcmat2'

        out_name_path = os.path.join(self.vc_mat_output_path_prefix, out_name)

        # input(f'Will write to {out_name_path}')

        with open(out_name_path,'w+') as of:
            pass


        p_iter = -1
        # n_paths = len(original_flat_path_list)

        vc_hop_load = [0 for _ in range(max_n_vcs)]
        vc_path_load = [0 for _ in range(max_n_vcs)]

        # for path in original_flat_path_list:
        with open(out_name_path,'a') as of:
            # Symmetry options:
            #   (A) symmetric + expand: input contains only canonical-source paths; we expand to all equivalent paths
            #   (B) symmetric only: cache VC decisions on canonicalized paths, but emit entries only for the input paths
            if self.symmetric and self.sym_expand:
                if self.my_tpuv4_symmetry is None:
                    raise RuntimeError('Symmetry is enabled but TPUv4_Symmetry is not initialized. Call setup_symmetry().')

                for path in self.stream_pathlist(pathlist_filepath):

                    path_src = path[0]
                    plen = len(path) - 1

                    # trivial paths
                    if plen <= 1:
                        out_line = (path_src, path_src, path_src, 0)
                        of.write(str(out_line) + '\n')
                        p_iter += 1
                        continue

                    if self.canonical_node_set is not None and path_src not in self.canonical_node_set:
                        # In expand mode, we expect canonical sources; skip to avoid duplicating work incorrectly.
                        continue

                    if p_iter % 1 == 0 or True:
                        print(f'paths {p_iter}')

                    # compute VC assignment once for the canonical path
                    vcs_along_path = self._get_vcs_for_canonical_path(path, vc_hop_load, vc_path_load)
                    if vcs_along_path is None:
                        print(f"ERROR: No VC assignment found. Exiting...")
                        quit()
                        p_iter += 1
                        continue

                    equivalent_paths = self.my_tpuv4_symmetry.get_all_equivalent_paths(path)

                    # update global loads weighted by number of equivalent flows
                    if online_load_balancing:
                        mult = len(equivalent_paths)
                        vc_hop_load, vc_path_load = self._update_vc_loads_mult(vcs_along_path, vc_hop_load, vc_path_load, mult=mult)

                    print(f"applying to equivalents")
                    # emit VC decisions for every equivalent path
                    for new_path in equivalent_paths:
                        new_src = new_path[0]
                        new_dest = new_path[-1]
                        for loc, vc in enumerate(vcs_along_path):
                            cur_node = new_path[loc]
                            out_line = (new_src, new_dest, cur_node, vc)
                            of.write(str(out_line) + '\n')

                    p_iter += len(equivalent_paths)
                    print(f"Found VCs for {path}")

            else:
                for path in self.stream_pathlist(pathlist_filepath):
                    if p_iter % 10 == 0:
                        print(f'paths {p_iter}')

                    path_src = path[0]
                    path_dest = path[-1]

                    plen = len(path) - 1
                    if plen <= 1:
                        out_line = (path_src, path_dest, path_src, 0)
                        of.write(str(out_line) + '\n')
                        p_iter += 1
                        continue

                    # In symmetry-cache mode, canonicalize the path w.r.t. src and reuse VC decisions.
                    if self.symmetric:
                        canon_path, _, _ = self._canonicalize_path(path)
                        vcs_along_path = self._get_vcs_for_canonical_path(canon_path, vc_hop_load, vc_path_load)
                    else:
                        if online_load_balancing:
                            _, vcs_along_path = self.calc_vcs_of_path_balanced(path, vc_hop_load)
                        else:
                            _, vcs_along_path = self.calc_vcs_of_path(path)

                    if vcs_along_path is None:
                        p_iter += 1
                        continue

                    if online_load_balancing and not self.symmetric:
                        vc_hop_load, vc_path_load = self.update_vc_loads_for_path(vcs_along_path, vc_hop_load, vc_path_load)
                    elif online_load_balancing and self.symmetric:
                        vc_hop_load, vc_path_load = self._update_vc_loads_mult(vcs_along_path, vc_hop_load, vc_path_load, mult=1)

                    for loc, vc in enumerate(vcs_along_path):
                        cur_node = path[loc]
                        out_line = (path_src, path_dest, cur_node, vc)
                        of.write(str(out_line) + '\n')

                    p_iter += 1
        print(f'Completed allocation to {out_name_path}')


        # verify correctness

        # print('\n' + '='*100)
        # print('Completed allocation. Checking for deadlock(s)' )
        # print('----------------------------------------------')

        # return vc_matrix

        # total_CDG = OmniCDG()
        # total_CDG.init_w_n_nodes(n_routers, n_vcs=2)

        # for path in original_flat_path_list:

        #     plen = len(path) - 1
        #     n_turns = plen - 1
        #     if n_turns <= 0:
        #         continue
            
        #     path_src = path[0]
        #     path_dest = path[-1]
        #     for i in range(n_turns):
        #         # let turn = a -> b -> c
        #         node_a = path[i]
        #         node_b = path[i+1]
        #         node_c = path[i+2]
        #         vc_ab = vc_matrix[path_src][path_dest][node_a]
        #         vc_bc = vc_matrix[path_src][path_dest][node_b]
        #         channel_a_w_vc = (node_a, node_b, vc_ab)
        #         channel_b_w_vc = (node_b, node_c, vc_bc)
        #         turn_w_vc = (channel_a_w_vc, channel_b_w_vc)
        #         total_CDG.add_turn(turn_w_vc)
        
        # cdg_cycles = total_CDG.networkx_get_cycle()

        # print(f'All VCs w/ cycles {cdg_cycles}')

        # if len(cdg_cycles) > 0:
        #     input('DEADLOCK?????')

        # else:        
        #     print(f'\tDeadlock free')
        
        # # input('complete?')
        
        # return vc_matrix

    # output
    ####################################################################################################
    def output_sequence(self, vc_matrix, vn_alg=None, cla_out_name=None):

        print('\n' + '='*100)
        print('Printing vcmat to file' )
        print('----------------------')


        out_name_base = self.get_base_name()
        out_name = f'{out_name_base}_{vn_alg}.vcmat'
        if cla_out_name is not None:
            out_name = f'{cla_out_name}.vcmat'

        out_name_path = os.path.join(self.vc_mat_output_path_prefix, out_name)

        self.output_vc_matrix(vc_matrix, out_name_path)

    def output_vc_matrix(self, vc_mat, out_name):
        assert(self.n_routers != -1)

        n_routers = self.n_routers

        with open(out_name,'w+') as of:
            for i in range(n_routers):
                for j in range(n_routers):

                    for e in vc_mat[i][j]:
                        l = f'{e} '
                        of.write(l)
                    of.write('\n')

        print(f'Wrote {out_name}')

# drivers
####################################################################################################

def drive_vnalloc(input_dict):

    my_VNAllocator = VNAllocator()

    # class variables
    if input_dict['vc_mat_dir'] is not None:
        my_VNAllocator.vc_mat_output_path_prefix = input_dict['vc_mat_dir']
    if input_dict['verbose']:
        my_VNAllocator.verbose = True

    # setups
    pl_name = input_dict['pl_name']
    my_VNAllocator.setup_w_pl(pl_name)
    atv_name = input_dict['atv_name']
    my_VNAllocator.ingest_allowed_turns(atv_name)


    max_n_vcs = input_dict['max_n_vcs']
    my_VNAllocator.max_n_vcs = max_n_vcs

    online_load_balancing = input_dict["load_balance"]
    my_VNAllocator.online_load_balancing = online_load_balancing

    # symmetry (optional)
    if input_dict.get('symmetric', False):
        xyzc_dims = input_dict.get('xyzc_dims', None)
        mc_dims = input_dict.get('mc_dims', None)
        sym_type = input_dict.get('sym_type', 'trans')
        expand_equivalents = input_dict.get('expand_equivalents', False)

        if xyzc_dims is None:
            raise RuntimeError('Symmetry requested but xyzc_dims not provided')

        my_VNAllocator.setup_symmetry(xyzc_dims, mc_dims=mc_dims, sym_type=sym_type, expand_equivalents=expand_equivalents)


    vc_matrix = None
    # if 'allowed_turns' in vn_alg:
    vc_matrix = my_VNAllocator.alloc_allowed_turns_w_vcs()


    # output
    # my_VNAllocator.output_sequence(vc_matrix, vn_alg=vn_alg, cla_out_name=input_dict['out_name'])



# main
####################################################################################################


def main():

    parser = argparse.ArgumentParser(description='Verify topology values')
    parser.add_argument('--chosen_paths_list','-cpl',type=str,help='.paths file to evaluate', required=True)
    parser.add_argument('--allowed_turns_w_vcs','-atv',type=str,help='.allowvcturns file to evaluate')
    parser.add_argument('--max_n_vcs','-mvcs',default=2,type=int,help='# vcs')

    parser.add_argument('--load_balance',action='store_true')

    # symmetry (optional)
    parser.add_argument('--symmetric', action='store_true', help='use symmetry to cache/expand VC allocation')
    parser.add_argument('--expand_equivalents', action='store_true', help='(requires --symmetric) assume input contains only canonical-source paths; expand to all equivalent paths in output')
    parser.add_argument('--xyzc_dims', nargs='+', type=int, help='global system x y z cube dims (required if --symmetric)')
    parser.add_argument('--mc_dims', nargs='+', type=int, help='mega cube dims x y z (optional)')
    parser.add_argument('--sym_type', type=str, choices=['trans','refl-trans'], default='trans', help='symmetry type for TPUv4_Symmetry')



    # parser.add_argument('--vn_alg','-va',default='fg_trans',type=str,help='which alg for vnalloc')
    parser.add_argument('--out_name','-o',type=str,help='output name (without extension)')

    # parser.add_argument('--is_tpuv4',action='store_true')
    # parser.add_argument('--xyzc_dims',nargs='+',type=int,help='type without parenthesis and use spaces, no commas')

    parser.add_argument('--verbose','-v',action='store_true',help='verbsoe for debugging')
    parser.add_argument('--vc_mat_dir',type=str,help='directory to output vc_mat')


    args = parser.parse_args()

    pl_name = args.chosen_paths_list
    atv_name = args.allowed_turns_w_vcs
    out_name = args.out_name
    max_n_vcs = args.max_n_vcs
    load_balance = args.load_balance

    symmetric = args.symmetric
    expand_equivalents = args.expand_equivalents
    xyzc_dims = tuple(args.xyzc_dims) if args.xyzc_dims else None
    mc_dims = tuple(args.mc_dims) if args.mc_dims else None
    sym_type = args.sym_type



    input_dict = {
                    'pl_name':pl_name,
                    'atv_name':atv_name,
                    'out_name':out_name,
                    'max_n_vcs':max_n_vcs,
                    "load_balance":load_balance,
                    'symmetric':symmetric,
                    'expand_equivalents':expand_equivalents,
                    'xyzc_dims':xyzc_dims,
                    'mc_dims':mc_dims,
                    'sym_type':sym_type,


                    'verbose':args.verbose,
                    'vc_mat_dir':args.vc_mat_dir
                    }
    
    drive_vnalloc(input_dict)


if __name__ == '__main__':
    main()