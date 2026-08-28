

'''

For ...


Based on https://dl.acm.org/doi/pdf/10.1145/77600.77620

'''

# Gurobi libs
import gurobipy as gp
from gurobipy import GRB

# regular libs
import argparse
from collections import deque
import networkx as nx
import ast
import os
import time

# constants
VERBOSE = False # for all
ASSERT_BINARY_MAP = True
INF = 999999 # for FW

# Regular Functions
# --------------------------------------------------------------------------------

def get_shape(nested_list):
    if isinstance(nested_list, list):
        return [len(nested_list)] + get_shape(nested_list[0])
    else:
        return []


def ingest_map(path_name):
    file_name = path_name.split('/')[-1]

    if VERBOSE:
        print(f'Ingesting filename = {file_name} ({path_name})')

    this_map = []

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

            this_map.append(r_conns)


    # quick sanitization
    n_routers = len(this_map)
    for i in range(n_routers):
        this_map[i][i] = 0

    # assert binary?
    if ASSERT_BINARY_MAP:
        for src_map in this_map:
            for conn in src_map:
                # instead, make binary
                if conn > 0.1:
                    conn = 1
                # assert(conn == 1 or conn == 0)

    if VERBOSE:
        print(f'read {this_map}')

    return this_map


def print_path(p):

    print(f'path {p[0]} to {p[-1]} (len {len(p)-1}): ',end='')

    l = len(p)
    for i in range(l-1):
        e = p[i]
        print(f'{e}->',end='')
    print(f'{p[-1]}')


def floyd(r_map):

    n_routers = len(r_map)

    if VERBOSE:
        print(f'begin floyd. n_routers={n_routers}')

    graph = [[item if item==1 else INF for item in row] for row in r_map]

    for i in range(0,n_routers):
        graph[i][i]=0

    dist = list(map(lambda p: list(map(lambda q: q, p)), graph))

    for r in range(n_routers):
        for p in range(n_routers):
            for q in range(n_routers):
                # shorter path through r
                if (dist[p][r]+ dist[r][q]) < dist[p][q]:
                    dist[p][q] = dist[p][r] + dist[r][q]


    if VERBOSE:
        print(f'dists = {dist}')

    return dist

def calculate_min_hop_paths(r_map, hop_dists):

    n_routers = len(r_map)

    short_paths = []

    for src in range(n_routers):

        short_paths.append([])

        if VERBOSE:
            print(f'Min hop paths for src {src}')

        for dest in range(n_routers):
            short_paths[src].append([])

            this_path_list = []

            if src == dest:
                # path is nonexistent
                this_path_list.append(src)
                short_paths[src][dest].append(this_path_list)
                continue


            # perform psuedo-BFS

            shortest_dist = hop_dists[src][dest]

            if VERBOSE:
                print(f'Searching for path {src}->{dest} of dist {shortest_dist}')


            queue = deque()

            path = []
            path.append(src)
            queue.append(path.copy())

            while queue:
                path = queue.popleft()
                last = path[-1]

                # only consider the minimal paths
                if len(path) - 1 > shortest_dist:
                    if VERBOSE:
                        print(f'path {path} (len {len(path)}) > shortest {shortest_dist}')
                    continue

                if last == dest:
                    this_path_list.append(path)
                    if VERBOSE:
                        print_path(path)

                for i in range(n_routers):

                    # only consider neighbors
                    if r_map[last][i] == 0:
                        continue

                    # if self.is_not_visited(i, path):
                    if not i in path:
                        new_path = path.copy()
                        new_path.append(i)
                        queue.append(new_path)


            short_paths[src][dest] = this_path_list.copy()

            if VERBOSE:
                print(f'Found {src}->{dest}={this_path_list}')

            # end dest loop

    if VERBOSE:
        print(f'done with min hop paths')

    if VERBOSE:
        for i, src_paths in enumerate(short_paths):
            print(f'{i}->')
            for j, paths in enumerate(src_paths):
                print(f'\t{j} : {paths}')

    return short_paths

def ingest_path_list(path_name, n_routers):

    if VERBOSE:
        print(f'Ingesting path list {path_name}')

    flat_path_list = []


    with open(path_name, 'r') as inf:

        line_num = 0

        for line in inf.readlines():
            # if '[' not in line:
            #     name = inf.readline()
            #     continue
            # print(f'line (type {type(line)}) = {line}')

            line_no_newline = line.strip('\n')

            line_w_commas =  ','.join(line_no_newline.split(' '))

            line_w_square_brackets = f'[ {line_w_commas} ]'

            as_list = ast.literal_eval(line_w_square_brackets)
            clean_as_list = [e for e in as_list]

            flat_path_list.append(clean_as_list)

            # if line_num % 1000 == 0:
            #     print(f'read {line_num}')

            line_num += 1

    allpath_list = [[ [] for _ in range(n_routers) ] for __ in range(n_routers)]

    for path in flat_path_list:
        s = path[0]
        d = path[-1]
        allpath_list[s][d].append(path)

    return allpath_list.copy()


def create_nwx_G_from_map(r_map, n_routers):


    # directed =  False

    G = nx.DiGraph()

    for src in range(n_routers):
        for dest in range(n_routers):

            if(src == dest):
                continue

            # if not directed and src > dest:
            #     continue

            if(r_map[src][dest] < 1):
                continue

            # print(f'connecting {src} -> {dest}')

            G.add_edge(src,dest)

    return G

def nwx_all_shortest_paths(r_map):

    n_routers = len(r_map)

    G = create_nwx_G_from_map(r_map, n_routers)

    all_paths = [ [ [] for _ in range(n_routers)] for __ in range(n_routers)]

    for src in range(n_routers):
        for dest in range(n_routers):

            if(src == dest):
                all_paths[src][dest].append([src])
                continue

            short_path_generator = nx.all_shortest_paths(G,src,dest)
            short_path_list = list()
            short_path_list += short_path_generator

            # input(f'{src}->{dest} : {short_path_generator} = {short_path_list}')
            all_paths[src][dest] = short_path_list

        # print(f'completed src {src}')

    print(f'Completed all short path creation')

    return all_paths

def output_pathlist( path_list, base_file_name, paths_output_path_prefix):

    full_name = f'{base_file_name}.paths'

    full_out_path = os.path.join(paths_output_path_prefix, \
            full_name)

    with open(full_out_path, 'w+') as of:
        of.write(base_file_name + '\n')
        for path in path_list:
            of.write(f'{path}\n')

    print(f'Wrote to {full_out_path}')


# Gurobi Functions
# --------------------------------------------------------------------------------

# Main(s)
# --------------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description='Verify topology values')
    parser.add_argument('--graph','-g',type=str,help='.map file to evaluate',default='files/map_files/example_6r_25ll.map')
    parser.add_argument('--allpath_list','-apl',type=str,help='shortcut over path creation')
    parser.add_argument('--destination_based',action='store_true',help='all paths with same destination follow same next router')

    parser.add_argument('--source_start',type=int,help='only paths with source node >= this val',default=0)
    parser.add_argument('--source_end',type=int,help='only paths with source node < this val',default=INF)
    parser.add_argument('--destination_start',type=int,help='only paths with destination node >= this val',default=0)
    parser.add_argument('--destination_end',type=int,help='only paths with destination node < this val',default=INF)
    parser.add_argument('--source_partition_step',type=int,help='...',default=INF)
    parser.add_argument('--destination_partition_step',type=int,help='...',default=INF)

    # direct Gurobi solver params
    parser.add_argument('--time_limit',type=int,help='time limit in minutes')
    parser.add_argument('--threads',type=int,help='# threads total')
    parser.add_argument('--concurrent_mip',type=int,help='# threads for concurrent')
    parser.add_argument('--heuristic_ratio',type=float,help='heuristic ratio [0,1]. 0=> none. 1=>all')
    parser.add_argument('--mip_focus',type=int,help='focus for MIP solver. 0=>balanced. 1=>feasible/first solution. 2=>optimality. 3=>bound')
    parser.add_argument('--symmetry_detection',type=int,help='control symmetry detection. -1 =>automatic. 0=>off. 1=>conservative. 2=>aggressive')
    parser.add_argument('--barrier_iter_limit',type=int,help='limit iterations of barrier algorithm')
    parser.add_argument('--iter_limit',type=int,help='limit iterations of something')
    parser.add_argument('--cut_passes',type=int,help='limit iterations of cut passes')
    parser.add_argument('--method',type=int,help='lp (root relax) method. -1=>auto. 0=>primal simplex. 1=>dual simplex. 2=>barrier. 3=>concurrent. 4=>deterministic concurrent. 5=>deterministic concurrent simplex')
    parser.add_argument('--node_method',type=int,help='-1=>auto. 0=>primal simplex. 1=>dual simplex. 2=>barrier')
    parser.add_argument('--crossover',type=int,help='')
    parser.add_argument('--no_rel_heur_time',type=int,help='')
    parser.add_argument('--presolve',type=int,help='Presolve aggressiveness. -1=>auto. 0=>off. 1=>conservative. 2=>aggressive')
    parser.add_argument('--presparsify',type=int,help='')
    parser.add_argument('--cuts',type=int,help='')
    parser.add_argument('--scale_flag',type=int,help='')
    parser.add_argument('--feas_tol',type=float,help='')
    parser.add_argument('--degen_moves',type=int,help='',default=-1)
    parser.add_argument('--write_presolved',action='store_true',help='presolve and write (presolved) model out as multiple/all formats')
    parser.add_argument('--read_presolved',type=str,help='read a presolved model of given name')


    args = parser.parse_args()

    map_filename = args.graph

    write_presolved = args.write_presolved

    read_presolved = False
    presolved_model_name = None
    if args.read_presolved is not None:
        read_presolved = True
        presolved_model_name = args.read_presolved

    destination_based = args.destination_based

    source_start = args.source_start
    source_end = args.source_end
    destination_start = args.destination_start
    destination_end = args.destination_end
    source_partition_step = args.source_partition_step
    destination_partition_step = args.destination_partition_step

    partitioned = False
    if source_start > 0 or source_end < INF or destination_start > 0 or destination_end < INF:
        partitioned = True

    if source_partition_step < INF or destination_partition_step < INF:
        partitioned = True

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

    if args.degen_moves is not None:
        print('here')
    solver_params.update({'DegenMoves':args.degen_moves})
    # input(f'degen = {args.degen_moves}')

    r_map = ingest_map(map_filename)

    n_routers = len(r_map)

    hop_dists = floyd(r_map)

    print(f'Completed hop dists')


    if not args.allpath_list:
        min_paths = nwx_all_shortest_paths(r_map)

    else:
        apl_name = args.allpath_list
        min_paths = ingest_path_list(apl_name, n_routers)

        # input(f'min_paths={min_paths}')

    print(f'Completed min hop paths')
    print(f'min_paths {len(min_paths)}x{len(min_paths[0])}')

    # solve
    ###########################################################################

    flat_route_pathlist = []

    if not read_presolved and not partitioned:
        route_pathlist = find_mclb(r_map, hop_dists, min_paths, solver_params=solver_params,write_presolved=write_presolved, destination_based=destination_based,source_start=source_start,source_end=source_end,destination_start=destination_start,destination_end=destination_end)

        for sr in range(n_routers):
            for dr in range(n_routers):
                if sr < source_start or sr >= source_end or dr < destination_start or dr >= destination_end:
                        continue
                flat_route_pathlist.append(route_pathlist[sr][dr])

    elif not read_presolved:
        
        for src_start in range(0,n_routers,source_partition_step):
            src_end = src_start + source_partition_step
            for dest_start in range(0,n_routers,destination_partition_step):
                dest_end = dest_start + destination_partition_step


                print(f'Working on partition src [{src_start},{src_end}), dest [{dest_start},{dest_end})')

                route_pathlist = find_mclb(r_map, hop_dists, min_paths, solver_params=solver_params,write_presolved=write_presolved, destination_based=destination_based,source_start=source_start,source_end=source_end,destination_start=destination_start,destination_end=destination_end)

                
                for sr in range(n_routers):
                    for dr in range(n_routers):
                        if sr < src_start or sr >= src_end or dr < dest_start or dr >= dest_end:
                                continue
                        flat_route_pathlist.append(route_pathlist[sr][dr])

                        min_paths[sr][dr] = [route_pathlist[sr][dr]]

                print(f'completed partition src [{src_start},{src_end}), dest [{dest_start},{dest_end})')

    else:
        print(f'Directly reading (presolved) model from {presolved_model_name}')
        model = gp.read(presolved_model_name)

        solver_params.update({'Presolve':0})
        route_pathlist = directly_solve(model, min_paths, solver_params)

        for sr in range(n_routers):
            for dr in range(n_routers):
                if sr < source_start or sr >= source_end or dr < destination_start or dr >= destination_end:
                        continue
                flat_route_pathlist.append(route_pathlist[sr][dr])


    # output
    ###########################################################################
    base_file_name = map_filename.split('/')[-1].replace('.map','_neu_mclb')

    if args.allpath_list:
        base_file_name = args.allpath_list.split('/')[-1].replace('.rallpaths','_neu_mclb')

    if destination_based:
        base_file_name += '_destbased'

    if source_start > 0 or source_end < INF or destination_start > 0 or destination_end < INF:
        base_file_name += f'_ss{source_start}'
        base_file_name += f'_se{source_end}'
        base_file_name += f'_ds{destination_start}'
        base_file_name += f'_de{destination_end}'

    if source_partition_step < INF or destination_partition_step < INF:
        base_file_name += f'_sps{source_partition_step}_dps{destination_partition_step}'

    output_pathlist( flat_route_pathlist, base_file_name, 'topologies_and_routing/routepath_lists')


def directly_solve(m, path_list, solver_params):

    try:

        # Parameters
        # --------------------------------------------------------------------------------
        for param_key, param_val in solver_params.items():
            model.setParam(param_key, param_val)


        m.update()

        # Optimize model
        m.optimize()

        # Output
        # --------------------------------------------------------------------------------


        # for v in m.getVars():
        #     try:
        #         print(f"{v.VarName} {v.X:g}")
        #     except:
        #         pass


        max_throughput_varname = 'max_cload'
        max_cload_var = m.getVarByName(max_throughput_varname)
        print(f'{max_throughput_varname} : {max_cload_var.X}')

        print(f"\t(aka obj: {m.ObjVal:g} )")

        chosen_paths = [ [ None for _ in range(n_routers)] for __ in range(n_routers) ]

        for sr in range(n_routers):
            for dr in range(n_routers):

                if sr==dr:
                    chosen_paths[sr][dr] = [sr]
                    continue

                # print(f'{sr} -> {dr}')

                paths = path_list[sr][dr]

                for p, path in enumerate(paths):
                    path_chosen_varname = f'var_path_chosen_{sr}r_{dr}r_{p}p'
                    path_chosen_var = m.getVarByName(path_chosen_varname)
                    path_chosen_val = path_chosen_var.X

                    # print(f'\t{path_chosen_varname} : {path_chosen_val}')

                    if path_chosen_val > 0:
                        chosen_path = path_list[sr][dr][p]
                        # print(f'{sr}->{dr} : {chosen_path}')

                        if chosen_paths[sr][dr] is not None:
                            input('overwrite!')

                        chosen_paths[sr][dr] = chosen_path
                    



        return chosen_paths

    # Error Handling
    # --------------------------------------------------------------------------------

    except AttributeError:
        print("Encountered an attribute error")

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")

def find_mclb(r_map, hop_dists, path_list, solver_params={},write_presolved=False, destination_based=False,source_start=0,source_end=INF,destination_start=0,destination_end=INF):

    # try:
        # Create a new model
        model_base_name = "neu_mclb"
        m = gp.Model(model_base_name)

        # Constants
        # --------------------------------------------------------------------------------
        n_routers = len(r_map)
        print(f'n_routers={n_routers}')

        demand = 1.0
        capacity = 1.0

        has_subpaths = {}
        is_subpath_of = {}
        if destination_based:
            for sr in range(n_routers):
                for dr in range(n_routers):

                    # if sr < source_start or sr >= source_end or dr < destination_start or dr >= destination_end:
                    #     continue

                    # flow (sr,dr)
                    for p,path in enumerate(path_list[sr][dr]):

                        outer_path_signature = (sr,dr,p)
                        sub_path_dest = dr
                        path_hop_len = len(path) - 1
                        for sub_idx in range(1, path_hop_len, 1):
                            sub_path = path[sub_idx:]

                            sub_path_src = sub_path[0]

                            idx_in_chosen_paths = -1
                            for inner_p, temp_path in enumerate(path_list[sub_path_src][sub_path_dest]):
                                if sub_path == temp_path:
                                    idx_in_chosen_paths = inner_p

                            # is extra
                            # IGNORE EXTRAS?
                            if idx_in_chosen_paths == -1 and False:
                                # add to path_list
                                path_list[sub_path_src][sub_path_dest].append(sub_path)
                                idx_in_chosen_paths = len(path_list[sub_path_src][sub_path_dest]) - 1

                            sub_path_signature = (sub_path_src, sub_path_dest, idx_in_chosen_paths)
                            is_subpath_of.update({ sub_path_signature : outer_path_signature })

                            if outer_path_signature not in has_subpaths.keys():
                                has_subpaths.update( { outer_path_signature : [] })

                            has_subpaths[outer_path_signature].append(sub_path_signature)


        # construct path set given edge
        # edge_paths[i][j][n] is nth path that crosses edge (i,j)
        links = []
        edge_paths = [[[] for _ in range(n_routers)] for __ in range(n_routers) ]
        edges_of_flow = [[[] for _ in range(n_routers)] for __ in range(n_routers) ]
        edges_of_flow_to_path_idx = [[ {} for _ in range(n_routers)] for __ in range(n_routers) ]
        for sr in range(n_routers):
            for dr in range(n_routers):

                # if sr < source_start or sr >= source_end or dr < destination_start or dr >= destination_end:
                #     continue

                # flow (sr,dr)
                for p,path in enumerate(path_list[sr][dr]):
                    path_len = len(path)
                    for i in range(path_len-1):
                        edge_src = path[i]
                        edge_dest = path[i+1]

                        path_signature = (sr,dr,p)

                        # edge_paths[edge_src][edge_dest].append(path)
                        edge_paths[edge_src][edge_dest].append(path_signature)

                        edge_signature = (edge_src, edge_dest) 

                        if edge_signature not in links:
                            links.append(edge_signature)
                        if edge_signature not in edges_of_flow[sr][dr]:
                            edges_of_flow[sr][dr].append(edge_signature)
                        
                        try:
                            edges_of_flow_to_path_idx[sr][dr][edge_signature].append(p)
                        except:
                            edges_of_flow_to_path_idx[sr][dr].update({ edge_signature : [p]  })


        # determine subpaths
        # FLAG
        # subpath [4, 7, 39] to [12, 4, 96, 64, 71, 39]


        # for i in range(n_routers):
        #     for j in range(n_routers):
        #         input(f'edge ({i},{j}) w/ paths {edge_paths[i][j]}')



        # Variables
        # --------------------------------------------------------------------------------

        n_links = len(links)
        n_flows = (n_routers**2) - n_routers
        min_cload = n_flows / n_links

        max_cload = m.addVar(lb=min_cload, ub=n_flows, vtype=GRB.INTEGER, name='max_cload')
        # max_cload = m.addVar(lb=min_cload, ub=n_flows, vtype=GRB.CONTINUOUS, name='max_cload')

        # cload = []
        # for i in range(n_routers):
        #     cload.append([])
        #     for j in range(n_routers):

        #         # if (i==j):
        #         #     continue

        #         myvarname = f'var_cload_{i}r_{j}r'
        #         # cload[i].append( m.addVar(lb=min_cload, ub=n_flows,vtype=GRB.INTEGER, name=myvarname) )
        #         cload[i].append( m.addVar(lb=min_cload, ub=n_flows,vtype=GRB.CONTINUOUS, name=myvarname) )


        known_paths = {(i,j) : None for i in range(n_routers) for j in range(n_routers)}
        path_chosen = [[[] for _ in range(n_routers)] for _ in range(n_routers)]
        for sr in range(n_routers):
            path_chosen.append([])
            for dr in range(n_routers):



                # if (sr==dr):
                #     continue

                path_chosen[sr].append([])

                paths = path_list[sr][dr]

                if len(paths) == 1:
                    path_chosen[sr][dr].append(1)
                    known_paths[(sr,dr)] = paths[0]
                    # print(f'Hardcoding {sr}->{dr} : {paths[0]}')
                    continue


                if sr < source_start or sr >= source_end or dr < destination_start or dr >= destination_end:
                    continue

                for p,path in enumerate(paths):
                    myvarname = f'var_path_chosen_{sr}r_{dr}r_{p}p'
                    path_chosen[sr][dr].append( m.addVar(vtype=GRB.BINARY, name=myvarname) )


        if VERBOSE:
            print(f'path_flow ({get_shape(path_flow)})')
            for k,v in path_flow.items():
                print(f'key = {k} : value = {v}')

        # quit()


        # Constraints
        # --------------------------------------------------------------------------------


        # # max_cload
        # for i in range(n_routers):
        #     for j in range(n_routers):
        #         if (i==j):
        #             continue

        #         cload_expr = gp.LinExpr()
        #         cload_expr += cload[i][j]

        #         myconstrname = f'constr_max_cload_{i}r_{j}r'
        #         m.addConstr(max_cload >= cload_expr , myconstrname)

        # start_t = time.time()

        # define cload
        for i in range(n_routers):
            for j in range(n_routers):
                if (i==j):
                    continue

                cload_expr = gp.LinExpr()

                for (sr,dr,p) in edge_paths[i][j]:

                    cload_expr += path_chosen[sr][dr][p]


                myconstrname = f'constr_cload_{i}r_{j}r'
                # m.addConstr(cload[i][j] >= cload_expr , myconstrname)
                m.addConstr(max_cload >= cload_expr , myconstrname)


        # single path
        for sr in range(n_routers):
            for dr in range(n_routers):
                if (sr==dr):
                    continue

                if sr < source_start or sr >= source_end or dr < destination_start or dr >= destination_end:
                    continue

                if known_paths[(sr,dr)]:
                    continue

                paths = path_list[sr][dr]

                path_expr = gp.LinExpr()

                for p, path in enumerate(paths):
                    path_expr += path_chosen[sr][dr][p]


                myconstrname = f'constr_path_chosen_{sr}r_{dr}r'
                m.addConstr( path_expr == 1, myconstrname)


        # # SOS single path
        # for sr in range(n_routers):
        #     for dr in range(n_routers):
        #         if (sr==dr):
        #             continue


        #         if known_paths[(sr,dr)]:
        #             continue

        #         # paths = path_list[sr][dr]

        #         related_vars = path_chosen[sr][dr]

        #         path_expr = gp.LinExpr()

        #         for p, path in enumerate(paths):
        #             path_expr += path_chosen[sr][dr][p]


        #         myconstrname = f'constr_path_chosen_{sr}r_{dr}r'
        #         m.addConstr( path_expr >= 1, myconstrname)
        #         m.addSOS(GRB.SOS_TYPE1, related_vars)

        if destination_based:
            for sr in range(n_routers):
                for dr in range(n_routers):
                    if (sr==dr):
                        continue

                    if sr < source_start or sr >= source_end or dr < destination_start or dr >= destination_end:
                        continue

                    n_paths = len(path_list[sr][dr])
                    for p in range(n_paths):

                        outer_path_signature = (sr,dr,p)

                        if outer_path_signature not in has_subpaths.keys():
                            continue

                        sub_path_signatures = has_subpaths[outer_path_signature]
                        for sub_path_sig in sub_path_signatures:

                            (sub_src, sub_dest, sub_idx) = sub_path_sig
                            m.addConstr(path_chosen[sr][dr][p] <= path_chosen[sub_src][sub_dest][sub_idx])


            # for sr in range(n_routers):
            #     for dr in range(n_routers):
            #         if (sr==dr):
            #             continue
            #         n_paths = len(path_list[sr][dr])
            #         for p in range(n_paths):

            #             sub_path_signature = (sr,dr,p)

            #             if sub_path_signature not in is_subpath_of.keys():
            #                 continue

            #             outer_path_signature = is_subpath_of[sub_path_signature]
            #             # for outer_path_signature in outer_path_signatures:

            #             (outer_src, outer_dest, outer_idx) = outer_path_signature
            #             m.addConstr(path_chosen[outer_src][outer_dest][outer_idx] <= path_chosen[sr][dr][p])



                            # if dr == 39:
                            #     input(f'subpath {path_list[sub_src][sub_dest][sub_idx]} to {path_list[sr][dr][p]}')

        # quit()

        # Objectives
        # --------------------------------------------------------------------------------

        m.setObjective(max_cload, GRB.MINIMIZE)


        # Params and Model Output
        write_model = False
        if write_model:
            out_model_name = f'files/models/{model_base_name}.lp'
            m.write(out_model_name)

        m.update()


        # Parameters
        # --------------------------------------------------------------------------------
        for param_key, param_val in solver_params.items():
            m.setParam(param_key, param_val)

        m.setParam('Crossover', 0)  # Turn it off completely (may lose some performance in some cases)
        # or
        m.setParam('CrossoverBasis', 0)  # Avoid trying to reuse the basis

        m.setParam('Cuts', 2)  # Try more aggressive cuts

        # Solve
        # --------------------------------------------------------------------------------

        if write_presolved:
            m.presolve()

            scratch_dir = '/scratch/negishi/green456'
            #scratch_dir = '.'

            out_presolved_model_name = f'{scratch_dir}/{model_base_name}_presolved.lp'

            try:
                m.write(out_presolved_model_name)
                print(f'Wrote to {out_presolved_model_name}')

            except Exception as e:
                print(f'{out_presolved_model_name} cannot be written')
                print(f'Error: {e}')



        # Optimize model
        m.optimize()

        # Output
        # --------------------------------------------------------------------------------


        # for v in m.getVars():
        #     try:
        #         print(f"{v.VarName} {v.X:g}")
        #     except:
        #         pass


        max_throughput_varname = 'max_cload'
        max_cload_var = m.getVarByName(max_throughput_varname)
        print(f'{max_throughput_varname} : {max_cload_var.X}')

        print(f"\t(aka obj: {m.ObjVal:g} )")

        chosen_paths = [ [ None for _ in range(n_routers)] for __ in range(n_routers) ]
        for sr in range(n_routers):
            for dr in range(n_routers):

                if sr==dr:
                    chosen_paths[sr][dr] = [sr]
                    continue

                if sr < source_start or sr >= source_end or dr < destination_start or dr >= destination_end:
                    continue

                # print(f'{sr} -> {dr}')

                if known_paths[(sr,dr)] is not None:
                    chosen_paths[sr][dr] = known_paths[(sr,dr)]
                    continue

                paths = path_list[sr][dr]

                for p, path in enumerate(paths):
                    path_chosen_varname = f'var_path_chosen_{sr}r_{dr}r_{p}p'
                    path_chosen_var = m.getVarByName(path_chosen_varname)
                    path_chosen_val = path_chosen_var.X

                    # print(f'\t{path_chosen_varname} : {path_chosen_val}')

                    if path_chosen_val > 0:
                        chosen_path = path_list[sr][dr][p]
                        # print(f'{sr}->{dr} : {chosen_path}')

                        if chosen_paths[sr][dr] is not None:
                            input('overwrite!')

                        chosen_paths[sr][dr] = chosen_path

        return chosen_paths

    # # Error Handling
    # # --------------------------------------------------------------------------------

    # except AttributeError:
    #     print("Encountered an attribute error")

    # except gp.GurobiError as e:
    #     print(f"Error code {e.errno}: {e}")

# Script Stuff
# --------------------------------------------------------------------------------

if __name__ == '__main__':

    main()
