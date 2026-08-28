#!/usr/bin/env python3.11


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

# constants
VERBOSE = False # for all
ASSERT_BINARY_MAP = True
INF = 999 # for FW

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

# Gurobi Functions
# --------------------------------------------------------------------------------

# Main(s)
# --------------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description='Verify topology values')
    parser.add_argument('--graph','-g',type=str,help='.map file to evaluate',default='files/map_files/example_6r_25ll.map')
    parser.add_argument('--allpath_list','-apl',type=str,help='shortcut over path creation')

    args = parser.parse_args()

    map_filename = args.graph

    r_map = ingest_map(map_filename)

    n_routers = len(r_map)

    hop_dists = floyd(r_map)

    print(f'Completed hop dists')

    if not args.allpath_list:
        min_paths = calculate_min_hop_paths(r_map, hop_dists)
    else:
        apl_name = args.allpath_list
        min_paths = ingest_path_list(apl_name, n_routers)

    # input(f'min_paths {len(min_paths)}x{len(min_paths[0])}')
    # input(f'min_paths={min_paths}')

    print(f'Completed min hop paths')

    find_mcf(r_map, hop_dists, min_paths)
 
def find_mcf(r_map, hop_dists, min_paths):

    try:
        # Create a new model
        model_base_name = "ilp_mcf"
        m = gp.Model(model_base_name)

        # Constants
        # --------------------------------------------------------------------------------
        n_routers = len(r_map)
        print(f'n_routers={n_routers}')

        demand = 1.0
        capacity = 1.0

        # construct path set given edge
        # edge_paths[i][j][n] is nth path that crosses edge (i,j)
        edge_paths = [[[] for _ in range(n_routers)] for __ in range(n_routers) ]
        for sr in range(n_routers):
            for dr in range(n_routers):
                # flow (sr,dr)
                for path in min_paths[sr][dr]:
                    path_len = len(path)
                    for i in range(path_len-1):
                        edge_src = path[i]
                        edge_dest = path[i+1]
                        edge_paths[edge_src][edge_dest].append(path)

        # for i in range(n_routers):
        #     for j in range(n_routers):
        #         input(f'edge ({i},{j}) w/ paths {edge_paths[i][j]}')



        # Variables
        # --------------------------------------------------------------------------------

        max_throughput = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='max_throughput')

        # path_flow = []
        # for sr in range(n_routers):
        #     path_flow.append([])
        #     for dr in range(n_routers):
        #         path_flow[sr].append([])

        #         n_paths = len(min_paths[sr][dr])

        #         for i in range(n_paths):
        #             myvarname = f'var_path_flow_{sr}r_{dr}r_{i}p'
        #             path_flow[sr][dr].append(m.addVar(vtype=GRB.CONTINUOUS, name=myvarname) )

        # indexed by path (a list cast as string)
        path_flow = {}
        for sr in range(n_routers):
            for dr in range(n_routers):

                paths = min_paths[sr][dr]
                for i, p in enumerate(paths):

                    myvarname = f'var_path_flow_{sr}r_{dr}r_{i}p'
                    path_flow.update({str(p) : m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=myvarname) })


        if VERBOSE:
            print(f'path_flow ({get_shape(path_flow)})')
            for k,v in path_flow.items():
                print(f'key = {k} : value = {v}')



        # Constraints
        # --------------------------------------------------------------------------------

        # partial demand

        # for sr in range(n_routers):
        #     for dr in range(n_routers):
        #         flow_sum = gp.quicksum(path_flow[sr][dr] )
        #         myconstrname = f'constr_demand_path_flow_{sr}r_{dr}r'
        #         m.addConstr(flow_sum - max_throughput*demand == 0 , myconstrname)

        #         print(f'path_flow[{sr}][{dr}]={path_flow[sr][dr]}')
        #         input(f'min_paths[{sr}][{dr}]={min_paths[sr][dr]}')
        for sr in range(n_routers):
            for dr in range(n_routers):
                if (sr==dr):
                    continue
                paths = min_paths[sr][dr]

                demand_sum = gp.quicksum(path_flow[str(p)] for p in paths)

                myconstrname = f'constr_demand_path_flow_{sr}r_{dr}r'
                m.addConstr(demand_sum - max_throughput*demand == 0 , myconstrname)

                # input(f'{myconstrname} : demand_sum = sum( {[path_flow[str(p)] for p in paths]} )')
        
        # partial capacity
        for i in range(n_routers):
            for j in range(n_routers):
                if (i==j):
                    continue

                # only consider valid edges (otherwise, no paths and thus max_throughput slammed to 0)
                if r_map[i][j] == 0:
                    continue

                paths = edge_paths[i][j]


                capacity_sum = gp.quicksum(path_flow[str(p)] for p in paths)


                myconstrname = f'constr_capacity_path_flow_{i}r_{j}r'
                m.addConstr(capacity_sum <= capacity , myconstrname)

        
        # Objectives
        # --------------------------------------------------------------------------------

        m.setObjective(max_throughput, GRB.MAXIMIZE)


        # Params and Model Output
        write_model = True
        if write_model:
            out_model_name = f'files/models/{model_base_name}.lp'
            m.write(out_model_name)


        # Solve
        # --------------------------------------------------------------------------------


        # Optimize model
        m.optimize()

        # Output
        # --------------------------------------------------------------------------------


        # for v in m.getVars():
        #     print(f"{v.VarName} {v.X:g}")

        max_throughput_varname = 'max_throughput'
        max_throughput_var = m.getVarByName(max_throughput_varname)
        print(f'{max_throughput_varname} : {max_throughput_var.X}')

        print(f"\t(aka obj: {m.ObjVal:g} )")

    # Error Handling
    # --------------------------------------------------------------------------------

    except AttributeError:
        print("Encountered an attribute error")

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")

# Script Stuff
# --------------------------------------------------------------------------------

if __name__ == '__main__':

    main()