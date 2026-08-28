
# std
import argparse
import time
import os
from collections import defaultdict

# pipd
import orjson


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

def stream_pathlist(path):

    if ".rallpaths" in path:
        with open(path, "rb", buffering=1024*1024) as inf:
            for bline in inf:
                bline = bline.strip()
                if not bline:
                    # skip empty
                    continue
                # split bytes into byte tokens, convert each to int
                row = [int(tok) for tok in bline.split()]
                yield row

    else:
        with open(path, "r", buffering=1024*1024) as inf:
            next(inf, None)  # skip header
            for line in inf:
                line = line.strip()
                if line:
                    yield orjson.loads(line)  # produce one row at a time

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

def calc_cload(pathlist_filepath):

    print(f'Calculating max channel load from file {pathlist_filepath}')

    cload_dict = defaultdict(int)

    n_routers = 0

    seen_flows = set()
    total_hops = 0

    for path in stream_pathlist(pathlist_filepath):
        path_src = path[0]
        path_dest = path[-1]

        if (path_src, path_dest) in seen_flows:
            continue

        seen_flows.add( (path_src, path_dest) )

        n_hops = len(path) - 1

        total_hops += n_hops

        # print(f'path {path_src}->..->{path_dest} = {path}')

        for i in range(n_hops):
            hop_src = path[i]
            hop_dest = path[i+1]

            cload_dict[(hop_src, hop_dest)] += 1

            n_routers = max(n_routers, hop_dest)

    n_routers += 1


    if ".rallpaths" not in pathlist_filepath:
        for s in range(n_routers):
            for d in range(n_routers):
                if s==d:
                    continue
                if (s,d) not in seen_flows:
                    input(f"ERROR: {(s,d)} not in pathlist")

    max_cload = 0
    maximally_loaded_edges = []
    for edge, cload in cload_dict.items():

        if cload > max_cload:
            max_cload = cload
            maximally_loaded_edges = [edge]
        elif cload == max_cload:
            maximally_loaded_edges.append(edge)

    n_to_print = min(5, len(maximally_loaded_edges)) - 1

    print(f"After everything, throughput is {1/max_cload} from max cload {max_cload}")
    print(f"\tFrom edges {maximally_loaded_edges[:n_to_print]}")

    print(f"Average number of hops: {total_hops / (n_routers*(n_routers-1))}")


def verify_pathlist(pathlist_filepath, topo_filepath):

    print(f'Verifying pathlist {pathlist_filepath} w/ topology {topo_filepath}')

    topo_adjmat, _topo_adjlist = ingest_map(topo_filepath)

    for path in stream_pathlist(pathlist_filepath):

        n_hops = len(path) - 1

        # print(f'path {path_src}->..->{path_dest} = {path}')

        for n in range(n_hops):
            i = path[n]
            j = path[n+1]

            if topo_adjmat[i][j] == 0:
                print(f"ERROR: disconnected edge ({i},{j}) in path {path}")
                quit()

    print(f"VALID")


def verify_allpathlist(allpathlist_filepath, topo_filepath):

    print(f'Verifying all paths list {allpathlist_filepath} w/ topology {topo_filepath}')

    topo_adjmat, _topo_adjlist = ingest_map(topo_filepath)

    for path in stream_raw_paths(allpathlist_filepath):

        n_hops = len(path) - 1

        # print(f'path {path_src}->..->{path_dest} = {path}')

        for n in range(n_hops):
            i = path[n]
            j = path[n+1]

            if topo_adjmat[i][j] == 0:
                print(f"ERROR: disconnected edge ({i},{j}) in path {path}")
                quit()

    print(f"VALID")

def main():
    parser = argparse.ArgumentParser(description='Convert pathlist into nrl (optimized for large files)')
    parser.add_argument('--pathlist',type=str,help='pathlist to convert to nrl')
    parser.add_argument("--topology", type=str,help=".map file (adjacency matrix format)")
    parser.add_argument("--all_paths_list", type=str,help="")

    args = parser.parse_args()

    if args.topology and args.all_paths_list:
        verify_allpathlist(args.all_paths_list, args.topology)

    elif args.topology:
        verify_pathlist(args.pathlist, args.topology)

    else:
        calc_cload(args.pathlist)



if __name__ == '__main__':
    main()

