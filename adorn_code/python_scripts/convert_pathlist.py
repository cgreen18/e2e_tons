
# std
import argparse
import time
import os

# pipd
import orjson


NRL_OUT_DIR = './topologies_and_routing/nr_lists'

def stream_pathlist(path):

    with open(path, "r", buffering=1024*1024) as inf:
        next(inf, None)  # skip header
        for line in inf:
            line = line.strip()
            if line:
                yield orjson.loads(line)  # produce one row at a time

def convert_pathlist_to_nrl(pathlist_filepath):

    print(f'constructing nr map from file {pathlist_filepath}')

    out_name_base = os.path.splitext(os.path.basename(pathlist_filepath))[0]
    nr_list_filepath = os.path.join(NRL_OUT_DIR, f'{out_name_base}.nrl2')

    print(f'Streaming to {nr_list_filepath}')    
    with open(nr_list_filepath,'w+') as of:
        pass

    n_completed = 0

    seen_flows = set()
    n_routers = 0
    prev_time = time.time()

    with open(nr_list_filepath,'a') as of:

        for path in stream_pathlist(pathlist_filepath):
            path_src = path[0]
            path_dest = path[-1]

            n_routers = max(n_routers,path_src)
            n_routers = max(n_routers,path_dest)

            seen_flows.add((path_src,path_dest))

            n_hops = len(path) - 1

            # print(f'path {path_src}->..->{path_dest} = {path}')

            for i in range(n_hops):
                hop_src = path[i]
                hop_dest = path[i+1]

                # of.write(str((hop_src, path_src, path_dest, hop_dest)) + '\n')
                of.write(str((path_src, path_dest, hop_src, hop_dest)) + '\n')

            cur_time = time.time()

            n_completed += 1
            if n_completed % 1_000_000 == 0 or ( False and n_completed % 1 == 0 and n_completed > 13026000):
                print(f'n_completed = {n_completed} in {round(cur_time - prev_time,2)}s')
                # print(f'path ({len(path)}) = {path}')
                prev_time = cur_time

    # adjust zero indexing
    n_routers = n_routers + 1

    print(f'Completed write to {nr_list_filepath}')

    for s in range(n_routers):
        for d in range(s+1,n_routers):
            if (s,d) not in seen_flows :
                print(f" {(s,d)} not in seen_flows ")
                quit()
            if (d,s) not in seen_flows :
                print(f" {(d,s)} not in seen_flows ")
                quit()

    print(f"Verified")

def main():
    parser = argparse.ArgumentParser(description='Convert pathlist into nrl (optimized for large files)')
    parser.add_argument('--pathlist',type=str,help='pathlist to convert to nrl',required=True)

    args = parser.parse_args()

    convert_pathlist_to_nrl(args.pathlist)

if __name__ == '__main__':
    main()

