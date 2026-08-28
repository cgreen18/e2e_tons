import itertools
import argparse
import os

def r_to_xyz(r, xyzc_dims):

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    x = r % x_dim
    y = (r // x_dim) % y_dim
    z = (r // (x_dim*y_dim)) % z_dim
    return x,y,z

def xyz_to_r(x,y,z, xyzc_dims):
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    return x + y*x_dim + z*x_dim*y_dim

def init_electrical_conns(xyzc_dims):

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    n_routers = x_dim*y_dim*z_dim

    electrical_conns_adj_list = [[] for _ in range(n_routers)]
    electrical_conns_by_dim = {'x':[],'y':[],'z':[]}


    for src in range(n_routers):

        src_x,src_y,src_z = r_to_xyz(src, xyzc_dims)

        # xpos
        # if not on edge then conn
        if(src_x % cube_dim != cube_dim - 1):
            targ_x = src_x + 1
            targ_y = src_y
            targ_z = src_z
            targ = xyz_to_r(targ_x, targ_y, targ_z, xyzc_dims)
            # electrical_conns[(src,targ)] = 1
            # electrical_conns.append( (src,targ) )
            electrical_conns_adj_list[src].append(targ)

        # xneg
        # if not on edge then conn
        if(src_x % cube_dim != 0):
            targ_x = src_x - 1
            targ_y = src_y
            targ_z = src_z
            targ = xyz_to_r(targ_x, targ_y, targ_z, xyzc_dims)
            # electrical_conns[(src,targ)] = 1
            # electrical_conns.append( (src,targ) )
            electrical_conns_adj_list[src].append(targ)


        # ypos
        # if not on edge then conn
        if(src_y % cube_dim != cube_dim - 1):
            targ_x = src_x
            targ_y = src_y + 1
            targ_z = src_z
            targ = xyz_to_r(targ_x, targ_y, targ_z, xyzc_dims)
            # electrical_conns[(src,targ)] = 1
            # electrical_conns.append( (src,targ) )
            electrical_conns_adj_list[src].append(targ)

        # yneg
        # if not on edge then conn
        if(src_y % cube_dim != 0):
            targ_x = src_x
            targ_y = src_y - 1
            targ_z = src_z
            targ = xyz_to_r(targ_x, targ_y, targ_z, xyzc_dims)
            # electrical_conns[(src,targ)] = 1
            # electrical_conns.append( (src,targ) )
            electrical_conns_adj_list[src].append(targ)


        # zpos
        # if not on edge then conn
        if(src_z % cube_dim != cube_dim - 1):
            targ_x = src_x
            targ_y = src_y
            targ_z = src_z + 1
            targ = xyz_to_r(targ_x, targ_y, targ_z, xyzc_dims)
            # electrical_conns[(src,targ)] = 1
            # electrical_conns.append( (src,targ) )
            electrical_conns_adj_list[src].append(targ)

        # zneg
        # if not on edge then conn
        if(src_z % cube_dim != 0):
            targ_x = src_x
            targ_y = src_y
            targ_z = src_z - 1
            targ = xyz_to_r(targ_x, targ_y, targ_z, xyzc_dims)
            # electrical_conns[(src,targ)] = 1
            # electrical_conns.append( (src,targ) )
            electrical_conns_adj_list[src].append(targ)

    return electrical_conns_adj_list

def write_bw_lats(latencies, bws, xyzc_dims, out_dir):
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    n_routers = x_dim*y_dim*z_dim
    with open(os.path.join(out_dir, f'{x_dim}x{y_dim}x{z_dim}_bw.txt'), 'w') as f:
        for i in range(n_routers):
            out_line = ""
            for j in range(n_routers):
                out_line += f'{bws[i][j]} '
            f.write(f'{out_line}\n')
    with open(os.path.join(out_dir, f'{x_dim}x{y_dim}x{z_dim}_lat.txt'), 'w') as f:
        for i in range(n_routers):
            out_line = ""
            for j in range(n_routers):
                out_line += f'{latencies[i][j]} '
            f.write(f'{out_line}\n')

def main():

    parser = argparse.ArgumentParser(description='Generate TPUv4 BW LATS')
    parser.add_argument('--xyzc_dims',nargs='+',type=int,help='type without parenthesis and use spaces, no commas')
    parser.add_argument('--electrical_latency',type=int,help='latency in cycles',default=5)
    parser.add_argument('--optical_latency',type=int,help='latency in cycles',default=25)
    parser.add_argument('--electrical_bandwidth',type=int,help='bandwidth in flits/cycle assuming 64B flits',default=3)
    parser.add_argument('--optical_bandwidth',type=int,help='bandwidth in flits/cycle assuming 64B flits',default=3)
    parser.add_argument('--verbose',action='store_true',help='debug prints')
    parser.add_argument('--out_dir',type=str,help='directory to output results',default='./tpuv4_bw_lats')

    args = parser.parse_args()

    electrical_latency = args.electrical_latency
    optical_latency = args.optical_latency
    electrical_bandwidth = args.electrical_bandwidth
    optical_bandwidth = args.optical_bandwidth
    verbose = args.verbose
    out_dir = args.out_dir

    if args.xyzc_dims is not None:
        xyzc_dims_list = [tuple(args.xyzc_dims)]
    else:
        vals = [4, 8, 12, 16, 24, 32]
        xyzc_dims_list = [tuple([x, y, z, 4]) for x, y, z in itertools.product(vals, repeat=3)]

    for xyzc_dims in xyzc_dims_list:
        electrical_conns_adj_list = init_electrical_conns(xyzc_dims)

        (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
        n_routers = x_dim*y_dim*z_dim

        if x_dim % 3 == 0 or x_dim >= 32:
            continue

        latencies = [[optical_latency for _ in range(n_routers)] for __ in range(n_routers)]
        bws = [[optical_bandwidth for _ in range(n_routers)] for __ in range(n_routers)]

        for i, conns in enumerate(electrical_conns_adj_list):
            for j in conns:
                latencies[i][j] = electrical_latency
                latencies[j][i] = electrical_latency
                bws[i][j] = electrical_bandwidth
                bws[j][i] = electrical_bandwidth

        write_bw_lats(latencies, bws, xyzc_dims, out_dir)
        print(f'Wrote {x_dim}x{y_dim}x{z_dim} bw and lat to {out_dir}')

if __name__ == "__main__":
    main()