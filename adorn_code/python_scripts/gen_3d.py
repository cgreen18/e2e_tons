
import argparse



import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionStyle
from mpl_toolkits.mplot3d import Axes3D

from itertools import combinations
import math
import operator as op
from functools import reduce


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2


def half_bw_sparsemap(sparse_map, combo, vb=False):

    n_routers = len(sparse_map)

    other_combo = [i for i in range(n_routers) if i not in combo]

    left_bw = 0

    for e in combo:

        for conn in sparse_map[e]:
            if conn in other_combo:
                left_bw += 1
    
    return left_bw

def calc_worst_cut(sparse_map ):

    n_routers = len(sparse_map)



def calc_bi_bw(sparse_map):

    n_routers = len(sparse_map)

    print(f'Will calculate bi bw')

    verbose = True

    routers = range(0,n_routers)
    half = n_routers // 2
    # all_combos = list()
    # all_combos += combinations(routers, half)
    all_combos_generator = combinations(routers, half)
    print(f'created combos')

    # defaults
    least_left_combo = []
    most_left_combo = []
    for a_combo in all_combos_generator:
        print(f'a_combo={a_combo}')
        least_left_combo = a_combo
        most_left_combo = a_combo
        break


    least_left_bw = half_bw_sparsemap(sparse_map, least_left_combo, vb=verbose)
    most_left_bw = least_left_bw
    # if verbose:
    #     print(f'first combo has bisection bw={least_left_bw} from 
    print(f'first combo has bisection bw={least_left_bw} from combo {least_left_combo}')




    new_combo = [x for x in range(half)]

    this_left_bw = half_bw_sparsemap(sparse_map, new_combo, vb=verbose)

    if this_left_bw < least_left_bw:
        print(f'lesser left bisection bw found: {this_left_bw} < {least_left_bw}')
        _ = half_bw_sparsemap(sparse_map, new_combo, vb=True)

        least_left_bw = this_left_bw
        least_left_combo = new_combo 

    # if verbose:
    #     print(f'first combo has bisection bw={least_left_bw} from 
    input(f'another combo has bisection bw={this_left_bw} from combo {new_combo}')



    # quit()
    n_combos = ncr(n_routers, half)
    print(f'n_combos={n_combos}')

    for i, combo in enumerate(all_combos_generator):

        if i % 10000 == 0:
            print(f'Working on combo # {i:03}/{n_combos:03}')
            print(f'\tleast left is still {least_left_bw} from {least_left_combo}')

        this_left_bw = half_bw_sparsemap(sparse_map, combo)

        if this_left_bw < least_left_bw:
            if verbose:
                print(f'lesser left bisection bw found: {this_left_bw} < {least_left_bw}')
                _ = half_bw_sparsemap(sparse_map, combo, vb=True)

            least_left_bw = this_left_bw
            least_left_combo = combo

        if this_left_bw > most_left_bw:
            if verbose:
                print(f'greater left bisection bw found: {this_left_bw} > {most_left_bw}')
                _ = half_bw_sparsemap(sparse_map, combo, vb=True)

            most_left_bw = this_left_bw
            most_left_combo = combo

    print(f'most left bisection bandwidth={most_left_bw} from combo {most_left_combo}')

    input(f'least left bisection bandwidth={least_left_bw} from combo {least_left_combo}')


def gen_r_to_xyz_dicts(x_dim,y_dim,z_dim):
    pos = {}
    # reverse_pos = [ [ [-1 for _z in range(z_dim)] for _y in range(y_dim) ] for _x in range(x_dim) ]
    reverse_pos = {}

    id = 0

    for k in range(z_dim):
        for j in range(y_dim):
            for i in range(x_dim):
                # print(f'router {id:02} at ({i},{j},{k})')
                pos.update({id : (i,j,k)})
                reverse_pos.update({ (i,j,k) : id})

                id += 1

    return pos, reverse_pos



########################################

def pdtt(x_dim, y_dim, z_dim, x_cube,y_cube,z_cube):

    wrap_types = {'x':{'on_y':0,'on_z':0},
                    'y':{'on_x':0,'on_z':0},
                    'z':{'on_x':0,'on_y':0}}

    n_cubes = (x_dim//x_cube)*(y_dim//y_cube)*(z_dim//z_cube)


    smallest_dim = min(x_dim,y_dim,z_dim)
    kval = smallest_dim // min(x_cube,y_cube,z_cube)

    # input(f'TODO: determine correct k value. Current={kval}. Applies for greater than 8x8x8 dims')


    # not expected but done
    # 2k x 2k x 3k
    if 2*z_dim == 3*y_dim and 2*z_dim == 3*x_dim:
        wrap_types['y']['on_z'] = y_cube*kval
        wrap_types['x']['on_z'] = x_cube*kval
        return pt_variable_twisting( x_dim, y_dim, z_dim, wrap_types, f'pdtt_{n_cubes}c')


    # 2k x k x k
    elif x_dim == 2*y_dim and x_dim == 2*z_dim:
        wrap_types['y']['on_x'] = x_cube*kval
        wrap_types['z']['on_x'] = x_cube*kval
        return pt_variable_twisting( x_dim, y_dim, z_dim, wrap_types, f'pdtt_{n_cubes}c')

    # k x k x 2k
    elif z_dim == 2*y_dim and z_dim == 2*x_dim:
        wrap_types['y']['on_z'] = x_cube*kval
        wrap_types['x']['on_z'] = x_cube*kval
        return pt_variable_twisting( x_dim, y_dim, z_dim, wrap_types, f'pdtt_{n_cubes}c')

    # THIS IS INCORRECT FOR kx2kx2k
    # # k x 2k x 2k
    # elif x_dim == y_dim//2 and x_dim == z_dim//2:
    #     wrap_types['x']['on_y'] = y_cube
    #     wrap_types['x']['on_z'] = z_cube
    #     wrap_types['y']['on_z'] = z_cube
    #     wrap_types['z']['on_y'] = y_cube
    #     return pt_variable_twisting( x_dim, y_dim, z_dim, wrap_types, f'pdtt_{n_cubes}c')

    # THIS IS CORRECT FOR kx2kx2k
    # k x 2k x 2k
    elif x_dim == y_dim//2 and x_dim == z_dim//2:
        wrap_types['x']['on_y'] = y_cube*kval
        wrap_types['x']['on_z'] = z_cube*kval
        # wrap_types['y']['on_z'] = z_cube
        # wrap_types['z']['on_y'] = y_cube
        return pt_variable_twisting( x_dim, y_dim, z_dim, wrap_types, f'pdtt_{n_cubes}c')

    # not expected but done
    # 3k x k x k
    elif x_dim == 3*y_dim and x_dim == 3*z_dim:
        wrap_types['y']['on_x'] = x_cube*kval
        wrap_types['z']['on_x'] = x_cube*kval
        return pt_variable_twisting( x_dim, y_dim, z_dim, wrap_types, f'pdtt_{n_cubes}c')



    # not expected but done
    # 4k x k x k
    elif x_dim == 4*y_dim and x_dim == 4*z_dim:
        wrap_types['y']['on_x'] = x_cube*kval
        wrap_types['z']['on_x'] = x_cube*kval
        return pt_variable_twisting( x_dim, y_dim, z_dim, wrap_types, f'pdtt_{n_cubes}c')


    # not known
    else:
        print(f'Unknown dimensions')



def pttt(x_dim, y_dim, z_dim, x_cube,y_cube,z_cube):

    wrap_types = {'x':{'on_y':0,'on_z':0},
                    'y':{'on_x':0,'on_z':0},
                    'z':{'on_x':0,'on_y':0}}

    n_cubes = (x_dim//x_cube)*(y_dim//y_cube)*(z_dim//z_cube)


    smallest_dim = min(x_dim,y_dim,z_dim)
    kval = smallest_dim // min(x_cube,y_cube,z_cube)

    # input(f'TODO: determine correct k value. Current={kval}. Applies for greater than 8x8x8 dims')

    # k x k x k
    if x_dim == y_dim and x_dim == z_dim:
        kval = kval // 2
        wrap_types['x']['on_y'] = y_cube*kval
        wrap_types['y']['on_z'] = z_cube*kval
        wrap_types['z']['on_x'] = x_cube*kval

        return pt_variable_twisting( x_dim, y_dim, z_dim, wrap_types, f'pttt_{n_cubes}c')

    # k x k x 2k
    elif z_dim == 2*y_dim and z_dim == 2*x_dim:


        # # lowest lat
        wrap_types['x']['on_y'] = y_cube*(kval//2)
        wrap_types['x']['on_z'] = z_cube*kval
        wrap_types['y']['on_x'] = x_cube*(kval//2)
        wrap_types['y']['on_z'] = z_cube*kval
        wrap_types['z']['on_x'] = x_cube*(kval//2)
        wrap_types['z']['on_y'] = y_cube*(kval//2)
        # wrap_types['z']['on_x'] = x_cube*kval
        # wrap_types['z']['on_y'] = y_cube*kval

        # wrap_types['x']['on_y'] = y_cube*(kval//2)
        # wrap_types['x']['on_z'] = z_cube*kval
        # wrap_types['y']['on_x'] = x_cube*(kval//2)
        # wrap_types['y']['on_z'] = z_cube*kval
        # wrap_types['z']['on_x'] = x_cube*(kval//2)
        # wrap_types['z']['on_y'] = y_cube*(kval//2)
        # # wrap_types['z']['on_x'] = x_cube*kval
        # # wrap_types['z']['on_y'] = y_cube*kval

        # # # lower lat
        # wrap_types['x']['on_y'] = y_cube
        # wrap_types['x']['on_z'] = z_cube*kval
        # # wrap_types['y']['on_x'] = x_cube
        # wrap_types['y']['on_z'] = z_cube*kval
        # # wrap_types['z']['on_x'] = x_cube*(kval*2)
        # # wrap_types['z']['on_y'] = y_cube*(kval*2)
        print(f'wrap_types={wrap_types}')
        return pt_variable_twisting( x_dim, y_dim, z_dim, wrap_types, f'pttt_{n_cubes}c')

    # k x 2k x 2k
    elif x_dim == y_dim//2 and x_dim == z_dim//2:
        wrap_types['x']['on_y'] = y_cube*kval
        wrap_types['x']['on_z'] = z_cube*kval
        wrap_types['y']['on_z'] = z_cube*kval
        wrap_types['z']['on_y'] = x_cube*kval
        return pt_variable_twisting( x_dim, y_dim, z_dim, wrap_types, f'pttt_{n_cubes}c')

    # not known
    else:
        print(f'Unknown dimensions')

def ptt(x_dim, y_dim, z_dim, x_cube,y_cube,z_cube):

    wrap_types = {'x':{'on_y':0,'on_z':0},
                    'y':{'on_x':0,'on_z':0},
                    'z':{'on_x':0,'on_y':0}}


    n_cubes = (x_dim//x_cube)*(y_dim//y_cube)*(z_dim//z_cube)


    smallest_dim = min(x_dim,y_dim,z_dim)
    kval = smallest_dim // min(x_cube,y_cube,z_cube)

    # 2k x k x k
    if x_dim == 2*y_dim and x_dim == 2*z_dim:
        wrap_types['y']['on_x'] = x_cube
        return pt_variable_twisting( x_dim, y_dim, z_dim, wrap_types, f'ptt_{n_cubes}c')
    # k x k x 4k
    elif z_dim == 4*y_dim and z_dim == 4*x_dim:
        wrap_types['y']['on_z'] = x_cube
        return pt_variable_twisting( x_dim, y_dim, z_dim, wrap_types, f'ptt_{n_cubes}c')
    # k x 2k x 2k
    elif x_dim == y_dim//2 and x_dim == z_dim//2:
        wrap_types['x']['on_y'] = y_cube*kval
        return pt_variable_twisting( x_dim, y_dim, z_dim, wrap_types, f'ptt_{n_cubes}c')
    # k x k x 2k
    elif z_dim == 2*y_dim and z_dim == 2*x_dim:
        print(f"type k x k 2k")
        wrap_types['x']['on_z'] = x_cube*kval
        return pt_variable_twisting( x_dim, y_dim, z_dim, wrap_types, f'ptt_{n_cubes}c')

    # not expected but done
    # 3k x k x k
    elif x_dim == 3*y_dim and x_dim == 3*z_dim:
        wrap_types['y']['on_x'] = y_cube
        return pt_variable_twisting( x_dim, y_dim, z_dim, wrap_types, f'ptt_{n_cubes}c')

    # not known
    else:
        print(f'Unknown dimensions')

def pt(x_dim, y_dim, z_dim, x_cube,y_cube,z_cube):


    wrap_types = {'x':{'on_y':0,'on_z':0},
                    'y':{'on_x':0,'on_z':0},
                    'z':{'on_x':0,'on_y':0}}

    n_cubes = (x_dim//x_cube)*(y_dim//y_cube)*(z_dim//z_cube)

    # 2k x k x k
    # k x k x k
    # 3k x k x k
    # k x 2k x 2k

    return pt_variable_twisting(x_dim, y_dim, z_dim, wrap_types, f'pt_{n_cubes}c')


def pt_variable_twisting(x_dim, y_dim, z_dim, wrap_types, topo_type):


    n_routers = x_dim*y_dim*z_dim
    # always (for now?)
    n_ports = 6

    sparse_map = []
    # x_sparse = []
    # y_sparse = []



    # pos, reverse_pos = gen_r_to_xyz_dict_mat(x_dim, y_dim, z_dim)
    pos, reverse_pos = gen_r_to_xyz_dicts(x_dim, y_dim, z_dim)


    for i in range(n_routers):
        these_conns = []

        (x,y,z) = pos[i]

        # x pos
        xprime = (x + 1) % x_dim
        yprime = y
        zprime = z
        # twisting x on y => yprime += (y_cube)
        # twisting x on z => zprime += (z_cube)
        if(x == x_dim - 1):
            yprime += wrap_types['x']['on_y']
            yprime = yprime % y_dim
            zprime += wrap_types['x']['on_z']
            zprime = zprime % z_dim
        _tc = reverse_pos[(xprime,yprime,zprime)]
        these_conns.append(_tc)

        # x neg
        xprime = (x - 1) % x_dim
        yprime = y
        zprime = z
        # twisting x on y => yprime += (y_cube)
        # twisting x on z => zprime += (z_cube)
        if(x == 0):
            yprime += wrap_types['x']['on_y']
            yprime = yprime % y_dim
            zprime += wrap_types['x']['on_z']
            zprime = zprime % z_dim
        _tc = reverse_pos[(xprime,yprime,zprime)]
        these_conns.append(_tc)

        # y pos
        xprime = x
        yprime = (y + 1) % y_dim
        zprime = z
        # twisting y on x => xprime += (x_cube)
        # twisting y on z => zprime += (z_cube)
        if(y == y_dim - 1):
            xprime += wrap_types['y']['on_x']
            xprime = xprime % x_dim
            zprime += wrap_types['y']['on_z']
            zprime = zprime % z_dim
        _tc = reverse_pos[(xprime,yprime,zprime)]
        these_conns.append(_tc)

        # y neg
        xprime = x
        yprime =  (y - 1) % y_dim
        zprime = z
        # twisting y on x => xprime += (x_cube)
        # twisting y on z => zprime += (z_cube)
        if(y == 0):
            xprime += wrap_types['y']['on_x']
            xprime = xprime % x_dim
            zprime += wrap_types['y']['on_z']
            zprime = zprime % z_dim
        _tc = reverse_pos[(xprime,yprime,zprime)]
        these_conns.append(_tc)

        # z pos
        xprime = x
        yprime = y
        zprime = (z + 1) % z_dim
        # twisting z on x => xprime += (x_cube)
        # twisting z on y => yprime += (y_cube)
        if(z == z_dim - 1):
            xprime += wrap_types['z']['on_x']
            xprime = xprime % x_dim
            yprime += wrap_types['z']['on_y']
            yprime = yprime % y_dim
        _tc = reverse_pos[(xprime,yprime,zprime)]
        these_conns.append(_tc)

        # z neg
        xprime = x
        yprime =  y
        zprime = (z - 1) % z_dim
        # twisting z on x => xprime += (x_cube)
        # twisting z on y => yprime += (y_cube)
        if(z == 0):
            xprime += wrap_types['z']['on_x']
            xprime = xprime % x_dim
            yprime += wrap_types['z']['on_y']
            yprime = yprime % y_dim
        _tc = reverse_pos[(xprime,yprime,zprime)]
        these_conns.append(_tc)

        sparse_map.append(these_conns)


    # input(f'sparse_map ({len(sparse_map)}) ={sparse_map}')

    # visualize_3d(sparse_map, x_dim, y_dim, z_dim)


    plt.show()

    out_name = f'./files/3d/{topo_type}_{n_routers}r_{n_ports}p_{x_dim}x{y_dim}x{z_dim}.map'
    input(f'output to: {out_name} ?')
    sparse_to_map_and_print(sparse_map, n_routers, n_ports, out_name)

    out_name = f'./files/paper_solutions/{n_routers}r/{topo_type}_{n_routers}r_{n_ports}p_{x_dim}x{y_dim}x{z_dim}.map'
    input(f'output to: {out_name} ?')
    sparse_to_map_and_print(sparse_map, n_routers, n_ports, out_name)

    return sparse_map

##############################################3



def tpuv4_base(n_routers, n_ports, x_dim,y_dim,z_dim):

    # n_ports = 6
    # n_routers = 64
    # x_dim = 8
    # y_dim = 4
    # z_dim = 4

    sparse_map = []
    # x_sparse = []
    # y_sparse = []


    # pos, reverse_pos = gen_r_to_xyz_dict_mat(x_dim, y_dim, z_dim)
    pos, reverse_pos = gen_r_to_xyz_dicts(x_dim, y_dim, z_dim)



    for i in range(n_routers):
        these_conns = []

        (x,y,z) = pos[i]

        # x pos
        xprime = (x + 1) % x_dim
        yprime = y
        zprime = z
        _tc = reverse_pos[(xprime,yprime,zprime)]
        if (x != x_dim - 1 and x != (x_dim//2) - 1 ):
            these_conns.append(_tc)

        # x neg
        xprime = (x - 1) % x_dim
        yprime = y
        zprime = z
        _tc = reverse_pos[(xprime,yprime,zprime)]
        if (x != 0 and x != (x_dim//2)  ):
            these_conns.append(_tc)

        # y pos
        xprime = x
        yprime = (y + 1) % y_dim
        zprime = z
        _tc = reverse_pos[(xprime,yprime,zprime)]
        if (y != y_dim - 1): #and y != (y_dim//2) - 1 ):
            these_conns.append(_tc)

        # y neg
        xprime = x
        yprime =  (y - 1) % y_dim
        zprime = z
        _tc = reverse_pos[(xprime,yprime,zprime)]
        if (y != 0) :#and y != (y_dim//2)  ):
            these_conns.append(_tc)

        # z pos
        xprime = x
        yprime = y
        zprime = (z + 1) % z_dim
        _tc = reverse_pos[(xprime,yprime,zprime)]
        if (z != z_dim - 1):
            these_conns.append(_tc)

        # z neg
        xprime = x
        yprime =  y
        zprime = (z - 1) % z_dim
        _tc = reverse_pos[(xprime,yprime,zprime)]
        if (z != 0) :
            these_conns.append(_tc)


        # input(f'router {i} w/ conns {these_conns}')

        sparse_map.append(these_conns)


    # input(f'sparse_map ({len(sparse_map)}) ={sparse_map}')

    # visualize_3d(sparse_map, x_dim, y_dim, z_dim)


    plt.show()

    out_name = f'./files/3d/tpuv4_base_{n_routers}r_{x_dim}x{y_dim}x{z_dim}_{n_ports}p.map'

    input(f'output to: {out_name} ?')


    sparse_to_map_and_print(sparse_map, n_routers, n_ports, out_name)

    return sparse_map

def visualize_3d(sparse_map, x_dim, y_dim, z_dim):


    n_routers = x_dim * y_dim * z_dim

    G = nx.DiGraph()


    for src, src_conns in enumerate(sparse_map):
        for dest in src_conns:

            # print(f'visualize_3d():: connecting {src} -> {dest}')

            G.add_edge(src,dest)


    pos, reverse_pos = gen_r_to_xyz_dicts(x_dim, y_dim, z_dim)



    visualize_graph_wpos(G, pos, reverse_pos, n_routers)



def visualize_graph_wpos(G, pos,reverse_pos, n_routers):

    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])


    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=100, ec="w")



    r2r_phys_dist = calc_distances(pos, n_routers)

    # Plot the edges
    for vizedge in edge_xyz:


        # src_xyz = vizedge[0]
        # dest_xyz = vizedge[1]

        # src = reverse_pos[(src_xyz[0],src_xyz[1],src_xyz[2])]
        # dest = reverse_pos[(dest_xyz[0],dest_xyz[1],dest_xyz[2])]

        # print(f'src = { src} @  {src_xyz} \ndest={dest} @ {dest_xyz}')


        # thisrad=0.0

        # dist = r2r_phys_dist[src][dest]

        # if dist > 1:
        #     thisrad = 0.4

        # input(f'vizedge ({type(vizedge)}) = {vizedge} w/ distance {dist}')


        # connectionstyle=f'arc3, rad={thisrad}',

        # edge_patch = ConnectionStyle.Arc3(rad=thisrad)
        # edge_path, edge_path_2 = edge_patch(src_xyz, dest_xyz)

        # input(f'edge_path = {edge_path}\nedge_path_2 = {edge_path_2}')

        # ax.add_patch(edge_patch)
        
        ax.plot(*vizedge.T, color="tab:gray",linestyle='dotted')


    def _format_axes(ax):
        """Visualization options for the 3D axes."""
        # Turn gridlines off
        ax.grid(False)
        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")


    options = {
        "font_size": 12,
        "node_size": 300,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1
    }


    _format_axes(ax)
    fig.tight_layout()

INF = 999


def calc_distances(pos, n_routers):
    r2r_phys_dist = [[INF for _ in range(n_routers)] for __ in range(n_routers)]

    for i in range(n_routers):
        for j in range(n_routers):
            if (i == j):
                r2r_phys_dist[i][j] = 0
                continue
            
            i_pos = pos[i]
            j_pos = pos[j]
            dx = abs( i_pos[0] -  j_pos[0])
            dy = abs( i_pos[1] -  j_pos[1])
            dz = abs( i_pos[2] -  j_pos[2])

            ds = (dx**2) + (dy**2) + (dy**2)
            d = math.sqrt(ds)
            r2r_phys_dist[i][j] = d
    return r2r_phys_dist

def sparse_to_map_and_print_map_file(sparse_map, n_routers, n_ports, out_name):
    r2r_map = []
    for i in range(0,n_routers):
        r2r_map.append([])
        for j in range(0,n_routers):
            r2r_map[i].append(0)

    for i in range(0,n_routers):
        for j in sparse_map[i]:
            r2r_map[i][j] = 1

    n_rows = len(r2r_map)

    with open(out_name,'w+') as out_file:
        for i, row in enumerate(r2r_map):
            line = ''
            for elem in row:
                line += str(elem) + ' '
            line = line[:-1] + '\n'

            if i != n_rows - 1:
                out_file.write(line)

    print(f'Wrote topology out to {out_name}')

def sparse_to_map_and_print(sparse_map, n_routers, n_ports, out_name):


    # print(f'n_routers={n_routers}')

    r2r_map = []
    for i in range(0,n_routers):
        r2r_map.append([])
        for j in range(0,n_routers):
            r2r_map[i].append(0)


    for i in range(0,n_routers):
        # print(f'i={i} sparse_map[i]={sparse_map[i]}')
        for j in sparse_map[i]:
            r2r_map[i][j] = 1

    # print(f'r2r_map({len(r2r_map)})={r2r_map}')

    n_links = 0
    for p in range(n_routers):

        for q in range(n_routers):

            if p==q:
                continue

            n_links += r2r_map[p][q]

    n_links = n_links / 2
    print(f"n_links={n_links}")

    with open(out_name,'w+') as out_file:
        # out_file.write(str(n_routers) + '\n')
        # out_file.write(str(n_ports) + '\n')
        for row in r2r_map:
            line = ''
            for elem in row:
                line += str(elem) + ' '
            line = line[:-1] + '\n'
            out_file.write(line)

    print(f'Wrote topology out to {out_name}')


def print_by_src(mat, n,m):

    i = 0
    for src in mat:
        print(f"r{i}")
        print(src)
        print_nxm(src, n,m)
        i+=1

def print_nxm(conns, n, m):
    for i in reversed(range(0,n)):
        for j in range(0,m):
            # print(f"{i*m+j}={conns[i*m+j]}",end = " ")
            print(f"{conns[i*m+j]}",end = " ")
        print("")
    print("")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate human/heuristic prismatic tori')
    parser.add_argument('--which',type=str,help='Topology to generate',choices=['pt','ptt','pdtt','pttt'],required=True)
    parser.add_argument('--xyzc_dims',nargs='+',type=int,help='type without parenthesis and use spaces, no commas', required=True)
    # parser.add_argument('--n_ports',type=int,default=6)

    args = parser.parse_args()

    which = args.which

    try:
        xyzc_dims = tuple(args.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    except:
        print(f'Error parsing xyzc_dims "{xyzc_dims}". Exiting...')
        quit()

    if(which =='pt'):
        pt(x_dim,y_dim,z_dim,cube_dim,cube_dim,cube_dim)
    elif(which =='ptt'):
        ptt(x_dim,y_dim,z_dim,cube_dim,cube_dim,cube_dim)
    elif(which =='pdtt'):
        pdtt(x_dim,y_dim,z_dim,cube_dim,cube_dim,cube_dim)
    elif(which =='pttt'):
        pttt(x_dim,y_dim,z_dim,cube_dim,cube_dim,cube_dim)
