
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch



DIRECTED = True #False            # set True if your matrix is directed and you want directed arcs
DRAW_LABELS = False
NODE_SIZE = 80
FONT_SIZE = 9
ARC_ALPHA = 0.35
ARC_LINEWIDTH = 1.0
RADIUS = 1.0                 # circle radius
CURVATURE_SCALE = 0.35       # controls how "tall" arcs are (0.2–0.6 are typical)
CURVATURE_SCALE = 0.6       # controls how "tall" arcs are (0.2–0.6 are typical)

# ========= REQUIRED: provide your calc_coord(i) =========
def calc_coord(i, xyzc_dims):
    """
    Return (x, y, z, c) for node i.
    Replace this with your real implementation.
    """
    x,y,z = r_to_xyz(i,xyzc_dims)
    c = which_cube(i,xyzc_dims)
    return (x, y, z, c)

def r_to_xyz(r,xyzc_dims):

    xd,yd,zd,cd = xyzc_dims

    xy_slice_size = xd*yd

    temp_r = r

    z = temp_r // xy_slice_size
    temp_r = temp_r % xy_slice_size
    y = temp_r // xd
    x = temp_r % xd

    return x,y,z

def which_cube(i, xyzc_dims):

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims


    n_x = (x_dim // cube_dim)
    n_y = (y_dim // cube_dim)
    n_z = (z_dim // cube_dim)

    # print(f'n_x,n_y,n_z = {n_x},{n_y},{n_z}')

    i_x,i_y,i_z = r_to_xyz(i,xyzc_dims)

    # rel_i_x = i_x % cube_dim
    # rel_i_y = i_y % cube_dim
    # rel_i_z = i_z % cube_dim

    n_i_x = i_x // cube_dim
    n_i_y = i_y // cube_dim
    n_i_z = i_z // cube_dim


    n_xy = n_x*n_y

    n_cube = (n_i_z)*n_xy + (n_i_y)*n_x + (n_i_x)

    # print(f'{i} @ ({i_x},{i_y},{i_z}) is cube # {n_cube}')

    return n_cube

def is_optical(i,j,xyzc_dims):

    if which_cube(i,xyzc_dims) != which_cube(j,xyzc_dims):
        return True
    
    ix,iy,iz = r_to_xyz(i,xyzc_dims)
    jx,jy,jz = r_to_xyz(j,xyzc_dims)

    is_opt = False


    if abs(ix-jx) > 1:
        is_opt = True
    if abs(iy-jy) > 1:
        is_opt = True
    if abs(iz-jz) > 1:
        is_opt = True

    # input(f'diff cubes. is_opt? {is_opt} : {i}->{j}. ix,iy,iz  = {(ix,iy,iz )}, jx,jy,jz = {(jx,jy,jz)}')

    return is_opt

# ========= Helpers =========
def load_adjacency_matrix(path):
    with open(path, "r") as f:
        rows = [list(map(float, line.strip().split())) for line in f if line.strip()]
    A = np.array(rows, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency must be square; got {A.shape}.")
    return A

def clockwise_angles(n):
    """
    Return n angles placed clockwise around the circle.
    We assign angles decreasing from 2π to 0 so that the given sequence is clockwise.
    """
    # Equally spaced angles; reverse to go clockwise
    base = np.linspace(0, 2*math.pi, num=n, endpoint=False)
    # Make it clockwise by reversing (or negative increment)
    return base[::-1]

def bezier_arc(p0, p1, h):
    """
    Quadratic Bezier control point that creates a nice arc between p0 and p1.
    p0, p1: endpoints (2D numpy arrays)
    h: arc "height" scale (positive => outward, negative => inward)
    Returns a PathPatch (quadratic Bezier).
    """
    mid = 0.5 * (p0 + p1)
    # Perpendicular unit vector to (p1 - p0)
    d = p1 - p0
    L = np.linalg.norm(d)
    if L == 0:
        L = 1e-9
    u = d / L
    # Rotate u by +90 degrees to get a perpendicular
    perp = np.array([-u[1], u[0]])
    control = mid + h * L * perp  # scale height with chord length

    verts = [tuple(p0), tuple(control), tuple(p1)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    return PathPatch(Path(verts, codes), fill=False, alpha=ARC_ALPHA, lw=ARC_LINEWIDTH)

def compute_order(n, xyzc_dims):
    """
    Compute node order based on coordinates.
    Order is increasing (c, z, y, x), then place nodes clockwise around the circle in that order.
    """
    coords = [calc_coord(i, xyzc_dims) for i in range(n)]  # (x,y,z,c)
    # Key = (c, z, y, x) (ascending)
    order = sorted(range(n), key=lambda i: (coords[i][3], coords[i][2], coords[i][1], coords[i][0]))
    return order, coords

def layout_positions(order, radius=RADIUS):
    """
    Map nodes (in the given order) onto points on a circle, clockwise.
    """
    n = len(order)
    thetas = clockwise_angles(n)
    pos = {}
    for k, node in enumerate(order):
        theta = thetas[k]
        pos[node] = np.array([radius * math.cos(theta), radius * math.sin(theta)])
    return pos

def draw_graph(A, pos, order, xyzc_dims, directed=False):
    n = A.shape[0]
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw nodes
    xs = [pos[i][0] for i in order]
    ys = [pos[i][1] for i in order]
    ax.scatter(xs, ys, s=NODE_SIZE, zorder=3)

    if DRAW_LABELS:
        for i in order:
            ax.text(pos[i][0], pos[i][1], str(i), fontsize=FONT_SIZE,
                    ha='center', va='center', zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # Draw circle outline (optional, helpful for orientation)
    circle = plt.Circle((0, 0), RADIUS, fill=False, alpha=0.25, lw=1.0)
    ax.add_patch(circle)

    # Draw edges as arcs
    # If undirected (default), we draw each i<j once if A[i,j] or A[j,i] is nonzero.
    # If directed, draw for every nonzero A[i,j].
    if directed:
        for i in range(n):
            for j in range(n):
                if i != j and A[i, j] != 0:
                    p0 = pos[i]
                    p1 = pos[j]
                    # Arc height: proportional to ring distance (shorter hops => smaller arc)
                    # Use minimal circular distance in the 'order' ring
                    pi = order.index(i)
                    pj = order.index(j)
                    ring_dist = min((pj - pi) % n, (pi - pj) % n)
                    height = CURVATURE_SCALE * (0.15 + ring_dist / n)
                    patch = bezier_arc(p0, p1, height)
                    patch.set_edgecolor("red" if is_optical(i, j,xyzc_dims) else "black")
                    ax.add_patch(patch)
    else:
        # Undirected-like drawing
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] != 0 or A[j, i] != 0:
                    p0 = pos[i]
                    p1 = pos[j]
                    pi = order.index(i)
                    pj = order.index(j)
                    ring_dist = min((pj - pi) % n, (pi - pj) % n)
                    height = CURVATURE_SCALE * (0.15 + ring_dist / n)
                    patch = bezier_arc(p0, p1, height)
                    patch.set_edgecolor("red" if is_optical(i, j, xyzc_dims) else "black")
                    ax.add_patch(patch)

    ax.set_aspect('equal')
    ax.axis('off')
    # ax.set_title("Circular layout (clockwise) with grouped order by (c, z, y, x)")
    plt.tight_layout()
    plt.show()

def main():

    parser = argparse.ArgumentParser(description='Plot topologies in a circle')

    # TODO make graph not required?
    parser.add_argument('--topology',type=str,help='.map file to evaluate',default='files/3d/pt_2c_128r_6p_8x4x4.map',required=True)
    parser.add_argument('--xyzc_dims',nargs='+',type=int,help='type without parenthesis and use spaces, no commas')

    args = parser.parse_args()

    topo_filepath = args.topology
    xyzc_dims = (8, 4, 4, 4)
    if args.xyzc_dims:
        xyzc_dims = tuple(args.xyzc_dims)
        assert(len(xyzc_dims) == 4)


    A = load_adjacency_matrix(topo_filepath)
    n = A.shape[0]
    order, coords = compute_order(n, xyzc_dims)
    pos = layout_positions(order, radius=RADIUS)
    draw_graph(A, pos, order, xyzc_dims, directed=DIRECTED)

if __name__ == "__main__":
    main()
