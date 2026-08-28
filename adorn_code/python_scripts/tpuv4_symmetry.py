import inspect
import argparse
from collections import defaultdict

import networkx as nx

def UNIMPLEMENTED(msg):
    # inspect.currentframe() → the current stack frame (inside callee)
    # .f_back → the previous frame on the stack (i.e., the caller)
    # .f_code.co_name → the name of the function associated with that frame
    caller_name = inspect.currentframe().f_back.f_code.co_name
    print(f"UNIMPLEMENTED :: {caller_name}() :: {msg}")
    quit()

# Function naming:
#   _ prefix => internal call only
#   _ suffix => no return

class TPUv4_Symmetry:

    verbose = False

    supported_sym_types = ["refl-trans","trans"]

    # initial and later setups
    ################################################################################

    def __init__(self, xyzc_dims, adj_list=None, mc_dims=None, sym_type='refl-trans'):
        self._set_xyzc_dims_(xyzc_dims)
        self.n_nodes = self.x_dim * self.y_dim * self.z_dim

        assert(sym_type in self.supported_sym_types)
        self.sym_type = sym_type

        if adj_list and mc_dims:
            self._set_adj_list_(adj_list)
            self._set_canonical_mega_cube_(mc_dims)
        elif adj_list:
            self._set_adj_list_(adj_list)
            mc_dims = self._detect_canonical_mega_cube()
            self._set_canonical_mega_cube_(mc_dims)
        elif mc_dims:
            self._set_canonical_mega_cube_(mc_dims)
        else:
            # assume basic cube
            _assumed_mc_dims = (self.cube_dim,self.cube_dim,self.cube_dim)
            self._set_canonical_mega_cube_(_assumed_mc_dims)


        # cache
        self.transform_cache = defaultdict(dict)
        self.get_transform_cache = defaultdict(dict)
        self.define_canonical_equivalents_()



    def _set_xyzc_dims_(self, xyzc_dims):
        assert(len(xyzc_dims) == 4)
        self.xyzc_dims = xyzc_dims
        (self.x_dim, self.y_dim, self.z_dim, self.cube_dim) = xyzc_dims
        self.dim_dict = {'x':self.x_dim,'y':self.y_dim,'z':self.z_dim}
        for dim in xyzc_dims:
            assert(isinstance(dim,int))
            assert(dim % self.cube_dim == 0)

    def _set_adj_list_(self, adj_list):
        assert(len(adj_list) == self.n_nodes)
        for conns in adj_list:
            for n in conns:
                assert(n >=0 and n < self.n_nodes)

        self.adj_list = adj_list

    def _set_canonical_mega_cube_(self, mc_dims):
        assert(len(mc_dims) == 3)
        for dim in mc_dims:
            assert(isinstance(dim,int))

        (self.mc_x, self.mc_y, self.mc_z) = mc_dims
        self.mc_dims = mc_dims

    def set_adj_list_(self, adj_list):
        self._set_adj_list_(adj_list)
        mc_dims = self._detect_canonical_mega_cube()
        self._set_canonical_mega_cube_(mc_dims)

    def set_canonical_mega_cube_(self, mc_dims):
        needs_reset = False if mc_dims == self.mc_dims else True
        self._set_canonical_mega_cube_(mc_dims)
        if needs_reset:
            self.define_canonical_equivalents_()

    # dimensional
    ################################################################################

    def _r_to_xyz(self,r):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = (self.x_dim, self.y_dim, self.z_dim, self.cube_dim)

        xy_slice_size = x_dim*y_dim

        temp_r = r

        z = temp_r // xy_slice_size
        temp_r = temp_r % xy_slice_size
        y = temp_r // x_dim
        x = temp_r % x_dim

        return x,y,z

    def _xyz_to_r(self,x,y,z):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = (self.x_dim, self.y_dim, self.z_dim, self.cube_dim)

        return x + y*x_dim + z*x_dim*y_dim

    # translations
    ################################################################################

    def _translate_r_to_rel_mc_xyz(self, r):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = (self.x_dim, self.y_dim, self.z_dim, self.cube_dim)
        assert(self.mc_dims)
        (mc_x, mc_y, mc_z) = (self.mc_x, self.mc_y, self.mc_z)

        r_x,r_y,r_z = self._r_to_xyz(r)

        rel_r_x = r_x % mc_x
        rel_r_y = r_y % mc_y
        rel_r_z = r_z % mc_z

        return rel_r_x, rel_r_y, rel_r_z

    def _translate_r_to_rel_xyz(self, r):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = (self.x_dim, self.y_dim, self.z_dim, self.cube_dim)

        r_x,r_y,r_z = self._r_to_xyz(r)

        rel_r_x = r_x % cube_dim
        rel_r_y = r_y % cube_dim
        rel_r_z = r_z % cube_dim

        return rel_r_x, rel_r_y, rel_r_z

    def _translate_r_to_rel_mc_r(self, r):
        (rel_x, rel_y, rel_z) = self._translate_r_to_rel_mc_xyz(r)
        rel_r = self._xyz_to_r(rel_x, rel_y, rel_z)
        return rel_r

    def _translate_r_to_rel_r(self, r):
        (rel_x, rel_y, rel_z) = self._translate_r_to_rel_xyz(r)
        rel_r = self._xyz_to_r(rel_x, rel_y, rel_z)
        return rel_r

    def calc_translation_delta(self, r_old, r_new):
        (r_x, r_y, r_z) = self._r_to_xyz(r_old)
        (ref_x, ref_y, ref_z) = self._r_to_xyz(r_new)

        d_x = (ref_x - r_x)
        d_y = (ref_y - r_y)
        d_z = (ref_z - r_z)

        return (d_x, d_y, d_z)

    def translate_to_mc(self, r):

        r_prime = self._translate_r_to_rel_mc_r(r)
        trans_delta = self.calc_translation_delta(r,r_prime)

        if self.verbose:
            print(f"{r} @ {self._r_to_xyz(r)} => {r_prime} @ {self._r_to_xyz(r_prime)} w/ trans_delta = {trans_delta}")

        return r_prime, trans_delta

    def apply_translation(self, r, trans_delta):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = (self.x_dim, self.y_dim, self.z_dim, self.cube_dim)

        (r_x, r_y, r_z) = self._r_to_xyz(r)
        (d_x, d_y, d_z) = trans_delta

        # modulo handles negative values correctly (e.g., (0 - 2) % 8 = 6 )
        r_prime_x = (r_x + d_x) % x_dim
        r_prime_y = (r_y + d_y) % y_dim
        r_prime_z = (r_z + d_z) % z_dim

        r_prime = self._xyz_to_r(r_prime_x, r_prime_y, r_prime_z)
        return r_prime

    def apply_reverse_translation(self, r, trans_delta):
        """Given r and trans_delta, return r_old such that apply_translation(r_old, trans_delta) == r."""
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = (self.x_dim, self.y_dim, self.z_dim, self.cube_dim)

        (r_x, r_y, r_z) = self._r_to_xyz(r)
        (d_x, d_y, d_z) = trans_delta

        r_old_x = (r_x - d_x) % x_dim
        r_old_y = (r_y - d_y) % y_dim
        r_old_z = (r_z - d_z) % z_dim

        r_old = self._xyz_to_r(r_old_x, r_old_y, r_old_z)
        return r_old

    # reflections
    ################################################################################

    def reflect_to_within_mc_hemisphere(self, r):
        assert(self.dim_dict)
        dim_dict = self.dim_dict
        cube_dim = self.cube_dim

        (r_x,r_y,r_z) = self._r_to_xyz(r)
        r_dims = {'x':r_x,'y':r_y,'z':r_z}

        if self.verbose:
            print(f'reflecting to hemisphere : {r} @ {r_dims}')

        # define hemispheres
        hemi_dict = {}
        for dim in ['x','y','z']:
            # max of cube_dim or (dim // 2) rounded/truncated to nearest cube_dim?
            # TODO
            h_dim = max(dim_dict[dim] // 2, cube_dim)
            hemi_dict[dim] = h_dim

        r_prime = r
        refl_dim = []
        for dim in ['x','y','z']:
            if r_dims[dim] >= hemi_dict[dim]:

                r_prime = self.apply_reflection(r_prime, dim)
                refl_dim.append(dim)
        
        if self.verbose:
            print(f"{r} @ {self._r_to_xyz(r)} => {r_prime} @ {self._r_to_xyz(r_prime)} w/ refl_dim = {refl_dim}")

        return r_prime, refl_dim

    def apply_reflection(self, r, refl_dim):
        assert(self.dim_dict)
        dim_dict = self.dim_dict

        (r_x,r_y,r_z) = self._r_to_xyz(r)
        r_dims = {'x':r_x,'y':r_y,'z':r_z}

        for dim in refl_dim:
            r_dim_prime = (dim_dict[dim] - 1) - r_dims[dim]

            if self.verbose:
                print(f"After reflecting over {dim}, the coordinate {r_dims[dim]}=>{r_dim_prime}")

            r_dims[dim] = r_dim_prime

        r_prime = self._xyz_to_r(r_dims['x'],r_dims['y'],r_dims['z'])
        return r_prime

    # transforms
    ################################################################################

    def calc_reflection_translation_delta(self, r, r_prime):

        # refl-trans: find (refl_dims, trans_delta) such that:
        #   r_mid  = reflect(r, refl_dims)
        #   r'     = translate(r_mid, trans_delta)
        #
        # Additionally, to respect the "mega-cube" translation symmetry that your
        # canonical equivalence construction implies, we require:
        #   trans_delta[x] % mc_x == 0, trans_delta[y] % mc_y == 0, trans_delta[z] % mc_z == 0

        assert(self.mc_dims)
        (mc_x, mc_y, mc_z) = self.mc_dims

        dims = ["x", "y", "z"]

        best_transform = None
        best_score = None  # lexicographic: (fewest reflections, smallest |dx|+|dy|+|dz|)

        # Enumerate all reflection subsets over {x,y,z} (8 total)
        for mask in range(8):

            refl_dims = []
            for i, dim in enumerate(dims):
                if (mask >> i) & 1:
                    refl_dims.append(dim)

            # Apply reflection first (consistent with apply_transformation())
            r_mid = self.apply_reflection(r, refl_dims)

            # Translation needed from r_mid -> r_prime
            trans_delta = self.calc_translation_delta(r_mid, r_prime)

            # Require translation to be on the mega-cube lattice
            if (trans_delta[0] % mc_x) != 0:
                continue
            if (trans_delta[1] % mc_y) != 0:
                continue
            if (trans_delta[2] % mc_z) != 0:
                continue

            # Sanity: must actually map exactly under your translation operator
            # (modulo x_dim/y_dim/z_dim inside apply_translation)
            if self.apply_translation(r_mid, trans_delta) != r_prime:
                continue

            # Deterministic choice:
            #  1) minimize number of reflected dimensions
            #  2) then minimize L1 magnitude of translation vector
            score = (len(refl_dims), abs(trans_delta[0]) + abs(trans_delta[1]) + abs(trans_delta[2]))

            if best_transform is None or score < best_score:
                best_transform = {"refl": refl_dims, "trans": trans_delta}
                best_score = score

        # If nothing matched the mega-cube lattice, fall back to a direct mapping.
        # This still satisfies apply_transformation(r, tform) == r_prime, but may not
        # correspond to a *valid* symmetry transform for your chosen mc_dims.
        if best_transform is None:
            if self.verbose:
                print(f"[WARN] calc_transform_delta(): no refl-trans symmetry move found on mc lattice.")
                print(f"       Falling back to pure translation delta for {r} -> {r_prime}")
            return {"refl": [], "trans": self.calc_translation_delta(r, r_prime)}

        return best_transform

    def _detect_canonical_mega_cube(self):
        UNIMPLEMENTED("Entire function")

    def define_canonical_equivalents_(self):
        # for each node, find it's equivalent in mega cube
        # AND find the reflectional and translational deltas for that transformation

        assert(self.n_nodes)
        assert(self.sym_type)
        n_nodes = self.n_nodes
        use_refl = True if self.sym_type == "refl-trans" else False

        self.canonical_equivalence_map = {}
        self.reverse_canonical_equivalence_map = {n : [] for n in self.get_canonical_nodes()}
        self.canonical_transformations = {}

        for i in range(n_nodes):

            i_hemi, refl_dim = i, []
            if use_refl:
                i_hemi, refl_dim = self.reflect_to_within_mc_hemisphere(i)

            i_prime, trans_delta = self.translate_to_mc(i_hemi)
            canon_trans = {'refl':refl_dim,'trans':trans_delta}

            self.canonical_equivalence_map[i] = i_prime
            self.reverse_canonical_equivalence_map[i_prime].append(i)
            self.canonical_transformations[i] = canon_trans
            tkey = str(canon_trans)
            self.transform_cache[i][tkey] = i_prime
            self.get_transform_cache[i][i_prime] = canon_trans

            if self.verbose:
                print(f"canonical equivalent : {i} -> {i_prime} by {self._r_to_xyz(i)} -> {self._r_to_xyz(i_prime)}")
                print(f"canonical transform  : {canon_trans}")
                print(f"    REFL    : {self._r_to_xyz(i)} => {i_hemi} @ {self._r_to_xyz(i_hemi)}")
                print(f"    TRANS   : {i_hemi} => {i_prime}")
                print()

    def calc_reverse_canonical_transform_delta(self, r, r_prime):
        assert(r_prime in self.get_canonical_nodes())
        assert(r_prime == self.canonical_equivalence_map[r])
        # r -> r_prime (r_prime in canonical)

        tform = self.canonical_transformations[r]

        (trot, tlate) = (tform["refl"], tform["trans"])

        tlate_prime = (tlate[0]*-1,tlate[1]*-1,tlate[2]*-1)

        tform_prime = {"refl":trot,"trans":tlate_prime}

        # print(f"reversing r {r} -> {r_prime} by tform_prime {tform_prime}")
        assert(r == self.apply_transformation(r_prime,tform_prime))

        return tform_prime

    def calc_transform_delta(self, r, r_prime):

        if r_prime in self.get_transform_cache[r]:
            return self.get_transform_cache[r][r_prime]

        if self.sym_type == "refl-trans":
            return self.calc_reflection_translation_delta(r,r_prime)
        
        else:
            return {"refl":[] , "trans":self.calc_translation_delta(r,r_prime)}

    def apply_transformation(self, r, transforms):
        tkey = str(transforms)

        if tkey in self.transform_cache[r]:
            return self.transform_cache[r][tkey]

        assert(self.sym_type)
        use_refl = True if self.sym_type == "refl-trans" else False

        r_mid = r
        if use_refl:
            r_mid = self.apply_reflection(r, transforms["refl"])
        r_prime = self.apply_translation(r_mid, transforms["trans"])
        self.transform_cache[r][tkey] = r_prime
        self.get_transform_cache[r][r_prime] = transforms
        return r_prime

    def apply_reverse_transformation(self, r, transforms):
        assert(self.sym_type)
        use_refl = True if self.sym_type == "refl-trans" else False

        r_mid = r
        if use_refl:
            r_mid = self.apply_reverse_reflection(r, transforms["refl"])
        r_prime = self.apply_reverse_translation(r_mid, transforms["trans"])

        return r_prime

    # queries
    ################################################################################

    def get_canonical_nodes(self):
        assert(self.mc_dims)
        (mc_x, mc_y, mc_z) = self.mc_dims

        canons = []
        for z in range(mc_z):
            for y in range(mc_y):
                for x in range(mc_x):
                    r = self._xyz_to_r(x,y,z)
                    canons.append(r)
        return canons

    def get_canonical_equivalent(self, r):
        r_prime = self.canonical_equivalence_map[r]
        r_transform = self.canonical_transformations[r]

        # cache
        self.transform_cache[r][str(r_transform)] = r_prime

        return r_prime, r_transform

    def get_all_noncanonical_equivalents(self, r):
        return self.reverse_canonical_equivalence_map[r]

    def get_canonical_equivalent_edge(self, i, j):
        i_prime, i_transform = self.get_canonical_equivalent(i)
        j_prime = self.apply_transformation(j, i_transform)
        return (i_prime, j_prime)

    def get_all_equivalent_paths(self, base_path):
        assert(len(base_path) > 1)

        (sr_canon, dr_canon) = (base_path[0], base_path[-1])
        all_sr_noncanons = self.get_all_noncanonical_equivalents(sr_canon)


        equivalent_paths = []

        for sr_noncanon in all_sr_noncanons:
            if sr_noncanon == sr_canon:
                continue

            # sr_canon -> sr_noncanon
            sr_canon_to_noncanon_tform = self.calc_transform_delta(sr_canon, sr_noncanon)

            dr_noncanon = self.apply_transformation(dr_canon, sr_canon_to_noncanon_tform)

            new_path = []
            for n_canon in base_path:

                n_noncanon = self.apply_transformation(n_canon, sr_canon_to_noncanon_tform)

                new_path.append(n_noncanon)


            # validation
            assert new_path[0] == sr_noncanon, f"sr_noncanon {sr_noncanon} not first node in new_path {new_path}"
            assert new_path[-1] == dr_noncanon, f"dr_noncanon {dr_noncanon} not last node in new_path {new_path}"

            equivalent_paths.append(new_path)

        return equivalent_paths


    def verify_symmetry_for_topology(self, adj_mat, verify_dist=False):
        n_nodes = len(adj_mat)

        for i, i_conns in enumerate(adj_mat):
            for j, conn in enumerate(i_conns):

                # uncomment if just want to check positive connections
                # if conn == 0:
                #     continue

                i_prime, i_transform = self.get_canonical_equivalent(i)
                j_prime = self.apply_transformation(j, i_transform)
                if adj_mat[i_prime][j_prime] != adj_mat[i][j]:
                    print(f"Symmetry mismatch!")
                    print(f"    Conn {i}->{j} (conn? {adj_mat[i][j]})")
                    print(f"        Becomes {i_prime}->{j_prime} (conn? {adj_mat[i_prime][j_prime]})")
                    print(f"        By transform {i_transform}")
                    print(f"            Moving i {i} @ {self._r_to_xyz(i)} to {self._r_to_xyz(i_prime)}")
                    print(f"            Moving j {j} @ {self._r_to_xyz(j)} to {self._r_to_xyz(j_prime)}")


                assert(adj_mat[i_prime][j_prime] == adj_mat[i][j])

        print(f"Success! Topology of dimensions {self.xyzc_dims} has {self.sym_type} edge symmetry with mega cube {self.mc_dims}")
        

        if not verify_dist:
            return

        dist_mat = self.calculate_all_pairs_distances(adj_mat)

        # for s, d_dict in dist_mat.items():
        #     print(f"{s} :")
        #     for d, dist in d_dict.items():
        #         print(f"\t{d} : {dist}")
        # input(f"dist_mat")

        for i in range(n_nodes):
            for j in range(n_nodes):

                # uncomment if just want to check positive connections
                # if conn == 0:
                #     continue

                i_prime, i_transform = self.get_canonical_equivalent(i)
                j_prime = self.apply_transformation(j, i_transform)
                if dist_mat[i_prime][j_prime] != dist_mat[i][j]:
                    print(f"Symmetry mismatch!")
                    print(f"    Conn {i}->{j} (dist? {dist_mat[i][j]})")
                    print(f"        Becomes {i_prime}->{j_prime} (dist? {dist_mat[i_prime][j_prime]})")
                    print(f"        By transform {i_transform}")
                    print(f"            Moving i {i} @ {self.r_to_xyz(i)} to {self.r_to_xyz(i_prime)}")
                    print(f"            Moving j {j} @ {self.r_to_xyz(j)} to {self.r_to_xyz(j_prime)}")


                assert(dist_mat[i_prime][j_prime] == dist_mat[i][j])

        print(f"Success! Topology of dimensions {self.xyzc_dims} has {self.sym_type} distance symmetry with mega cube {self.mc_dims}")

    def calculate_all_pairs_distances(self, adj_mat):
        n_nodes = len(adj_mat)

        G = self.create_an_nwx_G_from_adj_mat(adj_mat)
        hops_dict = dict(nx.all_pairs_shortest_path_length(G))

        return hops_dict

    def create_an_nwx_G_from_adj_mat(self, adj_mat):
        n_nodes = len(adj_mat)

        G = nx.DiGraph()
        for src in range(n_nodes):
            for dest in range(n_nodes):
                if src==dest:
                    continue
                if adj_mat[src][dest] == 0:
                    continue

                if self.verbose:
                    print(f'connecting {src} -> {dest}')

                G.add_edge(src,dest)

        return G


# standalone
################################################################################

def transform_conn(tpuv4_symmetry, conn):
    (i,j) = conn

    i_prime, i_transform = tpuv4_symmetry.get_canonical_equivalent(i)
    j_prime = tpuv4_symmetry.apply_transformation(j, i_transform)

    print(f"Conn {i}->{j}")
    print(f"    Becomes {i_prime}->{j_prime}")
    print(f"    By transform {i_transform}")
    print(f"        Moving i {i} @ {tpuv4_symmetry.r_to_xyz(i)} to {tpuv4_symmetry.r_to_xyz(i_prime)}")
    print(f"        Moving j {j} @ {tpuv4_symmetry.r_to_xyz(j)} to {tpuv4_symmetry.r_to_xyz(j_prime)}")

def ingest_map(path_name):
    file_name = path_name.split('/')[-1]

    if True:
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

    return this_map

def test_topology(file_path, xyzc_dims, mc_dims, sym_type):

    print(f"Verifying {file_path} of dimensions {xyzc_dims} with mega cube {mc_dims}")

    adj_mat = ingest_map(file_path)

    my_tpuv4_symmetry = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type=sym_type)
    my_tpuv4_symmetry.verify_symmetry_for_topology(adj_mat, verify_dist=True)

def test_by_hand():

    xyzc_dims = (16, 4, 4, 4)
    mc_dims = (8,4,4)
    print(f'xyzc_dims = {xyzc_dims}')
    my_tpuv4_symmetry = TPUv4_Symmetry(xyzc_dims)

    my_tpuv4_symmetry.set_canonical_mega_cube_(mc_dims)

    example_conns = [(0,1), (4,3), (11,12), (5,17), (48,4), (56,12)]
    for conn in example_conns:
        transform_conn(my_tpuv4_symmetry, conn)

def main():

    parser = argparse.ArgumentParser(description='Class for determining symmetries. Verify symmetry of topologies by direct call. PDTT kxkx2k => mc=(dx/2, dy/2, 1). kx2kx2k => mc=(dx/2, 1, 1)')
    parser.add_argument('--graph','-g',type=str,help='.map file to evaluate',required=True)
    parser.add_argument('--xyzc_dims',nargs='+',type=int,required=True,help='Global system x, y, z, and cube dimensions. Type without parenthesis and use spaces, no commas')
    parser.add_argument('--mc_dims',nargs='+',type=int,help='Minimum "cube" of symmetry\'s x, y, and z dimensions. If unspecified then assuming single (zeroth) cube (i.e., x,y,z==cube). Type without parenthesis and use spaces, no commas')
    parser.add_argument('--sym_type',type=str,choices=["refl-trans","trans"],default="trans",help="reflection and translation (in that order) or just translation)")

    args = parser.parse_args()

    map_filename = args.graph
    xyzc_dims = tuple(args.xyzc_dims)
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    mc_dims = (cube_dim,cube_dim,cube_dim)
    if args.mc_dims:
        mc_dims = tuple(args.mc_dims)
    # default in argparser
    sym_type = args.sym_type

    assert(len(xyzc_dims) == 4)
    for d in xyzc_dims:
        assert(isinstance(d,int))
        assert( d % cube_dim == 0)
        assert( d > 0)
    assert(len(mc_dims) == 3)
    for d in mc_dims:
        assert( d > 0)

    # test_by_hand()

    test_topology(map_filename, xyzc_dims, mc_dims, sym_type)

if __name__ == "__main__":
    main()