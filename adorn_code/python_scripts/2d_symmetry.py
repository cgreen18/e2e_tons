"""
2D translational symmetry for grid topologies with wrap-around (modulus).

Node index r is row-major: r = x + y * x_dim.
Canonical tile: nodes with (x % mc_x, y % mc_y) in [0, mc_x) x [0, mc_y).
Translation uses modulus for wrap-around: (x + dx) % x_dim, (y + dy) % y_dim.
"""


class Symmetry2D:
    """
    2D translational symmetry only. No reflections.
    Interface compatible with the parts of TPUv4_Symmetry used by gen_coll_comm_a2a_*_sym scripts.
    """

    verbose = False

    def __init__(self, xy_dims, mc_dims):
        """
        Args:
            xy_dims: (x_dim, y_dim) - global grid dimensions
            mc_dims: (mc_x, mc_y) - canonical tile dimensions (translation period)
        """
        self._set_xy_dims_(xy_dims)
        self._set_mc_dims_(mc_dims)
        self.define_canonical_equivalents_()

    def _set_xy_dims_(self, xy_dims):
        assert len(xy_dims) == 2
        self.xy_dims = tuple(xy_dims)
        self.x_dim, self.y_dim = self.xy_dims
        self.n_nodes = self.x_dim * self.y_dim
        for d in xy_dims:
            assert isinstance(d, int) and d > 0

    def _set_mc_dims_(self, mc_dims):
        assert len(mc_dims) == 2
        self.mc_dims = tuple(mc_dims)
        self.mc_x, self.mc_y = self.mc_dims
        for d in mc_dims:
            assert isinstance(d, int) and d > 0
        assert self.mc_x <= self.x_dim and self.mc_y <= self.y_dim
        assert self.x_dim % self.mc_x == 0 and self.y_dim % self.mc_y == 0

    # 2D coordinate conversion (row-major)
    def _r_to_xy(self, r):
        x = r % self.x_dim
        y = r // self.x_dim
        return x, y

    def _xy_to_r(self, x, y):
        return x + y * self.x_dim

    # Translation with wrap-around (modulus)
    def translate_to_mc(self, r):
        """
        Map node r to its canonical representative in the tile [0, mc_x) x [0, mc_y).
        Returns (r_prime, trans_delta) where trans_delta is (dx, dy) such that
        apply_translation(r, trans_delta) = r_prime.
        """
        x, y = self._r_to_xy(r)
        x_c = x % self.mc_x
        y_c = y % self.mc_y
        r_prime = self._xy_to_r(x_c, y_c)
        dx = (x_c - x) % self.x_dim
        dy = (y_c - y) % self.y_dim
        trans_delta = (dx, dy)
        return r_prime, trans_delta

    def apply_translation(self, r, trans_delta):
        x, y = self._r_to_xy(r)
        dx, dy = trans_delta
        x_prime = (x + dx) % self.x_dim
        y_prime = (y + dy) % self.y_dim
        return self._xy_to_r(x_prime, y_prime)

    def apply_reverse_translation(self, r, trans_delta):
        """Return r_old such that apply_translation(r_old, trans_delta) == r."""
        x, y = self._r_to_xy(r)
        dx, dy = trans_delta
        x_old = (x - dx) % self.x_dim
        y_old = (y - dy) % self.y_dim
        return self._xy_to_r(x_old, y_old)

    def apply_reverse_transformation(self, r, transforms):
        """Return r_old such that apply_transformation(r_old, transforms) == r."""
        return self.apply_reverse_translation(r, transforms["trans"])

    def calc_translation_delta(self, r_old, r_new):
        """Return (dx, dy) such that apply_translation(r_old, (dx, dy)) == r_new."""
        x_old, y_old = self._r_to_xy(r_old)
        x_new, y_new = self._r_to_xy(r_new)
        dx = (x_new - x_old) % self.x_dim
        dy = (y_new - y_old) % self.y_dim
        return (dx, dy)

    def define_canonical_equivalents_(self):
        """Build canonical_equivalence_map, reverse_canonical_equivalence_map, canonical_transformations."""
        n_nodes = self.n_nodes
        self.canonical_equivalence_map = {}
        canons = self.get_canonical_nodes()
        self.reverse_canonical_equivalence_map = {c: [] for c in canons}
        self.canonical_transformations = {}

        for r in range(n_nodes):
            r_prime, trans_delta = self.translate_to_mc(r)
            tform = {"refl": [], "trans": trans_delta}
            self.canonical_equivalence_map[r] = r_prime
            self.reverse_canonical_equivalence_map[r_prime].append(r)
            self.canonical_transformations[r] = tform

    def apply_transformation(self, r, transforms):
        """Apply translation only (transforms['refl'] is ignored for 2D)."""
        return self.apply_translation(r, transforms["trans"])

    def calc_transform_delta(self, r_old, r_new):
        """Return transform tform such that apply_transformation(r_old, tform) == r_new."""
        trans_delta = self.calc_translation_delta(r_old, r_new)
        return {"refl": [], "trans": trans_delta}

    def get_canonical_nodes(self):
        """Return list of node indices in the canonical tile [0, mc_x) x [0, mc_y)."""
        canons = []
        for y in range(self.mc_y):
            for x in range(self.mc_x):
                canons.append(self._xy_to_r(x, y))
        return canons

    def get_canonical_equivalent(self, r):
        """Return (canonical_node, transform) for node r."""
        r_prime = self.canonical_equivalence_map[r]
        tform = self.canonical_transformations[r]
        return r_prime, tform

    def verify_symmetry_for_topology(self, adj_mat):
        """Verify that the adjacency matrix is invariant under 2D translational symmetry."""
        n_nodes = len(adj_mat)
        assert n_nodes == self.n_nodes
        for i in range(n_nodes):
            i_prime, i_tform = self.get_canonical_equivalent(i)
            for j in range(n_nodes):
                if adj_mat[i][j] == 0:
                    continue
                j_prime = self.apply_transformation(j, i_tform)
                if adj_mat[i_prime][j_prime] != adj_mat[i][j]:
                    print(
                        f"Symmetry mismatch: edge ({i},{j}) -> ({i_prime},{j_prime}) "
                        f"by transform {i_tform}"
                    )
                    assert adj_mat[i_prime][j_prime] == adj_mat[i][j]
        print(
            f"Success! Topology of dimensions {self.xy_dims} has 2D translational "
            f"symmetry with tile {self.mc_dims}"
        )
