import random
import pytest

from tpuv4_symmetry import TPUv4_Symmetry


def xyz_to_r(x, y, z, x_dim, y_dim):
    return x + y * x_dim + z * x_dim * y_dim


def expected_canonical_trans(xyzc_dims, mc_dims, r):
    x_dim, y_dim, z_dim, cube_dim = xyzc_dims
    mc_x, mc_y, mc_z = mc_dims

    xy_slice = x_dim * y_dim
    z = r // xy_slice
    t = r % xy_slice
    y = t // x_dim
    x = t % x_dim

    cx = x % mc_x
    cy = y % mc_y
    cz = z % mc_z
    r_can = xyz_to_r(cx, cy, cz, x_dim, y_dim)

    trans = (cx - x, cy - y, cz - z)
    return r_can, {"refl": [], "trans": trans}


def expected_canonical_refl_trans(xyzc_dims, mc_dims, r):
    """
    This matches the CURRENT behavior in reflect_to_within_mc_hemisphere():
      hemi_dim = max(dim//2, cube_dim)
      reflect if coord >= hemi_dim
    then translate into mega-cube via % mc_dim.
    """
    x_dim, y_dim, z_dim, cube_dim = xyzc_dims
    mc_x, mc_y, mc_z = mc_dims

    xy_slice = x_dim * y_dim
    z = r // xy_slice
    t = r % xy_slice
    y = t // x_dim
    x = t % x_dim

    hemi_x = max(x_dim // 2, cube_dim)
    hemi_y = max(y_dim // 2, cube_dim)
    hemi_z = max(z_dim // 2, cube_dim)

    refl = []
    if x >= hemi_x:
        x = (x_dim - 1) - x
        refl.append("x")
    if y >= hemi_y:
        y = (y_dim - 1) - y
        refl.append("y")
    if z >= hemi_z:
        z = (z_dim - 1) - z
        refl.append("z")

    cx = x % mc_x
    cy = y % mc_y
    cz = z % mc_z
    r_can = xyz_to_r(cx, cy, cz, x_dim, y_dim)

    trans = (cx - x, cy - y, cz - z)
    return r_can, {"refl": refl, "trans": trans}


@pytest.mark.parametrize(
    "xyzc_dims,mc_dims,sym_type,r,expected_xyz,expected_refl,expected_trans",
    [
        # refl-trans examples (8x8x8, cube=4, mc=4x4x4)
        ((8, 8, 8, 4), (4, 4, 4), "refl-trans", xyz_to_r(6, 1, 2, 8, 8), (1, 1, 2), ["x"], (0, 0, 0)),
        ((8, 8, 8, 4), (4, 4, 4), "refl-trans", xyz_to_r(2, 6, 7, 8, 8), (2, 1, 0), ["y", "z"], (0, 0, 0)),
        ((8, 8, 8, 4), (4, 4, 4), "refl-trans", xyz_to_r(5, 5, 5, 8, 8), (2, 2, 2), ["x", "y", "z"], (0, 0, 0)),

        # trans-only example (16x8x8, cube=4, mc=8x4x4)
        ((16, 8, 8, 4), (8, 4, 4), "trans", xyz_to_r(9, 5, 6, 16, 8), (1, 1, 2), [], (-8, -4, -4)),
    ],
)
def test_known_canonical_examples(xyzc_dims, mc_dims, sym_type, r, expected_xyz, expected_refl, expected_trans):
    sym = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type=sym_type)

    r_can, tform = sym.get_canonical_equivalent(r)

    x_dim, y_dim, z_dim, cube_dim = xyzc_dims
    exp_r_can = xyz_to_r(*expected_xyz, x_dim, y_dim)

    assert r_can == exp_r_can
    assert tform["refl"] == expected_refl
    assert tform["trans"] == expected_trans

    # Transform must map r -> canonical
    assert sym.apply_transformation(r, tform) == r_can


def test_roundtrip_transformation_matches_calc_transform_delta_trans():
    xyzc_dims = (16, 8, 8, 4)
    mc_dims = (8, 4, 4)
    sym = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type="trans")

    for _ in range(200):
        r = random.randrange(sym.n_nodes)
        r_can, tform = sym.get_canonical_equivalent(r)

        # For trans, calc_transform_delta should match canonical-transform exactly.
        tform2 = sym.calc_transform_delta(r, r_can)
        assert tform2 == tform
        assert sym.apply_transformation(r, tform2) == r_can


def test_roundtrip_transformation_maps_correctly_refl_trans():
    xyzc_dims = (8, 8, 8, 4)
    mc_dims = (4, 4, 4)
    sym = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type="refl-trans")

    for _ in range(500):
        r = random.randrange(sym.n_nodes)
        r_can, _ = sym.get_canonical_equivalent(r)

        # For refl-trans, validate correctness-by-application (and lattice constraint).
        tform2 = sym.calc_transform_delta(r, r_can)
        assert sym.apply_transformation(r, tform2) == r_can

        dx, dy, dz = tform2["trans"]
        mc_x, mc_y, mc_z = mc_dims
        assert dx % mc_x == 0
        assert dy % mc_y == 0
        assert dz % mc_z == 0


def test_apply_reflection_accepts_list_and_string():
    xyzc_dims = (8, 8, 8, 4)
    mc_dims = (4, 4, 4)
    sym = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type="refl-trans")

    r = xyz_to_r(6, 1, 2, 8, 8)  # (6,1,2)

    # reflect over x should go to (1,1,2)
    r_list = sym.apply_reflection(r, ["x"])
    r_str = sym.apply_reflection(r, "x")  # current code iterates the string
    assert r_list == r_str
    assert r_list == xyz_to_r(1, 1, 2, 8, 8)


def test_calc_reverse_canonical_transform_delta_is_inverse():
    xyzc_dims = (16, 8, 8, 4)
    mc_dims = (8, 4, 4)
    sym = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type="trans")

    for _ in range(200):
        r = random.randrange(sym.n_nodes)
        r_can, tform = sym.get_canonical_equivalent(r)

        inv = sym.calc_reverse_canonical_transform_delta(r, r_can)
        assert sym.apply_transformation(r, tform) == r_can
        assert sym.apply_transformation(r_can, inv) == r


def test_apply_reverse_transformation_is_unimplemented_and_should_fail_fast():
    """
    apply_reverse_transformation() currently calls undefined methods:
      apply_reverse_reflection()
      apply_reverse_translation()
    So this should raise AttributeError until you implement them.
    """
    xyzc_dims = (8, 8, 8, 4)
    mc_dims = (4, 4, 4)
    sym = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type="refl-trans")

    with pytest.raises(AttributeError):
        sym.apply_reverse_transformation(0, {"refl": ["x"], "trans": (0, 0, 0)})


def test_expected_canonical_formulas_match_implementation_for_random_nodes():
    # This validates the canonical-mapping spec (as CURRENTLY CODED) over many nodes.
    for sym_type in ["trans", "refl-trans"]:
        xyzc_dims = (16, 8, 8, 4)
        mc_dims = (8, 4, 4)
        sym = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type=sym_type)

        for _ in range(500):
            r = random.randrange(sym.n_nodes)

            if sym_type == "trans":
                exp_can, exp_tform = expected_canonical_trans(xyzc_dims, mc_dims, r)
            else:
                exp_can, exp_tform = expected_canonical_refl_trans(xyzc_dims, mc_dims, r)

            r_can, tform = sym.get_canonical_equivalent(r)

            assert r_can == exp_can
            assert tform == exp_tform
            assert sym.apply_transformation(r, tform) == r_can
