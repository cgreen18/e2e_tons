#include "tpuv4_symmetry.hpp"
#include <sstream>
#include <algorithm>
#include <cmath>

namespace tpuv4 {

const std::vector<std::string> TPUv4_Symmetry::supported_sym_types = {"refl-trans", "trans"};

static std::string transform_to_key(const Transform& t) {
    std::ostringstream os;
    os << "refl:[";
    for (size_t i = 0; i < t.refl.size(); ++i) {
        if (i) os << ",";
        os << t.refl[i];
    }
    os << "]trans:(" << t.trans[0] << "," << t.trans[1] << "," << t.trans[2] << ")";
    return os.str();
}

TPUv4_Symmetry::TPUv4_Symmetry(const std::array<int, 4>& xyzc_dims,
                               const std::array<int, 3>* mc_dims_ptr,
                               const std::string& sym_type) {
    if (sym_type != "refl-trans" && sym_type != "trans")
        throw std::runtime_error("unsupported sym_type");
    sym_type_ = sym_type;
    set_xyzc_dims(xyzc_dims);
    n_nodes = x_dim_ * y_dim_ * z_dim_;
    if (mc_dims_ptr) {
        set_canonical_mega_cube(*mc_dims_ptr);
    } else {
        set_canonical_mega_cube({cube_dim_, cube_dim_, cube_dim_});
    }
    define_canonical_equivalents();
}

void TPUv4_Symmetry::set_xyzc_dims(const std::array<int, 4>& xyzc_dims) {
    xyzc_dims_ = xyzc_dims;
    x_dim_ = xyzc_dims[0];
    y_dim_ = xyzc_dims[1];
    z_dim_ = xyzc_dims[2];
    cube_dim_ = xyzc_dims[3];
    dim_arr_ = {x_dim_, y_dim_, z_dim_};
    for (int d : xyzc_dims)
        if (d % cube_dim_ != 0)
            throw std::runtime_error("xyzc dim not divisible by cube_dim");
}

void TPUv4_Symmetry::set_canonical_mega_cube(const std::array<int, 3>& mc_dims) {
    mc_x_ = mc_dims[0];
    mc_y_ = mc_dims[1];
    mc_z_ = mc_dims[2];
    mc_dims_ = mc_dims;
    for (int d : mc_dims)
        if (d % cube_dim_ != 0)
            throw std::runtime_error("mc_dim not divisible by cube_dim");
}

int TPUv4_Symmetry::xyz_to_r(int x, int y, int z) const {
    return x + y * x_dim_ + z * x_dim_ * y_dim_;
}

void TPUv4_Symmetry::r_to_xyz(int r, int& out_x, int& out_y, int& out_z) const {
    int xy_slice = x_dim_ * y_dim_;
    out_z = r / xy_slice;
    r %= xy_slice;
    out_y = r / x_dim_;
    out_x = r % x_dim_;
}

void TPUv4_Symmetry::translate_r_to_rel_mc_xyz(int r, int& rel_x, int& rel_y, int& rel_z) const {
    int rx, ry, rz;
    r_to_xyz(r, rx, ry, rz);
    rel_x = ((rx % mc_x_) + mc_x_) % mc_x_;
    rel_y = ((ry % mc_y_) + mc_y_) % mc_y_;
    rel_z = ((rz % mc_z_) + mc_z_) % mc_z_;
}

int TPUv4_Symmetry::translate_r_to_rel_mc_r(int r) const {
    int rel_x, rel_y, rel_z;
    translate_r_to_rel_mc_xyz(r, rel_x, rel_y, rel_z);
    return xyz_to_r(rel_x, rel_y, rel_z);
}

void TPUv4_Symmetry::calc_translation_delta(int r_old, int r_new, int& d_x, int& d_y, int& d_z) const {
    int rx, ry, rz, ref_x, ref_y, ref_z;
    r_to_xyz(r_old, rx, ry, rz);
    r_to_xyz(r_new, ref_x, ref_y, ref_z);
    d_x = ref_x - rx;
    d_y = ref_y - ry;
    d_z = ref_z - rz;
}

void TPUv4_Symmetry::translate_to_mc(int r, int& r_prime, int& d_x, int& d_y, int& d_z) {
    r_prime = translate_r_to_rel_mc_r(r);
    calc_translation_delta(r, r_prime, d_x, d_y, d_z);
}

int TPUv4_Symmetry::apply_translation(int r, int d_x, int d_y, int d_z) const {
    int rx, ry, rz;
    r_to_xyz(r, rx, ry, rz);
    int rpx = ((rx + d_x) % x_dim_ + x_dim_) % x_dim_;
    int rpy = ((ry + d_y) % y_dim_ + y_dim_) % y_dim_;
    int rpz = ((rz + d_z) % z_dim_ + z_dim_) % z_dim_;
    return xyz_to_r(rpx, rpy, rpz);
}

int TPUv4_Symmetry::apply_reverse_translation(int r, int d_x, int d_y, int d_z) const {
    int rx, ry, rz;
    r_to_xyz(r, rx, ry, rz);
    int rpx = ((rx - d_x) % x_dim_ + x_dim_) % x_dim_;
    int rpy = ((ry - d_y) % y_dim_ + y_dim_) % y_dim_;
    int rpz = ((rz - d_z) % z_dim_ + z_dim_) % z_dim_;
    return xyz_to_r(rpx, rpy, rpz);
}

int TPUv4_Symmetry::apply_reflection(int r, const std::vector<std::string>& refl_dim) const {
    int rx, ry, rz;
    r_to_xyz(r, rx, ry, rz);
    int r_dims[3] = {rx, ry, rz};
    const int dim_sizes[3] = {x_dim_, y_dim_, z_dim_};
    const char dim_names[3] = {'x', 'y', 'z'};
    for (const std::string& dim : refl_dim) {
        int idx = (dim == "x") ? 0 : (dim == "y") ? 1 : 2;
        r_dims[idx] = (dim_sizes[idx] - 1) - r_dims[idx];
    }
    return xyz_to_r(r_dims[0], r_dims[1], r_dims[2]);
}

void TPUv4_Symmetry::reflect_to_within_mc_hemisphere(int r, int& r_prime, std::vector<std::string>& refl_dim) {
    int rx, ry, rz;
    r_to_xyz(r, rx, ry, rz);
    int r_dims[3] = {rx, ry, rz};
    int hemi[3];
    hemi[0] = std::max(x_dim_ / 2, cube_dim_);
    hemi[1] = std::max(y_dim_ / 2, cube_dim_);
    hemi[2] = std::max(z_dim_ / 2, cube_dim_);
    refl_dim.clear();
    r_prime = r;
    for (int i = 0; i < 3; ++i) {
        if (r_dims[i] >= hemi[i]) {
            refl_dim.push_back(i == 0 ? "x" : (i == 1 ? "y" : "z"));
            r_dims[i] = (dim_arr_[i] - 1) - r_dims[i];
        }
    }
    r_prime = xyz_to_r(r_dims[0], r_dims[1], r_dims[2]);
}

Transform TPUv4_Symmetry::calc_reflection_translation_delta(int r, int r_prime) {
    const char* dims[] = {"x", "y", "z"};
    Transform best;
    best.trans = {0, 0, 0};
    bool found = false;
    int best_refl_len = 999;
    int best_l1 = 999999;

    for (int mask = 0; mask < 8; ++mask) {
        std::vector<std::string> refl_dims;
        for (int i = 0; i < 3; ++i)
            if ((mask >> i) & 1)
                refl_dims.push_back(dims[i]);
        int r_mid = apply_reflection(r, refl_dims);
        int dx, dy, dz;
        calc_translation_delta(r_mid, r_prime, dx, dy, dz);
        if ((dx % mc_x_) != 0 || (dy % mc_y_) != 0 || (dz % mc_z_) != 0)
            continue;
        if (apply_translation(r_mid, dx, dy, dz) != r_prime)
            continue;
        int l1 = std::abs(dx) + std::abs(dy) + std::abs(dz);
        int refl_len = static_cast<int>(refl_dims.size());
        if (!found || std::tie(refl_len, l1) < std::tie(best_refl_len, best_l1)) {
            found = true;
            best_refl_len = refl_len;
            best_l1 = l1;
            best.refl = refl_dims;
            best.trans = {dx, dy, dz};
        }
    }
    if (!found) {
        calc_translation_delta(r, r_prime, best.trans[0], best.trans[1], best.trans[2]);
        return best;
    }
    return best;
}

Transform TPUv4_Symmetry::calc_transform_delta(int r, int r_prime) {
    auto it = get_transform_cache_[r].find(r_prime);
    if (it != get_transform_cache_[r].end())
        return it->second;
    Transform t;
    if (sym_type_ == "refl-trans")
        t = calc_reflection_translation_delta(r, r_prime);
    else {
        int dx, dy, dz;
        calc_translation_delta(r, r_prime, dx, dy, dz);
        t.refl = {};
        t.trans = {dx, dy, dz};
    }
    get_transform_cache_[r][r_prime] = t;
    return t;
}

int TPUv4_Symmetry::apply_transformation(int r, const Transform& tform) {
    std::string tkey = transform_to_key(tform);
    auto it = transform_cache_[r].find(tkey);
    if (it != transform_cache_[r].end())
        return it->second;
    int r_mid = r;
    if (sym_type_ == "refl-trans" && !tform.refl.empty())
        r_mid = apply_reflection(r, tform.refl);
    int r_prime = apply_translation(r_mid, tform.trans[0], tform.trans[1], tform.trans[2]);
    transform_cache_[r][tkey] = r_prime;
    get_transform_cache_[r][r_prime] = tform;
    return r_prime;
}

void TPUv4_Symmetry::define_canonical_equivalents() {
    std::vector<int> canons = get_canonical_nodes();
    reverse_canonical_equivalence_map_.clear();
    for (int c : canons)
        reverse_canonical_equivalence_map_[c] = {};
    canonical_equivalence_map_.clear();
    canonical_transformations_.clear();
    get_transform_cache_.clear();
    transform_cache_.clear();

    bool use_refl = (sym_type_ == "refl-trans");
    for (int i = 0; i < n_nodes; ++i) {
        int i_hemi = i;
        std::vector<std::string> refl_dim;
        if (use_refl)
            reflect_to_within_mc_hemisphere(i, i_hemi, refl_dim);
        int i_prime;
        int dx, dy, dz;
        translate_to_mc(i_hemi, i_prime, dx, dy, dz);
        Transform canon_trans;
        canon_trans.refl = refl_dim;
        canon_trans.trans = {dx, dy, dz};

        canonical_equivalence_map_[i] = i_prime;
        reverse_canonical_equivalence_map_[i_prime].push_back(i);
        canonical_transformations_[i] = canon_trans;
        std::string tkey = transform_to_key(canon_trans);
        transform_cache_[i][tkey] = i_prime;
        get_transform_cache_[i][i_prime] = canon_trans;
    }
}

std::vector<int> TPUv4_Symmetry::get_canonical_nodes() const {
    std::vector<int> canons;
    for (int z = 0; z < mc_z_; ++z)
        for (int y = 0; y < mc_y_; ++y)
            for (int x = 0; x < mc_x_; ++x)
                canons.push_back(xyz_to_r(x, y, z));
    return canons;
}

const std::vector<int>& TPUv4_Symmetry::get_all_noncanonical_equivalents(int r) const {
    static const std::vector<int> empty;
    auto it = reverse_canonical_equivalence_map_.find(r);
    if (it == reverse_canonical_equivalence_map_.end())
        return empty;
    return it->second;
}

}  // namespace tpuv4
