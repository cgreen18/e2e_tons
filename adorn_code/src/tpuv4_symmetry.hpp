#ifndef TPUV4_SYMMETRY_HPP
#define TPUV4_SYMMETRY_HPP

#include <vector>
#include <string>
#include <array>
#include <map>
#include <tuple>
#include <cassert>
#include <stdexcept>

namespace tpuv4 {

struct Transform {
    std::vector<std::string> refl;  // dimension names "x", "y", "z"
    std::array<int, 3> trans;        // (dx, dy, dz)
};

inline bool operator<(const Transform& a, const Transform& b) {
    if (a.refl != b.refl) return a.refl < b.refl;
    return a.trans < b.trans;
}

class TPUv4_Symmetry {
public:
    static const std::vector<std::string> supported_sym_types;

    TPUv4_Symmetry(const std::array<int, 4>& xyzc_dims,
                   const std::array<int, 3>* mc_dims_ptr,
                   const std::string& sym_type = "trans");

    void r_to_xyz(int r, int& out_x, int& out_y, int& out_z) const;
    int xyz_to_r(int x, int y, int z) const;

    std::vector<int> get_canonical_nodes() const;
    const std::vector<int>& get_all_noncanonical_equivalents(int r) const;
    Transform calc_transform_delta(int r, int r_prime);
    int apply_transformation(int r, const Transform& tform);

    int n_nodes = 0;

private:
    void set_xyzc_dims(const std::array<int, 4>& xyzc_dims);
    void set_canonical_mega_cube(const std::array<int, 3>& mc_dims);
    void define_canonical_equivalents();

    std::array<int, 4> xyzc_dims_{};
    int x_dim_ = 0, y_dim_ = 0, z_dim_ = 0, cube_dim_ = 0;
    std::array<int, 3> dim_arr_{};  // [x_dim, y_dim, z_dim] for reflection
    std::array<int, 3> mc_dims_{};
    int mc_x_ = 0, mc_y_ = 0, mc_z_ = 0;
    std::string sym_type_;

    void translate_r_to_rel_mc_xyz(int r, int& rel_x, int& rel_y, int& rel_z) const;
    int translate_r_to_rel_mc_r(int r) const;
    void calc_translation_delta(int r_old, int r_new, int& d_x, int& d_y, int& d_z) const;
    void translate_to_mc(int r, int& r_prime, int& d_x, int& d_y, int& d_z);
    int apply_translation(int r, int d_x, int d_y, int d_z) const;
    int apply_reverse_translation(int r, int d_x, int d_y, int d_z) const;

    void reflect_to_within_mc_hemisphere(int r, int& r_prime, std::vector<std::string>& refl_dim);
    int apply_reflection(int r, const std::vector<std::string>& refl_dim) const;
    Transform calc_reflection_translation_delta(int r, int r_prime);

    std::map<int, int> canonical_equivalence_map_;
    std::map<int, std::vector<int>> reverse_canonical_equivalence_map_;
    std::map<int, Transform> canonical_transformations_;
    std::map<int, std::map<int, Transform>> get_transform_cache_;
    std::map<int, std::map<std::string, int>> transform_cache_;
};

}  // namespace tpuv4

#endif
