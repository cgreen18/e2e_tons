/*
 * pathlist_to_xml: translate pathlist (.paths) to MSCCL-style XML plan.
 * Mirrors python_scripts/pathlist_to_xml.py and uses tpuv4_symmetry for --symmetric.
 */

#include "tpuv4_symmetry.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace tpuv4;

// Python-compatible Mersenne Twister so shuffle order matches Python's random.Random(seed).shuffle()
namespace python_mt {
static const int N = 624;
static const int M = 397;
static uint32_t state[N];
static int next_idx = N;

static void init_genrand(uint32_t seed) {
    state[0] = seed & 0xffffffffu;
    for (int i = 1; i < N; i++)
        state[i] = (1812433253u * (state[i-1] ^ (state[i-1] >> 30)) + static_cast<uint32_t>(i)) & 0xffffffffu;
    next_idx = N;
}

static uint32_t genrand_uint32() {
    if (next_idx >= N) {
        static const uint32_t MATRIX_A = 0x9908b0dfu;
        for (int k = 0; k < N - M; k++) {
            uint32_t y = (state[k] & 0x80000000u) | (state[k+1] & 0x7fffffffu);
            state[k] = state[k+M] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
        }
        for (int k = N - M; k < N - 1; k++) {
            uint32_t y = (state[k] & 0x80000000u) | (state[k+1] & 0x7fffffffu);
            state[k] = state[k+M-N] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
        }
        uint32_t y = (state[N-1] & 0x80000000u) | (state[0] & 0x7fffffffu);
        state[N-1] = state[M-1] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
        next_idx = 0;
    }
    uint32_t y = state[next_idx++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

static double random_() {
    return (static_cast<double>(genrand_uint32() >> 5) / (1ULL << 26) +
            static_cast<double>(genrand_uint32() >> 6) / (1ULL << 53));
}

template<typename T>
static void shuffle(std::vector<T>& v) {
    for (size_t i = v.size(); i > 1; i--) {
        size_t j = static_cast<size_t>(random_() * i);
        if (j >= i) j = i - 1;
        std::swap(v[j], v[i-1]);
    }
}
}

// ---- JSON line parser: parse one line like "[0, 1, 2]" into vector<int> ----
static bool parse_json_array_line(const std::string& line, std::vector<int>& out) {
    out.clear();
    size_t i = 0;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
    if (i >= line.size() || line[i] != '[') return false;
    ++i;
    while (i < line.size()) {
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
        if (i >= line.size()) return false;
        if (line[i] == ']') return true;
        if (line[i] == ',') { ++i; continue; }
        if (!std::isdigit(static_cast<unsigned char>(line[i])) && line[i] != '-') return false;
        long val = 0;
        int sign = 1;
        if (line[i] == '-') { sign = -1; ++i; }
        while (i < line.size() && std::isdigit(static_cast<unsigned char>(line[i]))) {
            val = val * 10 + (line[i] - '0');
            ++i;
        }
        out.push_back(static_cast<int>(sign * val));
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
        if (i < line.size() && line[i] == ',') ++i;
    }
    return false;
}

// ---- Pathlist loading ----
using PathDict = std::map<std::pair<int, int>, std::vector<std::vector<int>>>;

static void load_pathlist_to_path_dict(
    const std::string& pathlist_filepath,
    bool skip_header,
    const std::set<int>& canon_set,
    PathDict& path_dict,
    std::vector<std::pair<int, int>>& path_key_order) {
    path_dict.clear();
    path_key_order.clear();
    std::ifstream inf(pathlist_filepath);
    if (!inf) throw std::runtime_error("cannot open pathlist: " + pathlist_filepath);
    std::string line;
    if (skip_header) std::getline(inf, line);
    std::set<std::pair<int, int>> seen;

    std::cout << "Loading pathlist to path dict" << std::endl;
    while (std::getline(inf, line)) {
        size_t start = 0;
        while (start < line.size() && (line[start] == ' ' || line[start] == '\r' || line[start] == '\n')) ++start;
        if (start >= line.size()) continue;
        size_t end = line.size();
        while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\r' || line[end - 1] == '\n')) --end;
        line = line.substr(start, end - start);
        if (line.empty()) continue;
        std::vector<int> path;
        if (!parse_json_array_line(line, path)) continue;
        if (path.size() < 2) continue;
        int s = path.front(), d = path.back();
        if (canon_set.count(s) == 0) continue;
        if (s == d) continue;
        if (s % 100 == 0 && d == 0) {
            std::cout << "on src " << s << std::endl;
        }
        auto key = std::make_pair(s, d);
        path_dict[key].push_back(path);
        if (seen.insert(key).second)
            path_key_order.push_back(key);
    }
}

static std::vector<std::vector<int>> adj_list_from_path_dict(
    const PathDict& path_dict,
    int n_nodes) {
    std::set<std::pair<int, int>> edges;
    for (const auto& kv : path_dict) {
        for (const auto& path : kv.second) {
            for (size_t i = 0; i + 1 < path.size(); ++i) {
                int u = path[i], v = path[i + 1];
                edges.insert({u, v});
            }
        }
    }
    std::vector<std::vector<int>> adj_list(static_cast<size_t>(n_nodes));
    for (const auto& e : edges) {
        int u = e.first, v = e.second;
        if (u < n_nodes)
            adj_list[static_cast<size_t>(u)].push_back(v);
    }
    for (auto& nbrs : adj_list) {
        std::set<int> uniq(nbrs.begin(), nbrs.end());
        nbrs.assign(uniq.begin(), uniq.end());
        std::sort(nbrs.begin(), nbrs.end());
    }
    return adj_list;
}

// ---- Transfer and compiler helpers ----
struct Transfer {
    int u, v, chan;
    std::string srcbuf, dstbuf;
    int srcoff, dstoff, cnt;
};

static int choose_channel(int u, int v, int n_channels) {
    if (n_channels <= 1) return 0;
    return (u + v) % n_channels;
}

static int static_scratch_slot(int s, int d, int q, int n_chunks, int n_nodes) {
    return n_chunks * (s + n_nodes * d) + q;
}

// Chunk assignment: (s, d, q, path) with path index 0
struct ChunkAssignment {
    int s, d, q;
    std::vector<int> path;
};

static std::vector<ChunkAssignment> quantize_pmcf_paths(
    const PathDict& path_dict,
    const std::vector<std::pair<int, int>>& path_key_order,
    int n_chunks,
    int n_nodes) {
    std::vector<ChunkAssignment> out;
    for (const auto& key : path_key_order) {
        auto it = path_dict.find(key);
        if (it == path_dict.end() || it->second.empty()) continue;
        const auto& paths = it->second;
        int s = key.first, d = key.second;
        for (int q = 0; q < n_chunks; ++q) {
            int pidx = 0;
            if (pidx < 0 || pidx >= static_cast<int>(paths.size())) pidx = 0;
            if (pidx >= static_cast<int>(paths.size())) pidx = static_cast<int>(paths.size()) - 1;
            out.push_back({s, d, q, paths[static_cast<size_t>(pidx)]});
        }
    }
    return out;
}

// Non-symmetric compiler
static void compile_pmcf_to_link_epochs(
    const std::vector<std::vector<int>>& adj_list,
    const PathDict& path_dict,
    const std::vector<std::pair<int, int>>& path_key_order,
    int n_chunks,
    int n_channels,
    int* max_epochs,
    unsigned seed,
    std::vector<std::vector<Transfer>>& epochs) {
    int n_nodes = static_cast<int>(adj_list.size());
    std::vector<ChunkAssignment> chunk_assignments = quantize_pmcf_paths(path_dict, path_key_order, n_chunks, n_nodes);

    struct State {
        int s, d, q;
        std::vector<int> path;
        size_t pos = 0;
        bool done = false;
        int t_ready = 0;
    };
    std::vector<State> states;
    for (auto& ca : chunk_assignments)
        states.push_back({ca.s, ca.d, ca.q, ca.path, 0, false, 0});

    python_mt::init_genrand(static_cast<uint32_t>(seed));
    std::set<size_t> active;
    for (size_t i = 0; i < states.size(); ++i) active.insert(i);
    int remaining = static_cast<int>(states.size());
    int t = 0;
    epochs.clear();

    while (remaining > 0) {
        if (max_epochs && t >= *max_epochs) break;
        std::set<std::pair<int, int>> used_links;
        std::vector<Transfer> transfers;
        std::vector<size_t> ready;
        for (size_t i : active) {
            if (states[i].t_ready <= t)
                ready.push_back(i);
        }
        python_mt::shuffle(ready);

        for (size_t idx : ready) {
            State& st = states[idx];
            const auto& path = st.path;
            size_t pos = st.pos;
            if (pos >= path.size() - 1) {
                st.done = true;
                active.erase(idx);
                continue;
            }
            int u = path[pos], v = path[pos + 1];
            if (used_links.count({u, v})) continue;
            used_links.insert({u, v});

            int slot = static_scratch_slot(st.s, st.d, st.q, n_chunks, n_nodes);
            std::string srcbuf, dstbuf;
            int srcoff, dstoff;
            if (u == st.s && pos == 0) {
                srcbuf = "i";
                srcoff = st.d * n_chunks + st.q;
            } else {
                srcbuf = "s";
                srcoff = slot;
            }
            if (v == st.d && pos == path.size() - 2) {
                dstbuf = "o";
                dstoff = st.s * n_chunks + st.q;
            } else {
                dstbuf = "s";
                dstoff = slot;
            }
            int chan = choose_channel(u, v, n_channels);
            transfers.push_back({u, v, chan, srcbuf, dstbuf, srcoff, dstoff, 1});
            st.pos++;
            if (st.pos >= path.size() - 1) {
                st.done = true;
                active.erase(idx);
                remaining--;
            } else {
                st.t_ready = t + 1;
            }
        }
        epochs.push_back(transfers);
        t++;
    }

    int expected = n_nodes * (n_nodes - 1) * n_chunks;
    int inj_cnt = 0, out_cnt = 0;
    for (const auto& tr_list : epochs)
        for (const auto& tr : tr_list) {
            if (tr.srcbuf == "i") inj_cnt += tr.cnt;
            if (tr.dstbuf == "o") out_cnt += tr.cnt;
        }
    if (inj_cnt != expected || out_cnt != expected)
        throw std::runtime_error("pMCF compiler A2A mismatch");
}

// Symmetric compiler: simple_compile_pmcf_to_link_epochs_sym
static void simple_compile_pmcf_to_link_epochs_sym(
    const std::vector<std::vector<int>>& adj_list,
    const PathDict& path_dict,
    const std::vector<std::pair<int, int>>& path_key_order_canonical,
    const std::set<int>& canon_set,
    TPUv4_Symmetry& sym,
    int n_chunks,
    int n_channels,
    int* max_epochs,
    unsigned seed,
    std::vector<std::vector<Transfer>>& epochs) {
    int n_nodes = static_cast<int>(adj_list.size());
    std::vector<ChunkAssignment> chunk_assignments = quantize_pmcf_paths(
        path_dict, path_key_order_canonical, n_chunks, n_nodes);

    std::vector<size_t> order(chunk_assignments.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    python_mt::init_genrand(static_cast<uint32_t>(seed));
    python_mt::shuffle(order);

    std::map<int, std::set<std::pair<int, int>>> used_edges_at_time;
    std::map<int, std::vector<Transfer>> transfers_at_time;

    int max_time = 0;

    for (size_t iter_num = 0; iter_num < order.size(); ++iter_num) {
        if (iter_num % 1000 == 0) {
            std::cout << "on iter " << iter_num << " / " << order.size() << " (" << round(100*iter_num / order.size()) << "%)" << std::endl;
        }
        size_t idx = order[iter_num];
        const ChunkAssignment& ca = chunk_assignments[idx];
        int sc = ca.s, d = ca.d, q = ca.q;
        const std::vector<int>& path = ca.path;
        if (canon_set.count(sc) == 0) continue;
        if (path.size() < 2) continue;
        int ready_t = 0;
        for (size_t pos = 0; pos < path.size() - 1; ++pos) {
            int u = path[pos], v = path[pos + 1];
            bool is_first_hop = (pos == 0);
            bool is_last_hop = (pos == path.size() - 2);
            const std::vector<int>& equivalents = sym.get_all_noncanonical_equivalents(sc);
            std::vector<std::tuple<int, int, int, int>> equivalent_edges;  // s, d_prime, uc, vc
            for (int s : equivalents) {
                int uc, vc, d_prime;
                if (s == sc) {
                    uc = u; vc = v; d_prime = d;
                } else {
                    Transform tform = sym.calc_transform_delta(sc, s);
                    uc = sym.apply_transformation(u, tform);
                    vc = sym.apply_transformation(v, tform);
                    d_prime = sym.apply_transformation(d, tform);
                }
                equivalent_edges.push_back({s, d_prime, uc, vc});
            }
            std::set<std::pair<int, int>> edges_for_equivalents;
            for (const auto& t : equivalent_edges)
                edges_for_equivalents.insert({std::get<2>(t), std::get<3>(t)});
            int t = ready_t;
            for (;;) {
                if (max_epochs && t >= *max_epochs) break;
                auto& used = used_edges_at_time[t];
                bool conflict = false;
                for (const auto& e : edges_for_equivalents)
                    if (used.count(e)) { conflict = true; break; }
                if (!conflict) break;
                t++;
            }

            if (max_time < t){
                max_time = t;
            }
            if (max_epochs && t >= *max_epochs) continue;
            used_edges_at_time[t].insert(edges_for_equivalents.begin(), edges_for_equivalents.end());
            for (const auto& tup : equivalent_edges) {
                int s = std::get<0>(tup), d_prime = std::get<1>(tup), uc = std::get<2>(tup), vc = std::get<3>(tup);
                if (s == d_prime) continue;
                int slot = static_scratch_slot(s, d_prime, q, n_chunks, n_nodes);
                std::string srcbuf, dstbuf;
                int srcoff, dstoff;
                if (is_first_hop) {
                    srcbuf = "i";
                    srcoff = d_prime * n_chunks + q;
                } else {
                    srcbuf = "s";
                    srcoff = slot;
                }
                if (is_last_hop) {
                    dstbuf = "o";
                    dstoff = s * n_chunks + q;
                } else {
                    dstbuf = "s";
                    dstoff = slot;
                }
                int chan = choose_channel(uc, vc, n_channels);
                transfers_at_time[t].push_back({uc, vc, chan, srcbuf, dstbuf, srcoff, dstoff, 1});
            }
            ready_t = t + 1;
        }
    }
    std::cout << "# epochs: " << (max_time+1) << std::endl;

    std::cout << "Finished compiling" << std::endl;
    std::cout << "Writing epochs" << std::endl;
    epochs.clear();
    for (const auto& kv : transfers_at_time)
        epochs.push_back(kv.second);

    int expected = n_nodes * (n_nodes - 1) * n_chunks;
    int inj_cnt = 0, out_cnt = 0;
    for (const auto& tr_list : epochs)
        for (const auto& tr : tr_list) {
            if (tr.srcbuf == "i") inj_cnt += tr.cnt;
            if (tr.dstbuf == "o") out_cnt += tr.cnt;
        }
    if (inj_cnt != expected || out_cnt != expected)
        throw std::runtime_error("pMCF compiler A2A mismatch (sym)");
}

static std::vector<std::vector<int>> build_inverse_adj_list(const std::vector<std::vector<int>>& adj_list) {
    int n_nodes = static_cast<int>(adj_list.size());
    std::vector<std::set<int>> in_nei(static_cast<size_t>(n_nodes));
    for (int u = 0; u < n_nodes; ++u)
        for (int v : adj_list[static_cast<size_t>(u)])
            in_nei[static_cast<size_t>(v)].insert(u);
    std::vector<std::vector<int>> out(static_cast<size_t>(n_nodes));
    for (int v = 0; v < n_nodes; ++v)
        out[static_cast<size_t>(v)].assign(in_nei[static_cast<size_t>(v)].begin(), in_nei[static_cast<size_t>(v)].end());
    return out;
}

static void write_msccl_xml_from_link_epochs(
    const std::vector<std::vector<int>>& adj_list,
    const std::vector<std::vector<Transfer>>& epochs,
    const std::string& out_xml_path,
    int n_chunks,
    int n_channels) {
    int n_nodes = static_cast<int>(adj_list.size());
    std::vector<std::vector<int>> in_nei = build_inverse_adj_list(adj_list);
    int max_in_deg = 0;
    for (int v = 0; v < n_nodes; ++v) {
        int d = static_cast<int>(in_nei[static_cast<size_t>(v)].size());
        if (d > max_in_deg) max_in_deg = d;
    }
    int s_chunks = std::max(2, 2 * max_in_deg);
    int i_chunks = n_nodes * n_chunks;
    int o_chunks = n_nodes * n_chunks;

    std::vector<std::map<int, int>> recv_tb_id(static_cast<size_t>(n_nodes));
    std::vector<std::map<int, int>> send_tb_id(static_cast<size_t>(n_nodes));
    std::vector<std::map<int, std::tuple<int, int, int>>> tb_info(static_cast<size_t>(n_nodes));
    std::vector<std::map<int, std::vector<std::string>>> tb_steps(static_cast<size_t>(n_nodes));

    for (int g = 0; g < n_nodes; ++g) {
        int tbid = 0;
        std::set<int> in_u(in_nei[static_cast<size_t>(g)].begin(), in_nei[static_cast<size_t>(g)].end());
        for (int u : in_u) {
            int chan = choose_channel(u, g, n_channels);
            recv_tb_id[g][u] = tbid;
            tb_info[g][tbid] = std::make_tuple(-1, u, chan);
            tbid++;
        }
        std::set<int> out_v(adj_list[static_cast<size_t>(g)].begin(), adj_list[static_cast<size_t>(g)].end());
        for (int v : out_v) {
            int chan = choose_channel(g, v, n_channels);
            send_tb_id[g][v] = tbid;
            tb_info[g][tbid] = std::make_tuple(v, -1, chan);
            tbid++;
        }
    }

    for (size_t t = 0; t < epochs.size(); ++t) {
        for (const auto& tr : epochs[t]) {
            int u = tr.u, v = tr.v;
            auto it_s = send_tb_id[static_cast<size_t>(u)].find(v);
            if (it_s != send_tb_id[static_cast<size_t>(u)].end()) {
                int tbid_s = it_s->second;
                std::ostringstream step;
                step << "      <step s=\"" << t << "\" type=\"s\" srcbuf=\"" << tr.srcbuf << "\" srcoff=\"" << tr.srcoff << "\" "
                     << "dstbuf=\"" << tr.dstbuf << "\" dstoff=\"" << tr.dstoff << "\" cnt=\"" << tr.cnt << "\" "
                     << "depid=\"-1\" deps=\"-1\" hasdep=\"0\"/>";
                tb_steps[u][tbid_s].push_back(step.str());
            }
            auto it_r = recv_tb_id[static_cast<size_t>(v)].find(u);
            if (it_r != recv_tb_id[static_cast<size_t>(v)].end()) {
                int tbid_r = it_r->second;
                std::ostringstream step;
                step << "      <step s=\"" << t << "\" type=\"r\" srcbuf=\"" << tr.srcbuf << "\" srcoff=\"" << tr.srcoff << "\" "
                     << "dstbuf=\"" << tr.dstbuf << "\" dstoff=\"" << tr.dstoff << "\" cnt=\"" << tr.cnt << "\" "
                     << "depid=\"-1\" deps=\"-1\" hasdep=\"0\"/>";
                tb_steps[v][tbid_r].push_back(step.str());
            }
        }
    }

    std::ofstream outf(out_xml_path);
    if (!outf) throw std::runtime_error("cannot write XML: " + out_xml_path);
    outf << "<algo name=\"alltoall_compiled\" proto=\"Simple\" nchannels=\"" << n_channels << "\" "
         << "nchunksperloop=\"" << i_chunks << "\" ngpus=\"" << n_nodes << "\" coll=\"alltoall\" inplace=\"0\">\n";
    for (int g = 0; g < n_nodes; ++g) {
        outf << "  <gpu id=\"" << g << "\" i_chunks=\"" << i_chunks << "\" o_chunks=\"" << o_chunks << "\" s_chunks=\"" << s_chunks << "\">\n";
        std::vector<int> tbids;
        for (const auto& kv : tb_info[g]) tbids.push_back(kv.first);
        std::sort(tbids.begin(), tbids.end());
        for (int tbid : tbids) {
            const auto& info = tb_info[g][tbid];
            int send_val = std::get<0>(info), recv_val = std::get<1>(info), chan_val = std::get<2>(info);
            outf << "    <tb id=\"" << tbid << "\" send=\"" << send_val << "\" recv=\"" << recv_val << "\" chan=\"" << chan_val << "\">\n";
            for (const auto& step_line : tb_steps[g][tbid])
                outf << step_line << "\n";
            outf << "    </tb>\n";
        }
        outf << "  </gpu>\n";
    }
    outf << "</algo>\n";
}

static bool has_arg(const std::vector<std::string>& args, const std::string& key) {
    return std::find(args.begin(), args.end(), key) != args.end();
}
static std::string get_arg(const std::vector<std::string>& args, const std::string& key) {
    auto it = std::find(args.begin(), args.end(), key);
    if (it == args.end() || it + 1 == args.end()) return "";
    return *(it + 1);
}
static int get_arg_int(const std::vector<std::string>& args, const std::string& key, int default_val) {
    std::string s = get_arg(args, key);
    if (s.empty()) return default_val;
    return std::stoi(s);
}
static std::vector<int> get_arg_ints(const std::vector<std::string>& args, const std::string& key, size_t count) {
    auto it = std::find(args.begin(), args.end(), key);
    std::vector<int> out;
    if (it == args.end() || it + 1 + static_cast<ptrdiff_t>(count) > args.end()) return out;
    for (size_t i = 0; i < count; ++i)
        out.push_back(std::stoi(*(it + 1 + i)));
    return out;
}

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    std::string pathlist = get_arg(args, "--pathlist");
    std::string xml_out = get_arg(args, "--xml");
    int n_nodes = get_arg_int(args, "--n_nodes", -1);
    if (pathlist.empty() || xml_out.empty() || n_nodes <= 0) {
        std::cerr << "Usage: pathlist_to_xml --pathlist <file> --xml <file> --n_nodes <N> [--symmetric --mc_dims X Y Z --xyzc_dims X Y Z C] [--n_chunks 1] [--n_channels 1] [--skip_header|--no_skip_header] [--max_epochs N] [--sym_type trans|refl-trans]\n";
        return 1;
    }
    bool symmetric = has_arg(args, "--symmetric");
    std::vector<int> mc_dims = get_arg_ints(args, "--mc_dims", 3);
    std::vector<int> xyzc_dims = get_arg_ints(args, "--xyzc_dims", 4);
    if (symmetric && (mc_dims.size() != 3 || xyzc_dims.size() != 4)) {
        std::cerr << "--symmetric requires --mc_dims <mcx> <mcy> <mcz> and --xyzc_dims <x> <y> <z> <c>\n";
        return 1;
    }
    int n_chunks = get_arg_int(args, "--n_chunks", 1);
    int n_channels = get_arg_int(args, "--n_channels", 1);
    int max_epochs_val = get_arg_int(args, "--max_epochs", 0);
    int* max_epochs = (max_epochs_val > 0) ? &max_epochs_val : nullptr;
    bool skip_header = has_arg(args, "--no_skip_header") ? false : (pathlist.size() >= 6 && pathlist.substr(pathlist.size() - 6) == ".paths");
    std::string sym_type = get_arg(args, "--sym_type");
    if (sym_type.empty()) sym_type = "trans";

    std::set<int> canon_set;
    for (int i = 0; i < n_nodes; ++i) canon_set.insert(i);
    std::unique_ptr<TPUv4_Symmetry> sym;
    std::vector<int> canonical_sources;
    if (symmetric) {
        std::array<int, 4> xyzc = {{xyzc_dims[0], xyzc_dims[1], xyzc_dims[2], xyzc_dims[3]}};
        std::array<int, 3> mc = {{mc_dims[0], mc_dims[1], mc_dims[2]}};
        sym.reset(new TPUv4_Symmetry(xyzc, &mc, sym_type));
        canonical_sources = sym->get_canonical_nodes();
        canon_set.clear();
        for (int c : canonical_sources) canon_set.insert(c);
    }

    PathDict path_dict;
    std::vector<std::pair<int, int>> path_key_order;
    load_pathlist_to_path_dict(pathlist, skip_header, canon_set, path_dict, path_key_order);
    if (path_dict.empty()) {
        std::cerr << "No paths loaded; pathlist is empty or invalid.\n";
        return 1;
    }

    std::vector<std::vector<int>> adj_list = adj_list_from_path_dict(path_dict, n_nodes);
    std::vector<std::vector<Transfer>> epochs;
    if (symmetric) {
        simple_compile_pmcf_to_link_epochs_sym(
            adj_list, path_dict, path_key_order, canon_set, *sym,
            n_chunks, n_channels, max_epochs, 0u, epochs);
    } else {
        compile_pmcf_to_link_epochs(
            adj_list, path_dict, path_key_order,
            n_chunks, n_channels, max_epochs, 0u, epochs);
    }
    std::cout << "# epochs: " << epochs.size() << std::endl;
    std::cout << "Writing XML" << std::endl;
    write_msccl_xml_from_link_epochs(adj_list, epochs, xml_out, n_chunks, n_channels);
    std::cout << "Wrote MSCCL XML to: " << xml_out << "\n";

    return 0;
}