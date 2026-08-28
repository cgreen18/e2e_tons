/*
--------------------------------------------------------------------------------
Copyright (c) 2025 Purdue University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met: redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer;
redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution;
neither the name of the copyright holders nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author(s): Conor Green

--------------------------------------------------------------------------------
Description: TODO
*/


// at least c++ 20 for CLA parsing with contains()
// to compile: g++ src/heuristic_tpuv4.cpp -std=c++20 -O3 -pthread -fopenmp -DNDEBUG -march=native -flto -fuse-linker-plugin -ffast-math -fno-math-errno -fno-trapping-math -freciprocal-math
// acutally, -flto -fuse-linker-plugin make it slower
// -ffast-math is blazing fast!

// defines like g++ src/heuristic_tpuv4.cpp -std=c++23 -O3 -DNDEBUG -march=native -ffast-math -fno-math-errno -fno-trapping-math -freciprocal-math -pthread -fopenmp -o bin/heuristic_tpuv4

// std
#include <vector>
#include <map>
#include <unordered_map>
#include <utility>
#include <iostream>
#include <queue>
#include <limits>
#include <chrono>
#include <cassert>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <cstring>

// lib
#include <omp.h>

// local
#include "heuristic_tpuv4_valid_topology.hpp"
#include "heuristic_tpuv4_graph_functions.hpp"
#include "heuristic_tpuv4_file_console.hpp"


// using namespace std;
using std::vector;
using std::tuple;
using std::map;
using std::unordered_map;
using std::pair;
using std::queue;
using std::string;
using std::priority_queue;

using std::sort;
using std::min;
using std::max;
using std::make_pair;
using std::make_tuple;
using std::to_string;
using std::stoi;

using std::cout;
using std::cerr;
using std::cin;
using std::endl;
using std::flush;
using std::getline;


#define OPENMP_PARALLEL_FOR_PRAGMA 16

// // convenient
// typedef pair<int,int> Edge;

typedef pair<int,int> Flow;

// "globals"
static int MAX_THREADS = 128;
static bool PARALLELIZE_DELTA = true;

// constants
// const int INF = std::numeric_limits<int>::max();
// const int INF = -1;
extern const int INF;
const string OUT_MAP_DIR = "files/heuristic_maps/";
const string OUT_LOG_DIR = "files/timeline_logs/";

// // enums
// enum class DIRECTION {
//     X_POS = 0,
//     X_NEG = 1,
//     Y_POS = 2,
//     Y_NEG = 3,
//     Z_POS = 4,
//     Z_NEG = 5
// };

// for top k edge selection
struct Candidate {
    double delta;
    Edge edge;
};

// Comparator for a max-heap by delta (largest on top).
struct WorseFirst {
    bool operator()(const Candidate& a, const Candidate& b) const {
        return a.delta < b.delta; // reverse: larger delta has higher priority
    }
};

// Dump heap -> ascending vector
static vector<Candidate> heap_to_sorted_vector(priority_queue<Candidate, vector<Candidate>, WorseFirst>& pq) {
    vector<Candidate> out;
    out.reserve(pq.size());
    while (!pq.empty()) { out.push_back(pq.top()); pq.pop(); }
    sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b){ return a.delta < b.delta; });
    return out;
}

enum class ParallelBackend { Single, OpenMP, Threads };
ParallelBackend PARALLEL_BACKEND = ParallelBackend::Single;

static int clamp_threads(int want) {
    if (want <= 1) return 1;
    unsigned hw = std::thread::hardware_concurrency();
    int max_hw = hw ? (int)hw : want;
    return max(1, min(want, max_hw));
}

// selection algs
////////////////////////////////////////////////////////////////////////////////

// int calc_delta_old(const vector<vector<int>>& dists, const Edge new_edge){
//     int n = dists.size();

//     auto [i,j] = new_edge;

//     int delta = 0;
//     // assume symmetric
//     for (int s=0; s<n; s++){
//         for (int d=s+1; d<n; d++){

//             int d_si = dists[s][i];
//             int d_jd = dists[j][d];

//             int new_dist = dists[s][d]; // assume old
//             // avoid overflow
//             if (d_si < INF && d_jd < INF){
//                 // quadrangle inequality
//                 new_dist = min(dists[s][d], dists[s][i] + 1 + dists[j][d]);
//             }
//             delta += 2*(new_dist - dists[s][d]);
//         }
//     }
//     return delta;
// }

inline bool is_inf(int x) { return (x == std::numeric_limits<int>::max() || x == INF); }

// top ONE delta
double calc_delta_scalar(const vector<vector<int>>& dists, Edge e)
{
    const int n = (int)dists.size();
    const int i = e.first, j = e.second;
    const auto& row_j = dists[j];

    double delta = 0.0;

    for (int s = 0; s < n; ++s) {
        const int dsi = dists[s][i];     // column i (one elem)
        for (int d = s + 1; d < n; ++d) { // only upper triangle
            const int dsd = dists[s][d];
            const int jdd = row_j[d];

            int nd = dsd;
            if (!is_inf(dsi) && !is_inf(jdd)) {
                const int via = dsi + 1 + jdd; // safe if not INF
                if (via < nd) nd = via;
            }
            // else{
            //     cout << "dsi = "<<dsi<<" or jdd = "<<jdd<<" is infinity"<<endl;
            // }
            delta += 2.0 * (nd - dsd);
        }
    }
    return delta;
}

// top ONE delta
double calc_delta_parallel_inner(const vector<vector<int>>& dists,Edge e,int max_threads)
{
    const int n = (int)dists.size();
    const int i = e.first, j = e.second;
    const auto& row_j = dists[j];

    double delta = 0.0;

    omp_set_num_threads(max(1, min(max_threads, MAX_THREADS)));
    #pragma omp parallel for reduction(+:delta) schedule(static)
    for (int s = 0; s < n; ++s) {
        const int dsi = dists[s][i];
        double local = 0.0;
        for (int d = s + 1; d < n; ++d) {
            const int dsd = dists[s][d];
            const int jdd = row_j[d];
            int nd = dsd;
            if (!is_inf(dsi) && !is_inf(jdd)) {
                const int via = dsi + 1 + jdd;
                if (via < nd) nd = via;
            }
            local += 2.0 * (nd - dsd);
        }
        delta += local;
    }

    return delta;
}

// top K deltas
vector<Candidate> topk_scalar_over_candidates(const vector<vector<int>>& dists, const vector<Edge>& flat_valid_conns, int k, bool parallel_inside_delta,int max_threads)
{
    priority_queue<Candidate, vector<Candidate>, WorseFirst> heap;
    heap = {};

    for (const auto& e : flat_valid_conns) {
        double d = parallel_inside_delta
            ? calc_delta_parallel_inner(dists, e, max_threads) // parallel *inside* one delta
            : calc_delta_scalar(dists, e);                     // plain scalar delta

        if ((int)heap.size() < k) heap.push({d, e});
        else if (d < heap.top().delta) { heap.pop(); heap.push({d, e}); }
    }
    return heap_to_sorted_vector(heap); // ascending by delta
}

// top ONE best conn
Edge calc_best_conn_parallel(const vector<vector<int>>& dists,const vector<Edge>& flat_valid_conns,int max_threads)
{
    const int S = (int)flat_valid_conns.size();
    if (S == 0){
        cerr << "ERROR :: calc_best_conn_parallel() :: empty valid conns."<<endl;
        return {-1, -1};
    }

    int T = max(1, min<int>(max_threads, S));  // cap threads
    const int chunk = (S + T - 1) / T;

    struct Best { int delta; Edge edge; };
    vector<Best> locals(T, { std::numeric_limits<int>::infinity(), {-1,-1} });
    vector<std::thread> threads;
    threads.reserve(T);

    for (int t = 0; t < T; ++t) {
        int start = t * chunk;
        int end   = min(start + chunk, S);
        if (start >= end) { locals[t] = { std::numeric_limits<int>::infinity(), {-1,-1} }; continue; }

        threads.emplace_back([&, t, start, end](){
            int best_delta = std::numeric_limits<int>::infinity();
            Edge best_edge = {-1,-1};

            for (int idx = start; idx < end; ++idx) {
                const auto& entry = flat_valid_conns[idx];
                int delta = calc_delta_scalar(dists, entry);   // heavy O(n^2)

                if (delta < best_delta) {
                    best_delta = delta;
                    best_edge  = entry;
                }
            }

            locals[t] = { best_delta, best_edge };
        });
    }

    for (auto& th : threads) th.join();

    // Reduce thread-local bests
    int global_best = std::numeric_limits<int>::infinity();
    Edge global_edge = {-1,-1};
    for (const auto& b : locals) {
        if (b.delta < global_best) {
            global_best = b.delta;
            global_edge = b.edge;
        }
    }
    return global_edge;
}

// top ONE best conn
Edge calc_best_conn_omp(const vector<vector<int>>& dists,const vector<Edge>& flat_valid_conns,int max_threads)
{
    const int S = (int)flat_valid_conns.size();
    if (S == 0) return {-1,-1};

    omp_set_num_threads(max(1, max_threads));

    int global_best = std::numeric_limits<int>::infinity();
    Edge global_edge{-1,-1};

    // Each thread keeps a local best; reduce at the end.
    #pragma omp parallel
    {
        int best = std::numeric_limits<int>::infinity();
        Edge edge{-1,-1};

        #pragma omp for schedule(dynamic, OPENMP_PARALLEL_FOR_PRAGMA) nowait
        for (int idx = 0; idx < S; ++idx) {
            const auto e = flat_valid_conns[idx];
            // IMPORTANT: use the scalar (single-thread) calc_delta here to avoid nested oversubscription.
            int d = calc_delta_scalar(dists, e);
            if (d < best) { best = d; edge = e; }
        }

        #pragma omp critical
        {
            if (best < global_best) { global_best = best; global_edge = edge; }
        }
    }
    return global_edge;
}

// top ONE best conn
Edge calc_best_conn(const vector<vector<int>>& dists, const vector<Edge>& flat_valid_conns, const bool delta_parallel, const int max_threads)
{

    int best = std::numeric_limits<int>::infinity();
    Edge edge{-1,-1};
    for (const auto& e : flat_valid_conns) {
        int d;
        if(delta_parallel){
            d = calc_delta_parallel_inner(dists, e, max_threads);
        }
        else{
            d = calc_delta_scalar(dists, e);
        }
        if (d < best) { best = d; edge = e; }
    }
    return edge;
}

// any INF conn
Edge calc_inf_conn(const vector<vector<int>>& adj_matrix, const vector<vector<int>>& dists, const vector<Edge>& flat_valid_conns){

    int n = (int)dists.size();


    vector<int> order(flat_valid_conns.size());
    std::iota(order.begin(), order.end(), 0);

    std::mt19937 rng(std::random_device{}());
    std::shuffle(order.begin(), order.end(), rng);

    // lower is better
    Edge best_pair; 
    for (int idx : order) {
        const auto& [i, j] = flat_valid_conns[idx];
        if (dists[i][j] >= n){
            cout << "conn ("<<i<<", "<<j<<") w/ dist "<<dists[i][j] << endl;
            best_pair = make_pair(i,j);
            break;
        }
    }

    auto [i,j] = best_pair;
    cout << "Inf dist edge " << i << " -> " << j << endl;


    return best_pair;
}

Edge calc_best_edge_histoed(const std::vector<std::vector<int>> &dists,
                            const std::vector<Edge> &flat_valid_conns,
                            int diameter)
{
    const int n       = static_cast<int>(dists.size());
    const int n_cands = static_cast<int>(flat_valid_conns.size());

    using Bucket     = std::vector<int>;
    using DistHisto  = std::vector<Bucket>;
    using NodeHistos = std::vector<DistHisto>;

    NodeHistos H_to_a(n);      // flows (*, a) by distance
    NodeHistos H_from_b(n);    // flows (b, *) by distance
    std::vector<bool> built_to_a(n, false);
    std::vector<bool> built_from_b(n, false);

    // Timer timer;
    // timer.reset();

    // Build histograms per endpoint, once
    for (const Edge &cand : flat_valid_conns) {
        const int a = cand.first;
        const int b = cand.second;

        if (!built_to_a[a]) {
            built_to_a[a] = true;
            auto &hist_to = H_to_a[a];
            hist_to.assign(diameter + 1, Bucket{});
            for (int i = 0; i < n; ++i) {
                const int dist = dists[i][a];
                // Optionally check: assert(0 <= dist && dist <= diameter);
                hist_to[dist].push_back(i);
            }
        }

        if (!built_from_b[b]) {
            built_from_b[b] = true;
            auto &hist_from = H_from_b[b];
            hist_from.assign(diameter + 1, Bucket{});
            for (int j = 0; j < n; ++j) {
                const int dist = dists[b][j];
                // Optionally check: assert(0 <= dist && dist <= diameter);
                hist_from[dist].push_back(j);
            }
        }
    }

    // const long long expected_calcs =
    //     static_cast<long long>(n_cands) * n * n;
    // std::cout << "Expect less than " << expected_calcs << " calcs\n";

    // const double time_histoify = timer.s();
    // timer.reset();

    std::vector<long long> deltas(static_cast<std::size_t>(n) * n, 0);

    // const double time_init = timer.s();
    // timer.reset();

    long long total_calcs = 0;

    for (const Edge &cand : flat_valid_conns) {
        const int a = cand.first;
        const int b = cand.second;

        const DistHisto &H_to   = H_to_a[a];
        const DistHisto &H_from = H_from_b[b];

        long long &delta_ab =
            deltas[static_cast<std::size_t>(a) * n + b];

        for (int p = 0; p < diameter - 1; ++p) {
            const Bucket &bucket_p = H_to[p];
            if (bucket_p.empty()) continue;

            for (int i : bucket_p) {
                const std::vector<int> &dists_i = dists[i];

                const int q_max = diameter - p - 1;
                for (int q = 0; q < q_max; ++q) {
                    if (p == 0 && q == 0) continue;

                    const Bucket &bucket_q = H_from[q];
                    if (bucket_q.empty()) continue;

                    const int d_prime = p + q + 1;

                    for (int j : bucket_q) {
                        ++total_calcs;

                        const int r_star = dists_i[j];
                        if (r_star > d_prime) {
                            delta_ab += static_cast<long long>(r_star - d_prime);
                        }
                    }
                }
            }
        }
    }

    // std::cout << "Total calcs " << total_calcs << '\n';
    // std::cout << "Efficiency "
    //           << (static_cast<long double>(total_calcs) /
    //               static_cast<long double>(expected_calcs)) * 100.0L
    //           << "%\n";

    // const double time_deltas = timer.s();
    // timer.reset();

    // higher is better
    Edge      best_edge{};
    long long best_delta = 0;

    for (const Edge &cand : flat_valid_conns) {
        const int a   = cand.first;
        const int b   = cand.second;
        const int idx = a * n + b;
        if (deltas[idx] > best_delta) {
            best_delta = deltas[idx];
            best_edge  = cand;
        }
    }

    // const double time_select_best = timer.s();
    // timer.reset();

    // std::cout << "calc_best_edge_histoed:: Chose edge ("
    //           << best_edge.first << ", " << best_edge.second
    //           << ") w/ delta " << best_delta << '\n';
    // std::cout << "PERFORMANCE:: histoify : " << time_histoify
    //           << ". init "                 << time_init
    //           << ". deltas "               << time_deltas
    //           << ". select_best "          << time_select_best
    //           << '\n';

    return best_edge;
}


// Edge calc_best_edge_histoed_openmped(const std::vector<std::vector<int>> &dists,
//                                      const std::vector<Edge> &flat_valid_conns,
//                                      int diameter,
//                                      int num_threads = 1)
// {
//     using NodeId = std::uint16_t;  // n <= 20000 -> fits safely
//     using Dist   = std::uint16_t;  // distances < n <= 20000

//     const int n       = static_cast<int>(dists.size());
//     const int n_cands = static_cast<int>(flat_valid_conns.size());

//     // sanity in case someone ever calls this with bigger n
//     // (you can remove this in the hot build)
//     if (n > static_cast<int>(std::numeric_limits<NodeId>::max())) {
//         throw std::runtime_error("n exceeds NodeId capacity");
//     }
//     if (diameter > static_cast<int>(std::numeric_limits<Dist>::max())) {
//         throw std::runtime_error("diameter exceeds Dist capacity");
//     }

//     using Bucket     = std::vector<NodeId>;     // list of node IDs
//     using DistHisto  = std::vector<Bucket>;     // index = distance
//     using NodeHistos = std::vector<DistHisto>;  // index = node ID

//     NodeHistos H_to_a(n);      // flows (*, a) by distance
//     NodeHistos H_from_b(n);    // flows (b, *) by distance
//     std::vector<bool> built_to_a(n, false);
//     std::vector<bool> built_from_b(n, false);

//     Timer timer;
//     timer.reset();

//     // ---- 1. Build histograms per endpoint (sequential) ----
//     for (const Edge &cand : flat_valid_conns) {
//         const int a = cand.first;
//         const int b = cand.second;

//         if (!built_to_a[a]) {
//             built_to_a[a] = true;
//             auto &hist_to = H_to_a[a];
//             hist_to.assign(static_cast<std::size_t>(diameter) + 1U, Bucket{});
//             for (int i = 0; i < n; ++i) {
//                 // distance fits in Dist by assumption
//                 const Dist dist = static_cast<Dist>(dists[i][a]);
//                 hist_to[dist].push_back(static_cast<NodeId>(i));
//             }
//         }

//         if (!built_from_b[b]) {
//             built_from_b[b] = true;
//             auto &hist_from = H_from_b[b];
//             hist_from.assign(static_cast<std::size_t>(diameter) + 1U, Bucket{});
//             for (int j = 0; j < n; ++j) {
//                 const Dist dist = static_cast<Dist>(dists[b][j]);
//                 hist_from[dist].push_back(static_cast<NodeId>(j));
//             }
//         }
//     }

//     const long long expected_calcs =
//         static_cast<long long>(n_cands) * n * n;

//     struct alignas(64) PaddedDelta {
//         long long value;
//     };
//     std::vector<PaddedDelta> deltas_cand(static_cast<std::size_t>(n_cands));

//     if (num_threads <= 0) {
//         num_threads = omp_get_max_threads();
//     }
//     omp_set_dynamic(0);  // keep thread count fixed

//     long long total_calcs = 0;

//     // if you expect uniform work, static is usually best;
//     // you can switch back to dynamic if needed.
//     int chunk = std::max(1, (n_cands + num_threads - 1) / num_threads);

//     #pragma omp parallel for schedule(static, chunk) reduction(+:total_calcs) num_threads(num_threads)
//     #pragma omp for schedule(dynamic, OPENMP_PARALLEL_FOR_PRAGMA) nowait
//     for (int k = 0; k < n_cands; ++k) {
//         const Edge &cand = flat_valid_conns[k];
//         const int a = cand.first;
//         const int b = cand.second;

//         const DistHisto &H_to   = H_to_a[a];
//         const DistHisto &H_from = H_from_b[b];

//         long long local_delta = 0;  // thread-local accumulator

//         // p, q, r_star, d_prime all fit in Dist (<= diameter <= n <= 20000)
//         for (Dist p = 0; p < static_cast<Dist>(diameter - 1); ++p) {
//             const Bucket &bucket_p = H_to[p];
//             if (bucket_p.empty()) continue;

//             for (NodeId i_id : bucket_p) {
//                 const int i = static_cast<int>(i_id);
//                 const std::vector<int> &dists_i = dists[i];

//                 const Dist q_max = static_cast<Dist>(diameter - 1 - p);
//                 for (Dist q = 0; q < q_max; ++q) {
//                     if (p == 0 && q == 0) continue;

//                     const Bucket &bucket_q = H_from[q];
//                     if (bucket_q.empty()) continue;

//                     const Dist d_prime = static_cast<Dist>(p + q + 1);

                    
//                     for (NodeId j_id : bucket_q) {
//                         const int j = static_cast<int>(j_id);
//                         ++total_calcs;

//                         const Dist r_star = static_cast<Dist>(dists_i[j]);
//                         if (r_star > d_prime) {
//                             local_delta += static_cast<long long>(r_star - d_prime);
//                         }
//                     }
//                 }
//             }
//         }

//         deltas_cand[static_cast<std::size_t>(k)].value = local_delta;
//     }

//     std::cout << "Time : " << timer.s() << "s. "
//               << "|S|=" << n_cands << " w/ efficiency "
//               << (static_cast<long double>(total_calcs) /
//                   static_cast<long double>(expected_calcs)) * 100.0L
//               << "%\n";

//     // ---- 3. Pick best edge (sequential) ----
//     Edge      best_edge{};
//     long long best_delta = 0;

//     for (int k = 0; k < n_cands; ++k) {
//         const long long delta = deltas_cand[static_cast<std::size_t>(k)].value;
//         if (delta > best_delta) {
//             best_delta = delta;
//             best_edge  = flat_valid_conns[k];
//         }
//     }

//     return best_edge;
// }

Edge calc_best_edge_histoed_openmped(const std::vector<std::vector<int>> &dists,
                                     const std::vector<Edge> &flat_valid_conns,
                                     int diameter,
                                     int num_threads = 1)
{
    using NodeId = std::uint16_t;  // n <= 20000 -> fits safely
    using Dist   = std::uint16_t;  // distances < n <= 20000

    const int n       = static_cast<int>(dists.size());
    const int n_cands = static_cast<int>(flat_valid_conns.size());

    if (n_cands == 0) {
        return Edge{-1, -1};
    }

    // sanity
    if (n > static_cast<int>(std::numeric_limits<NodeId>::max())) {
        throw std::runtime_error("n exceeds NodeId capacity");
    }
    if (diameter > static_cast<int>(std::numeric_limits<Dist>::max())) {
        throw std::runtime_error("diameter exceeds Dist capacity");
    }

    // --------------------------------------------------------------------
    // Compact CSR-like histogram per endpoint:
    //   nodes   : all node IDs for this endpoint, grouped by distance
    //   offsets : size = diameter+2
    //             offsets[d]   = start index for distance d
    //             offsets[d+1] = end   index for distance d
    // So "bucket d" = nodes[offsets[d] .. offsets[d+1])
    // --------------------------------------------------------------------
    struct EndpointHisto {
        std::vector<NodeId> nodes;
        std::vector<int>    offsets;  // length = diameter + 2
    };

    using EndpointHistos = std::vector<EndpointHisto>;

    EndpointHistos H_to_a(n);      // flows (*, a) by distance
    EndpointHistos H_from_b(n);    // flows (b, *) by distance
    std::vector<bool> built_to_a(n, false);
    std::vector<bool> built_from_b(n, false);

    Timer timer;
    timer.reset();

    auto build_hist_to = [&](int a) {
        built_to_a[a] = true;
        EndpointHisto &hist = H_to_a[a];

        // offsets[d+1] counts number of nodes at distance d
        hist.offsets.assign(static_cast<std::size_t>(diameter) + 2U, 0);
        const auto &d_col = dists;  // we read dists[i][a]

        // 1) Count
        for (int i = 0; i < n; ++i) {
            Dist dist = static_cast<Dist>(d_col[i][a]);
            // assume diameter is a correct upper bound; if not, clamp/check
            if (dist > static_cast<Dist>(diameter)) {
                // you can throw or clamp; for safety we clamp here
                dist = static_cast<Dist>(diameter);
            }
            ++hist.offsets[static_cast<std::size_t>(dist) + 1U];
        }

        // 2) Prefix sum -> offsets[d] = start index of bucket d
        for (int d = 0; d <= diameter; ++d) {
            hist.offsets[static_cast<std::size_t>(d) + 1U] += hist.offsets[static_cast<std::size_t>(d)];
        }

        const int total_nodes = hist.offsets[static_cast<std::size_t>(diameter) + 1U];
        hist.nodes.resize(static_cast<std::size_t>(total_nodes));

        // 3) Fill nodes; within each bucket, NodeIds are in increasing order
        std::vector<int> cursor = hist.offsets;  // copy
        for (int i = 0; i < n; ++i) {
            Dist dist = static_cast<Dist>(d_col[i][a]);
            if (dist > static_cast<Dist>(diameter)) {
                dist = static_cast<Dist>(diameter);
            }
            int pos = cursor[static_cast<std::size_t>(dist)]++;
            hist.nodes[static_cast<std::size_t>(pos)] = static_cast<NodeId>(i);
        }
    };

    auto build_hist_from = [&](int b) {
        built_from_b[b] = true;
        EndpointHisto &hist = H_from_b[b];

        hist.offsets.assign(static_cast<std::size_t>(diameter) + 2U, 0);
        const auto &d_row = dists[b];  // we read dists[b][j]

        // 1) Count
        for (int j = 0; j < n; ++j) {
            Dist dist = static_cast<Dist>(d_row[j]);
            if (dist > static_cast<Dist>(diameter)) {
                dist = static_cast<Dist>(diameter);
            }
            ++hist.offsets[static_cast<std::size_t>(dist) + 1U];
        }

        // 2) Prefix sum
        for (int d = 0; d <= diameter; ++d) {
            hist.offsets[static_cast<std::size_t>(d) + 1U] += hist.offsets[static_cast<std::size_t>(d)];
        }

        const int total_nodes = hist.offsets[static_cast<std::size_t>(diameter) + 1U];
        hist.nodes.resize(static_cast<std::size_t>(total_nodes));

        // 3) Fill nodes
        std::vector<int> cursor = hist.offsets;
        for (int j = 0; j < n; ++j) {
            Dist dist = static_cast<Dist>(d_row[j]);
            if (dist > static_cast<Dist>(diameter)) {
                dist = static_cast<Dist>(diameter);
            }
            int pos = cursor[static_cast<std::size_t>(dist)]++;
            hist.nodes[static_cast<std::size_t>(pos)] = static_cast<NodeId>(j);
        }
    };

    // ---- 1. Build histograms per endpoint (sequential, but cache-friendly) ----
    for (const Edge &cand : flat_valid_conns) {
        const int a = cand.first;
        const int b = cand.second;

        if (!built_to_a[a])   build_hist_to(a);
        if (!built_from_b[b]) build_hist_from(b);
    }

    const long long expected_calcs =
        static_cast<long long>(n_cands) * n * n;

    struct alignas(64) PaddedDelta {
        long long value;
    };
    std::vector<PaddedDelta> deltas_cand(static_cast<std::size_t>(n_cands));

    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_dynamic(0);
    omp_set_num_threads(std::max(1, num_threads));

    long long total_calcs      = 0;
    long long global_best_diff = 0;
    int       global_best_idx  = -1;
    Edge      global_best_edge{};

    int per_chunk_divisor = 100;
    if (n > 8000){
        per_chunk_divisor = 1000;
    }

    // const int chunk = std::max(1, (n_cands + num_threads - 1) / num_threads);
    const int chunk = std::max(1, (n_cands + num_threads - 1) / (per_chunk_divisor*num_threads));

    // ---- 2. Parallel compute deltas and best edge (CSR histos) ----
    #pragma omp parallel
    {
        long long thread_best_diff = 0;
        int       thread_best_idx  = -1;
        Edge      thread_best_edge{};

        #pragma omp for schedule(dynamic, chunk) reduction(+:total_calcs) nowait
        for (int k = 0; k < n_cands; ++k) {
            const Edge &cand = flat_valid_conns[k];
            const int a = cand.first;
            const int b = cand.second;

            const EndpointHisto &H_to   = H_to_a[a];
            const EndpointHisto &H_from = H_from_b[b];

            long long local_delta = 0;

            const Dist p_max = (diameter > 0)
                               ? static_cast<Dist>(diameter - 1)
                               : static_cast<Dist>(0);

            for (Dist p = 0; p < p_max; ++p) {
                const int start_p = H_to.offsets[static_cast<std::size_t>(p)];
                const int end_p   = H_to.offsets[static_cast<std::size_t>(p) + 1U];
                if (start_p == end_p) continue;

                for (int idx_i = start_p; idx_i < end_p; ++idx_i) {
                    const int i = static_cast<int>(H_to.nodes[static_cast<std::size_t>(idx_i)]);
                    const std::vector<int> &dists_i = dists[i];
                    const int * __restrict dists_i_ptr = dists_i.data();

                    const Dist q_max = static_cast<Dist>(diameter - 1 - p);
                    for (Dist q = 0; q < q_max; ++q) {
                        if (p == 0 && q == 0) continue;

                        const int start_q = H_from.offsets[static_cast<std::size_t>(q)];
                        const int end_q   = H_from.offsets[static_cast<std::size_t>(q) + 1U];
                        if (start_q == end_q) continue;

                        const Dist d_prime = static_cast<Dist>(p + q + 1);

                        for (int idx_j = start_q; idx_j < end_q; ++idx_j) {
                            const int j = static_cast<int>(H_from.nodes[static_cast<std::size_t>(idx_j)]);
                            ++total_calcs;

                            const Dist r_star = static_cast<Dist>(dists_i_ptr[j]);
                            if (r_star > d_prime) {
                                local_delta += static_cast<long long>(r_star - d_prime);
                            }
                        }
                    }
                }
            }

            deltas_cand[static_cast<std::size_t>(k)].value = local_delta;

            // Thread-local best, same semantics as before (baseline 0, only >0 matters)
            if (local_delta > thread_best_diff) {
                thread_best_diff = local_delta;
                thread_best_idx  = k;
                thread_best_edge = cand;
            } else if (local_delta == thread_best_diff &&
                       local_delta > 0 &&
                       (thread_best_idx < 0 || k < thread_best_idx)) {
                // deterministic tie-breaking: lowest index
                thread_best_idx  = k;
                thread_best_edge = cand;
            }
        } // omp for

        #pragma omp critical
        {
            if (thread_best_diff > global_best_diff) {
                global_best_diff = thread_best_diff;
                global_best_idx  = thread_best_idx;
                global_best_edge = thread_best_edge;
            } else if (thread_best_diff == global_best_diff &&
                       thread_best_diff > 0 &&
                       thread_best_idx >= 0 &&
                       (global_best_idx < 0 || thread_best_idx < global_best_idx)) {
                global_best_idx  = thread_best_idx;
                global_best_edge = thread_best_edge;
            }
        }
    } // omp parallel

    std::cout << "(v3) Time : " << timer.s() << "s. "
              << "|S|=" << n_cands << " w/ efficiency "
              << (static_cast<long double>(total_calcs) /
                  static_cast<long double>(expected_calcs)) * 100.0L
              << "%\n";

    return global_best_edge;
}


// Edge calc_best_edge_histoed_openmped(const std::vector<std::vector<int>> &dists,
//                                      const std::vector<Edge> &flat_valid_conns,
//                                      int diameter,
//                                      int num_threads = 1)
// {
//     using NodeId = std::uint16_t;  // n <= 20000 -> fits safely
//     using Dist   = std::uint16_t;  // distances < n <= 20000

//     const int n       = static_cast<int>(dists.size());
//     const int n_cands = static_cast<int>(flat_valid_conns.size());

//     if (n_cands == 0) {
//         return Edge{-1, -1};
//     }

//     // sanity in case someone ever calls this with bigger n
//     if (n > static_cast<int>(std::numeric_limits<NodeId>::max())) {
//         throw std::runtime_error("n exceeds NodeId capacity");
//     }
//     if (diameter > static_cast<int>(std::numeric_limits<Dist>::max())) {
//         throw std::runtime_error("diameter exceeds Dist capacity");
//     }

//     using Bucket     = std::vector<NodeId>;     // list of node IDs
//     using DistHisto  = std::vector<Bucket>;     // index = distance
//     using NodeHistos = std::vector<DistHisto>;  // index = node ID

//     NodeHistos H_to_a(n);      // flows (*, a) by distance
//     NodeHistos H_from_b(n);    // flows (b, *) by distance
//     std::vector<bool> built_to_a(n, false);
//     std::vector<bool> built_from_b(n, false);

//     Timer timer;
//     timer.reset();

//     // ---- 1. Build histograms per endpoint (sequential) ----
//     for (const Edge &cand : flat_valid_conns) {
//         const int a = cand.first;
//         const int b = cand.second;

//         if (!built_to_a[a]) {
//             built_to_a[a] = true;
//             auto &hist_to = H_to_a[a];
//             hist_to.assign(static_cast<std::size_t>(diameter) + 1U, Bucket{});
//             for (int i = 0; i < n; ++i) {
//                 const Dist dist = static_cast<Dist>(dists[i][a]);
//                 hist_to[dist].push_back(static_cast<NodeId>(i));
//             }
//         }

//         if (!built_from_b[b]) {
//             built_from_b[b] = true;
//             auto &hist_from = H_from_b[b];
//             hist_from.assign(static_cast<std::size_t>(diameter) + 1U, Bucket{});
//             for (int j = 0; j < n; ++j) {
//                 const Dist dist = static_cast<Dist>(dists[b][j]);
//                 hist_from[dist].push_back(static_cast<NodeId>(j));
//             }
//         }
//     }

//     const long long expected_calcs =
//         static_cast<long long>(n_cands) * n * n;

//     struct alignas(64) PaddedDelta {
//         long long value;
//     };
//     std::vector<PaddedDelta> deltas_cand(static_cast<std::size_t>(n_cands));

//     // Thread config same style as calc_best_conn_omp
//     if (num_threads <= 0) {
//         num_threads = omp_get_max_threads();
//     }
//     omp_set_dynamic(0);
//     omp_set_num_threads(std::max(1, num_threads));

//     long long total_calcs      = 0;
//     long long global_best_diff = 0;   // same as original best_delta init
//     int       global_best_idx  = -1;  // to preserve "first best" semantics
//     Edge      global_best_edge{};

//     // Reasonable chunk size; you can tune this if work per cand is skewed
//     const int chunk = std::max(1, (n_cands + num_threads - 1) / num_threads);

//     // ---- 2. Parallel compute deltas and best edge ----
//     #pragma omp parallel
//     {
//         long long thread_best_diff = 0;
//         int       thread_best_idx  = -1;
//         Edge      thread_best_edge{};

//         #pragma omp for schedule(dynamic, chunk) reduction(+:total_calcs) nowait
//         for (int k = 0; k < n_cands; ++k) {
//             const Edge &cand = flat_valid_conns[k];
//             const int a = cand.first;
//             const int b = cand.second;

//             const DistHisto &H_to   = H_to_a[a];
//             const DistHisto &H_from = H_from_b[b];

//             long long local_delta = 0;

//             // p, q, r_star, d_prime all fit in Dist (<= diameter <= n <= 20000)
//             const Dist p_max = static_cast<Dist>(diameter > 0 ? diameter - 1 : 0);
//             for (Dist p = 0; p < p_max; ++p) {
//                 const Bucket &bucket_p = H_to[p];
//                 if (bucket_p.empty()) continue;

//                 for (NodeId i_id : bucket_p) {
//                     const int i = static_cast<int>(i_id);
//                     const std::vector<int> &dists_i = dists[i];
//                     const int * __restrict dists_i_ptr = dists_i.data();

//                     const Dist q_max = static_cast<Dist>(diameter - 1 - p);
//                     for (Dist q = 0; q < q_max; ++q) {
//                         if (p == 0 && q == 0) continue;

//                         const Bucket &bucket_q = H_from[q];
//                         if (bucket_q.empty()) continue;

//                         const Dist d_prime = static_cast<Dist>(p + q + 1);

//                         for (NodeId j_id : bucket_q) {
//                             const int j = static_cast<int>(j_id);
//                             ++total_calcs;

//                             const Dist r_star = static_cast<Dist>(dists_i_ptr[j]);
//                             if (r_star > d_prime) {
//                                 local_delta += static_cast<long long>(r_star - d_prime);
//                             }
//                         }
//                     }
//                 }
//             }

//             // keep per-candidate deltas as before (optional but cheap)
//             deltas_cand[static_cast<std::size_t>(k)].value = local_delta;

//             // Thread-local best, preserving "only >0" behavior and
//             // deterministic tie-breaking by smallest k.
//             if (local_delta > thread_best_diff) {
//                 thread_best_diff = local_delta;
//                 thread_best_idx  = k;
//                 thread_best_edge = cand;
//             } else if (local_delta == thread_best_diff &&
//                        local_delta > 0 &&                       // ignore delta==0 ties w/ baseline
//                        (thread_best_idx < 0 || k < thread_best_idx)) {
//                 thread_best_idx  = k;
//                 thread_best_edge = cand;
//             }
//         } // omp for

//         // Reduce thread-local bests into a single global best
//         #pragma omp critical
//         {
//             if (thread_best_diff > global_best_diff) {
//                 global_best_diff = thread_best_diff;
//                 global_best_idx  = thread_best_idx;
//                 global_best_edge = thread_best_edge;
//             } else if (thread_best_diff == global_best_diff &&
//                        thread_best_diff > 0 &&
//                        thread_best_idx >= 0 &&
//                        (global_best_idx < 0 || thread_best_idx < global_best_idx)) {
//                 global_best_idx  = thread_best_idx;
//                 global_best_edge = thread_best_edge;
//             }
//         }
//     } // omp parallel

//     std::cout << "Time : " << timer.s() << "s. "
//               << "|S|=" << n_cands << " w/ efficiency "
//               << (static_cast<long double>(total_calcs) /
//                   static_cast<long double>(expected_calcs)) * 100.0L
//               << "%\n";

//     // If no edge achieved delta > 0, this matches original behavior:
//     // best_edge stayed default initialized.
//     return global_best_edge;
// }


vector<Candidate> compute_all_deltas_auto(const vector<vector<int>>& dists,
                        const vector<Edge>& conns,
                        ParallelBackend backend,
                        int max_threads)
{
    const int S = (int)conns.size();
    const int threads = max_threads;

    vector<Candidate> out(S);

    // Heuristic:
    // - If S is large enough to feed threads, parallelize OVER candidates (each delta scalar).
    // - Otherwise, keep outer loop scalar and parallelize INSIDE each delta.
    const bool parallel_over_candidates =
        (S >= 2 * threads);

    if (parallel_over_candidates) {

        if(backend == ParallelBackend::OpenMP){
            omp_set_num_threads(threads);
            #pragma omp parallel for schedule(dynamic, OPENMP_PARALLEL_FOR_PRAGMA)
            for (int idx = 0; idx < S; ++idx) {
                const auto e = conns[idx];
                const double d = calc_delta_scalar(dists, e); // scalar kernel
                out[idx] = Candidate{d, e};
            }
        }
        else if(backend == ParallelBackend::Threads){
            const int T = std::max(1, threads);
            const int chunk = (S + T - 1) / T;

            vector<std::thread> ths;
            ths.reserve(T);

            for (int t = 0; t < T; ++t) {
                const int start = t * chunk;
                const int end   = min(start + chunk, S);
                if (start >= end) break;

                ths.emplace_back([&, start, end]() {
                    for (int idx = start; idx < end; ++idx) {
                        const auto e = conns[idx];
                        const double d = calc_delta_scalar(dists, e); // scalar kernel
                        out[idx] = Candidate{d, e};
                    }
                });
            }
            for (auto& th : ths) th.join();
        }
        else{
            for (int idx = 0; idx < S; ++idx) {
                const auto e = conns[idx];
                const double d = calc_delta_scalar(dists, e);
                out[idx] = Candidate{d, e};
            }
        }
    } else {
        // Few candidates but huge n: parallelize inside the kernel
        for (int idx = 0; idx < S; ++idx) {
            const auto e = conns[idx];
            const double d = calc_delta_parallel_inner(dists, e, threads);
            out[idx] = Candidate{d, e};
        }
    }
    return out;
}

// Select up to k edges: most-negative-first, skipping conflicts greedily
vector<Edge> select_topk_with_conflicts(vector<Candidate>& cand, int k, const int x_dim,const int y_dim,const int z_dim,const int cube_dim)
{
    if (k <= 0 || cand.empty()) return {};

    // Sort by delta ascending (most negative first = highest priority)
    sort(cand.begin(), cand.end(),
              [](const Candidate& a, const Candidate& b){
                  if (a.delta != b.delta) return a.delta < b.delta;
                  // stable tie-breaker (optional): smaller (i,j) first
                  if (a.edge.first != b.edge.first) return a.edge.first < b.edge.first;
                  return a.edge.second < b.edge.second;
              });

    int max_n_picked = min(k,(int)cand.size());
    vector<Edge> picked;
    picked.reserve(max_n_picked);

    for (const auto& c : cand) {
        bool ok = true;
        for (const auto& p : picked) {
            if (are_mutually_exclusive(c.edge, p, x_dim, y_dim, z_dim, cube_dim)) { ok = false; break; }
        }
        if (!ok) continue;

        picked.push_back(c.edge);
        if ((int)picked.size() == k) break;
    }
    return picked;
}

// One-call helper: compute deltas (parallel), then greedy select respecting conflicts.
vector<Edge> topk_best_deltas_with_conflicts(const vector<vector<int>>& dists,
                                const vector<Edge>& conns,
                                int k,
                                const int x_dim,
                                const int y_dim,
                                const int z_dim,
                                const int cube_dim,
                                ParallelBackend backend,
                                int max_threads)
{
    auto all = compute_all_deltas_auto(dists, conns, backend, max_threads);
    return select_topk_with_conflicts(all, k, x_dim, y_dim, z_dim, cube_dim);
}

// driver
////////////////////////////////////////////////////////////////////////////////

void hops_gen(
            const int x_dim,
            const int y_dim,
            const int z_dim,
            const int cube_dim,
            bool diameter_prune=false,
            bool super_prune=false,
            int recalc_interval=1,
            bool use_histo_method=true){

    Timer global_timer;
    global_timer.reset();

    Timer timer;
    timer.reset();

    const int n_routers = x_dim*y_dim*z_dim;
    const int n_ports = 6;
    const int n_cubes = (x_dim/cube_dim)*(y_dim/cube_dim)*(z_dim/cube_dim);

    if (super_prune) diameter_prune = true;

    if (PARALLELIZE_DELTA && !(PARALLEL_BACKEND == ParallelBackend::Threads || PARALLEL_BACKEND == ParallelBackend::OpenMP)){
        cout << "NOTE :: hops_gen() :: Disallowing PARALLELIZE_DELTA as it is not openmp or threaded"<<endl;
        PARALLELIZE_DELTA = false;
    }

    cout << "Creating hops heuristic topology for " << n_routers << " routers ("<<x_dim<<", "<<y_dim<<", "<<z_dim<<") w/ "<<n_cubes<<" cubes"<<endl;
    cout << "   BACKEND (0=>single,1=>openmp,2=>threads?        "<<static_cast<int>(PARALLEL_BACKEND)<<endl;
    cout << "   MAX_THREADS?                                    "<<MAX_THREADS<<endl;
    cout << "   PARALLELIZE_DELTA?                              "<<PARALLELIZE_DELTA<<endl;
    cout << "   diameter prune?                                 "<<diameter_prune<<endl;
    cout << "   super_prune?                                    "<<super_prune<<endl;
    cout << "   histo_method?                                   "<<use_histo_method<<endl;

    string base_name;
    base_name = "hops_cpp_" + to_string(n_routers) + "r";
    if(diameter_prune && super_prune){
        base_name = base_name + "_superdiamprune";
    }
    else if(diameter_prune){
        base_name = base_name + "_diamprune";
    }
    if(recalc_interval > 1){
        base_name = base_name + "_" + to_string(recalc_interval) + "recalc";
    }

    string out_map_name = OUT_MAP_DIR + base_name + ".map";
    string out_log_name = OUT_LOG_DIR + base_name + ".txt";

    auto current_adj_mat = init_known_conns(x_dim, y_dim, z_dim, cube_dim);
    auto dists = all_pairs_hops(current_adj_mat);
    // print_twod_vector(dists);
    double avg_hops;
    int diameter;
    auto auto_avg_hops_and_diam = average_hops(dists);
    avg_hops= auto_avg_hops_and_diam.first;
    diameter = auto_avg_hops_and_diam.second;


    int n_conns = count_edges(current_adj_mat);
    int n_total = n_routers*n_ports;
    int n_remaining = n_total - n_conns;
    int max_n_conns = n_cubes*(x_dim*y_dim + x_dim*z_dim + z_dim*y_dim);

    auto valid_conns = init_valid_conns(x_dim, y_dim, z_dim, cube_dim);
    auto flat_valid_conns = flatten_valid_conns(valid_conns);
    int n_possible = 2*flat_valid_conns.size();


    print_status(n_conns, n_remaining, n_possible, avg_hops, timer, "Initial (electrical) connections");
    timer.reset();

    int n_iters = 0, last_print_iter = 0;
    const int n_tot_iters = (n_remaining / (2*recalc_interval)) + (n_cubes - 1);
    while (n_remaining > 0){

        // copy constructor
        vector<Edge> pruned_flat_valid_conns = flat_valid_conns;
        if(diameter_prune){
            bool super_pruned = false;
            bool basic_pruned = false;
            auto peripheries = periphery_from_dists(dists);
            if(super_prune){
                // modifies in place
                super_prune_flat_valid_conns(pruned_flat_valid_conns, peripheries);
                if ( pruned_flat_valid_conns.size() > 0){
                    // success
                    super_pruned = true;
                }
            }
            if(!super_pruned){
                // modifies in place
                prune_flat_valid_conns(pruned_flat_valid_conns, peripheries);
                if ( pruned_flat_valid_conns.size() > 0){
                    // success
                    basic_pruned = true;
                }
            }

            if(pruned_flat_valid_conns.size() == 0){
                pruned_flat_valid_conns = flat_valid_conns;
                basic_pruned = false;
                super_pruned = false;
                // cout << "Un- pruned"<<endl;
            }

            // cout << "Super? "<<super_pruned<<", basic? "<<basic_pruned<<", un? "<<!(super_pruned || basic_pruned)<<endl;
            // cout << "Before pruning, |S|="<<flat_valid_conns.size()<<". After pruning, |S|="<<pruned_flat_valid_conns.size()<<endl;
        }

        // for (auto edge : pruned_flat_valid_conns){
        //     auto& [i,j] = edge;
        //     cout << "    possible conn : ("<<i<<", "<<j<<")"<<endl;
        // }

        // cout << "avg hops "<<avg_hops<<" diameter "<<diameter<<" and n_routers "<<n_routers<<endl;

        bool parallelize_delta = false;
        // only even allow parallelizing delta if PARALLELIZE_DELTA
        // modify dynamically. better to parallelize delta for small |S|
        if(PARALLELIZE_DELTA && pruned_flat_valid_conns.size() < 2*MAX_THREADS){
            parallelize_delta = true;
        }
        else{
            parallelize_delta = false;
        }

        vector<Edge> best_edges;

        // trivial
        if(pruned_flat_valid_conns.size() == 1){
            auto best_edge = pruned_flat_valid_conns.front();
            auto [i,j] = best_edge;
            // cout << "No choice. Connecting "<<i<<"->"<<j<<" w/ dist "<<dists[i][j]<<endl;
            best_edges.push_back(best_edge);
        }
        else if(avg_hops >= n_routers && false){
            // for unconnected, just connect anything with infinite distance?
            auto best_edge = calc_inf_conn(current_adj_mat, dists, pruned_flat_valid_conns);
            auto [i,j] = best_edge;
            cout << "Unconnected graph. Connecting "<<i<<"->"<<j<<" w/ dist "<<dists[i][j]<<endl;
            best_edges.push_back(best_edge);
        }

        else if (use_histo_method){
            Edge best_edge;
            if(PARALLEL_BACKEND == ParallelBackend::OpenMP){
                best_edge = calc_best_edge_histoed_openmped(dists, pruned_flat_valid_conns, diameter,MAX_THREADS);
            }
            else{
                best_edge = calc_best_edge_histoed(dists, pruned_flat_valid_conns, diameter);
            }
            best_edges.push_back(best_edge);
        }

        // top-k
        else if(recalc_interval > 1){
            best_edges = topk_best_deltas_with_conflicts(dists, pruned_flat_valid_conns,recalc_interval, x_dim,y_dim,z_dim, cube_dim,PARALLEL_BACKEND, MAX_THREADS);
            // cout << "TOP K"<<endl;
        }
        // top 1
        else{
            if(PARALLEL_BACKEND == ParallelBackend::Threads){
                auto best_edge = calc_best_conn_parallel( dists, pruned_flat_valid_conns, MAX_THREADS);
                best_edges.push_back(best_edge);
            }
            else if(PARALLEL_BACKEND == ParallelBackend::OpenMP){
                auto best_edge = calc_best_conn_omp( dists, pruned_flat_valid_conns, MAX_THREADS);
                best_edges.push_back(best_edge);
            }
            else{
                // decides on scalar or parallel delta based on arguments
                auto best_edge = calc_best_conn( dists, pruned_flat_valid_conns, parallelize_delta, MAX_THREADS );
                best_edges.push_back(best_edge);
            }
        }

        for (auto best_edge : best_edges){
            auto [i,j] = best_edge;
            current_adj_mat[i][j] = 1;
            current_adj_mat[j][i] = 1;

            // valid_conns is passed by reference (pbr) => udpates in place
            update_valid_conns_pbr(best_edge, valid_conns, x_dim, y_dim, z_dim, cube_dim);
            flat_valid_conns = flatten_valid_conns(valid_conns);


            n_conns += 2;
            assert(count_edges(current_adj_mat) == n_conns);
            n_remaining -= 2;
            n_possible = 2*flat_valid_conns.size();
        }

        dists = all_pairs_hops(current_adj_mat);
        auto auto_avg_hops_and_diam = average_hops(dists);
        avg_hops= auto_avg_hops_and_diam.first;
        diameter = auto_avg_hops_and_diam.second;
        if (n_iters % 100 == 0 || (last_print_iter - n_iters > 100)){
            print_status(n_conns, n_remaining, n_possible, avg_hops, timer, "Iteration and updates completed");
            timer.reset();
            last_print_iter = 0;
        }

        // wait_for_enter();
        // exit(-1);
        n_iters++;
        if(n_iters % 10 == 0)
        cout << "Iteration "<<n_iters<<"/"<<n_tot_iters<<endl;

        auto cur_time = global_timer.ms();
        log_metrics(out_log_name,n_iters,cur_time,avg_hops,n_conns);

    }

    print_status(n_conns, n_remaining, n_possible, avg_hops, timer, "Iteration and updates completed");


    write_adj_matrix(current_adj_mat, out_map_name);
    cout << "Wrote out to " << out_map_name << endl;

}

// main and CLAs
////////////////////////////////////////////////////////////////////////////////


int main(int argc, char* argv[]){

    // default
    int x_dim=4, y_dim=4, z_dim=8, cube_dim=4;
    // int x_dim=8, y_dim=8, z_dim=8, cube_dim=4;

    bool diameter_prune = false;
    bool super_prune = false;
    int recalc_interval = 1;
    bool histo_method = true;

    // Parse CLAs
    // ------------------------------------------------------------------
    for(int i=1; i<argc; i++){
        if( strcmp(argv[i], "--xyzc_dims") == 0 )
        {
            x_dim = stoi(argv[i+1]);
            y_dim = stoi(argv[i+2]);
            z_dim = stoi(argv[i+3]);
            cube_dim = stoi(argv[i+4]);
            i += 4;
        }
        else if (strcmp(argv[i], "--diameter_prune") == 0){
            diameter_prune = true;
        }
        else if (strcmp(argv[i], "--super_prune") == 0){
            super_prune = true;
        }
        else if(strcmp(argv[i], "--recalc_interval") == 0){
            recalc_interval = stoi(argv[i+1]);
            i++;
        }

        // operational args
        else if (strcmp(argv[i], "--max_threads") == 0){
            MAX_THREADS = stoi(argv[i+1]);
            i++;
        }
        else if (strcmp(argv[i], "--openmped") == 0){
            PARALLEL_BACKEND = ParallelBackend::OpenMP;
        }
        else if (strcmp(argv[i], "--threaded") == 0){
            PARALLEL_BACKEND = ParallelBackend::Threads;
        }
        else if (strcmp(argv[i], "--disallow_parallelize_delta") == 0){
            PARALLELIZE_DELTA = false;
        }
        else if (strcmp(argv[i], "--disallow_histo_method") == 0){
            histo_method = false;
        }
        // default for now
        else{
            cout << "Unrecognized argument: "<<argv[i]<< endl<<endl;
            // usage(argv[0]);
            exit(-1);
        }
    }

    if(PARALLEL_BACKEND == ParallelBackend::Threads && ( MAX_THREADS != clamp_threads(MAX_THREADS))){
        int new_val = clamp_threads(MAX_THREADS);
        cout << "NOTE :: main() :: Modifying max threads from "<<MAX_THREADS<<" to "<<new_val<<endl;
        MAX_THREADS = new_val;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    hops_gen(x_dim, y_dim, z_dim, cube_dim, diameter_prune, super_prune, recalc_interval, histo_method);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    cout << "hops_gen took " << elapsed_ms/1000.0 << "sec ("<<elapsed_ms/(1000.0*60.0)<<"min)"<<endl; 
}