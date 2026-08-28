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

/*
Compilation (on Negishi)

Test that hipcc is working
apptainer exec --rocm -B $PWD /scratch/negishi/green456/images/pytorch-latest hipcc --version

apptainer exec --rocm -B $PWD /scratch/negishi/green456/images/pytorch-latest hipcc -O3 -std=c++23 --amdgpu-target=gfx90a -ffast-math -fno-exceptions -fno-rtti -mllvm -amdgpu-num-vgpr=64 -mllvm -amdgpu-post-RA-scheduler=true -mllvm -amdgpu-post-RA-scheduler=true src/heuristic_tpuv4_gpu.cpp -o gpu_deltas
apptainer exec --rocm -B $PWD /scratch/negishi/green456/images/pytorch-latest hipcc -O3 -std=c++23 --amdgpu-target=gfx90a src/heuristic_tpuv4_gpu.cpp -o gpu_deltas


apptainer exec --rocm -B $PWD /scratch/negishi/green456/images/pytorch-latest hipcc -O3 -std=c++23 --offload-arch=gfx90a -ffast-math -fno-exceptions -fno-rtti -mllvm -amdgpu-num-vgpr=64 -mllvm -amdgpu-post-RA-scheduler=true -mllvm -amdgpu-post-RA-scheduler=true src/heuristic_tpuv4_gpu.cpp -o gpu_deltas

# works!
apptainer exec --rocm -B "$PWD" /scratch/negishi/green456/images/pytorch-latest hipcc -O3 -std=c++23 --offload-arch=gfx90a -ffast-math -fno-exceptions -fno-rtti -Wno-pass-failed src/heuristic_tpuv4_gpu.cpp -o gpu_deltas_v2


*/


// std
#include <vector>
#include <map>
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

#include <hip/hip_runtime.h>

// local
#include "heuristic_tpuv4_valid_topology.hpp"
#include "heuristic_tpuv4_graph_functions.hpp"
#include "heuristic_tpuv4_file_console.hpp"


// using namespace std;
using std::vector;

using std::min;
using std::sort;
using std::to_string;
using std::stoi;
using std::iota;

extern const int INF;
extern Edge;
const string OUT_MAP_DIR = "files/heuristic_maps/";
const string RUNNING_OUT_MAP_DIR = "files/gpu_running_solutions/";
const string OUT_LOG_DIR = "files/timeline_logs/";

// #ifndef T
// #define T 128   // tile size over the distance matrix
// #endif
// #ifndef U
// #define U 32    // tile size over candidates
// #endif
// #ifndef BS     // block size. BS^2 = threads per block
// #define BS 16
// #endif

typedef uint64_t OUTPUT_TYPE;

#ifndef T
#define T 16   // tile size over the distance matrix
#endif
#ifndef U
#define U 128    // tile size over candidates
#endif
#ifndef BX     // block x size
#define BX 32
#endif
#ifndef BY     // block y size
#define BY 32
#endif

constexpr int BLOCK_SIZE      = 256;  // 4 wavefronts on MI210
static int ROWS_PER_BLOCK  = 256; 
// constexpr int ROWS_PER_BLOCK  = 8;    // tune this (4–16 is reasonable)

static int N_GPUS = 3;

// Types & helpers
////////////////////////////////////////////////////////////////////////////////

#define HIPCHK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)

constexpr uint16_t INFU16 = std::numeric_limits<uint16_t>::max();

// --- NEW STRUCT ---
// We'll use this to pass results back from each thread
struct GpuResult {
    float min_delta = std::numeric_limits<float>::max();
    int global_index = -1; // The index in the *original* h_s vector
    int candidate_a = -1;
    int candidate_b = -1;
};

// Kernels
////////////////////////////////////////////////////////////////////////////////

// TODO : decide if able to skip lower triangle tiles



__global__
void calc_deltas_kernel_histo(const std::uint16_t* __restrict d_dists,
                        int n,
                        int diameter,
                        const std::uint16_t* __restrict d_cand_a,
                        const std::uint16_t* __restrict d_cand_b,
                        OUTPUT_TYPE* __restrict d_deltas,
                        int n_cands,
                        int rows_per_block = 256)
{
    const int cand_idx = blockIdx.x;                  // which candidate
    const int base_row = blockIdx.y * rows_per_block; // starting row i

    if (cand_idx >= n_cands) return;

    const std::uint16_t a = d_cand_a[cand_idx];
    const std::uint16_t b = d_cand_b[cand_idx];

    OUTPUT_TYPE local_sum = 0;

    const long long expected_calcs =
        static_cast<long long>(n_cands) * n * n;
    long long total_calcs = 0;

    // Each block processes ROWS_PER_BLOCK rows (i) for this candidate
    for (int r = 0; r < rows_per_block; ++r) {
        const int i = base_row + r;
        if (i >= n) break;

        const int row_i_offset = i * n;

        // p = dist(i, a)
        const std::uint16_t p = d_dists[row_i_offset + a];

        // If p is too large, no q will satisfy p+q+1 < diameter
        if (p >= static_cast<std::uint16_t>(diameter - 1)) {
            continue;
        }

        // Threads in the block stride over j
        #pragma unroll
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            // q = dist(b, j)
            const int row_b_offset = b * n;
            const std::uint16_t q = d_dists[row_b_offset + j];

            if (p == 0 && q == 0) continue;

            const int d_prime_int = static_cast<int>(p) + static_cast<int>(q) + 1;
            if (d_prime_int >= diameter) continue;

            const std::uint16_t d_prime = static_cast<std::uint16_t>(d_prime_int);

            // r_star = dist(i, j)
            const std::uint16_t r_star = d_dists[row_i_offset + j];

            if (r_star > d_prime) {
                local_sum += static_cast<OUTPUT_TYPE>(r_star - d_prime);
            }
            ++total_calcs;
        }
    }

    // efficiency = (static_cast<long double>(total_calcs) /
    //               static_cast<long double>(expected_calcs)) * 100.0L;
    // std::cout << "    # candidates "<< n_cands<<" w/ efficiency "
    //           << (static_cast<long double>(total_calcs) /
    //               static_cast<long double>(expected_calcs)) * 100.0L
    //           << "%\n";


    // Block-level reduction of local_sum into shared memory
    __shared__ OUTPUT_TYPE sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Tree reduction
    for (int offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // One atomic add per block per candidate
    if (threadIdx.x == 0) {
        atomicAdd(reinterpret_cast<OUTPUT_TYPE*>(&d_deltas[cand_idx]), static_cast<OUTPUT_TYPE>(sdata[0]));

    }
}

__global__
void deltas_tiled_basic_kernel(const uint16_t* __restrict__ D,   // (n*n) row-major => d_ij = D[i*n + j]
                               int n,
                               const int2* __restrict__ conns, // (S)
                               int S,
                               OUTPUT_TYPE* __restrict__ partials,
                               const uint16_t diameter) // (n_tiles_per_dim*n_tiles_per_dim)*S => partials[tile_id*S + cand_id]
{
    const int tile_x = blockIdx.x;      // 0..n_tiles_per_dim-1 (columns)
    const int tile_y = blockIdx.y;      // 0..n_tiles_per_dim-1 (rows)
    const int cand_batch  = blockIdx.z;      // 0..ceil(S/U)-1

    const int n_tiles_per_dim = gridDim.x;        // == gridDim.y

    const int row_start = tile_y * T;          // row start of this tile
    const int col_start = tile_x * T;          // col start of this tile

    // rows out of range (shouldnt happen)
    if (row_start >= n || col_start >= n) return;
    // lower triangle
    if (tile_y > tile_x) return;


    // within grid
    // working on distances [tile_id*T, (tile_id+1)*T)
    const int tile_id = tile_y * n_tiles_per_dim + tile_x;

    // working on candidates [k0,k1) = [cand_batch*U, (cand_batch+1)*U)
    const int k0 = cand_batch * U;
    const int k1 = min(k0 + U, S);

    // within block
    // this is thread (tx,ty) working within the tile of size (BX,BY)
    const int tx = threadIdx.x, ty = threadIdx.y;
    // const int BX = blockDim.x,  BY = blockDim.y;

    // OUTPUT_TYPE local[U] = {0};

    // Iterate the T×T square submatrix; use only upper triangle (s<d)


    // for (int uidx = 0; uidx < U; ++uidx) {
    //     const int k = k0 + uidx;
    //     if (k >= k1) break;

    //     const int a = conns[k].x;
    //     const int b = conns[k].y;

    //     OUTPUT_TYPE total_improvement = 0;

    //     for (int row_delta = ty; row_delta < T; row_delta += BY) {
    //         const int i = row_start + row_delta;
    //         if (i >= n) continue;

    //         #pragma unroll
    //         for (int col_delta = tx; col_delta < T; col_delta += BX) {
    //             const int j = col_start + col_delta;

    //             // only accumulate when j > i (upper tri)
    //             if (j >= n || i > j) continue;

    //             // if (j > i) {
    //             // const auto dist_ij = (uint32_t)D[i*n + j];
    //             const auto dist_ij = D[i*n + j];

    //             const auto dist_ia = D[i*n + a];
    //             const auto dist_bj = D[b * n + j];

    //             // if (dist_ia == INFU16 || dist_bj == INFU16) continue;
    //             // if (dist_ia == (diameter-2) || dist_bj == (diameter-2)) continue;
    //             if (dist_ia >= (diameter-2) || dist_bj >= (diameter-2)) continue;
    //             if(dist_ij <= 3) continue;

    //             const uint16_t via_cand = dist_ia + 1u + dist_bj;
    //             if (via_cand >= dist_ij) continue;
    //             // via_cand less
    //             const OUTPUT_TYPE improvement = (OUTPUT_TYPE)dist_ij - (OUTPUT_TYPE)via_cand;
    //             // local[uidx] += improvement;
    //             total_improvement += improvement;
    //         }
    //     }

    //     atomicAdd(
    //     reinterpret_cast<OUTPUT_TYPE*>(&partials[tile_id * S + k]), static_cast<OUTPUT_TYPE>(total_improvement));
    // }



    // #pragma unroll
    for (int row_delta = ty; row_delta < T; row_delta += BY) {
        const int i = row_start + row_delta;
        if (i >= n) continue;

        // #pragma unroll
        for (int col_delta = tx; col_delta < T; col_delta += BX) {
            const int j = col_start + col_delta;

            // only accumulate when j > i (upper tri)
            if (j >= n || i > j) continue;

            // if (j > i) {
            // const auto dist_ij = (uint32_t)D[i*n + j];
            const auto dist_ij = D[i*n + j];

            if(dist_ij <= 3) continue;

            #pragma unroll
            for (int uidx = 0; uidx < U; ++uidx) {
                const int k = k0 + uidx;
                if (k >= k1) break;

                const int a = conns[k].x;
                const int b = conns[k].y;

                const auto dist_ia = D[i*n + a];
                const auto dist_bj = D[b * n + j];

                // if (dist_ia == INFU16 || dist_bj == INFU16) continue;
                if (dist_ia >= (diameter-2) || dist_bj >= (diameter-2)) continue;

                const uint16_t via_cand = dist_ia + 1u + dist_bj;
                if (via_cand >= dist_ij) continue;
                // via_cand less
                const OUTPUT_TYPE improvement = (OUTPUT_TYPE)dist_ij - (OUTPUT_TYPE)via_cand;
                // local[uidx] += improvement;


                atomicAdd(
                reinterpret_cast<OUTPUT_TYPE*>(&partials[tile_id * S + k]), static_cast<OUTPUT_TYPE>(improvement));

            }
        }
    }


    return;

    // // At this point local[uidx] holds this thread's contribution to delta
    // // for each candidate k in [k0, k1) for this tile.
    // // Accumulate directly into global partials using atomics.
    // #pragma unroll
    // for (int uidx = 0; uidx < U; ++uidx) {
    //     const int k = k0 + uidx;
    //     if (k >= k1) break;

    //     const OUTPUT_TYPE val = local[uidx];
    //     if (val != 0) {
    //         // partials layout: [tile_id][k]
    //         // where tile_id in [0, n_tiles_per_dim*n_tiles_per_dim)
    //         atomicAdd(
    //             reinterpret_cast<unsigned OUTPUT_TYPE*>(
    //                 &partials[tile_id * S + k]),
    //             static_cast<unsigned OUTPUT_TYPE>(val));
    //     }
    // }
    // // Reduce per-candidate partials across threads
    // // this is shared memory array that is used to accumulate across threads
    // extern __shared__ OUTPUT_TYPE ssum[];            // U * (BX*BY)
    // const int lane  = ty * BX + tx;
    // const int n_lanes = BX * BY;


    // #pragma unroll
    // for (int uidx = 0; uidx < U; ++uidx) {
    //     const int k = k0 + uidx;
    //     if (k >= k1) break;
    //     ssum[uidx * n_lanes + lane] = local[uidx];
    // }
    // __syncthreads();

    // for (int stride = n_lanes >> 1; stride > 0; stride >>= 1) {
    //     if (lane < stride) {
    //         #pragma unroll
    //         for (int uidx = 0; uidx < U; ++uidx) {
    //             const int k = k0 + uidx;
    //             if (k >= k1) break;
    //             ssum[uidx * n_lanes + lane] += ssum[uidx * n_lanes + lane + stride];
    //         }
    //     }
    //     __syncthreads();
    // }

    // if (lane == 0) {
    //     for (int uidx = 0; uidx < U; ++uidx) {
    //         const int k = k0 + uidx;
    //         if (k >= k1) break;
    //         // partials layout: [tile_id][k]  where tile_id in [0, n_tiles_per_dim*n_tiles_per_dim)
    //         partials[tile_id * S + k] = ssum[uidx * n_lanes + 0];
    //     }
    // }
}


// Host wrappers
////////////////////////////////////////////////////////////////////////////////


Edge calc_best_edge_histoed_multi_gpu(const std::vector<std::vector<int>> &dists,
                                      const std::vector<Edge> &flat_valid_conns,
                                      int diameter,
                                      int max_devices = -1)
{
    using NodeId = std::uint16_t;
    using Dist   = std::uint16_t;

    const int n       = static_cast<int>(dists.size());
    const int n_cands = static_cast<int>(flat_valid_conns.size());

    if (n == 0 || n_cands == 0) {
        return Edge{-1, -1};
    }

    if (n > static_cast<int>(std::numeric_limits<NodeId>::max())) {
        throw std::runtime_error("n exceeds NodeId capacity (uint16_t)");
    }
    if (diameter > static_cast<int>(std::numeric_limits<Dist>::max())) {
        throw std::runtime_error("diameter exceeds Dist capacity (uint16_t)");
    }

    // ---- 0. How many GPUs do we have / want? ----
    int device_count = 0;
    hipGetDeviceCount(&device_count);
    if (device_count <= 0) {
        throw std::runtime_error("No HIP devices found");
    }

    if (max_devices > 0 && max_devices < device_count) {
        device_count = max_devices;
    }
    if (device_count > n_cands) {
        // more GPUs than candidates → just use n_cands
        device_count = n_cands;
    }


    // ---- 1. Pack distances into flat uint16_t buffer (shared by all GPUs) ----
    const std::size_t n2 = static_cast<std::size_t>(n) * n;
    std::vector<Dist> h_dists(n2);

    for (int i = 0; i < n; ++i) {
        const auto &row = dists[i];
        for (int j = 0; j < n; ++j) {
            const int d = row[j];
            if (d < 0 || d > n) {
                throw std::runtime_error("distance out of expected range [0, n)");
            }
            h_dists[static_cast<std::size_t>(i) * n + j] =
                static_cast<Dist>(d);
        }
    }

    // ---- 2. Pack all candidates into uint16_t arrays on host ----
    std::vector<NodeId> h_cand_a(n_cands), h_cand_b(n_cands);
    for (int k = 0; k < n_cands; ++k) {
        const Edge &e = flat_valid_conns[k];
        if (e.first < 0 || e.first >= n || e.second < 0 || e.second >= n) {
            throw std::runtime_error("candidate endpoint out of range");
        }
        h_cand_a[k] = static_cast<NodeId>(e.first);
        h_cand_b[k] = static_cast<NodeId>(e.second);
    }

    // ---- 3. Partition candidates across devices ----
    std::vector<int> cand_begin(device_count);
    std::vector<int> cand_end(device_count);

    {
        const int base_chunk = n_cands / device_count;
        int extra = n_cands % device_count;
        int offset = 0;
        for (int dev = 0; dev < device_count; ++dev) {
            int chunk = base_chunk + (extra > 0 ? 1 : 0);
            if (extra > 0) --extra;

            cand_begin[dev] = offset;
            cand_end[dev]   = offset + chunk;
            offset += chunk;
        }
    }

    // ---- 4. Allocate per-device buffers and launch kernels ----
    struct DeviceBuffers {
        std::uint16_t *d_dists   = nullptr;
        NodeId        *d_cand_a  = nullptr;
        NodeId        *d_cand_b  = nullptr;
        OUTPUT_TYPE     *d_deltas  = nullptr;
        int            local_n_cands = 0;
    };

    std::vector<DeviceBuffers> dev_bufs(device_count);

    for (int dev = 0; dev < device_count; ++dev) {
        hipSetDevice(dev);

        DeviceBuffers &buf = dev_bufs[dev];

        const int begin = cand_begin[dev];
        const int end   = cand_end[dev];
        const int local_n_cands = end - begin;
        buf.local_n_cands = local_n_cands;

        if (local_n_cands == 0) continue;

        // Allocate memory
        hipMalloc(&buf.d_dists,  n2 * sizeof(Dist));
        hipMalloc(&buf.d_cand_a, local_n_cands * sizeof(NodeId));
        hipMalloc(&buf.d_cand_b, local_n_cands * sizeof(NodeId));
        hipMalloc(&buf.d_deltas, local_n_cands * sizeof(OUTPUT_TYPE));

        // Copy global distance matrix
        hipMemcpy(buf.d_dists, h_dists.data(),
                  n2 * sizeof(Dist), hipMemcpyHostToDevice);

        // Copy this device's slice of candidates
        hipMemcpy(buf.d_cand_a,
                  h_cand_a.data() + begin,
                  local_n_cands * sizeof(NodeId),
                  hipMemcpyHostToDevice);

        hipMemcpy(buf.d_cand_b,
                  h_cand_b.data() + begin,
                  local_n_cands * sizeof(NodeId),
                  hipMemcpyHostToDevice);

        // Initialize deltas to zero
        hipMemset(buf.d_deltas, 0, local_n_cands * sizeof(OUTPUT_TYPE));

        // Launch kernel on this device (no sync yet)
        dim3 block(BLOCK_SIZE, 1, 1);
        const int grid_y = (n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
        dim3 grid(local_n_cands, grid_y, 1);

        hipLaunchKernelGGL(
            calc_deltas_kernel_histo,
            grid,
            block,
            0,      // shared memory size
            0,      // stream
            buf.d_dists,
            n,
            diameter,
            buf.d_cand_a,
            buf.d_cand_b,
            buf.d_deltas,
            local_n_cands,
            ROWS_PER_BLOCK
        );


        // std::cout << "    GPU "<<dev <<" w/ # candidates "<< local_n_cands<<" w/ efficiency "
        //           << local_efficiency
        //           << "%\n";

    }

    // ---- 5. Gather results from all devices ----
    std::vector<OUTPUT_TYPE> h_deltas(n_cands, 0);

    for (int dev = 0; dev < device_count; ++dev) {
        hipSetDevice(dev);
        DeviceBuffers &buf = dev_bufs[dev];
        const int begin = cand_begin[dev];
        const int local_n_cands = buf.local_n_cands;

        if (local_n_cands == 0) continue;

        hipDeviceSynchronize(); // wait for this device

        std::vector<OUTPUT_TYPE> tmp(local_n_cands);
        hipMemcpy(tmp.data(), buf.d_deltas,
                  local_n_cands * sizeof(OUTPUT_TYPE),
                  hipMemcpyDeviceToHost);

        // Scatter into global deltas
        for (int i = 0; i < local_n_cands; ++i) {
            h_deltas[begin + i] = tmp[i];
        }

        // Free device memory
        hipFree(buf.d_dists);
        hipFree(buf.d_cand_a);
        hipFree(buf.d_cand_b);
        hipFree(buf.d_deltas);
    }

    // ---- 6. Pick best edge on host ----
    Edge      best_edge{-1, -1};
    OUTPUT_TYPE best_delta = 0;

    for (int k = 0; k < n_cands; ++k) {
        if (h_deltas[k] > best_delta) {
            best_delta = h_deltas[k];
            best_edge  = flat_valid_conns[k];
        }
    }

    return best_edge;
}

Edge calc_best_edge_histoed_gpu(const std::vector<std::vector<int>> &dists,
                                const std::vector<Edge> &flat_valid_conns,
                                int diameter,
                                int device_id = 0)
{
    using NodeId = std::uint16_t;
    using Dist   = std::uint16_t;

    const int n       = static_cast<int>(dists.size());
    const int n_cands = static_cast<int>(flat_valid_conns.size());

    if (n == 0 || n_cands == 0) {
        return Edge{-1, -1};
    }

    if (n > static_cast<int>(std::numeric_limits<NodeId>::max())) {
        throw std::runtime_error("n exceeds NodeId capacity (uint16_t)");
    }
    if (diameter > static_cast<int>(std::numeric_limits<Dist>::max())) {
        throw std::runtime_error("diameter exceeds Dist capacity (uint16_t)");
    }

    // ---- 1. Pack distances into flat uint16_t buffer ----
    const std::size_t n2 = static_cast<std::size_t>(n) * n;
    std::vector<Dist> h_dists(n2);

    for (int i = 0; i < n; ++i) {
        const auto &row = dists[i];
        for (int j = 0; j < n; ++j) {
            int d = row[j];
            if (d < 0 || d > n) {
                throw std::runtime_error("distance out of expected range [0, n)");
            }
            h_dists[static_cast<std::size_t>(i) * n + j] =
                static_cast<Dist>(d);
        }
    }

    // ---- 2. Pack candidates into uint16_t arrays ----
    std::vector<NodeId> h_cand_a(n_cands), h_cand_b(n_cands);
    for (int k = 0; k < n_cands; ++k) {
        const Edge &e = flat_valid_conns[k];
        if (e.first < 0 || e.first >= n || e.second < 0 || e.second >= n) {
            throw std::runtime_error("candidate endpoint out of range");
        }
        h_cand_a[k] = static_cast<NodeId>(e.first);
        h_cand_b[k] = static_cast<NodeId>(e.second);
    }

    hipSetDevice(device_id);

    // ---- 3. Allocate device memory ----
    Dist    *d_dists   = nullptr;
    NodeId  *d_cand_a  = nullptr;
    NodeId  *d_cand_b  = nullptr;
    OUTPUT_TYPE *d_deltas = nullptr;

    hipMalloc(&d_dists,   n2 * sizeof(Dist));
    hipMalloc(&d_cand_a,  n_cands * sizeof(NodeId));
    hipMalloc(&d_cand_b,  n_cands * sizeof(NodeId));
    hipMalloc(&d_deltas,  n_cands * sizeof(OUTPUT_TYPE));

    // ---- 4. Copy data to device ----
    hipMemcpy(d_dists,  h_dists.data(),   n2 * sizeof(Dist),    hipMemcpyHostToDevice);
    hipMemcpy(d_cand_a, h_cand_a.data(),  n_cands * sizeof(NodeId), hipMemcpyHostToDevice);
    hipMemcpy(d_cand_b, h_cand_b.data(),  n_cands * sizeof(NodeId), hipMemcpyHostToDevice);

    // Initialize deltas to zero
    hipMemset(d_deltas, 0, n_cands * sizeof(OUTPUT_TYPE));

    // ---- 5. Launch kernel ----
    dim3 block(BLOCK_SIZE, 1, 1);
    const int grid_y = (n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    dim3 grid(n_cands, grid_y, 1);


    hipLaunchKernelGGL(
        calc_deltas_kernel_histo,
        grid,
        block,
        0,      // shared memory size
        0,      // stream
        d_dists,
        n,
        diameter,
        d_cand_a,
        d_cand_b,
        d_deltas,
        n_cands,
        ROWS_PER_BLOCK
    );

        // std::cout << "    GPU "<<0 <<" w/ # candidates "<< n_cands<<" w/ efficiency "
        //           << local_efficiency
        //           << "%\n";

    hipDeviceSynchronize();

    // ---- 6. Copy deltas back and pick best on host ----
    std::vector<OUTPUT_TYPE> h_deltas(n_cands);
    hipMemcpy(h_deltas.data(), d_deltas,
              n_cands * sizeof(OUTPUT_TYPE),
              hipMemcpyDeviceToHost);

    Edge      best_edge = {-1, -1};
    OUTPUT_TYPE best_delta = 0;

    for (int k = 0; k < n_cands; ++k) {
        if (h_deltas[k] > best_delta) {
            best_delta = h_deltas[k];
            best_edge  = flat_valid_conns[k];
        }
    }

    // ---- 7. Cleanup ----
    hipFree(d_dists);
    hipFree(d_cand_a);
    hipFree(d_cand_b);
    hipFree(d_deltas);

    // best_edge is for the undirected (a,b) as before
    return best_edge;
}

void compute_deltas_tiled_basic_square(const std::vector<std::vector<int>>& dists_host,
                                       const std::vector<std::pair<int,int>>& conns_host,
                                       std::vector<uint32_t>& deltas_out)
{
    // basic setup
    // -----------
    const int n = (int)dists_host.size();
    const int S = (int)conns_host.size();
    deltas_out.assign(S, 0);
    if (n == 0 || S == 0) return;

    Timer timer;
    timer.reset();

    uint16_t diameter = 0;

    // Flatten and convert to uint16 distances
    std::vector<uint16_t> D_rm(n * n);
    for (int r=0; r<n; ++r){
        for (int c=r+1; c<n; ++c){
            uint16_t val = INFU16;
            if (dists_host[r][c] < n){
                val = static_cast<uint16_t>(dists_host[r][c]);
            }
            D_rm[r*n + c] = val;
            D_rm[c*n + r] = val; //sym
            if (val > diameter) diameter = val;
        }
    }

    // Pack candidates
    std::vector<int2> E(S);
    for (int k=0; k<S; ++k) E[k] = int2{ conns_host[k].first, conns_host[k].second };

    // Device buffers
    uint16_t* d_D = nullptr; HIPCHK( hipMalloc(&d_D, n*n*sizeof(uint16_t)) );
    int2*  d_E = nullptr; HIPCHK( hipMalloc(&d_E, S*sizeof(int2)) );
    HIPCHK( hipMemcpy(d_D, D_rm.data(), n*n*sizeof(uint16_t), hipMemcpyHostToDevice) );
    HIPCHK( hipMemcpy(d_E, E.data(),     S*sizeof(int2), hipMemcpyHostToDevice) );

    cout << "    PERFORMANCE:: Flatten and input memcpy took "<<timer.s()<<"s"<<endl;
    timer.reset();

    // Threadblock dimension setup
    // ---------------------------

    // Grid definition (square n_tiles_per_dim)
    // n_tiles_per_dim = ceil(n / T) ; grid = (n_tiles_per_dim, n_tiles_per_dim, n_batches)
    const int n_tiles_per_dim   = (n + T - 1) / T;         // == tiles_x == tiles_y
    const int n_batches = (S + U - 1) / U;
    dim3 grid(n_tiles_per_dim, n_tiles_per_dim, n_batches);


    // Block definition
    // Use a 2D block to cover the T×T tile
    dim3 block(BX, BY, 1);

    // cout << "Defined distance grid of ("<<T<<"x"<<T<<") for total # tiles "<<n_tiles_per_dim*n_tiles_per_dim<<endl;
    // cout << "Defined candidate grid of ("<<U<<") for total # tiles "<<n_batches<<endl;
    // cout << "Therefore, total # blocks "<<n_tiles_per_dim*n_tiles_per_dim*n_batches<<" but ~1/2 due to symmetry"<<endl;
    // cout << "Each block, # threads "<<BS*BS<<" where each thread handles "<<T/BS<<" distances and all "<<U<<" candidates"<<endl;


    // Output setup
    // ------------

    // Partials buffer: one int64 per (tile_id, candidate)
    const int num_tiles = n_tiles_per_dim * n_tiles_per_dim;

    cout << "Launching "<<num_tiles*n_batches<<" threadblocks"<< ". Expect "<<T*T*U<<" cpmpute and "<<T*T + 2*T*U<<" memory for ratio "<<(T*T*U)/(T*T + 2*T*U) <<endl;


    OUTPUT_TYPE* d_partials = nullptr;
    HIPCHK( hipMalloc(&d_partials, num_tiles * S * sizeof(OUTPUT_TYPE)) );
    HIPCHK( hipMemset(d_partials, 0, num_tiles * S * sizeof(OUTPUT_TYPE)) );

    // Shared memory for reduction: U * (BX*BY)
    const size_t n_lanes       = BX * BY;

    // comment out for atomic
    // const size_t shmem_bytes = U * n_lanes * sizeof(OUTPUT_TYPE);
    const size_t shmem_bytes = 0;


    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    size_t maxShmem = prop.sharedMemPerBlock;   // gfx90a: 64*1024

    if (shmem_bytes > maxShmem) {
        fprintf(stderr, "ERROR: requested shared memory %zu > max %zu. "
                        "Reduce U or block size.\n", shmem_bytes, maxShmem);
        std::abort();
    }

    if (BX*BY > 1024){
        fprintf(stderr, "ERROR: requested block size %zu * %zu > max 1024. "
                        "Reduce BX/BY.\n", BX,BY, maxShmem);
        std::abort();
    }

    // Launch
    hipLaunchKernelGGL(deltas_tiled_basic_kernel, grid, block, shmem_bytes, 0,
                       d_D, n, d_E, S, d_partials, diameter);
    HIPCHK( hipGetLastError() );
    HIPCHK( hipDeviceSynchronize() );


    cout << "    PERFORMANCE:: Kernel took "<<timer.s()<<"s"<<endl;
    timer.reset();


    // Reduce over all n_tiles_per_dim on host: delta[k] = sum_t partials[t,k]
    std::vector<OUTPUT_TYPE> partials_host(num_tiles * S);
    HIPCHK( hipMemcpy(partials_host.data(), d_partials,
                      num_tiles * S * sizeof(OUTPUT_TYPE),
                      hipMemcpyDeviceToHost) );

    for (int k=0; k<S; ++k) {
        OUTPUT_TYPE acc = 0;
        for (int t=0; t<num_tiles; ++t)
            acc += partials_host[t * S + k];
        deltas_out[k] = acc;
    }

    hipFree(d_partials);
    hipFree(d_E);
    hipFree(d_D);
}

// Pick the single best (most negative) edge
Edge pick_best_gpu(const vector<vector<int>>& dists,
                   const vector<Edge>& conns)
{
    Timer timer;        timer.reset();
    std::vector<uint32_t> deltas;
    // in place modifies deltas

    int n = (int)dists.size();
    int diameter = 0;

    for(int i=0; i<n; i++){
        for (int j=0; j<n; j++){
            if (dists[i][j] > diameter) diameter=dists[i][j];
        }
    }




    timer.reset();

    Edge best_edge;
    if(N_GPUS > 1) best_edge = calc_best_edge_histoed_multi_gpu(dists,conns,diameter,N_GPUS);
    else best_edge = calc_best_edge_histoed_gpu(dists,conns,diameter);

    std::cout << "    PERFORMANCE:: Kernel took "<<timer.s()<<"s. Best edge: (" << best_edge.first << "," << best_edge.second
              << ")\n";
    return best_edge;

    // timer.reset();

    // compute_deltas_tiled_basic_square(dists, conns, deltas);
    // // compute_deltas_tiled_per_candidate(dists, conns, deltas);

    // // auto it = std::min_element(deltas.begin(), deltas.end());
    // auto it = std::max_element(deltas.begin(), deltas.end());
    // const int best_idx = int(std::distance(deltas.begin(), it));
    // const auto best_edge = conns[best_idx];
    // const uint32_t best_delta = *it;

    // std::cout << "    PERFORMANCE:: Selection took "<<timer.s()<<"s. Best edge: (" << best_edge.first << "," << best_edge.second
    //           << "), delta=" << best_delta << "\n";

    // return best_edge;
}


// any INF conn
Edge calc_inf_conn(const vector<vector<int>>& adj_matrix, const vector<vector<int>>& dists, const vector<Edge>& flat_valid_conns){

    vector<int> order(flat_valid_conns.size());
    std::iota(order.begin(), order.end(), 0);

    std::mt19937 rng(std::random_device{}());
    std::shuffle(order.begin(), order.end(), rng);

    // lower is better
    Edge best_pair; 
    for (int idx : order) {
        const auto& [i, j] = flat_valid_conns[idx];
        if (dists[i][j] >= INF){
            // cout << "is infinite" << endl;
            best_pair = make_pair(i,j);
            break;
        }
    }

    // auto [i,j] = best_pair;
    // cout << "Inf dist edge " << i << " -> " << j << endl;


    return best_pair;
}

void ingest_start_map(const string& filename, vector< vector<int> >& adj_mat){
    std::ifstream start_map_file(filename);
    if (!start_map_file.is_open()) {
        std::cerr << "Error: could not open file '" << filename << "'\n";
        return;
    }

    int n_routers = (int)adj_mat.size();
    for(int i=0; i<n_routers; i++){
        for(int j=0; j<n_routers; j++){
            start_map_file >> adj_mat[i][j];
        }
    }

    cout << "Ingested start map "<<filename<<endl;
    // for(int i=0; i<n_routers; i++){
    //     cout << i <<" : ";
    //     for(int j=0; j<n_routers; j++){
    //         cout << adj_mat[i][j] << " ";
    //     }
    //     cout << endl;
    // }
}

// Driver
////////////////////////////////////////////////////////////////////////////////
void hops_gen(
            const int x_dim,
            const int y_dim,
            const int z_dim,
            const int cube_dim,
            bool diameter_prune=false,
            bool super_prune=false,
            int recalc_interval=1,
            bool start_map_given = false,
            string start_file_name = "")
{
    Timer global_timer; global_timer.reset();
    Timer timer;        timer.reset();

    const int n_routers = x_dim * y_dim * z_dim;
    const int n_ports   = 6;
    const int n_cubes   = (x_dim / cube_dim) * (y_dim / cube_dim) * (z_dim / cube_dim);

    if (super_prune) diameter_prune = true;

    cout << "Creating hops heuristic topology for " << n_routers
         << " routers ("<<x_dim<<", "<<y_dim<<", "<<z_dim<<") w/ "<<n_cubes<<" cubes\n";
    cout << "   diameter prune?                                 " << diameter_prune << "\n";
    cout << "   super_prune?                                    " << super_prune << "\n";
    cout << "   start_map?                                      " << start_map_given << "\n";

    string base_name = "hops_gpu_cpp_" + to_string(n_routers) + "r";
    if      (diameter_prune && super_prune) base_name += "_superdiamprune";
    else if (diameter_prune)                base_name += "_diamprune";
    if (recalc_interval > 1)                base_name += "_" + to_string(recalc_interval) + "recalc";

    const string out_map_name = OUT_MAP_DIR + base_name + ".map";
    const string out_log_name = OUT_LOG_DIR + base_name + ".txt";

    // Initial state (CPU)
    auto current_adj_mat = init_known_conns(x_dim, y_dim, z_dim, cube_dim);

    if (start_map_given){
        ingest_start_map(start_file_name, current_adj_mat);
    }

    auto dists           = all_pairs_hops(current_adj_mat);
    // auto avg_hops        = average_hops(dists);

    double avg_hops;
    int diameter;
    auto auto_avg_hops_and_diam = average_hops(dists);
    avg_hops= auto_avg_hops_and_diam.first;
    diameter = auto_avg_hops_and_diam.second;

    int n_conns     = count_edges(current_adj_mat);
    const int n_total = n_routers * n_ports;
    int n_remaining = n_total - n_conns;
    const int max_n_conns = n_cubes * (x_dim*y_dim + x_dim*z_dim + z_dim*y_dim);

    auto valid_conns       = init_valid_conns(x_dim, y_dim, z_dim, cube_dim);
    if(start_map_given){
        for(int i=0; i<n_routers; i++){
            for(int j=0; j<n_routers; j++){
                if (current_adj_mat[i][j] == 0) continue;
                Edge e = {i,j};
                update_valid_conns_pbr(e, valid_conns, x_dim, y_dim, z_dim, cube_dim);
            }
        }
    }
    auto flat_valid_conns  = flatten_valid_conns(valid_conns);
    int  n_possible        = 2 * (int)flat_valid_conns.size();

    print_status(n_conns, n_remaining, n_possible, avg_hops, timer, "Initial (electrical) connections");
    timer.reset();

    Timer all_else_timer;
    

    int n_iters = 0, last_print_iter = 0;
    const int n_tot_iters = (n_remaining / (2*recalc_interval)) + (n_cubes - 1);

    while (n_remaining > 0) {
        // --- optional diameter-based pruning (CPU) ---
        vector<Edge> pruned_flat_valid_conns = flat_valid_conns; // copy
        if (diameter_prune) {
            bool super_pruned = false, basic_pruned = false;
            auto peripheries = periphery_from_dists(dists);
            if (super_prune) {
                super_prune_flat_valid_conns(pruned_flat_valid_conns, peripheries);
                super_pruned = !pruned_flat_valid_conns.empty();
            }
            if (!super_pruned) {
                prune_flat_valid_conns(pruned_flat_valid_conns, peripheries);
                basic_pruned = !pruned_flat_valid_conns.empty();
            }
            if (pruned_flat_valid_conns.empty()) {
                pruned_flat_valid_conns = flat_valid_conns; // fall back
            }
        }

        vector<Edge> best_edges;
        cout << "    PERFORMANCE:: All else took "<<all_else_timer.s()<<endl;

        // cout << "# candidates "<<pruned_flat_valid_conns.size()<<endl;

        // --- selection policy ---
        if (pruned_flat_valid_conns.size() == 1) {
            best_edges.push_back(pruned_flat_valid_conns.front());
        }
        else if (avg_hops >= n_routers && false) {
            // disconnected: choose a bridging edge using current CPU distances
            Edge e = calc_inf_conn(current_adj_mat, dists, pruned_flat_valid_conns);
            best_edges.push_back(e);
        }
        // else if (recalc_interval > 1) {
        //     // top-k on CPU (your existing function)
        //     best_edges = topk_best_deltas_with_conflicts(dists,
        //                                                  pruned_flat_valid_conns,
        //                                                  recalc_interval,
        //                                                  x_dim, y_dim, z_dim, cube_dim);
        // } 
        else {
            // top-1 via GPU delta computation (uses existing host-based entry point)
            // NOTE: compute_all_deltas_gpu returns vector<Candidate> {delta, edge}
            // auto cand = compute_all_deltas_gpu(dists, pruned_flat_valid_conns);
            auto cand = pick_best_gpu(dists, pruned_flat_valid_conns);
            best_edges.push_back(cand); // Edge is pair<int,int>
        }
        all_else_timer.reset();

        // --- apply chosen edge(s) ---
        for (const Edge& e : best_edges) {
            const auto [i, j] = e;  // pair<int,int>
            current_adj_mat[i][j] = 1;
            current_adj_mat[j][i] = 1;

            // Update valid connections (CPU mirror) and re-flatten
            update_valid_conns_pbr(e, valid_conns, x_dim, y_dim, z_dim, cube_dim);
            flat_valid_conns = flatten_valid_conns(valid_conns);

            n_conns    += 2;
            assert(count_edges(current_adj_mat) == n_conns);
            n_remaining -= 2;
            n_possible  = 2 * (int)flat_valid_conns.size();
        }

        // --- recompute CPU APSP for pruning/metrics ---
        dists    = all_pairs_hops(current_adj_mat);
        // avg_hops = average_hops(dists);

        auto auto_avg_hops_and_diam = average_hops(dists);
        avg_hops= auto_avg_hops_and_diam.first;
        diameter = auto_avg_hops_and_diam.second;

        n_iters++;
        if (n_iters % 100 == 0 || (last_print_iter - n_iters > 100)) {
            print_status(n_conns, n_remaining, n_possible, avg_hops, timer, "Iteration and updates completed");
            timer.reset();
            last_print_iter = 0;
	}

        // const string running_out_map_name = RUNNING_OUT_MAP_DIR + base_name + "_iter" + to_string(n_iters) + ".map";
        const string running_out_map_name = RUNNING_OUT_MAP_DIR + base_name + ".map";

        write_adj_matrix(current_adj_mat, running_out_map_name);
        cout << "Wrote out to " << out_map_name << "\n";
        

        // if (n_iters % 1000 == 0 || (last_print_iter - n_iters > 1000)) {

        //     const string running_out_map_name = OUT_MAP_DIR + base_name + "_iter" + to_string(n_iters) + ".map";

        //     write_adj_matrix(current_adj_mat, out_map_name);
        //     cout << "Wrote out to " << out_map_name << "\n";
        // }


        if (n_iters % 10 == 0)
            cout << "Iteration " << n_iters << "/" << n_tot_iters << "\n";

        const auto cur_time = global_timer.ms();
        log_metrics(out_log_name, n_iters, cur_time, avg_hops, n_conns);
    }

    print_status(n_conns, n_remaining, n_possible, avg_hops, timer, "Script completed");
    write_adj_matrix(current_adj_mat, out_map_name);
    cout << "Wrote out to " << out_map_name << "\n";
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

    const int MAX_STR_LEN = 1024;
    string start_file_name;
    bool start_map_given = false;

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
        else if(strcmp(argv[i], "--rows_per_block") == 0){
            ROWS_PER_BLOCK = stoi(argv[i+1]);
            i++;
        }
        else if(strcmp(argv[i], "--start_map") == 0){
            start_file_name = argv[i+1];
            start_map_given = true;
            i++;
        }


        // default for now
        else{
            cout << "Unrecognized argument: "<<argv[i]<< endl<<endl;
            // usage(argv[0]);
            exit(-1);
        }
    }

    // avoid any problems later
    if (x_dim*y_dim*z_dim >= 65535){
        cout << "ERROR :: distances assumed as uint16_t. Large graph may cause issues. Exiting...";
        exit(0);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    hops_gen(x_dim, y_dim, z_dim, cube_dim, diameter_prune, super_prune, recalc_interval, start_map_given, start_file_name);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    cout << "hops_gen took " << elapsed_ms/1000.0 << "sec ("<<elapsed_ms/(1000.0*60.0)<<"min)"<<endl; 
}
