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

#pragma once

#include <vector>
#include <iostream>
#include <queue>
#include <limits>

// using namespace std;
using std::vector;
using std::queue;

using std::cout;
using std::cerr;
using std::cin;
using std::endl;
using std::flush;
using std::getline;

const int INF = std::numeric_limits<int>::max();


// graph functions
////////////////////////////////////////////////////////////////////////////////

// adj_matrix[i][j] != 0 means there's an edge.
vector<vector<int>> build_adj_list(const vector<vector<int>>& adj_matrix) {
    int n = adj_matrix.size();
    vector<vector<int>> adj_list(n);

    for (int i = 0; i < n; i++) {
        adj_list[i].reserve(8); // small hint since radix ~6
        for (int j = 0; j < n; j++) {
            if (adj_matrix[i][j] != 0) {
                adj_list[i].push_back(j);
            }
        }
    }
    return adj_list;
}

// BFS from a single source to get hop distances
vector<int> bfs_dists_one_source(const vector<vector<int>>& adj_list, int src) {
    int n = adj_list.size();
    vector<int> dist(n, INF);
    queue<int> q;

    dist[src] = 0;
    q.push(src);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        int du = dist[u];

        for (int v : adj_list[u]) {
            if (dist[v] == INF) {
                dist[v] = du + 1;
                q.push(v);
            }
        }
    }

    return dist;
}

vector<vector<int>> all_pairs_hops(const vector<vector<int>>& adj_matrix) {
    int n = adj_matrix.size();

    // Step 1: adjacency list for fast BFS
    vector<vector<int>> adj_list = build_adj_list(adj_matrix);

    // Step 2: run BFS from every node
    vector<vector<int>> dist_matrix;
    dist_matrix.reserve(n);

    for (int s = 0; s < n; s++) {
        dist_matrix.push_back(bfs_dists_one_source(adj_list, s));

        for (int d=0; d < n; d++){
            if (dist_matrix[s][d] > n){
                dist_matrix[s][d] = n;
            }
        }
    }

    return dist_matrix;
}

pair<double,int> average_hops(const vector<vector<int>>& dists) {
    long long total = 0;   // sum of all finite distances
    long long count = 0;   // number of pairs counted

    int n = dists.size();

    int diam = 0;

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {  // only i<j to avoid double-counting
            int d = dists[i][j];

            // if (d != INF) {
            total += d;
            count += 1;
            // }

            if (d > diam) diam = d;
        }
    }

    if (count == 0) {
        // graph is fully disconnected or n<2
        return make_pair(0.0,n);
    }

    return make_pair(static_cast<double>(total) / static_cast<double>(count), diam);
}

int count_edges(const vector<vector<int>>& adj_matrix) {
    int n = adj_matrix.size();
    int edges = 0;

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) { // only upper triangle
            if (adj_matrix[i][j] != 0) {
                edges += 2;
            }
        }
    }

    return edges;
}

vector<int> periphery_from_dists(const vector<vector<int>>& dists) {
    const int n = (int)dists.size();
    vector<int> ecc(n, 0);
    int diameter = 0;

    // eccentricity(i) = max finite distance from i to any j in its component
    for (int i = 0; i < n; ++i) {
        int mx = 0;
        for (int j = 0; j < n; ++j) {
            int dij = dists[i][j];
            // if (dij != INF && dij > mx) mx = dij;
            // dont ignore infinite distances
            if (dij > mx) mx = dij;
        }
        ecc[i] = mx;
        if (mx > diameter) diameter = mx;
    }

    // cout<<"diameter = "<<diameter<<endl;

    // periphery = nodes with ecc == diameter
    vector<int> periph;
    for (int i = 0; i < n; ++i)
        if (ecc[i] == diameter) periph.push_back(i);

    return periph;
}