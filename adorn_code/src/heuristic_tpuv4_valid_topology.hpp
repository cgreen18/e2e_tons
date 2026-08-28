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
#include <map>
#include <iostream>


// using namespace std;
using std::vector;
using std::tuple;
using std::map;
using std::pair;

using std::make_pair;
using std::make_tuple;

using std::cout;
using std::cerr;
using std::cin;
using std::endl;
using std::flush;
using std::getline;

// enums
enum class DIRECTION {
    X_POS = 0,
    X_NEG = 1,
    Y_POS = 2,
    Y_NEG = 3,
    Z_POS = 4,
    Z_NEG = 5
};

// convenient
typedef pair<int,int> Edge;




// TPU functions
////////////////////////////////////////////////////////////////////////////////

tuple<int,int,int> r_to_xyz(const int r, const int xd, const int yd, const int zd){
    int xy_slice_size = xd*yd;

    int temp_r = r;

    int z = temp_r / xy_slice_size;
    temp_r = temp_r % xy_slice_size;
    int y = temp_r / xd;
    int x = temp_r % xd;

    return make_tuple(x,y,z);
}

tuple<int,int,int> r_to_rel_xyz(const int r, const int xd, const int yd, const int zd, const int cd){

    auto [r_x,r_y,r_z] = r_to_xyz(r, xd, yd, zd);

    int rel_r_x = r_x % cd;
    int rel_r_y = r_y % cd;
    int rel_r_z = r_z % cd;

    return make_tuple(rel_r_x, rel_r_y, rel_r_z);
}

inline int xyz_to_r(const int x, const int y, const int z, const int xd, const int yd, const int zd){
    return x + y*xd + z*xd*yd;
}

vector< vector<int> > init_known_conns(const int x_dim,const int y_dim,const int z_dim,const int cube_dim){
    const int n_routers = x_dim*y_dim*z_dim;

    vector< vector<int>> known_conns(n_routers, vector<int>(n_routers,0));

    for (int src=0; src<n_routers; src++ ){

        auto [src_x,src_y,src_z] = r_to_xyz(src, x_dim, y_dim, z_dim);

        // xpos
        // if not on edge then conn
        if(src_x % cube_dim != cube_dim - 1){
            int targ_x = src_x + 1;
            int targ_y = src_y;
            int targ_z = src_z;
            int targ = xyz_to_r(targ_x, targ_y, targ_z, x_dim, y_dim, z_dim);
            known_conns[src][targ] = 1;
        }

        // xneg
        // if not on edge then conn
        if(src_x % cube_dim != 0){
            int targ_x = src_x - 1;
            int targ_y = src_y;
            int targ_z = src_z;
            int targ = xyz_to_r(targ_x, targ_y, targ_z, x_dim, y_dim, z_dim);
            known_conns[src][targ] = 1;
        }

        // ypos
        // if not on edge then conn
        if(src_y % cube_dim != cube_dim - 1){
            int targ_x = src_x;
            int targ_y = src_y + 1;
            int targ_z = src_z;
            int targ = xyz_to_r(targ_x, targ_y, targ_z, x_dim, y_dim, z_dim);
            known_conns[src][targ] = 1;
        }

        // yneg
        // if not on edge then conn
        if(src_y % cube_dim != 0){
            int targ_x = src_x;
            int targ_y = src_y - 1;
            int targ_z = src_z;
            int targ = xyz_to_r(targ_x, targ_y, targ_z, x_dim, y_dim, z_dim);
            known_conns[src][targ] = 1;
        }

        // zpos
        // if not on edge then conn
        if(src_z % cube_dim != cube_dim - 1){
            int targ_x = src_x;
            int targ_y = src_y;
            int targ_z = src_z + 1;
            int targ = xyz_to_r(targ_x, targ_y, targ_z, x_dim, y_dim, z_dim);
            known_conns[src][targ] = 1;
        }

        // zneg
        // if not on edge then conn
        if(src_z % cube_dim != 0){
            int targ_x = src_x;
            int targ_y = src_y;
            int targ_z = src_z - 1;
            int targ = xyz_to_r(targ_x, targ_y, targ_z, x_dim, y_dim, z_dim);
            known_conns[src][targ] = 1;
        }

    }


    return known_conns;
}

vector<int> iter_rel_xyz_across_cubes(const int rel_x,const int rel_y,const int rel_z,const int x_dim,const int y_dim,const int z_dim,const int cube_dim){

    int n_x_cubes = x_dim / cube_dim;
    int n_y_cubes = y_dim / cube_dim;
    int n_z_cubes = z_dim / cube_dim;

    vector<int> targs;

    for (int xc=0; xc < n_x_cubes; xc++){
        for (int yc=0; yc<n_y_cubes; yc++){
            for (int zc=0; zc<n_z_cubes; zc++){
                int xprime = rel_x + cube_dim*xc;
                int yprime = rel_y + cube_dim*yc;
                int zprime = rel_z + cube_dim*zc;
                int targ = xyz_to_r(xprime, yprime, zprime, x_dim, y_dim, z_dim);
                targs.push_back(targ);
            }
        }
    }

    return targs;
}

map<DIRECTION,vector<int>> poss_conns_for_r(const int r, const int x_dim,const int y_dim,const int z_dim,const int cube_dim){

    auto [rel_r_x, rel_r_y, rel_r_z] = r_to_rel_xyz(r, x_dim, y_dim, z_dim, cube_dim);

    // cout << "poss conns for " << r << " at rel coords ("<<rel_r_x<<", "<<rel_r_y<<", "<<rel_r_z<<")"<<endl;

    int n_x_cubes = x_dim / cube_dim;
    int n_y_cubes = y_dim / cube_dim;
    int n_z_cubes = z_dim / cube_dim;

    // map for simplicity of not needing to define DIRECTION hash
    map<DIRECTION, vector<int>> poss_conns;

    // x+
    if (rel_r_x == cube_dim - 1)
        poss_conns[DIRECTION::X_POS] = iter_rel_xyz_across_cubes(0, rel_r_y, rel_r_z, x_dim, y_dim, z_dim, cube_dim);
    // x-
    if (rel_r_x == 0)
        poss_conns[DIRECTION::X_NEG] = iter_rel_xyz_across_cubes(cube_dim - 1, rel_r_y, rel_r_z, x_dim, y_dim, z_dim, cube_dim);

    // y+
    if (rel_r_y == cube_dim - 1)
        poss_conns[DIRECTION::Y_POS] = iter_rel_xyz_across_cubes(rel_r_x, 0, rel_r_z, x_dim, y_dim, z_dim, cube_dim);
    // y-
    if (rel_r_y == 0)
        poss_conns[DIRECTION::Y_NEG] = iter_rel_xyz_across_cubes(rel_r_x, cube_dim - 1, rel_r_z, x_dim, y_dim, z_dim, cube_dim);

    // z+
    if (rel_r_z == cube_dim - 1)
        poss_conns[DIRECTION::Z_POS] = iter_rel_xyz_across_cubes(rel_r_x, rel_r_y, 0, x_dim, y_dim, z_dim, cube_dim);
    // z-
    if (rel_r_z == 0)
        poss_conns[DIRECTION::Z_NEG] = iter_rel_xyz_across_cubes(rel_r_x, rel_r_y, cube_dim - 1, x_dim, y_dim, z_dim, cube_dim);

    return poss_conns;
}

vector< vector<int> > init_valid_conns(const int x_dim,const int y_dim,const int z_dim,const int cube_dim){
    const int n_routers = x_dim*y_dim*z_dim;

    // cout << "init_valid_conns for " << n_routers << " routers ("<<x_dim<<", "<<y_dim<<", "<<z_dim<<")"<<endl;

    vector< vector<int> > valid_conns(n_routers, vector<int>(n_routers,0));

    for (int src=0; src<n_routers; src++){
        auto poss_conns = poss_conns_for_r(src, x_dim, y_dim, z_dim, cube_dim);

        // cout << "Init valid conns :: "<<endl;

        for (auto &entry : poss_conns) {
            DIRECTION dir = entry.first;
            // cout << "   dir "<<static_cast<int>(dir)<<" :: "<<endl;
            vector<int> &vals = entry.second;
            for (int conn : vals){
                valid_conns[src][conn] = 1;
                // cout << "   " << src << " -> "<<conn<<" allowed"<<endl;
            }
        }
    }

    return valid_conns;
}

DIRECTION calc_conn_dim_w_pos_neg(const int i, const int j, const int x_dim,const int y_dim,const int z_dim,const int cube_dim){

    auto [i_x,i_y,i_z] = r_to_xyz(i,x_dim,y_dim,z_dim);
    int rel_i_x = i_x % cube_dim;
    int rel_i_y = i_y % cube_dim;
    int rel_i_z = i_z % cube_dim;


    auto [j_x,j_y,j_z] = r_to_xyz(j,x_dim,y_dim,z_dim);
    int rel_j_x = j_x % cube_dim;
    int rel_j_y = j_y % cube_dim;
    int rel_j_z = j_z % cube_dim;

    DIRECTION conn_type;

    if(rel_i_y == rel_j_y && rel_i_z == rel_j_z){
        // pos face to neg face
        if ((rel_i_x == cube_dim - 1) && (rel_j_x == 0))
            conn_type = DIRECTION::X_POS;
        // neg face
        else if ((rel_i_x == 0) && (rel_j_x == cube_dim - 1))
            conn_type = DIRECTION::X_NEG;
        // intracube conn
        else if (j_x >= i_x)
            conn_type = DIRECTION::X_POS;
        else
            conn_type = DIRECTION::X_NEG;
    }
    else if(rel_i_x == rel_j_x && rel_i_z == rel_j_z){
        // pos face to neg face
        if ((rel_i_y == cube_dim - 1) && (rel_j_y == 0))
            conn_type = DIRECTION::Y_POS;
        // neg face
        else if ((rel_i_y == 0) && (rel_j_y == cube_dim - 1))
            conn_type = DIRECTION::Y_NEG;
        // intracube conn
        else if (j_y >= i_y)
            conn_type = DIRECTION::Y_POS;
        else
            conn_type = DIRECTION::Y_NEG;
    }
    else if(rel_i_x == rel_j_x && rel_i_y == rel_j_y){
        // pos face to neg face
        if ((rel_i_z == cube_dim - 1) && (rel_j_z == 0))
            conn_type = DIRECTION::Z_POS;
        // neg face
        else if ((rel_i_z == 0) && (rel_j_z == cube_dim - 1))
            conn_type = DIRECTION::Z_NEG;
        // intracube conn
        else if (j_z >= i_z)
            conn_type = DIRECTION::Z_POS;
        else
            conn_type = DIRECTION::Z_NEG;
    }
    else{
        cout << "ERROR :: calc_conn_dim_w_pos_neg() :: conn_type unknown. Exiting..."<<endl;
        exit(-1);
    }

    return conn_type;
}

// valid_conns is passed by reference (and thus modified in place)
void update_valid_conns_pbr(const Edge& new_conn,vector<vector<int>>& valid_conns,const int x_dim,const int y_dim,const int z_dim,const int cube_dim){

    auto [i,j] = new_conn;

    // cout << "Updating valid conns for new connection " << i <<"->"<<j<<endl;

    // get conn type and void all of those
    auto i_to_j_type = calc_conn_dim_w_pos_neg( i,j,x_dim,y_dim,z_dim,cube_dim);
    auto j_to_i_type = calc_conn_dim_w_pos_neg( j,i,x_dim,y_dim,z_dim,cube_dim);

    // cout << "i->j conn type "<<static_cast<int>(i_to_j_type)<<endl;
    // cout << "j->i conn type "<<static_cast<int>(j_to_i_type)<<endl;

    auto i_poss_conns = poss_conns_for_r(i, x_dim, y_dim, z_dim, cube_dim);
    auto j_poss_conns = poss_conns_for_r(j, x_dim, y_dim, z_dim, cube_dim);


    for (int i_conn : i_poss_conns[i_to_j_type]){
        // if (i_conn == j) continue;
        valid_conns[i][i_conn] = 0;
        valid_conns[i_conn][i] = 0;
        // cout << "Now, i_conn "
    }
    for (int j_conn : j_poss_conns[j_to_i_type]){
        // if (j_conn == i) continue;
        valid_conns[j][j_conn] = 0;
        valid_conns[j_conn][j] = 0;
    }
}

// note, only creates (i,j) where i<j
vector<Edge> flatten_valid_conns(const vector<vector<int>>& valids, int max_conns = -1) {
    int n = valids.size();
    vector<Edge> result;

    if (max_conns > -1)
    result.reserve(max_conns);

    for (int i = 0; i < n; ++i) {
        // assuming each row has size n
        for (int j = i+1; j < n; ++j) {
            if (valids[i][j] == 1) {
                result.emplace_back(i, j);
            }
        }
    }

    return result;
}

void prune_flat_valid_conns(vector<Edge>& pruned_flat_valid_conns, const vector<int> peripheries){

    for (size_t i = 0; i < pruned_flat_valid_conns.size(); ) {
        auto [s,d] = pruned_flat_valid_conns[i];

        bool is_in_periphery = false;
        for (auto p : peripheries){
            if (s == p || d == p){
                is_in_periphery = true;
                break;
            }
        }

        if (!is_in_periphery) {
            pruned_flat_valid_conns[i] = std::move(pruned_flat_valid_conns.back());
            pruned_flat_valid_conns.pop_back();
            // do not ++i here; inspect the swapped-in element next
        } else {
            ++i;
        }
    }

}

void super_prune_flat_valid_conns(vector<Edge>& pruned_flat_valid_conns, const vector<int> peripheries){

    for (size_t i = 0; i < pruned_flat_valid_conns.size(); ) {
        auto [s,d] = pruned_flat_valid_conns[i];

        bool s_in_periphery = false;
        bool d_in_periphery = false;
        for (auto p : peripheries){
            if (s == p){
                s_in_periphery = true;
            }
            if (d == p){
                d_in_periphery = true;
            }
            if (s_in_periphery && d_in_periphery) break;
        }

        if (!s_in_periphery || !d_in_periphery) {
            pruned_flat_valid_conns[i] = std::move(pruned_flat_valid_conns.back());
            pruned_flat_valid_conns.pop_back();
            // do not ++i here; inspect the swapped-in element next
        } else {
            ++i;
        }
    }

}

bool are_mutually_exclusive(const Edge e0, const Edge e1, const int x_dim,const int y_dim,const int z_dim,const int cube_dim){

    auto [i,j] = e0;
    auto [k,l] = e1;

    // symmetric exclusion. just work with e0
    auto i_to_j_type = calc_conn_dim_w_pos_neg( i,j,x_dim,y_dim,z_dim,cube_dim);
    auto j_to_i_type = calc_conn_dim_w_pos_neg( j,i,x_dim,y_dim,z_dim,cube_dim);
    auto i_poss_conns = poss_conns_for_r(i, x_dim, y_dim, z_dim, cube_dim);
    auto j_poss_conns = poss_conns_for_r(j, x_dim, y_dim, z_dim, cube_dim);

    for (int i_conn : i_poss_conns[i_to_j_type]){
        if (i_conn == k || i_conn == l) return true;
    }
    for (int j_conn : j_poss_conns[j_to_i_type]){
        if (j_conn == k || j_conn == l) return true;
    }

    return false;
}
