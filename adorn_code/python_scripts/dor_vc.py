# Copyright (c) 2024 Purdue University
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Authors: Conor Green

"""

Description:
    Implementation of Nue (https://dl.acm.org/doi/10.1145/2907294.2907313)

    Uses "multilevel k-way partitioning algorithm"
        (https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=d545a4e5a5e00935e59141f26041531ba1aa97a0)

"""

# std
import argparse
import ast
import os

# pipd
import orjson

# locals
from cdg import CDG
from routing import Routing

class VNAllocator:

    # class vars
    ############

    verbose = False
    slow  = False

    INF = 999

    vc_mat_output_path_prefix = './topologies_and_routing/vc_mats'


    def __init__(self):

        # object vars
        #############

        # # topology vars
        # -------------
        self.n_routers = -1
        self.r_map = None

        # path vars
        # ---------
        # held in routing

        # chosen path vars
        # ----------------
        self.chosen_paths_flat = None
        self.chosen_paths_twod = None

        # Routing object
        # --------------
        self.my_Routing = None

        # tpuv4
        # -----
        self.x_dim = -1
        self.y_dim = -1
        self.z_dim = -1
        self.cube_dim = -1

        # name
        # ----
        self.base_name = None
    
    # setups
    ####################################################################################################

    def setup_w_apl(self, apl_file_path):

        if self.my_Routing is None:
            self.my_Routing = Routing()
        self.my_Routing.setup_given_all_path_list(apl_file_path)

        # required values
        if self.n_routers == -1:
            self.n_routers = self.my_Routing.get_n_routers()
        if self.r_map is None:
            self.r_map = self.my_Routing.r_map
        self.base_name = self.my_Routing.get_base_name()

    def setup_w_rmap(self, rmap_file_path, binary_r_map=False):

        if self.my_Routing is None:
            self.my_Routing = Routing()
        self.my_Routing.setup_given_r_map(rmap_file_path, binary_r_map=False)

        # required values
        if self.n_routers == -1:
            self.n_routers = self.my_Routing.get_n_routers()
        self.base_name = self.my_Routing.get_base_name()
        self.r_map = self.my_Routing.r_map


    # def setup(self, rmap_file_path, apl_file_path, binary_r_map=False):

    #     self.my_Routing = Routing()
    #     self.my_Routing.setup_given_both(rmap_file_path, apl_file_path, binary_r_map=False)

    def setup_w_pl(self, pl_file_path):

        if self.my_Routing is None:
            self.my_Routing = Routing()
        self.my_Routing.setup_given_path_list( pl_file_path)

        # required values
        if self.n_routers == -1:
            self.n_routers = self.my_Routing.get_n_routers()
        if self.r_map is None:
            self.r_map = self.my_Routing.r_map
        self.chosen_paths_flat, self.chosen_paths_twod = self.my_Routing.get_chosen_paths()
        self.base_name = self.my_Routing.get_base_name()

    # getters and setters
    ####################################################################################################

    def set_xyzc_dims(self, dims_tuple):

        self.x_dim, self.y_dim, self.z_dim, self.cube_dim = dims_tuple
        self.dim_str_dict = {'x':self.x_dim, 'y':self.y_dim, 'z':self.z_dim}

    def get_base_name(self):
        return self.base_name



    # tpuv4 funcs
    ####################################################################################################

    def c_to_xyz_cubes(self,c):
        assert(self.x_dim != -1)
        assert(self.y_dim != -1)
        assert(self.z_dim != -1)
        assert(self.cube_dim != -1)
        x_dim = self.x_dim
        y_dim = self.y_dim
        z_dim = self.z_dim
        cube_dim = self.cube_dim

        n_x_cube = x_dim // cube_dim
        n_y_cube = y_dim // cube_dim
        n_z_cube = z_dim // cube_dim

        xy_cube_slice = n_x_cube*n_y_cube

        temp_c = c

        z_cube = temp_c // xy_cube_slice
        temp_c = temp_c % xy_cube_slice
        y_cube = temp_c // n_x_cube
        x_cube = temp_c % n_x_cube

        # print(f'cube {c} => cube dims {(x_cube,y_cube,z_cube)}')

        return x_cube,y_cube,z_cube

    def rel_xyz_and_c_to_r(self,rel_x,rel_y,rel_z, c):
        x,y,z = self.rel_xyz_and_c_to_abs_xyz(rel_x,rel_y,rel_z, c)
        r = self.xyz_to_r(x,y,z)

        # print(f'rel_x/y/z {(rel_x,rel_y,rel_z)} on c {c} => x/y/z {(x,y,z)}')
        # print(f'\t=>r {r}')

        return r

    def rel_xyz_and_c_to_abs_xyz(self,rel_x,rel_y,rel_z, c):
        assert(self.x_dim != -1)
        assert(self.y_dim != -1)
        assert(self.z_dim != -1)
        assert(self.cube_dim != -1)
        x_dim = self.x_dim
        y_dim = self.y_dim
        z_dim = self.z_dim
        cube_dim = self.cube_dim

        x_cube,y_cube,z_cube = self.c_to_xyz_cubes(c)

        x = rel_x + cube_dim*x_cube
        y = rel_y + cube_dim*y_cube
        z = rel_z + cube_dim*z_cube

        return x,y,z

    def xyz_to_r(self,x,y,z):
        assert(self.x_dim != -1)
        assert(self.y_dim != -1)
        assert(self.z_dim != -1)
        assert(self.cube_dim != -1)
        x_dim = self.x_dim
        y_dim = self.y_dim
        z_dim = self.z_dim

        r = x + y*x_dim + z*x_dim*y_dim

        return r

    def r_to_xyz(self, r):
        assert(self.x_dim != -1)
        assert(self.y_dim != -1)
        assert(self.z_dim != -1)
        xd = self.x_dim
        yd = self.y_dim
        zd = self.z_dim

        xy_slice_size = xd*yd

        temp_r = r

        z = temp_r // xy_slice_size
        temp_r = temp_r % xy_slice_size
        y = temp_r // xd
        x = temp_r % xd

        return x,y,z

    def r_to_xyz_as_dict(self, r):
        x, y, z = self.r_to_xyz(r)
        as_dict = {'x':x,'y':y,'z':z}

        return as_dict

    def which_cube(self, i):

        assert(self.x_dim != -1)
        assert(self.y_dim != -1)
        assert(self.z_dim != -1)
        assert(self.cube_dim != -1)
        x_dim = self.x_dim
        y_dim = self.y_dim
        z_dim = self.z_dim
        cube_dim = self.cube_dim

        n_x = (x_dim // cube_dim)
        n_y = (y_dim // cube_dim)
        n_z = (z_dim // cube_dim)

        # print(f'n_x,n_y,n_z = {n_x},{n_y},{n_z}')

        i_x,i_y,i_z = self.r_to_xyz(i)

        # rel_i_x = i_x % cube_dim
        # rel_i_y = i_y % cube_dim
        # rel_i_z = i_z % cube_dim

        n_i_x = i_x // cube_dim
        n_i_y = i_y // cube_dim
        n_i_z = i_z // cube_dim


        n_xy = n_x*n_y

        n_cube = (n_i_z)*n_xy + (n_i_y)*n_x + (n_i_x)

        # print(f'{i} @ ({i_x},{i_y},{i_z}) is cube # {n_cube}')

        return n_cube

    def calc_conn_dim_w_pos_neg(self, i,j):
        assert(self.cube_dim != -1)
        cube_dim = self.cube_dim
        
        i_x,i_y,i_z = self.r_to_xyz(i)

        rel_i_x = i_x % cube_dim
        rel_i_y = i_y % cube_dim
        rel_i_z = i_z % cube_dim


        j_x,j_y,j_z = self.r_to_xyz(j)

        rel_j_x = j_x % cube_dim
        rel_j_y = j_y % cube_dim
        rel_j_z = j_z % cube_dim

        # should be just one type!
        # conn_types = []
        conn_type = None

        if(rel_i_y == rel_j_y and rel_i_z == rel_j_z):
            # conn_types.append('x')

            # pos face to neg face
            if (rel_i_x == cube_dim - 1) and (rel_j_x == 0):
                conn_type = 'x+'
            # neg face
            elif (rel_i_x == 0) and (rel_j_x == cube_dim - 1):
                conn_type = 'x-'
            # intracube conn
            elif j_x >= i_x:
                conn_type = 'x+'
            else:
                conn_type = 'x-'

        if(rel_i_x == rel_j_x and rel_i_z == rel_j_z):
            # conn_types.append('y')
            
            # pos face to neg face
            if (rel_i_y == cube_dim - 1) and (rel_j_y == 0):
                conn_type = 'y+'
            # neg face
            elif (rel_i_y == 0) and (rel_j_y == cube_dim - 1):
                conn_type = 'y-'
            # intracube conn
            elif j_y >= i_y:
                conn_type = 'y+'
            else:
                conn_type = 'y-'

        if(rel_i_x == rel_j_x and rel_i_y == rel_j_y):
            # conn_types.append('z')
            # pos face to neg face
            if (rel_i_z == cube_dim - 1) and (rel_j_z == 0):
                conn_type = 'z+'
            # neg face
            elif (rel_i_z == 0) and (rel_j_z == cube_dim - 1):
                conn_type = 'z-'
            # intracube conn
            elif j_z >= i_z:
                conn_type = 'z+'
            else:
                conn_type = 'z-'
        
        if conn_type == None:
            input(f'{i} ({rel_i_x},{rel_i_y},{rel_i_z}) -> {j} ({rel_j_x},{rel_j_y},{rel_j_z}) => conn_type = {conn_type}')
        # if self.verbose:
        #     print(f'{i} ({rel_i_x},{rel_i_y},{rel_i_z}) -> {j} ({rel_j_x},{rel_j_y},{rel_j_z}) => conn_type = {conn_type}')

        return conn_type

    def calc_conn_dim(self, i,j):
        assert(self.cube_dim != -1)
        cube_dim = self.cube_dim
        i_x,i_y,i_z = self.r_to_xyz(i)

        rel_i_x = i_x % cube_dim
        rel_i_y = i_y % cube_dim
        rel_i_z = i_z % cube_dim

        j_x,j_y,j_z = self.r_to_xyz(j)

        rel_j_x = j_x % cube_dim
        rel_j_y = j_y % cube_dim
        rel_j_z = j_z % cube_dim

        if self.verbose:
            print(f'{i} ({i_x},{i_y},{i_z}) -> {j} ({j_x},{j_y},{j_z}) ')


        # should be just one type!
        conn_type = None

        if(rel_i_y == rel_j_y and rel_i_z == rel_j_z):
            conn_type = 'x'

        if(rel_i_x == rel_j_x and rel_i_z == rel_j_z):
            conn_type = 'y'

        if(rel_i_x == rel_j_x and rel_i_y == rel_j_y):
            conn_type = 'z'
        
        if conn_type == None:
            input(f'{i} ({rel_i_x},{rel_i_y},{rel_i_z}) -> {j} ({rel_j_x},{rel_j_y},{rel_j_z}) => conn_type = {conn_type}')

        # if self.verbose:
        #     print(f'{i} ({rel_i_x},{rel_i_y},{rel_i_z}) -> {j} ({rel_j_x},{rel_j_y},{rel_j_z}) => conn_type = {conn_type}')

        return conn_type

    def calc_conn_dim_w_twists(self, i,j):
        assert(self.cube_dim != -1)
        cube_dim = self.cube_dim
        i_x,i_y,i_z = self.r_to_xyz(i)

        rel_i_x = i_x % cube_dim
        rel_i_y = i_y % cube_dim
        rel_i_z = i_z % cube_dim

        j_x,j_y,j_z = self.r_to_xyz(j)

        rel_j_x = j_x % cube_dim
        rel_j_y = j_y % cube_dim
        rel_j_z = j_z % cube_dim

        # should be just one type!
        conn_type = None
        is_twist_conn = False

        if(rel_i_y == rel_j_y and rel_i_z == rel_j_z):
            conn_type = 'x'
            if (i_y != j_y or i_z != j_z):
                is_twist_conn = True

        if(rel_i_x == rel_j_x and rel_i_z == rel_j_z):
            conn_type = 'y'
            if (i_x != j_x or i_z != j_z):
                is_twist_conn = True

        if(rel_i_x == rel_j_x and rel_i_y == rel_j_y):
            conn_type = 'z'
            if (i_y != j_y or i_x != j_x):
                is_twist_conn = True

        if conn_type == None:
            input(f'{i} ({rel_i_x},{rel_i_y},{rel_i_z}) -> {j} ({rel_j_x},{rel_j_y},{rel_j_z}) => conn_type = {conn_type} and is twist {is_twist_conn}')

        # if self.verbose:
        #     print(f'{i} ({rel_i_x},{rel_i_y},{rel_i_z}) -> {j} ({rel_j_x},{rel_j_y},{rel_j_z}) => conn_type = {conn_type}')


        return conn_type, is_twist_conn


    def path_crosses_dateline_considering_twist(self, path, dateline_dim='x'):
        assert(self.dim_str_dict is not None)
        assert(self.cube_dim != -1)
        dim_str_dict = self.dim_str_dict
        cube_dim = self.cube_dim

        dim_of_interest = dim_str_dict[dateline_dim]
        rel_dim_of_interest = dim_of_interest % cube_dim

        if self.verbose:
            print('-'*50)
            print(f'consdiering {path}')

        crosses = False
        cross_idx = -1

        plen = len(path) - 1
        for i in range(plen):

            src = path[i]
            dest = path[i+1]





            src_coord_dict = self.r_to_xyz_as_dict(src)
            dest_coord_dict = self.r_to_xyz_as_dict(dest)

            rel_src_coord_dict = {k:v%cube_dim for k,v in src_coord_dict.items()}
            rel_dest_coord_dict = {k:v%cube_dim for k,v in dest_coord_dict.items()}

            src_cube = self.which_cube(src)
            dest_cube = self.which_cube(dest)

            src_coord_of_interest = src_coord_dict[dateline_dim]
            dest_coord_of_interest = dest_coord_dict[dateline_dim]
            rel_src_coord_of_interest = src_coord_of_interest % cube_dim
            rel_dest_coord_of_interest = dest_coord_of_interest % cube_dim

            conn_type, is_twist = self.calc_conn_dim_w_twists(src, dest)

            if self.verbose:
                print(f'\tdl dim {dateline_dim}. dim_of_interest = {dim_of_interest}')
                print(f'\tsrc {src} cube {src_cube} @({src_coord_dict}) and rel ({rel_src_coord_dict})')
                print(f'\tdest {dest} cube {dest_cube} @({dest_coord_dict}) and rel ({rel_dest_coord_dict})')
                print(f'\tdl dim {dateline_dim}. src_coord_of_interest={src_coord_of_interest}. dest_coord_of_interest={dest_coord_of_interest}')
                print(f'\trel_src_coord_of_interest={rel_src_coord_of_interest}. rel_dest_coord_of_interest={rel_dest_coord_of_interest}')

                print(f'\tconn_type={conn_type} and is_twist={is_twist}')

            # dateline iff
            # coming out neg dir and same cube
            # coming out neg dir and to/from 0

            # if is an optical link
            is_optical_link = False
            if rel_src_coord_of_interest == cube_dim - 1 and rel_dest_coord_of_interest == 0:
                is_optical_link = True

            if rel_src_coord_of_interest == 0 and rel_dest_coord_of_interest == cube_dim - 1 :
                is_optical_link = True
            
            if self.verbose:
                print(f'is optical? {is_optical_link}')

            # ignore non optical links
            if not is_optical_link:
                pass
            elif src_cube == dest_cube:
                if rel_src_coord_of_interest == 0 or rel_dest_coord_of_interest == 0:
                    if self.verbose:
                        print(f'same cube, to/from 0 coord')
                    crosses = True
                    cross_idx = i
            else:
                if rel_src_coord_of_interest == 0 and src_cube < dest_cube:
                    if self.verbose:
                        print(f'diff cube, src < dest cube')
                    crosses = True
                    cross_idx = i
                if rel_dest_coord_of_interest == 0 and dest_cube < src_cube:
                    if self.verbose:
                        print(f'diff cube, dest < src cube')
                    crosses = True
                    cross_idx = i

        if self.verbose:
            input(f'path {path} crosses? {crosses}')

        return crosses, cross_idx

    def path_crosses_dateline(self, path, dateline_dim='x'):
        assert(self.dim_str_dict is not None)
        assert(self.cube_dim != -1)
        dim_str_dict = self.dim_str_dict
        cube_dim = self.cube_dim

        dim_of_interest = dim_str_dict[dateline_dim]
        rel_dim_of_interest = dim_of_interest % cube_dim

        crosses = False
        cross_idx = -1

        plen = len(path) - 1
        for i in range(plen):

            src = path[i]
            dest = path[i+1]

            src_coord_dict = self.r_to_xyz_as_dict(src)
            dest_coord_dict = self.r_to_xyz_as_dict(dest)

            src_coord_of_interest = src_coord_dict[dateline_dim]
            dest_coord_of_interest = dest_coord_dict[dateline_dim]
            rel_src_coord_of_interest = src_coord_of_interest % cube_dim
            rel_dest_coord_of_interest = dest_coord_of_interest % cube_dim

            # print(f'dl dim {dateline_dim}. src_coord_of_interest={src_coord_of_interest}. dest_coord_of_interest={dest_coord_of_interest}')

            # pos wraparound
            if src_coord_of_interest == (dim_of_interest - 1) and dest_coord_of_interest == 0:
                cross_idx = i
                crosses = True
            if rel_src_coord_of_interest == (dim_of_interest - 1) % cube_dim and rel_dest_coord_of_interest == 0:
                cross_idx = i
                crosses = True
            # neg wraparound
            if src_coord_of_interest == 0 and dest_coord_of_interest == (dim_of_interest - 1) :
                cross_idx = i
                crosses = True
            if rel_src_coord_of_interest == 0 and rel_dest_coord_of_interest == (dim_of_interest - 1) % cube_dim :
                cross_idx = i
                crosses = True

        return crosses, cross_idx


    def path_crosses_dateline_arbitrary(self, path, datelines_by_dim_dict, dateline_dim=None):

        # datelines_by_dim_dict[dateline_dim] =
        #       list of absolute coords pairs (src,dest) that imply
        #       if src, dest OR dest,src coords are equal
        #       and the conn is on the dateline_dim
        #       then it crosses



        # assert(self.dim_str_dict is not None)
        # assert(self.cube_dim != -1)
        # dim_str_dict = self.dim_str_dict
        # cube_dim = self.cube_dim

        # dim_of_interest = dim_str_dict[dateline_dim]
        # rel_dim_of_interest = dim_of_interest % cube_dim

        relevant_datelines = datelines_by_dim_dict[dateline_dim]
        # print(f'datelines for {dateline_dim} are {relevant_datelines}')

        crosses = False
        cross_idx = -1

        plen = len(path) - 1
        for i in range(plen):

            src = path[i]
            dest = path[i+1]

            src_coord = self.r_to_xyz(src)
            dest_coord = self.r_to_xyz(dest)

            conn_type = self.calc_conn_dim(src, dest)

            # print(f'{src} ({src_coord}) -> {dest} ({dest_coord}) of type {conn_type}')

            if dateline_dim != conn_type:
                # input(f'ignoring as wrong dim')
                continue
            
            # datelines are calculated where sources are of that dateline
            if (src_coord,dest_coord) in relevant_datelines:
                # input(f'src,dest matches a dateline')
                cross_idx = i
                crosses = True

            # datelines are calculated where sources are of that dateline
            if (dest_coord,src_coord) in relevant_datelines:
                # input(f'dest,src matches a dateline')
                cross_idx = i
                crosses = True

        return crosses, cross_idx

    def determine_datelines_multicube(self):
        assert(self.x_dim != -1)
        assert(self.y_dim != -1)
        assert(self.z_dim != -1)
        assert(self.cube_dim != -1)
        assert(self.r_map is not None)
        x_dim = self.x_dim
        y_dim = self.y_dim
        z_dim = self.z_dim
        cube_dim = self.cube_dim
        r_map = self.r_map

        n_x_cube = x_dim // cube_dim
        n_y_cube = y_dim // cube_dim
        n_z_cube = z_dim // cube_dim

        n_cubes = n_x_cube*n_y_cube*n_z_cube


        # print(f'Working on total problem of {(x_dim,y_dim,z_dim)} w/ # cubes {n_cubes}\n')

        datelines_by_dim_dict = {'x':[],'y':[],'z':[]}

        # x pos
        # -----
        src_rel_x = cube_dim - 1
        dest_rel_x = 0
        # src_rel_x = 0
        # dest_rel_x = cube_dim - 1
        for rel_y in range(cube_dim):
            for rel_z in range(cube_dim):


                conns_list = []
                conns_dict = {}

                if self.verbose:
                    print(f'\nrel_x/y/z {(src_rel_x, rel_y, rel_z)} -> {(dest_rel_x, rel_y, rel_z)}')

                for src_cube in range(n_cubes):

                    src_r = self.rel_xyz_and_c_to_r(src_rel_x, rel_y, rel_z, src_cube)
                    if self.verbose:
                        print(f'\tsrc {src_r:02} @ cube {src_cube}')
                    

                    for dest_cube in range(n_cubes):
                        dest_r = self.rel_xyz_and_c_to_r(dest_rel_x, rel_y, rel_z, dest_cube)

                        if r_map[src_r][dest_r] > 0:
                            if self.verbose:
                                print(f'\t\t-> dest {dest_r:02} @ cube {dest_cube}')
                            conns_list.append((src_cube,dest_cube))
                            conns_dict.update({src_cube : dest_cube})
                
                # input(f'conns_dict={conns_dict}')
                
                queue = list(range(n_cubes))
                loops_list = []

                while queue:
                    cur_cube = queue.pop()
                    next_cube = conns_dict[cur_cube]

                    # new loop
                    loops_list.append([cur_cube])
                    # print(f'new q iter. cur={cur_cube}, next={next_cube}, and loops_list={loops_list}, queue={queue}')

                    while next_cube != cur_cube:
                        loops_list[-1].append(next_cube)
                        queue.remove(next_cube)
                        next_cube = conns_dict[next_cube]

                        # print(f'\tcur={cur_cube}, next={next_cube}, and loops_list={loops_list}, queue={queue}')
                if self.verbose:
                    input(f'loops_list = {loops_list}')

                # list of abs coord tuples
                datelines = []
                dateline_cubes = []
                for loop in loops_list:
                    # select enter/exit zeroth cube as dateline
                    src_dateline_cube = loop[0]
                    try:
                        dest_dateline_cube = loop[1]
                    except:
                        dest_dateline_cube = loop[0]
                    # # select enter/exit min numbered cube as dateline
                    # dateline_cube = min(loop)
                    src_dateline_coord = self.rel_xyz_and_c_to_abs_xyz(src_rel_x, rel_y, rel_z, src_dateline_cube)
                    dest_dateline_coord = self.rel_xyz_and_c_to_abs_xyz(dest_rel_x, rel_y, rel_z, dest_dateline_cube)
                    dateline_coords = (src_dateline_coord, dest_dateline_coord)
                    datelines.append(dateline_coords)
                    dateline_cubes.append((src_dateline_cube,dest_dateline_cube))

                if self.verbose:
                    input(f'dateline(s) : {datelines}  ({dateline_cubes})')
                datelines_by_dim_dict['x'] += datelines
        
        # input(f"datelines_by_dim_dict['x']={datelines_by_dim_dict['x']}")

        # y pos
        # -----
        src_rel_y = cube_dim - 1
        dest_rel_y = 0
        for rel_x in range(cube_dim):
            for rel_z in range(cube_dim):


                conns_list = []
                conns_dict = {}

                if self.verbose:
                    print(f'\nrel_x/y/z {(rel_x, src_rel_y, rel_z)} -> {(rel_x, dest_rel_y, rel_z)}')


                for src_cube in range(n_cubes):

                    src_r = self.rel_xyz_and_c_to_r(rel_x, src_rel_y, rel_z, src_cube)                    
                    if self.verbose:
                        print(f'\tsrc {src_r:02} @ cube {src_cube}')
                    for dest_cube in range(n_cubes):
                        dest_r = self.rel_xyz_and_c_to_r(rel_x, dest_rel_y, rel_z, dest_cube)

                        if r_map[src_r][dest_r] > 0:
                            if self.verbose:
                                print(f'\t\t-> dest {dest_r:02} @ cube {dest_cube}')
                            conns_list.append((src_cube,dest_cube))
                            conns_dict.update({src_cube : dest_cube})
                
                queue = list(range(n_cubes))
                loops_list = []

                while queue:
                    cur_cube = queue.pop()
                    next_cube = conns_dict[cur_cube]

                    # new loop
                    loops_list.append([cur_cube])

                    while next_cube != cur_cube:
                        loops_list[-1].append(next_cube)
                        queue.remove(next_cube)
                        next_cube = conns_dict[next_cube]
                
                if self.verbose:
                    input(f'loops_list = {loops_list}')

                # list of abs coord tuples
                datelines = []
                dateline_cubes= []
                for loop in loops_list:
                    # select enter/exit zeroth cube as dateline
                    src_dateline_cube = loop[0]
                    try:
                        dest_dateline_cube = loop[1]
                    except:
                        dest_dateline_cube = loop[0]
                    # # select enter/exit min numbered cube as dateline
                    # dateline_cube = min(loop)
                    src_dateline_coord = self.rel_xyz_and_c_to_abs_xyz(rel_x, src_rel_y, rel_z, src_dateline_cube)
                    dest_dateline_coord = self.rel_xyz_and_c_to_abs_xyz(rel_x, dest_rel_y, rel_z, dest_dateline_cube)
                    dateline_coords = (src_dateline_coord, dest_dateline_coord)
                    datelines.append(dateline_coords)
                    dateline_cubes.append((src_dateline_cube,dest_dateline_cube))

                if self.verbose:
                    input(f'dateline(s) : {datelines}  ({dateline_cubes})')
                datelines_by_dim_dict['y'] += datelines
    
        # input(f"datelines_by_dim_dict['y']={datelines_by_dim_dict['y']}")

        # z pos
        # -----
        src_rel_z = cube_dim - 1
        dest_rel_z = 0
        for rel_x in range(cube_dim):
            for rel_y in range(cube_dim):


                conns_list = []
                conns_dict = {}

                if self.verbose:
                    print(f'\nrel_x/y/z {(rel_x, rel_y, src_rel_z)} -> {(rel_x, rel_y, dest_rel_z)}')

                for src_cube in range(n_cubes):

                    src_r = self.rel_xyz_and_c_to_r(rel_x, rel_y, src_rel_z, src_cube)
                    if self.verbose:
                        print(f'\tsrc {src_r:02} @ cube {src_cube}')

                    for dest_cube in range(n_cubes):
                        dest_r = self.rel_xyz_and_c_to_r(rel_x, rel_y, dest_rel_z, dest_cube)

                        if r_map[src_r][dest_r] > 0:
                            if self.verbose:
                                print(f'\t\t-> dest {dest_r:02} @ cube {dest_cube}')
                            conns_list.append((src_cube,dest_cube))
                            conns_dict.update({src_cube : dest_cube})

                queue = list(range(n_cubes))
                loops_list = []

                while queue:
                    cur_cube = queue.pop()
                    next_cube = conns_dict[cur_cube]

                    # new loop
                    loops_list.append([cur_cube])

                    while next_cube != cur_cube:
                        loops_list[-1].append(next_cube)
                        queue.remove(next_cube)
                        # cur_cube = next_cube
                        next_cube = conns_dict[next_cube]
                
                if self.verbose:
                    input(f'loops_list = {loops_list}')

                # list of abs coord tuples
                datelines = []
                dateline_cubes= []
                for loop in loops_list:
                    # select enter/exit zeroth cube as dateline
                    src_dateline_cube = loop[0]
                    try:
                        dest_dateline_cube = loop[1]
                    except:
                        dest_dateline_cube = loop[0]
                    # # select enter/exit min numbered cube as dateline
                    # dateline_cube = min(loop)
                    src_dateline_coord = self.rel_xyz_and_c_to_abs_xyz(rel_x, rel_y, src_rel_z, src_dateline_cube)
                    dest_dateline_coord = self.rel_xyz_and_c_to_abs_xyz(rel_x, rel_y, dest_rel_z, dest_dateline_cube)
                    dateline_coords = (src_dateline_coord, dest_dateline_coord)
                    datelines.append(dateline_coords)
                    dateline_cubes.append((src_dateline_cube,dest_dateline_cube))

                if self.verbose:
                    input(f'dateline(s) : {datelines}  ({dateline_cubes})')
                datelines_by_dim_dict['z'] += datelines
        # input(f"datelines_by_dim_dict['z']={datelines_by_dim_dict['z']}")

        # quit()

        return datelines_by_dim_dict

    # File I/O
    ####################################################################################################
    def stream_pathlist(self, path):

        with open(path, "r", buffering=1024*1024) as inf:
            next(inf, None)  # skip header
            for line in inf:
                line = line.strip()
                if line:
                    yield orjson.loads(line)  # produce one row at a time


    # big worker(s)
    ####################################################################################################
    def alloc_dateline_dor_tpuv4_vns(self, pathlist_filepath, validate_after=False):

        print('\n' + '='*100)
        print('Alloc dateline dor for tpuv4' )
        print('----------------------------')


        assert(self.n_routers != -1)
        n_routers = self.n_routers

        n_vcs = 2



        out_name_base = os.path.splitext(os.path.basename(pathlist_filepath))[0]
        out_name = f'{out_name_base}.vcmat2'

        print(f'out_name = {out_name}')

        out_name_path = os.path.join(self.vc_mat_output_path_prefix, out_name)



        # if n_routers > 128:
        print(f'Determining datelines')
        datelines_by_dim_dict = self.determine_datelines_multicube()
        print(f'Determined datelines')

        # input(f'datelines_by_dim_dict={datelines_by_dim_dict}')


        p_iter = -1
        n_paths = n_routers**2

        # clear file
        with open(out_name_path,'w+') as of:
            pass

        with open(out_name_path,'a') as of:
            for path in self.stream_pathlist(pathlist_filepath):


                if p_iter % 100000 == 0:
                    p_done = 100*(p_iter/n_paths)
                    print(f'paths {p_iter}/{n_paths} ({round(p_done)}%)')

                if self.verbose:
                    print('-'*10)

                path_src = path[0]
                path_dest = path[-1]

                plen = len(path) - 1
                if plen <= 1:
                    # vc_matrix[path_src][path_dest][path_src] = 0
                    out_line = (path_src, path_dest, path_src, 0)
                    of.write(str(out_line) + '\n')
                    continue

                # dim_subpaths = {'x+':None,
                #             'y+':None,
                #             'z+':None,
                #             'x-':None,
                #             'y-':None,
                #             'z-':None}

                dim_subpaths = {'x':[],
                                'y':[],
                                'z':[]}

                path_dims_w_pos_neg = [ self.calc_conn_dim_w_pos_neg(path[i],path[i+1])  for i in range(plen) ]
                path_dims = [ self.calc_conn_dim(path[i],path[i+1])  for i in range(plen) ]
                path_dims_w_twists = [ self.calc_conn_dim_w_twists(path[i],path[i+1])  for i in range(plen) ]


                if self.verbose:
                    print(f'='*100)
                    print(f'path {path}')
                    print(f'dims {path_dims}')
                    print(f'dims_w_twists {path_dims_w_twists}')
                    print(f'path_dims_w_pos_neg {path_dims_w_pos_neg}')

                cur_dim = path_dims[0]
                cur_idx = 0
                
                for i in range(1,plen):

                    next_dim = path_dims[i]

                    if cur_dim == next_dim:
                        continue
                    else:
                        subpath = path[cur_idx:i+1]


                        # SUBPATHS
                        dim_subpaths[cur_dim].append( subpath)

                        # PATHS
                        # combine into megapath
                        # dim_subpaths[cur_dim] += subpath


                        cur_dim = next_dim
                        cur_idx = i
                
                # print(f'after, cur_dim={cur_dim} and cur_idx={cur_idx}')
                
                # after
                if cur_idx != plen:
                    subpath = path[cur_idx:]

                    # SUBPATHS
                    dim_subpaths[cur_dim].append( subpath)

                    # FULL PATHS
                    # combine into megapath
                    # dim_subpaths[cur_dim] += subpath



                if self.verbose:
                    print(f'dim_subpaths {dim_subpaths}')

                # SUBPATHS
                for dim, subpaths in dim_subpaths.items():
                    for subpath in subpaths:
                        # print(f'subpath={subpath}')
                        # does_cross = self.path_crosses_dateline(subpath, dateline_dim=dim)
                        # does_cross, cross_idx = self.path_crosses_dateline(subpath, dateline_dim=dim)
                        does_cross, cross_idx = self.path_crosses_dateline_arbitrary(subpath, datelines_by_dim_dict, dateline_dim=dim)

                        # does_cross, cross_idx = self.path_crosses_dateline_considering_twist(subpath, dateline_dim=dim)



                        if self.verbose:
                            print(f'subpath {subpath} on dim {dim} crosses? {does_cross}')

                        vc = 0
                        # if does_cross:
                        #     vc = 1

                        subplen = len(subpath) - 1
                        for i in range(subplen):
                            cur_node = subpath[i]

                            if does_cross and i >= cross_idx:
                                vc = 1



                            out_line = (path_src, path_dest, cur_node, vc)
                            of.write(str(out_line) + '\n')


                if self.verbose:
                    input('cont?')

                p_iter += 1
        

        print(f'Wrote out to {out_name_path}')

        if not validate_after:
            print(f'\tSkipping validation!!!')
            return

        print(f'\tCompleted allocation. Checking CDG for deadlocks')


        # now construct cdgs to verify
        turns_lists = [ [], [] ]
        turn_to_paths_dict = {}
        for path in original_flat_path_list:

            path_src = path[0]
            path_dest = path[-1]

            plen = len(path) - 1
            if plen < 2:
                continue

            for i in range(plen - 1):
                node = path[i]
                this_vc = vc_matrix[path_src][path_dest][node]

                next_node = path[i+1]
                next_vc = vc_matrix[path_src][path_dest][next_node]

                if this_vc != next_vc:
                    continue

                next_next_node = path[i+2]
                
                this_turn = ( (node,next_node ),(next_node, next_next_node) )
                turns_lists[this_vc].append(this_turn)

                try:
                    turn_to_paths_dict[this_turn].append(path)
                except:
                    turn_to_paths_dict.update({ this_turn : [path] })


        for vc, turns in enumerate(turns_lists):
            print(f'\t# turns in vc {vc} : {len(turns)}')

        cdgs_list = [ CDG(), CDG()]
        for vc, turns in enumerate(turns_lists):
            this_cdg = cdgs_list[vc]
            this_cdg.init_cdg_from_turns(turns)

            cdg_cycles = this_cdg.networkx_get_cycle()

            print(f'vc {vc} w/ cycles {cdg_cycles}')

            if len(cdg_cycles) > 0:
                this_cdg.print_cycle_as_map_nodes(cdg_cycles)

                these_turns_as_nodes = this_cdg.translate_cycle_as_turns_of_nodes(cdg_cycles)

                print(f'turns as nodes {these_turns_as_nodes}')
                print(f'w/ dims...')
                for turn in these_turns_as_nodes:
                    c1 = turn[0]
                    c2 = turn[1]
                    print(f'{self.calc_conn_dim_w_pos_neg(c1[0],c1[1])}, {self.calc_conn_dim_w_pos_neg(c2[0],c2[1])} ',end='')
                print('')                

                print(f'related paths')
                for turn in these_turns_as_nodes:
                    print(f'\t{turn}...')
                    for path in turn_to_paths_dict[turn]:
                        plen = len(path) - 1
                        path_dims = [ self.calc_conn_dim(path[i],path[i+1])  for i in range(plen) ]
                        psrc = path[0]
                        pdest= path[-1]
                        path_vcs = [vc_matrix[psrc][pdest][pcur] for pcur in path]
                        print(f'\t\t{path} : {path_dims} : {path_vcs}')
                input('Deadlocky')

            # this_cdg.visualize_cdg()
        
        print(f'\tDeadlock free')
        
        return

# drivers
####################################################################################################

def drive_vnalloc(input_dict):

    my_VNAllocator = VNAllocator()

    # class variables
    if input_dict['vc_mat_dir'] is not None:
        my_VNAllocator.vc_mat_output_path_prefix = input_dict['vc_mat_dir']
    if input_dict['verbose']:
        my_VNAllocator.verbose = True

    my_VNAllocator.set_xyzc_dims( input_dict['xyzc_dims'])

    # my_VNAllocator.tmp()

    # setups
    if input_dict['fname'] is not None:
        my_VNAllocator.setup_w_rmap(input_dict['fname'])



    # TODO allow modify dateline through CLA
    my_VNAllocator.alloc_dateline_dor_tpuv4_vns(input_dict['pl_name'])


# main
####################################################################################################


def main():

    parser = argparse.ArgumentParser(description='Verify topology values')
    parser.add_argument('--filename','-f',type=str,help='.map file to evaluate', required=True)
    parser.add_argument('--chosen_paths_list','-pl',type=str,help='.paths file to evaluate', required=True)
    parser.add_argument('--out_name','-o',type=str,help='output name (without extension)')
    parser.add_argument('--xyzc_dims',nargs='+',type=int,help='type without parenthesis and use spaces, no commas',required=True)

    parser.add_argument('--validate_cdg',action='store_true',help='validate via CDG afterwards')

    parser.add_argument('--verbose','-v',action='store_true',help='verbsoe for debugging')
    parser.add_argument('--vc_mat_dir',type=str,help='directory to output vc_mat')


    args = parser.parse_args()

    fname = args.filename
    pl_name = args.chosen_paths_list
    out_name = args.out_name

    xyzc_dims = tuple(args.xyzc_dims)

    input_dict = {'fname':fname,
                    'pl_name':pl_name,
                    'out_name':out_name,
                    'xyzc_dims':xyzc_dims,

                    'verbose':args.verbose,
                    'vc_mat_dir':args.vc_mat_dir
                    }
    
    drive_vnalloc(input_dict)

if __name__ == '__main__':
    main()