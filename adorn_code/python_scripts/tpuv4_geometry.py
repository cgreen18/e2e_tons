# Copyright (c) 2026 Purdue University
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


class TPUv4_Geometry():

    # class vars
    ############

    verbose = False

    def __init__(self, xyzc_dims):
        

        self.set_xyzc_dims()


    def set_xyzc_dims(self, dims_tuple):
        self.xyzc_dims = dims_tuple
        self.x_dim, self.y_dim, self.z_dim, self.cube_dim = dims_tuple
        dim_str_dict = {'x':self.x_dim, 'y':self.y_dim, 'z':self.z_dim}

        min_ordered_dim_list = []
        min_ordered_dim_list.append( min([k for k,v in dim_str_dict.items()]) )
        min_ordered_dim_list.append( min([k for k,v in dim_str_dict.items() if k not in min_ordered_dim_list]) )
        min_ordered_dim_list.append( min([k for k,v in dim_str_dict.items() if k not in min_ordered_dim_list]) )

        max_ordered_dim_list = min_ordered_dim_list.copy()
        max_ordered_dim_list.reverse()

        self.dim_str_dict = dim_str_dict
        self.min_ordered_dim_list = min_ordered_dim_list
        self.max_ordered_dim_list = max_ordered_dim_list

    # TPUv4 geometry
    # ################################################################################

    def r_to_xyz(self, r):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        xy_slice_size = x_dim*y_dim

        temp_r = r

        z = temp_r // xy_slice_size
        temp_r = temp_r % xy_slice_size
        y = temp_r // x_dim
        x = temp_r % x_dim

        return x,y,z

    def xyz_to_r(self, x,y,z):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        return x + y*x_dim + z*x_dim*y_dim

    def r_to_rel_xyz(self, r):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        r_x,r_y,r_z = self.r_to_xyz(r)

        rel_r_x = r_x % cube_dim
        rel_r_y = r_y % cube_dim
        rel_r_z = r_z % cube_dim

        return rel_r_x, rel_r_y, rel_r_z

    def rel_xyz_is_on_face(self, rel_x, rel_y, rel_z):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        if rel_x == 0 or rel_x == cube_dim - 1:
            return True
        if rel_y == 0 or rel_y == cube_dim - 1:
            return True
        if rel_z == 0 or rel_z == cube_dim - 1:
            return True
        return False

    def calc_opt_conn_type(self, s, d):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        rel_s_x, rel_s_y, rel_s_z = self.r_to_rel_xyz(s)
        rel_d_x, rel_d_y, rel_d_z = self.r_to_rel_xyz(d)
        
        
        if (not self.rel_xyz_is_on_face(rel_s_x, rel_s_y, rel_s_z)) or (not self.rel_xyz_is_on_face(rel_d_x, rel_d_y, rel_d_z)):
            return None

        # x+
        if rel_s_x == cube_dim - 1 and rel_d_x == 0:
            return 'x+'
        # x-
        if rel_s_x == 0 and rel_d_x == cube_dim - 1:
            return 'x-'

        # y+
        if rel_s_y == cube_dim - 1 and rel_d_y == 0:
            return 'y+'
        # y-
        if rel_s_y == 0 and rel_d_y == cube_dim - 1:
            return 'y-'

        # z+
        if rel_s_z == cube_dim - 1 and rel_d_z == 0:
            return 'z+'
        # z-
        if rel_s_z == 0 and rel_d_z == cube_dim - 1:
            return 'z-'

    def calc_conn_type(self,asdf)

    def ocs_id(self, i, j):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        ij_conn_type = self.calc_opt_conn_type(i,j)

        if ij_conn_type is None:
            return -1

        nodes_per_face = cube_dim**2

        # + => i is representative
        # - => j is representative

        representative = i
        if '-' in ij_conn_type:
            representative = j

        rel_x, rel_y, rel_z = self.r_to_rel_xyz(representative)

        if 'x' in ij_conn_type:
            base_val = 0
            return base_val + rel_y + cube_dim*rel_z

        elif 'y' in ij_conn_type:
            base_val = nodes_per_face
            return base_val + rel_x + cube_dim*rel_z

        elif 'z' in ij_conn_type:
            base_val = 2*nodes_per_face
            return base_val + rel_x + cube_dim*rel_y

