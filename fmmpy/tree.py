from math import floor, sqrt

import compyle.array as ary
import numpy as np
from compyle.api import Elementwise, Reduction, Scan, annotate, declare
from compyle.low_level import atomic_inc, cast
from compyle.sort import radix_sort
from compyle.template import Template


@annotate(int='i, max_index', index='gintp', gfloatp='x, y, z',
          double='length, x_min, y_min, z_min')
def get_part_index(i, index, x, y, z, max_index, length, x_min, y_min, z_min):
    """Get the Morton z-index of particles using particles' coordinates

    Args:
        i (int): iterator for Elementwise function
        index (int[]): Morton z-index of particles at finest level
        x (float[]): particle x-coordinates
        y (float[]): particle y-coordinates
        z (float[]): particle z-coordinates
        max_index (int): 2 ** max_level
        length (double): length of the domain
        x_min (double): minimum x-coordinate of the domain
        y_min (double): minimum y-coordinate of the domain
        z_min (double): minimum z-coordinate of the domain
    """
    nx, ny, nz = declare('int', 3)
    nx = cast(floor((max_index * (x[i] - x_min)) / length), 'int')
    ny = cast(floor((max_index * (y[i] - y_min)) / length), 'int')
    nz = cast(floor((max_index * (z[i] - z_min)) / length), 'int')

    nx = (nx | (nx << 16)) & 0x030000FF
    nx = (nx | (nx << 8)) & 0x0300F00F
    nx = (nx | (nx << 4)) & 0x030C30C3
    nx = (nx | (nx << 2)) & 0x09249249

    ny = (ny | (ny << 16)) & 0x030000FF
    ny = (ny | (ny << 8)) & 0x0300F00F
    ny = (ny | (ny << 4)) & 0x030C30C3
    ny = (ny | (ny << 2)) & 0x09249249

    nz = (nz | (nz << 16)) & 0x030000FF
    nz = (nz | (nz << 8)) & 0x0300F00F
    nz = (nz | (nz << 4)) & 0x030C30C3
    nz = (nz | (nz << 2)) & 0x09249249

    index[i] = (nz << 2) | (ny << 1) | nx


class CopyArrays(Template):
    """Copy arrays in parallel
    """
    def __init__(self, name, arrays):
        super(CopyArrays, self).__init__(name=name)
        self.arrays = arrays
        self.number = len(arrays)

    def extra_args(self):
        return self.arrays, {'intp': ','.join(self.arrays)}

    @annotate(i='int')
    def template(self, i):
        '''
        % for t in range(obj.number//2):
        ${obj.arrays[t]}[i] = ${obj.arrays[obj.number//2+t]}[i]
        % endfor
        '''


class ReverseArrays(Template):
    """Reverse arrays in parallel
    """
    def __init__(self, name, arrays):
        super(ReverseArrays, self).__init__(name=name)
        self.arrays = arrays
        self.number = len(arrays)

    def extra_args(self):
        return self.arrays + ['length'], {'intp': ','.join(self.arrays),
                                          'length': 'int'}

    @annotate(i='int')
    def template(self, i):
        '''
        % for t in range(obj.number//2):
        ${obj.arrays[t]}[length-i-1] = ${obj.arrays[obj.number//2+t]}[i]
        % endfor
        '''


@annotate(i='int', gintp='leaf_sfc, leaf_sfc_a, bin_count, bin_idx')
def single_node(i, leaf_sfc, leaf_sfc_a, bin_count, bin_idx):
    """Merge the particles having the same Morton z-index into one bin

    Args:
        i (int): iterator for Elementwise function
        leaf_sfc (int[]): Morton z-index of particles at finest level
        leaf_sfc_a (int[]): Morton z-index at finest level shifted by one place
        bin_count (int[]): number of particles in each bin
        bin_idx (int[]): marks bins which are repeated
    """
    j = declare('int')
    if leaf_sfc[i] != leaf_sfc_a[i]:
        return
    elif i != 0 and leaf_sfc[i - 1] == leaf_sfc_a[i]:
        return
    else:
        j = 1
        while leaf_sfc[i] == leaf_sfc[i + j]:
            bin_count[i] += 1
            bin_idx[i + j] = 1
            j += 1


@annotate(i='int', x='gintp')
def map_sum(i, x):
    """Reduction to get the sum of the array
    """
    return x[i]


@annotate(i='int', in_arr='gintp', return_='int')
def input_expr(i, in_arr):
    """ Prefix sum of the array
    """
    if i == 0:
        return 0
    else:
        return in_arr[i - 1]


@annotate(int='i, item', out_arr='gintp')
def output_expr(i, item, out_arr):
    """ Prefix sum of the array
    """
    out_arr[i] = item


@annotate(gintp='sfc1, sfc2, level1, level2, lca_sfc, lca_level',
          int='i, dimension')
def inter_nodes(i, sfc1, sfc2, level1, level2, lca_sfc, lca_level, dimension):
    """Find the lowest common ancestor of all nodes at the finest level

    Args:
        i (int): iterator for Elementwise function
        sfc1 (int[]): Morton z-index at finest level
        sfc2 (int[]): Morton z-index at finest level shifted by one place
        level1 (int[]): level of nodes at finest level
        level2 (int[]): level of nodes at finest level shifted by one place
        lca_sfc (int[]): Morton z-index of internal nodes
        lca_level (int[]): level of internal nodes
        dimension (int): dimension of the problem
    """
    level_diff, xor_id, i1, i2, level, j = declare('int', 6)
    level_diff = cast(abs(level1[i] - level2[i]), 'int')

    if level1[i] - level2[i] > 0:
        i1 = sfc1[i]
        i2 = sfc2[i] << dimension * level_diff
        level = level1[i]
    elif level1[i] - level2[i] < 0:
        i1 = sfc1[i] << dimension * level_diff
        i2 = sfc2[i]
        level = level2[i]
    else:
        i1 = sfc1[i]
        i2 = sfc2[i]
        level = level1[i]

    xor_id = i1 ^ i2

    if xor_id == 0:
        lca_sfc[i] = i1 >> dimension * level_diff
        lca_level[i] = level - level_diff
        return

    for j in range(level + 1, 0, -1):
        if xor_id > ((1 << (j - 1) * dimension) - 1):
            lca_sfc[i] = i1 >> dimension * j
            lca_level[i] = level - j
            return

    lca_sfc[i] = 0
    lca_level[i] = 0
    return


@annotate(gintp='sfc1, sfc2, level1, level2, lca_sfc, all_idx, lca_level, '
          'temp_idx', int='i, dimension')
def find_parents(i, sfc1, sfc2, level1, level2, all_idx, lca_sfc, lca_level,
                 temp_idx, dimension):
    """Find the parents of all the nodes and internal nodes

    Args:
        i (int): iterator for Elementwise function
        sfc1 (int[]): Morton z-index of nodes for calculating the parents
        sfc2 (int[]): Morton z-index of nodes shifted by one place
        level1 (int[]): level of nodes for calculating the parents
        level2 (int[]): level of nodes shifted by one place
        all_idx (int[]): relative index of the nodes in tree
        lca_sfc (int[]): Morton z-index of parents
        lca_level (int[]): level of parents
        temp_idx (int[]): temporary array to store the relative index of the
                          child of the current parent node
        dimension (int): dimension of the problem
    """
    level_diff, xor_id, i1, i2, level, j = declare('int', 6)
    level_diff = cast(abs(level1[i] - level2[i]), 'int')

    if level1[i] - level2[i] > 0:
        i1 = sfc1[i]
        i2 = sfc2[i] << dimension * level_diff
        level = level1[i]
    elif level1[i] - level2[i] < 0:
        i1 = sfc1[i] << dimension * level_diff
        i2 = sfc2[i]
        level = level2[i]
    else:
        i1 = sfc1[i]
        i2 = sfc2[i]
        level = level1[i]

    xor_id = i1 ^ i2

    if xor_id == 0:
        lca_sfc[i] = i1 >> dimension * level_diff
        lca_level[i] = level - level_diff
        temp_idx[i] = all_idx[i]
        return

    for j in range(level + 1, 0, -1):
        if xor_id > ((1 << (j - 1) * dimension) - 1):
            lca_sfc[i] = i1 >> dimension * j
            lca_level[i] = level - j
            temp_idx[i] = all_idx[i]
            return

    lca_sfc[i] = 0
    lca_level[i] = 0
    temp_idx[i] = all_idx[i]
    return


@annotate(int='i, max_level, dimension', gintp='sfc, level')
def sfc_same(i, sfc, level, max_level, dimension):
    """Converts the sfc to the same level as the max_level

    Args:
        i (int): iterator for Elementwise function
        sfc (int[]): Morton z-index of nodes
        level (int[]): levl of nodes
        max_level (int): finest level of the tree
        dimension (int): dimension of the problem
    """
    sfc[i] = ((sfc[i] + 1) << dimension * (max_level - level[i])) - 1


@annotate(int='i, max_level, dimension', gintp='sfc, level')
def sfc_real(i, sfc, level, max_level, dimension):
    """Converts the sfc to the real level

    Args:
        i (int): iterator for Elementwise function
        sfc (int[]): Morton z-index of nodes
        level (int[]): levl of nodes
        max_level (int): finest level of the tree
        dimension (int): dimension of the problem
    """
    sfc[i] = ((sfc[i] + 1) >> dimension * (max_level - level[i])) - 1


@annotate(i='int', gintp='sfc, level, dp_idx')
def id_duplicates(i, sfc, level, dp_idx):
    """Identifies the duplicate nodes

    Args:
        i (int): iterator for Elementwise function
        sfc (int[]): Morton z-index of nodes
        level (int[]): levl of nodes
        dp_idx (int[]): mark duplicates nodes
    """
    if i == 0:
        dp_idx[i] = 0

    if sfc[i] == sfc[i + 1] and level[i] == level[i + 1]:
        dp_idx[i + 1] = 1


@annotate(i='int', gintp='dp_idx, sfc, level')
def remove_duplicates(i, dp_idx, sfc, level):
    """Removes the duplicate nodes

    Args:
        i (int): iterator for Elementwise function
        dp_idx (int[]): marked duplicates nodes
        sfc (int[]): Morton z-index of nodes
        level (int[]): levl of nodes
    """
    if dp_idx[i] == 1:
        sfc[i] = -1
        level[i] = -1


@annotate(i='int', gintp='pc_sfc, pc_level, temp_idx, rel_idx, parent_idx, '
          'child_idx')
def get_rel(i, pc_sfc, pc_level, temp_idx, rel_idx, parent_idx, child_idx):
    """Finds the relative index of the child of the current parent node

    Args:
        i (int): iterator for Elementwise function
        pc_sfc (int[]): Morton z-index of all nodes including generated copies
        pc_level (int[]): level of all nodes including generated copies
        temp_idx (int[]): [description]
        rel_idx (int[]): [description]
        parent_idx (int[]): parent relative index of all nodes
        child_idx (int[]): child relative index of all nodes
    """
    j = declare('int')

    if pc_sfc[i] == -1 or temp_idx[i] != -1 or i == 0:
        return

    for j in range(8):
        if (pc_sfc[i] != pc_sfc[i - j - 1] or
            pc_level[i] != pc_level[i - j - 1] or
                temp_idx[i - j - 1] == -1):
            return
        else:
            parent_idx[temp_idx[i - j - 1]] = rel_idx[i]
            child_idx[8 * rel_idx[i] + j] = temp_idx[i - j - 1]


@annotate(i='int', gintp='level, idx, parent, level_diff, move_up')
def find_level_diff(i, level, idx, parent, level_diff, move_up):
    """Find the level difference between the current node and its parent

    Args:
        i (int): iterator for Elementwise function
        level (int[]): level of nodes
        idx (int[]): particle index of nodes
        parent (int[]): parent relative index of nodes
        level_diff (int[]): level difference between nodes and parent
        move_up (int[]): move up flag
    """
    if parent[i] != -1:
        level_diff[i] = level[i] - level[parent[i]] - 1

    if idx[i] != -1:
        move_up[i] = level_diff[i]
        level_diff[i] = 0


@annotate(gintp='level_diff, sfc, level, idx, parent, child, sfc_n, level_n, '
          'idx_n, parent_n, child_n, cumsum_diff, move_up', int='i, dimension')
def complete_tree(i, level_diff, cumsum_diff, sfc, level, idx, parent, child,
                  sfc_n, level_n, idx_n, parent_n, child_n, dimension,
                  move_up):
    """Add nodes between parent and child with a level difference of 2 or more
       and remove cells if childless nodes are found with a level difference

    Args:
        i (int): iterator for Elementwise function
        level_diff (int[]): level difference between nodes and parent
        cumsum_diff (int[]): cumulative level difference
        sfc (int[]): Morton z-index of nodes
        level (int[]): level of nodes
        idx (int[]): particle index of nodes
        parent (int[]): parent relative index of nodes
        child (int[]): child relative index of nodes
        sfc_n (int[]): Morton z-index of  new nodes
        level_n (int[]): level of new nodes
        idx_n (int[]): particle index of new nodes
        parent_n (int[]): parent relative index of new nodes
        child_n (int[]): child relative index of new nodes
        dimension (int): dimension of the problem
        move_up (int[]): move up flag
    """
    cid, j, k = declare('int', 3)

    cid = i + cumsum_diff[i]
    sfc_n[cid] = sfc[i]
    level_n[cid] = level[i]
    idx_n[cid] = idx[i]
    if parent[i] != -1:
        parent_n[cid] = parent[i] + cumsum_diff[parent[i]]
    else:
        parent_n[cid] = -1
    for j in range(8):
        if child[8 * i + j] != -1:
            child_n[8 * cid + j] = (child[8 * i + j] +
                                    cumsum_diff[child[8 * i + j] + 1])
        else:
            break

    if level_diff[i] == 0 and move_up[i] == 0:
        return
    elif move_up[i] != 0:
        sfc_n[cid] = sfc_n[cid] >> dimension * move_up[i]
        level_n[cid] = level_n[cid] - move_up[i]
        return
    else:
        for k in range(level_diff[i]):
            cid += 1
            sfc_n[cid] = sfc_n[cid - 1] >> dimension
            level_n[cid] = level_n[cid - 1] - 1
            idx_n[cid] = -1
            parent_n[cid] = parent_n[cid - 1]
            parent_n[cid - 1] = cid
            child_n[8 * cid] = cid - 1


@annotate(i='int', gintp='idx, bin_count, start_idx, part2bin, p2b_offset, '
                         'leaf_idx')
def p2bin(i, idx, bin_count, start_idx, part2bin, p2b_offset, leaf_idx):
    """Find the node relative index of each particle

    Args:
        i (int): iterator for Elementwise function
        idx (int[]): particle index of nodes
        bin_count (int[]): number of particles in each childless node
        start_idx (int[]): start index of each childless node
        part2bin (int[]): node relative index of each particle
        p2b_offset (int[]): offset of each particle
        leaf_idx (int[]): leaf index of each particle
    """
    n = declare('int')
    if idx[i] == -1:
        return
    else:
        for n in range(bin_count[idx[i]]):
            part2bin[leaf_idx[start_idx[idx[i]] + n]] = i
            p2b_offset[leaf_idx[start_idx[idx[i]] + n]] = n


@annotate(i='int', gintp='new_count, idx, bin_count')
def set_new_count(i, new_count, idx, bin_count):
    """Set the new count of each node

    Args:
        i (int): iterator for Elementwise function
        new_count (int[]): number of particles in each node
        idx (int[]): particle index of nodes
        bin_count (int[]): number of particles in each childless nodes
    """
    if idx[i] == -1:
        return
    else:
        new_count[i] = bin_count[idx[i]]


@annotate(i='int', gintp='new_count, idx, bin_count, level')
def find_num_part(i, new_count, idx, level, bin_count):
    """Sums up the number of particles in each node using children nodes

    Args:
        i (int): iterator for Elementwise function
        new_count (int[]): number of particles in each node after summing up
        idx (int[]): particle index of nodes
        level (int[]): level of nodes
        bin_count (int[]): number of particles in each childless nodes
    """
    j = declare('int')
    j = 1
    if i == 0:
        return
    elif level[i] < level[i - 1]:
        while level[i] < level[i - j] and i - j >= 0:
            if idx[i - j] != -1:
                new_count[i] += new_count[i - j]
            j += 1
    else:
        return


@annotate(int='i, n_max', gintp='new_count, level, idx, merge_idx, parent, '
          'child')
def merge_mark(i, idx, level, new_count, merge_idx, n_max, parent, child):
    """Mark nodes which are to be merged such that maximum number of particles
       in a node does not exceed n_max

    Args:
        i (int): iterator for Elementwise function
        idx (int[]): particle index of nodes
        level (int[]): level of nodes
        new_count (int[]): number of particles in each node after summing up
        merge_idx (int[]): mark nodes which needs to be merged
        n_max (int): maximum number of particles in a node
        parent (int[]): parent relative index of nodes
        child (int[]): children relative index of nodes
    """
    j, k, cid, pid, nid = declare('int', 5)
    if new_count[i] > n_max or idx[i] != -1:
        return
    elif new_count[i] <= n_max and idx[i] == -1:
        pid = parent[i]
        cid = i
        nid = 0
        if new_count[pid] > n_max:
            for j in range(8):
                if child[8 * pid + j] == cid:
                    if j != 0:
                        nid = child[8 * pid + j - 1]
                    else:
                        while idx[cid] == -1:
                            pid = cid
                            cid = child[8 * pid]
                        nid = cid - 1
                    for k in range(nid + 1, i):
                        merge_idx[k] = 1
                    return


@annotate(i='int', gintp='sfc_n, level_n, idx_n, parent_n, child_n, sfc, '
          'level, temp_idx, parent, child, cumsum_merge, merge_idx, '
          'temp_count, new_count')
def merge(i, sfc_n, level_n, idx_n, parent_n, child_n, sfc, level, temp_idx,
          parent, child, cumsum_merge, merge_idx, temp_count, new_count):
    """Merge nodes which are marked for merging

    Args:
        i (int): iterator for Elementwise function
        sfc_n (int[]): Morton z-index of new tree nodes
        level_n (int[]): level of new tree nodes
        idx_n (int[]): particle index of new tree nodes
        parent_n (int[]): parent relative index of new tree nodes
        child_n (int[]): children relative index of new tree nodes
        sfc (int[]): Morton z-index of nodes
        level (int[]): level of nodes
        temp_idx (int[]): marks whether a node is childless or not
        parent (int[]): parent relative index of nodes
        child (int[]): children relative index of nodes
        cumsum_merge (int[]): cumulative sum of nodes to be merged
        merge_idx (int[]): marked nodes which needs to be merged
        temp_count (int[]): number of particles in each new childless node
        new_count (int[]): number of particles in each node after summing up
    """
    n, j, chid = declare('int', 3)
    if merge_idx[i] == 0:
        n = i - cumsum_merge[i]
        sfc[n] = sfc_n[i]
        level[n] = level_n[i]
        parent[n] = parent_n[i] - cumsum_merge[parent_n[i]]

        if (i != 0 and merge_idx[i - 1] == 1) or idx_n[i] != -1:
            temp_idx[n] = 1
            temp_count[n] = new_count[i]
        elif idx_n[i] == -1:
            temp_idx[n] = 0
            for j in range(8):
                chid = child_n[8 * i + j]
                if chid != -1:
                    child[8 * n + j] = chid - cumsum_merge[chid]


@annotate(i='int', gintp='idx, temp_idx, cumsum_idx, bin_count, temp_count')
def correct_idx(i, idx, temp_idx, cumsum_idx, bin_count, temp_count):
    """Setup the particle index and bin count of childless nodes

    Args:
        i (int): iterator for Elementwise function
        idx (int[]): particle index of new nodes
        temp_idx (int[]): marked nodes which are childless
        cumsum_idx (int[]): cumulative sum of nodes which are childless
        bin_count (int[]): number of particles in each childless nodes
        temp_count (int[]): number of particles calculated in previous function
    """
    if temp_idx[i] == 0:
        idx[i] = -1
    else:
        idx[i] = cumsum_idx[i]
        bin_count[idx[i]] = temp_count[i]


@annotate(x='int', coeff='gintp', return_='int')
def deinterleave(x, coeff):
    """Deinterleave the Morton z-index

    Args:
        x (int): Morton z-index
        coeff (int[]): coefficients for deinterleaving

    Returns:
        int: deinterleaved index
    """
    x = x & coeff[0]
    x = (x | (x >> 2)) & coeff[1]
    x = (x | (x >> 4)) & coeff[2]
    x = (x | (x >> 8)) & coeff[3]
    x = (x | (x >> 16)) & coeff[4]
    return x


@annotate(i='int', gintp='sfc, level, coeff', gfloatp='cx, cy, cz',
          double='x_min, y_min, z_min, length')
def calc_center(i, sfc, level, cx, cy, cz, x_min, y_min, z_min, length, coeff):
    """Calculate the center of a node

    Args:
        i (int): iterator for Elementwise function
        sfc (int[]): Morton z-index of nodes
        level (int[]): level of nodes
        cx (float[]): center x-coordinate of nodes
        cy (float[]): center y-coordinate of nodes
        cz (float[]): center z-coordinate of nodes
        x_min (double): minimum x-coordinate of the domain
        y_min (double): minimum y-coordinate of the domain
        z_min (double): minimum z-coordinate of the domain
        length (double): length of the domain
        coeff (int[]): coefficients for deinterleaving
    """
    x, y, z = declare('int', 3)
    x = deinterleave(sfc[i], coeff)
    y = deinterleave(sfc[i] >> 1, coeff)
    z = deinterleave(sfc[i] >> 2, coeff)

    cx[i] = x_min + length * (x + 0.5) / (2.0 ** level[i])
    cy[i] = y_min + length * (y + 0.5) / (2.0 ** level[i])
    cz[i] = z_min + length * (z + 0.5) / (2.0 ** level[i])


@annotate(int='i, num_p2', gintp='level, index', double='length, out_r, in_r',
          gfloatp='cx, cy, cz, out_x, out_y, out_z, in_x, in_y, in_z, sph_pts')
def setting_p2(i, out_x, out_y, out_z, in_x, in_y, in_z, sph_pts, cx, cy, cz,
               out_r, in_r, length, level, num_p2, index):
    """Set up the pseudo-particles for all nodes

    Args:
        i (int): iterator for Elementwise function
        out_x (float[]): x-coordinate of the outer spheres
        out_y (float[]): y-coordinate of the outer spheres
        out_z (float[]): z-coordinate of the outer spheres
        in_x (float[]): x-coordinate of the inner spheres
        in_y (float[]): y-coordinate of the inner spheres
        in_z (float[]): z-coordinate of the inner spheres
        sph_pts (float[]): spherical t-design points for unit sphere
        cx (float[]): center x-coordinate of nodes
        cy (float[]): center y-coordinate of nodes
        cz (float[]): center z-coordinate of nodes
        out_r (double): radius of the outer spheres
        in_r (double): radius of the inner spheres
        length (double): length of the domain
        level (int[]): level of nodes
        num_p2 (int): number of pseudo-particles for each node
        index (int[]): mapping from tree relative index to childless-wise index
    """
    cid, sid = declare('int', 2)
    sz_cell = declare('double')
    cid = cast(floor(i * 1.0 / num_p2), 'int')
    cid = index[cid]
    sid = i % num_p2
    sz_cell = sqrt(3.0) * length / (2.0**(level[cid] + 1))
    out_x[i] = cx[cid] + out_r * sz_cell * sph_pts[3 * sid]
    out_y[i] = cy[cid] + out_r * sz_cell * sph_pts[3 * sid + 1]
    out_z[i] = cz[cid] + out_r * sz_cell * sph_pts[3 * sid + 2]
    in_x[i] = cx[cid] + in_r * sz_cell * sph_pts[3 * sid]
    in_y[i] = cy[cid] + in_r * sz_cell * sph_pts[3 * sid + 1]
    in_z[i] = cz[cid] + in_r * sz_cell * sph_pts[3 * sid + 2]


@annotate(int='i, max_depth', gintp='level, lev_n, idx')
def level_info(i, level, idx, lev_n, max_depth):
    """Find number of nodes at each level and childless nodes are counted in
       max_depth level

    Args:
        i (int): iterator for Elementwise function
        level (int[]): level of nodes
        idx (int[]): whether the node is childless or not
        lev_n (int[]): number of nodes at each level
        max_depth (int): maximum depth of the tree
    """
    if idx[i] == -1:
        _ = atomic_inc(lev_n[level[i]])
    else:
        _ = atomic_inc(lev_n[max_depth])


@annotate(i='int', gintp='level, lev_n')
def levwise_info(i, level, lev_n):
    """Find number of nodes at each level

    Args:
        i (int): iterator for Elementwise function
        level (int[]): level of nodes
        lev_n (int[]): number of nodes at each level
    """
    _ = atomic_inc(lev_n[level[i]])


def build(N, max_depth, n_max, part_x, part_y, part_z, x_min, y_min, z_min,
          out_r, in_r, length, num_p2, backend, dimension, sph_pts,
          deleave_coeff):
    """Build the tree

    Args:
        N (int): number of particles
        max_depth (int): maximum depth of the tree
        n_max (int): maximum number of particles in a node
        part_x (float[]): x-coordinate of particles
        part_y (float[]): y-coordinate of particles
        part_z (float[]): z-coordinate of particles
        x_min (double): minimum x-coordinate of the domain
        y_min (double): minimum y-coordinate of the domain
        z_min (double): minimum z-coordinate of the domain
        out_r (double): radius of the outer spheres
        in_r (double): radius of the inner spheres
        length (double): length of the domain
        num_p2 (int): number of pseudo-particles for each node
        backend (str): backend ['cython', 'opencl', 'cuda']
        dimension (int): dimension of the problem
        sph_pts (float[]): spherical t-design points for unit sphere
        deleave_coeff (int[]): coefficients for deinterleaving

    Returns:
        int: Total number of nodes
        int[]: Morton z-index of each node
        int[]: level of each node
        int[]: particle index of each node
        int[]: number of particles in each node
        int[]: start index of each node
        int[]: leaf index of each node
        int[]: parent relative index of each node
        int[]: children relative index of each node
        int[]: node relative index of each particle
        int[]: offset of each particle
        int[]: number of nodes in each level with childless cells at max_depth
        int[]: number of nodes in each level
        int[]: mapping from tree relative index to childless-wise index
        int[]: reverse-mapping from tree relative index to childless index
        int[]: mapping from tree relative index to level-wise index
        int[]: reverse-mapping from tree relative index to level-wise index
        float[]: x-coordinate of the center of each node
        float[]: y-coordinate of the center of each node
        float[]: z-coordinate of the center of each node
        float[]: x-coordinate of the outer spheres
        float[]: y-coordinate of the outer spheres
        float[]: z-coordinate of the outer spheres
        float[]: x-coordinate of the inner spheres
        float[]: y-coordinate of the inner spheres
        float[]: z-coordinate of the inner spheres
        float[]: charge of all pseudo-particles in the outer spheres
        float[]: approximate potential at the inner spheres
    """

    max_index = 2 ** max_depth

    # defining the arrays
    leaf_sfc = ary.zeros(N, dtype=np.int32, backend=backend)
    leaf_idx = ary.arange(0, N, 1, dtype=np.int32, backend=backend)
    bin_count = ary.ones(N, dtype=np.int32, backend=backend)
    bin_idx = ary.zeros(N, dtype=np.int32, backend=backend)
    # start_idx = ary.zeros(N, dtype=np.int32, backend=backend)

    # different functions start from here
    eget_part_index = Elementwise(get_part_index, backend=backend)
    esingle_node = Elementwise(single_node, backend=backend)
    cumsum = Scan(input_expr, output_expr, 'a+b',
                  dtype=np.int32, backend=backend)
    reduction = Reduction('a+b', map_func=map_sum, dtype_out=np.int32,
                          backend=backend)

    copy2 = CopyArrays('copy2', [
        'a1', 'a2',
        'b1', 'b2']).function
    copy4 = CopyArrays('copy4', [
        'a1', 'a2', 'a3', 'a4',
        'b1', 'b2', 'b3', 'b4']).function
    copy5 = CopyArrays('copy5', [
        'a1', 'a2', 'a3', 'a4', 'a5',
        'b1', 'b2', 'b3', 'b4', 'b5']).function

    ecopy2 = Elementwise(copy2, backend=backend)
    ecopy4 = Elementwise(copy4, backend=backend)
    ecopy5 = Elementwise(copy5, backend=backend)

    einter_nodes = Elementwise(inter_nodes, backend=backend)

    reverse2 = ReverseArrays('reverse2', [
        'a1', 'a2',
        'b1', 'b2']).function
    reverse3 = ReverseArrays('reverse3', [
        'a1', 'a2', 'a3',
        'b1', 'b2', 'b3']).function
    reverse5 = ReverseArrays('reverse5', [
        'a1', 'a2', 'a3', 'a4', 'a5',
        'b1', 'b2', 'b3', 'b4', 'b5']).function

    ereverse2 = Elementwise(reverse2, backend=backend)
    ereverse3 = Elementwise(reverse3, backend=backend)
    ereverse5 = Elementwise(reverse5, backend=backend)

    esfc_same = Elementwise(sfc_same, backend=backend)
    esfc_real = Elementwise(sfc_real, backend=backend)

    eid_duplicates = Elementwise(id_duplicates, backend=backend)
    eremove_duplicates = Elementwise(remove_duplicates, backend=backend)

    efind_parents = Elementwise(find_parents, backend=backend)
    eget_rel = Elementwise(get_rel, backend=backend)

    efind_level_diff = Elementwise(find_level_diff, backend=backend)
    ecomplete_tree = Elementwise(complete_tree, backend=backend)

    ep2bin = Elementwise(p2bin, backend=backend)
    ecalc_center = Elementwise(calc_center, backend=backend)
    esetting_p2 = Elementwise(setting_p2, backend=backend)
    elev_info = Elementwise(level_info, backend=backend)
    elevwise_info = Elementwise(levwise_info, backend=backend)

    # calculations
    eget_part_index(leaf_sfc, part_x, part_y, part_z, max_index, length, x_min,
                    y_min, z_min)

    [leaf_sfc_sorted, leaf_idx_sorted], _ = radix_sort(
        [leaf_sfc, leaf_idx], backend=backend)

    esingle_node(leaf_sfc_sorted[:-1], leaf_sfc_sorted[1:], bin_count, bin_idx)

    [bin_idx_sorted, bin_count_sorted, leaf_sfc], _ = radix_sort(
        [bin_idx, bin_count, leaf_sfc_sorted], backend=backend)

    # cumsum(in_arr=bin_count_sorted, out_arr=start_idx)
    repeated = reduction(bin_idx)
    M = N - repeated

    leaf_sfc.resize(M)
    # start_idx.resize(M)
    bin_count = bin_count_sorted[:M]
    leaf_idx = leaf_idx_sorted[:]

    leaf_sfc_sorted.resize(0)
    leaf_idx_sorted.resize(0)
    bin_count_sorted.resize(0)
    bin_idx.resize(0)
    bin_idx_sorted.resize(0)

    # setting up the arrays
    leaf_idx_pointer = ary.arange(0, M, 1, dtype=np.int32, backend=backend)
    leaf_nds_idx = ary.arange(0, 2 * M - 1, 1, dtype=np.int32, backend=backend)
    leaf_level = ary.empty(M, dtype=np.int32, backend=backend)
    leaf_level.fill(max_depth)

    nodes_sfc = ary.zeros(M - 1, dtype=np.int32, backend=backend)
    nodes_level = ary.zeros(M - 1, dtype=np.int32, backend=backend)

    dp_idx = ary.zeros(M - 1, dtype=np.int32, backend=backend)

    sfc = ary.zeros(2 * M - 1, dtype=np.int32, backend=backend)
    level = ary.zeros(2 * M - 1, dtype=np.int32, backend=backend)
    idx = ary.empty(2 * M - 1, dtype=np.int32, backend=backend)
    idx.fill(-1)

    pc_sfc = ary.empty(4 * M - 2, dtype=np.int32, backend=backend)
    pc_level = ary.empty(4 * M - 2, dtype=np.int32, backend=backend)
    pc_idx = ary.empty(4 * M - 2, dtype=np.int32, backend=backend)
    temp_idx = ary.empty(4 * M - 2, dtype=np.int32, backend=backend)
    parent = ary.empty(2 * M - 1, dtype=np.int32, backend=backend)
    child = ary.empty(8 * (2 * M - 1), dtype=np.int32, backend=backend)
    rel_idx = ary.empty(4 * M - 2, dtype=np.int32, backend=backend)
    pc_sfc.fill(-1)
    pc_level.fill(-1)
    pc_idx.fill(-1)
    temp_idx.fill(-1)
    parent.fill(-1)
    child.fill(-1)
    rel_idx.fill(-1)

    pc_sfc_s = ary.zeros(4 * M - 2, dtype=np.int32, backend=backend)
    pc_level_s = ary.zeros(4 * M - 2, dtype=np.int32, backend=backend)
    pc_idx_s = ary.zeros(4 * M - 2, dtype=np.int32, backend=backend)
    rel_idx_s = ary.zeros(4 * M - 2, dtype=np.int32, backend=backend)
    temp_idx_s = ary.zeros(4 * M - 2, dtype=np.int32, backend=backend)

    level_diff = ary.zeros(2 * M - 1, dtype=np.int32, backend=backend)
    cumsum_diff = ary.zeros(2 * M - 1, dtype=np.int32, backend=backend)
    move_up = ary.zeros(2 * M - 1, dtype=np.int32, backend=backend)

    einter_nodes(leaf_sfc[:-1], leaf_sfc[1:], leaf_level[:-1], leaf_level[1:],
                 nodes_sfc, nodes_level, dimension)

    [nodes_level_sorted, nodes_sfc_sorted], _ = radix_sort(
        [nodes_level, nodes_sfc], backend=backend)

    ereverse2(nodes_level, nodes_sfc, nodes_level_sorted,
              nodes_sfc_sorted, M - 1)

    esfc_same(nodes_sfc, nodes_level, max_depth, dimension)

    [nodes_sfc_sorted, nodes_level_sorted], _ = radix_sort(
        [nodes_sfc, nodes_level], backend=backend)

    ecopy2(nodes_sfc, nodes_level, nodes_sfc_sorted, nodes_level_sorted)

    eid_duplicates(nodes_sfc[:-1], nodes_level[:-1], dp_idx)

    eremove_duplicates(dp_idx, nodes_sfc, nodes_level)

    [dp_idx_sorted, nodes_sfc_sorted, nodes_level_sorted], _ = radix_sort(
        [dp_idx, nodes_sfc, nodes_level], backend=backend)

    ecopy2(nodes_sfc, nodes_level, nodes_sfc_sorted, nodes_level_sorted)

    node_repeated = reduction(dp_idx_sorted)
    cells = 2 * M - 1 - node_repeated

    ecopy5(sfc[:M], sfc[M:], level[:M], level[M:], idx[:M], leaf_sfc,
           nodes_sfc, leaf_level, nodes_level, leaf_idx_pointer)

    [sfc_s, level_s, idx_s], _ = radix_sort([sfc, level, idx], backend=backend)

    esfc_real(sfc_s, level_s, max_depth, dimension)

    sfc_s.resize(cells)
    level_s.resize(cells)
    idx_s.resize(cells)
    parent.resize(cells)
    child.resize(8 * cells)
    level_diff.resize(cells)
    cumsum_diff.resize(cells)
    move_up.resize(cells)

    ecopy4(pc_sfc[:cells], pc_level[:cells], pc_idx[:cells], rel_idx[:cells],
           sfc_s, level_s, idx_s, leaf_nds_idx)

    efind_parents(sfc_s[:-1], sfc_s[1:], level_s[:-1], level_s[1:],
                  leaf_nds_idx[:-1], pc_sfc[2 * M - 1:], pc_level[2 * M - 1:],
                  temp_idx[2 * M - 1:], dimension)

    [pc_level_s, pc_sfc_s, pc_idx_s, rel_idx_s, temp_idx_s], _ = radix_sort(
        [pc_level, pc_sfc, pc_idx, rel_idx, temp_idx], backend=backend)

    ereverse5(pc_level, pc_sfc, pc_idx, rel_idx, temp_idx, pc_level_s,
              pc_sfc_s, pc_idx_s, rel_idx_s, temp_idx_s, 4 * M - 2)

    esfc_same(pc_sfc[2 * node_repeated + 1:], pc_level[2 * node_repeated + 1:],
              max_depth, dimension)

    [pc_sfc_s, pc_level_s, pc_idx_s, rel_idx_s, temp_idx_s], _ = radix_sort(
        [pc_sfc, pc_level, pc_idx, rel_idx, temp_idx], backend=backend)

    esfc_real(pc_sfc_s[:-(2 * node_repeated + 1)],
              pc_level_s[:-(2 * node_repeated + 1)], max_depth, dimension)

    eget_rel(pc_sfc_s, pc_level_s, temp_idx_s, rel_idx_s, parent, child)

    efind_level_diff(level_s, idx_s, parent, level_diff, move_up)
    cumsum(in_arr=level_diff, out_arr=cumsum_diff)
    add_cells = reduction(level_diff)

    total_cells = cells + add_cells

    sfc_n = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    level_n = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    idx_n = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    parent_n = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    child_n = ary.empty(8 * total_cells, dtype=np.int32, backend=backend)
    child_n.fill(-1)

    ecomplete_tree(level_diff, cumsum_diff, sfc_s, level_s, idx_s, parent,
                   child, sfc_n, level_n, idx_n, parent_n, child_n,
                   dimension, move_up)

    new_count = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    merge_idx = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    cumsum_merge = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    eset_new_count = Elementwise(set_new_count, backend=backend)
    efind_num_part = Elementwise(find_num_part, backend=backend)
    emerge_mark = Elementwise(merge_mark, backend=backend)
    emerge = Elementwise(merge, backend=backend)
    ecorrect_idx = Elementwise(correct_idx, backend=backend)
    eset_new_count(new_count, idx_n, bin_count)
    efind_num_part(new_count, idx_n, level_n, bin_count)
    emerge_mark(idx_n, level_n, new_count, merge_idx, n_max, parent_n, child_n)
    remove_cells = reduction(merge_idx)
    cumsum(in_arr=merge_idx, out_arr=cumsum_merge)
    total_cells = total_cells - remove_cells
    sfc_a = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    level_a = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    idx_a = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    temp_idx = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    temp_count = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    cumsum_idx = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    parent_a = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    child_a = ary.empty(8 * total_cells, dtype=np.int32, backend=backend)
    child_a.fill(-1)
    emerge(sfc_n, level_n, idx_n, parent_n, child_n, sfc_a, level_a, temp_idx,
           parent_a, child_a, cumsum_merge, merge_idx, temp_count, new_count)
    cumsum(in_arr=temp_idx, out_arr=cumsum_idx)
    ecorrect_idx(idx_a, temp_idx, cumsum_idx, bin_count, temp_count)

    size_bin = cumsum_idx[-1]

    bin_count = bin_count[:size_bin]
    start_idx = ary.zeros(size_bin, dtype=np.int32, backend=backend)
    cumsum(in_arr=bin_count, out_arr=start_idx)

    leaf_sfc.resize(0)
    leaf_idx_pointer.resize(0)
    leaf_level.resize(0)
    nodes_sfc.resize(0)
    nodes_level.resize(0)
    dp_idx.resize(0)
    leaf_nds_idx.resize(0)
    pc_sfc.resize(0)
    pc_sfc_s.resize(0)
    pc_level.resize(0)
    pc_level_s.resize(0)
    pc_idx.resize(0)
    pc_idx_s.resize(0)
    temp_idx.resize(0)
    temp_idx_s.resize(0)
    rel_idx.resize(0)
    rel_idx_s.resize(0)
    level_diff.resize(0)
    cumsum_diff.resize(0)
    move_up.resize(0)

    index = ary.arange(0, total_cells, 1, dtype=np.int32, backend=backend)

    s1_index = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    s1r_index = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    s2_index = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    s1_lev = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    s2_lev = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    s1_idx = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    s2_idx = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    lev_index = ary.zeros(total_cells, dtype=np.int32, backend=backend)
    lev_index_r = ary.zeros(total_cells, dtype=np.int32, backend=backend)

    lev_n = ary.zeros(max_depth + 1, dtype=np.int32, backend=backend)
    levwise_n = ary.zeros(max_depth + 1, dtype=np.int32, backend=backend)
    lev_nr = ary.zeros(max_depth + 1, dtype=np.int32, backend=backend)
    levwise_nr = ary.zeros(max_depth + 1, dtype=np.int32, backend=backend)
    lev_cs = ary.zeros(max_depth + 1, dtype=np.int32, backend=backend)
    levwise_cs = ary.zeros(max_depth + 1, dtype=np.int32, backend=backend)

    cx = ary.zeros(total_cells, dtype=np.float32, backend=backend)
    cy = ary.zeros(total_cells, dtype=np.float32, backend=backend)
    cz = ary.zeros(total_cells, dtype=np.float32, backend=backend)

    out_x = ary.zeros(total_cells * num_p2, dtype=np.float32, backend=backend)
    out_y = ary.zeros(total_cells * num_p2, dtype=np.float32, backend=backend)
    out_z = ary.zeros(total_cells * num_p2, dtype=np.float32, backend=backend)
    out_vl = ary.zeros(total_cells * num_p2, dtype=np.float32, backend=backend)
    in_x = ary.zeros(total_cells * num_p2, dtype=np.float32, backend=backend)
    in_y = ary.zeros(total_cells * num_p2, dtype=np.float32, backend=backend)
    in_z = ary.zeros(total_cells * num_p2, dtype=np.float32, backend=backend)
    in_vl = ary.zeros(total_cells * num_p2, dtype=np.float32, backend=backend)

    part2bin = ary.zeros(N, dtype=np.int32, backend=backend)
    p2b_offset = ary.zeros(N, dtype=np.int32, backend=backend)

    ep2bin(idx_a, bin_count, start_idx, part2bin, p2b_offset, leaf_idx)

    ecalc_center(sfc_a, level_a, cx, cy, cz, x_min, y_min, z_min, length,
                 deleave_coeff)

    [s1_lev, s1_idx, s1_index], _ = radix_sort([level_a, idx_a, index],
                                               backend=backend)
    ereverse3(s2_idx, s2_lev, s2_index, s1_idx, s1_lev, s1_index, total_cells)

    lev_index = s2_index[:]
    [_, lev_index_r], _ = radix_sort([lev_index, index], backend=backend)

    [s1_idx, s1_lev, s1_index], _ = radix_sort([s2_idx, s2_lev, s2_index],
                                               backend=backend)

    [s2_index, s1r_index], _ = radix_sort([s1_index, index], backend=backend)

    s2_idx.resize(0)
    s2_lev.resize(0)
    s2_index.resize(0)

    s1_lev.resize(0)
    s1_idx.resize(0)
    index.resize(0)

    elev_info(level_a, idx_a, lev_n, max_depth)
    elevwise_info(level_a, levwise_n)
    ereverse2(lev_nr, levwise_nr, lev_n, levwise_n, max_depth + 1)
    cumsum(in_arr=lev_nr, out_arr=lev_cs)
    cumsum(in_arr=levwise_nr, out_arr=levwise_cs)
    ereverse2(lev_n, levwise_n, lev_cs, levwise_cs, max_depth + 1)

    esetting_p2(out_x, out_y, out_z, in_x, in_y, in_z, sph_pts, cx, cy, cz,
                out_r, in_r, length, level_a, num_p2, s1_index)

    lev_nr.resize(0)
    levwise_nr.resize(0)

    return (total_cells, sfc_a, level_a, idx_a, bin_count, start_idx, leaf_idx,
            parent_a, child_a, part2bin, p2b_offset, lev_n, levwise_n,
            s1_index, s1r_index, lev_index, lev_index_r, cx, cy, cz, out_x,
            out_y, out_z, in_x, in_y, in_z, out_vl, in_vl)
