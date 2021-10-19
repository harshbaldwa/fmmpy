# TODO: resize all the waste arrays to 0
# TODO: rearrange all the function variables
# LATER: resize if certain cell is filled with particles
# LATER: add legendre polynomial list as physical file

import importlib.resources
import time
from math import floor, sqrt

import compyle.array as ary
import numpy as np
import yaml
from compyle.api import (Elementwise, Reduction, Scan, annotate, declare,
                         get_config, wrap)
from compyle.low_level import atomic_inc, cast
from compyle.sort import radix_sort
from compyle.template import Template

np.set_printoptions(linewidth=np.inf)


# TODO: Make dimension independent
@annotate(i="int", index="gintp", gfloatp="x, y, z",
          double="max_index, length, x_min, y_min, z_min")
def get_particle_index(i, index, x, y, z, max_index,
                       length, x_min, y_min, z_min):
    nx, ny, nz = declare("int", 3)
    nx = cast(floor((max_index * (x[i] - x_min)) / length), "int")
    ny = cast(floor((max_index * (y[i] - y_min)) / length), "int")
    nz = cast(floor((max_index * (z[i] - z_min)) / length), "int")

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
    def __init__(self, name, arrays):
        super(CopyArrays, self).__init__(name=name)
        self.arrays = arrays
        self.number = len(arrays)

    def extra_args(self):
        return self.arrays, {"intp": ','.join(self.arrays)}

    @annotate(i='int')
    def template(self, i):
        '''
        % for t in range(obj.number//2):
        ${obj.arrays[t]}[i] = ${obj.arrays[obj.number//2+t]}[i]
        % endfor
        '''


class ReverseArrays(Template):
    def __init__(self, name, arrays):
        super(ReverseArrays, self).__init__(name=name)
        self.arrays = arrays
        self.number = len(arrays)

    def extra_args(self):
        return self.arrays + ["length"], {"intp": ','.join(self.arrays),
                                          "length": "int"}

    @annotate(i='int')
    def template(self, i):
        '''
        % for t in range(obj.number//2):
        ${obj.arrays[t]}[length-i-1] = ${obj.arrays[obj.number//2+t]}[i]
        % endfor
        '''


@annotate(i="int", gintp="leaf_sfc, leaf_sfc_a, bin_count, bin_idx")
def single_node(i, leaf_sfc, leaf_sfc_a, bin_count, bin_idx):
    j = declare("int")
    if leaf_sfc[i] != leaf_sfc_a[i]:
        return
    elif i != 0 and leaf_sfc[i-1] == leaf_sfc_a[i]:
        return
    else:
        j = 1
        while leaf_sfc[i] == leaf_sfc[i+j]:
            bin_count[i] += 1
            bin_idx[i+j] = 1
            j += 1


@annotate(i="int", x="gintp")
def map_sum(i, x):
    return x[i]


@annotate(i="int", in_arr="gintp", return_="int")
def input_expr(i, in_arr):
    if i == 0:
        return 0
    else:
        return in_arr[i - 1]


@annotate(int="i, item", out_arr="gintp")
def output_expr(i, item, out_arr):
    out_arr[i] = item


@annotate(gintp="sfc1, sfc2, level1, level2, lca_sfc, lca_level",
          int="i, dimension")
def internal_nodes(i, sfc1, sfc2, level1, level2, lca_sfc, lca_level,
                   dimension):
    level_diff, xor_id, i1, i2, level, j = declare("int", 6)
    level_diff = cast(abs(level1[i] - level2[i]), "int")

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


@annotate(gintp="sfc1, sfc2, level1, level2, lca_sfc, "
                "all_idx, lca_level, temp_idx", int="i, dimension")
def find_parents(i, sfc1, sfc2, level1, level2, all_idx, lca_sfc,
                 lca_level, temp_idx, dimension):
    level_diff, xor_id, i1, i2, level, j = declare("int", 6)
    level_diff = cast(abs(level1[i] - level2[i]), "int")

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


@annotate(int="i, max_level, dimension", gintp="sfc, level")
def sfc_same(i, sfc, level, max_level, dimension):
    sfc[i] = ((sfc[i] + 1) << dimension * (max_level - level[i])) - 1


@annotate(int="i, max_level, dimension", gintp="sfc, level")
def sfc_real(i, sfc, level, max_level, dimension):
    sfc[i] = ((sfc[i] + 1) >> dimension * (max_level - level[i])) - 1


@annotate(i="int", gintp="sfc, level, dp_idx")
def id_duplicates(i, sfc, level, dp_idx):
    if i == 0:
        dp_idx[i] = 0

    if sfc[i] == sfc[i+1] and level[i] == level[i+1]:
        dp_idx[i+1] = 1


@annotate(i="int", gintp="dp_idx, sfc, level")
def remove_duplicates(i, dp_idx, sfc, level):
    if dp_idx[i] == 1:
        sfc[i] = -1
        level[i] = -1


@annotate(i="int", gintp="pc_sfc, pc_level, temp_idx, rel_idx, "
                         "parent_idx, child_idx")
def get_relations(i, pc_sfc, pc_level, temp_idx, rel_idx,
                  parent_idx, child_idx):
    j = declare("int")

    if pc_sfc[i] == -1 or temp_idx[i] != -1 or i == 0:
        return

    for j in range(8):
        if (pc_sfc[i] != pc_sfc[i-j-1] or
            pc_level[i] != pc_level[i-j-1] or
                temp_idx[i-j-1] == -1):
            return
        else:
            parent_idx[temp_idx[i-j-1]] = rel_idx[i]
            child_idx[8*rel_idx[i] + j] = temp_idx[i-j-1]


@annotate(i="int", gintp="level, parent_idx, level_diff")
def find_level_diff(i, level, parent_idx, level_diff):
    if parent_idx[i] != -1:
        level_diff[i] = level[i] - level[parent_idx[i]] - 1


@annotate(int="i, dimension", gintp="level_diff, sfc, level")
def complete_tree(i, level_diff, sfc, level, dimension):
    if level_diff[i] < 1:
        return
    else:
        sfc[i] = sfc[i] >> dimension * level_diff[i]
        level[i] = level[i] - level_diff[i]


@annotate(i="int", gintp="idx, bin_count, start_idx, part2bin, p2b_offset, "
                         "leaf_idx")
def p2bin(i, idx, bin_count, start_idx, part2bin, p2b_offset, leaf_idx):
    n = declare("int")
    if idx[i] == -1:
        return
    else:
        for n in range(bin_count[idx[i]]):
            part2bin[leaf_idx[start_idx[idx[i]] + n]] = i
            p2b_offset[leaf_idx[start_idx[idx[i]] + n]] = n


# TODO: make dimension a parameter
@annotate(x="int", return_="int")
def deinterleave(x):
    x = x & 0x49249249
    x = (x | (x >> 2)) & 0xC30C30C3
    x = (x | (x >> 4)) & 0xF00F00F
    x = (x | (x >> 8)) & 0xFF0000FF
    x = (x | (x >> 16)) & 0x0000FFFF
    return x


@annotate(i="int", gintp="sfc, level", gfloatp="cx, cy, cz",
          double="x_min, y_min, z_min, length")
def calc_center(i, sfc, level, cx, cy, cz,
                x_min, y_min, z_min, length):
    x, y, z = declare("int", 3)
    x = deinterleave(sfc[i])
    y = deinterleave(sfc[i] >> 1)
    z = deinterleave(sfc[i] >> 2)

    cx[i] = x_min + length*(x + 0.5)/(2.0 ** level[i])
    cy[i] = y_min + length*(y + 0.5)/(2.0 ** level[i])
    cz[i] = z_min + length*(z + 0.5)/(2.0 ** level[i])


@annotate(int="i, num_p2", gintp="level, index", double="length, out_r, in_r",
          gfloatp="cx, cy, cz, out_x, out_y, out_z, in_x, in_y, in_z, "
                  "sph_pts")
def setting_p2(i, out_x, out_y, out_z, in_x, in_y, in_z, sph_pts, cx, cy, cz,
               out_r, in_r, length, level, num_p2, index):
    cid, sid = declare("int", 2)
    sz_cell = declare("double")
    cid = cast(floor(i*1.0/num_p2), "int")
    cid = index[cid]
    sid = i % num_p2
    sz_cell = sqrt(3.0)*length/(2.0**(level[cid]+1))
    out_x[i] = cx[cid] + out_r*sz_cell*sph_pts[3*sid]
    out_y[i] = cy[cid] + out_r*sz_cell*sph_pts[3*sid+1]
    out_z[i] = cz[cid] + out_r*sz_cell*sph_pts[3*sid+2]
    in_x[i] = cx[cid] + in_r*sz_cell*sph_pts[3*sid]
    in_y[i] = cy[cid] + in_r*sz_cell*sph_pts[3*sid+1]
    in_z[i] = cz[cid] + in_r*sz_cell*sph_pts[3*sid+2]


@annotate(int="i, max_depth", gintp="level, lev_n, idx")
def level_info(i, level, idx, lev_n, max_depth):
    ix = declare("int")
    if idx[i] == -1:
        ix = atomic_inc(lev_n[level[i]])
    else:
        ix = atomic_inc(lev_n[max_depth])


@annotate(i="int", gintp="level, lev_n")
def levwise_info(i, level, lev_n):
    ix = declare("int")
    ix = atomic_inc(lev_n[level[i]])


def build(N, max_depth, part_val, part_x, part_y, part_z, x_min, y_min, z_min,
          out_r, in_r, length, num_p2, backend, dimension):

    max_index = 2 ** max_depth

    # defining the arrays
    leaf_sfc = ary.zeros(N, dtype=np.int32, backend=backend)
    leaf_idx = ary.arange(0, N, 1, dtype=np.int32, backend=backend)
    bin_count = ary.ones(N, dtype=np.int32, backend=backend)
    bin_idx = ary.zeros(N, dtype=np.int32, backend=backend)
    start_idx = ary.zeros(N, dtype=np.int32, backend=backend)

    # with importlib.resources.open_text("fmm", "t_design.yaml") as file:
    #     data = yaml.load(file)[num_p2]
    data = yaml.load(open("t_design.yaml"), Loader=yaml.FullLoader)[num_p2]
    sph_pts = np.array(data['array'], dtype=np.float32)
    order = data['order']
    sph_pts = wrap(sph_pts, backend=backend)

    # different functions start from here
    eget_particle_index = Elementwise(get_particle_index, backend=backend)
    esingle_node = Elementwise(single_node, backend=backend)
    cumsum = Scan(input_expr, output_expr, 'a+b',
                  dtype=np.int32, backend=backend)
    reduction = Reduction('a+b', map_func=map_sum, dtype_out=np.int32,
                          backend=backend)

    copy2 = CopyArrays('copy2', [
        'a1', 'a2',
        'b1', 'b2']).function
    copy3 = CopyArrays('copy3', [
        'a1', 'a2', 'a3',
        'b1', 'b2', 'b3']).function
    copy4 = CopyArrays('copy4', [
        'a1', 'a2', 'a3', 'a4',
        'b1', 'b2', 'b3', 'b4']).function
    copy5 = CopyArrays('copy5', [
        'a1', 'a2', 'a3', 'a4', 'a5',
        'b1', 'b2', 'b3', 'b4', 'b5']).function
    copy6 = CopyArrays('copy6', [
        'a1', 'a2', 'a3', 'a4', 'a5', 'a6',
        'b1', 'b2', 'b3', 'b4', 'b5', 'b6']).function

    ecopy2 = Elementwise(copy2, backend=backend)
    ecopy3 = Elementwise(copy3, backend=backend)
    ecopy4 = Elementwise(copy4, backend=backend)
    ecopy5 = Elementwise(copy5, backend=backend)
    ecopy6 = Elementwise(copy6, backend=backend)

    einternal_nodes = Elementwise(internal_nodes, backend=backend)

    reverse1 = ReverseArrays('reverse1', ['a', 'b']).function
    reverse2 = ReverseArrays('reverse2', [
        'a1', 'a2',
        'b1', 'b2']).function
    reverse3 = ReverseArrays('reverse3', [
        'a1', 'a2', 'a3',
        'b1', 'b2', 'b3']).function
    reverse5 = ReverseArrays('reverse5', [
        'a1', 'a2', 'a3', 'a4', 'a5',
        'b1', 'b2', 'b3', 'b4', 'b5']).function

    ereverse1 = Elementwise(reverse1, backend=backend)
    ereverse2 = Elementwise(reverse2, backend=backend)
    ereverse3 = Elementwise(reverse3, backend=backend)
    ereverse5 = Elementwise(reverse5, backend=backend)

    esfc_same = Elementwise(sfc_same, backend=backend)
    esfc_real = Elementwise(sfc_real, backend=backend)

    eid_duplicates = Elementwise(id_duplicates, backend=backend)
    eremove_duplicates = Elementwise(remove_duplicates, backend=backend)

    efind_parents = Elementwise(find_parents, backend=backend)
    eget_relations = Elementwise(get_relations, backend=backend)

    efind_level_diff = Elementwise(find_level_diff, backend=backend)
    ecomplete_tree = Elementwise(complete_tree, backend=backend)

    ep2bin = Elementwise(p2bin, backend=backend)
    ecalc_center = Elementwise(calc_center, backend=backend)
    esetting_p2 = Elementwise(setting_p2, backend=backend)
    elev_info = Elementwise(level_info, backend=backend)
    elevwise_info = Elementwise(levwise_info, backend=backend)

    # calculations
    tree_start = time.time()

    eget_particle_index(leaf_sfc, part_x, part_y,
                        part_z, max_index, length,
                        x_min, y_min, z_min)

    [leaf_sfc_sorted, leaf_idx_sorted], _ = radix_sort(
        [leaf_sfc, leaf_idx], backend=backend)

    esingle_node(leaf_sfc_sorted[:-1], leaf_sfc_sorted[1:], bin_count, bin_idx)

    [bin_idx_sorted, bin_count_sorted, leaf_sfc], _ = radix_sort(
        [bin_idx, bin_count, leaf_sfc_sorted], backend=backend)

    cumsum(in_arr=bin_count_sorted, out_arr=start_idx)
    repeated = reduction(bin_idx)
    M = N - repeated

    leaf_sfc.resize(M)
    start_idx.resize(M)
    bin_count = bin_count_sorted[:M]
    leaf_idx = leaf_idx_sorted[:]

    leaf_sfc_sorted.resize(0)
    leaf_idx_sorted.resize(0)
    bin_count_sorted.resize(0)
    bin_idx.resize(0)
    bin_idx_sorted.resize(0)

    # setting up the arrays
    leaf_idx_pointer = ary.arange(0, M, 1, dtype=np.int32, backend=backend)
    leaf_nodes_idx = ary.arange(0, 2*M-1, 1, dtype=np.int32, backend=backend)
    leaf_level = ary.empty(M, dtype=np.int32, backend=backend)
    leaf_level.fill(max_depth)

    nodes_sfc = ary.zeros(M-1, dtype=np.int32, backend=backend)
    nodes_level = ary.zeros(M-1, dtype=np.int32, backend=backend)

    dp_idx = ary.zeros(M-1, dtype=np.int32, backend=backend)

    sfc = ary.zeros(2*M-1, dtype=np.int32, backend=backend)
    level = ary.zeros(2*M-1, dtype=np.int32, backend=backend)
    idx = ary.empty(2*M-1, dtype=np.int32, backend=backend)
    idx.fill(-1)

    pc_sfc = ary.empty(4*M-2, dtype=np.int32, backend=backend)
    pc_level = ary.empty(4*M-2, dtype=np.int32, backend=backend)
    pc_idx = ary.empty(4*M-2, dtype=np.int32, backend=backend)
    temp_idx = ary.empty(4*M-2, dtype=np.int32, backend=backend)
    parent = ary.empty(2*M-1, dtype=np.int32, backend=backend)
    child = ary.empty(8*(2*M-1), dtype=np.int32, backend=backend)
    rel_idx = ary.empty(4*M-2, dtype=np.int32, backend=backend)
    pc_sfc.fill(-1)
    pc_level.fill(-1)
    pc_idx.fill(-1)
    temp_idx.fill(-1)
    parent.fill(-1)
    child.fill(-1)
    rel_idx.fill(-1)

    pc_sfc_s = ary.zeros(4*M-2, dtype=np.int32, backend=backend)
    pc_level_s = ary.zeros(4*M-2, dtype=np.int32, backend=backend)
    pc_idx_s = ary.zeros(4*M-2, dtype=np.int32, backend=backend)
    rel_idx_s = ary.zeros(4*M-2, dtype=np.int32, backend=backend)
    temp_idx_s = ary.zeros(4*M-2, dtype=np.int32, backend=backend)

    level_diff = ary.zeros(2*M-1, dtype=np.int32, backend=backend)
    cumsum_diff = ary.zeros(2*M-1, dtype=np.int32, backend=backend)

    einternal_nodes(leaf_sfc[:-1], leaf_sfc[1:], leaf_level[:-1],
                    leaf_level[1:], nodes_sfc, nodes_level, dimension)

    [nodes_level_sorted, nodes_sfc_sorted], _ = radix_sort(
        [nodes_level, nodes_sfc], backend=backend)

    ereverse2(nodes_level, nodes_sfc, nodes_level_sorted,
              nodes_sfc_sorted, M-1)

    esfc_same(nodes_sfc, nodes_level, max_depth, dimension)

    [nodes_sfc_sorted, nodes_level_sorted], _ = radix_sort(
        [nodes_sfc, nodes_level], backend=backend)

    ecopy2(nodes_sfc, nodes_level, nodes_sfc_sorted, nodes_level_sorted)

    eid_duplicates(nodes_sfc[:-1], nodes_level[:-1], dp_idx)

    eremove_duplicates(dp_idx, nodes_sfc, nodes_level)

    [dp_idx_sorted, nodes_sfc_sorted, nodes_level_sorted], _ = radix_sort(
        [dp_idx, nodes_sfc, nodes_level],
        backend=backend)

    ecopy2(nodes_sfc, nodes_level, nodes_sfc_sorted, nodes_level_sorted)

    node_repeated = reduction(dp_idx_sorted)
    cells = 2*M-1 - node_repeated

    ecopy5(sfc[:M], sfc[M:], level[:M], level[M:], idx[:M],
           leaf_sfc, nodes_sfc, leaf_level, nodes_level, leaf_idx_pointer)

    [sfc_s, level_s, idx_s], _ = radix_sort(
        [sfc, level, idx], backend=backend)

    esfc_real(sfc_s, level_s, max_depth, dimension)

    sfc_s.resize(cells)
    level_s.resize(cells)
    idx_s.resize(cells)
    leaf_nodes_idx.resize(cells)
    parent.resize(cells)
    child.resize(8*cells)
    level_diff.resize(cells)

    ecopy4(pc_sfc[:cells], pc_level[:cells], pc_idx[:cells], rel_idx[:cells],
           sfc_s, level_s, idx_s, leaf_nodes_idx)

    efind_parents(sfc_s[:-1], sfc_s[1:], level_s[:-1], level_s[1:],
                  leaf_nodes_idx[:-1], pc_sfc[2*M-1:], pc_level[2*M-1:],
                  temp_idx[2*M-1:], dimension)

    [pc_level_s, pc_sfc_s, pc_idx_s,
     rel_idx_s, temp_idx_s], _ = radix_sort(
         [pc_level, pc_sfc, pc_idx, rel_idx, temp_idx], backend=backend)

    ereverse5(pc_level, pc_sfc, pc_idx, rel_idx,
              temp_idx, pc_level_s, pc_sfc_s,
              pc_idx_s, rel_idx_s, temp_idx_s, 4*M-2)

    esfc_same(pc_sfc[2*node_repeated+1:], pc_level[2*node_repeated+1:],
              max_depth, dimension)

    [pc_sfc_s, pc_level_s, pc_idx_s,
     rel_idx_s, temp_idx_s], _ = radix_sort(
         [pc_sfc, pc_level, pc_idx, rel_idx, temp_idx], backend=backend)

    esfc_real(pc_sfc_s[:-(2*node_repeated+1)],
              pc_level_s[:-(2*node_repeated+1)],
              max_depth, dimension)

    eget_relations(pc_sfc_s, pc_level_s, temp_idx_s, rel_idx_s,
                   parent, child)

    efind_level_diff(level_s, parent, level_diff)

    ecomplete_tree(level_diff, sfc_s, level_s, dimension)

    index = ary.arange(0, cells, 1, dtype=np.int32, backend=backend)

    s1_index = ary.zeros(cells, dtype=np.int32, backend=backend)
    s1r_index = ary.zeros(cells, dtype=np.int32, backend=backend)
    s2_index = ary.zeros(cells, dtype=np.int32, backend=backend)
    s1_lev = ary.zeros(cells, dtype=np.int32, backend=backend)
    s2_lev = ary.zeros(cells, dtype=np.int32, backend=backend)
    s1_idx = ary.zeros(cells, dtype=np.int32, backend=backend)
    s2_idx = ary.zeros(cells, dtype=np.int32, backend=backend)
    lev_index = ary.zeros(cells, dtype=np.int32, backend=backend)
    lev_index_r = ary.zeros(cells, dtype=np.int32, backend=backend)

    lev_n = ary.zeros(max_depth+1, dtype=np.int32, backend=backend)
    levwise_n = ary.zeros(max_depth+1, dtype=np.int32, backend=backend)
    lev_nr = ary.zeros(max_depth+1, dtype=np.int32, backend=backend)
    levwise_nr = ary.zeros(max_depth+1, dtype=np.int32, backend=backend)
    lev_cs = ary.zeros(max_depth+1, dtype=np.int32, backend=backend)
    levwise_cs = ary.zeros(max_depth+1, dtype=np.int32, backend=backend)

    cx = ary.zeros(cells, dtype=np.float32, backend=backend)
    cy = ary.zeros(cells, dtype=np.float32, backend=backend)
    cz = ary.zeros(cells, dtype=np.float32, backend=backend)

    out_x = ary.zeros(cells*num_p2, dtype=np.float32, backend=backend)
    out_y = ary.zeros(cells*num_p2, dtype=np.float32, backend=backend)
    out_z = ary.zeros(cells*num_p2, dtype=np.float32, backend=backend)
    out_vl = ary.zeros(cells*num_p2, dtype=np.float32, backend=backend)
    in_x = ary.zeros(cells*num_p2, dtype=np.float32, backend=backend)
    in_y = ary.zeros(cells*num_p2, dtype=np.float32, backend=backend)
    in_z = ary.zeros(cells*num_p2, dtype=np.float32, backend=backend)
    in_vl = ary.zeros(cells*num_p2, dtype=np.float32, backend=backend)

    assoc = ary.zeros(cells*26, dtype=np.int32, backend=backend)
    part2bin = ary.zeros(N, dtype=np.int32, backend=backend)
    p2b_offset = ary.zeros(N, dtype=np.int32, backend=backend)

    ep2bin(idx_s, bin_count, start_idx, part2bin, p2b_offset, leaf_idx)

    ecalc_center(sfc_s, level_s, cx, cy, cz,
                 x_min, y_min, z_min, length)

    # FIXME: Remove unnecessary copies
    [s1_lev, s1_idx, s1_index], _ = radix_sort([level_s, idx_s, index],
                                               backend=backend)
    ereverse3(s2_idx, s2_lev, s2_index, s1_idx, s1_lev, s1_index, cells)

    lev_index = s2_index[:]
    [_, lev_index_r], _ = radix_sort([lev_index, index], backend=backend)

    [s1_idx, s1_lev, s1_index], _ = radix_sort([s2_idx, s2_lev, s2_index],
                                               backend=backend)

    [s2_index, s1r_index], _ = radix_sort([s1_index, index],
                                          backend=backend)

    s2_idx.resize(0)
    s2_lev.resize(0)
    s2_index.resize(0)

    s1_lev.resize(0)
    s1_idx.resize(0)
    index.resize(0)

    elev_info(level_s, idx_s, lev_n, max_depth)
    elevwise_info(level_s, levwise_n)
    ereverse2(lev_nr, levwise_nr, lev_n, levwise_n, max_depth+1)
    cumsum(in_arr=lev_nr, out_arr=lev_cs)
    cumsum(in_arr=levwise_nr, out_arr=levwise_cs)
    ereverse2(lev_n, levwise_n, lev_cs, levwise_cs, max_depth+1)

    esetting_p2(out_x, out_y, out_z, in_x, in_y, in_z, sph_pts, cx, cy, cz,
                out_r, in_r, length, level_s, num_p2, s1_index)

    tree_stop = time.time()

    return (cells, sfc_s, level_s, idx_s, bin_count, start_idx, leaf_idx,
            parent, child, part2bin, p2b_offset, lev_n, levwise_n, s1_index,
            s1r_index, lev_index, lev_index_r, cx, cy, cz, out_x, out_y, out_z,
            in_x, in_y, in_z, out_vl, in_vl, order, tree_stop - tree_start)
