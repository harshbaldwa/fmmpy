import importlib.resources
from math import floor

import compyle.array as ary
import numpy as np
import yaml
from compyle.api import (Elementwise, Reduction, Scan, annotate, declare,
                         get_config, wrap)
from compyle.low_level import cast, atomic_inc
from compyle.sort import radix_sort
from compyle.template import Template

np.set_printoptions(linewidth=np.inf)


@annotate(i="int", index="gintp", gdoublep="x, y, z",
          double="max_index, length, x_min, y_min, z_min"
          )
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
        ${obj.arrays[obj.number//2+t]}[i] = ${obj.arrays[t]}[i]
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


@annotate(i="int",
          gintp="sfc1, sfc2, level1, level2, lca_sfc, lca_level, lca_idx"
          )
def internal_nodes(i, sfc1, sfc2, level1, level2, lca_sfc, lca_level,
                   lca_idx):
    level_diff, xor_id, i1, i2, level, j = declare("int", 6)
    level_diff = cast(abs(level1[i] - level2[i]), "int")

    if level1[i] - level2[i] > 0:
        i1 = sfc1[i]
        i2 = sfc2[i] << 3 * level_diff
        level = level1[i]
    elif level1[i] - level2[i] < 0:
        i1 = sfc1[i] << 3 * level_diff
        i2 = sfc2[i]
        level = level2[i]
    else:
        i1 = sfc1[i]
        i2 = sfc2[i]
        level = level1[i]

    xor_id = i1 ^ i2

    if xor_id == 0:
        lca_sfc[i] = i1 >> 3 * level_diff
        lca_level[i] = level - level_diff
        lca_idx[i] = -1
        return

    for j in range(level + 1, 0, -1):
        if xor_id > ((1 << (j - 1) * 3) - 1):
            lca_sfc[i] = i1 >> 3 * j
            lca_level[i] = level - j
            lca_idx[i] = -1
            return

    lca_sfc[i] = 0
    lca_level[i] = 0
    lca_idx[i] = -1
    return


@annotate(i="int",
          gintp="sfc1, sfc2, level1, level2, lca_sfc, all_idx, "
                "lca_level, lca_idx, temp_idx"
          )
def find_parents(i, sfc1, sfc2, level1, level2, all_idx, lca_sfc,
                 lca_level, lca_idx, temp_idx):
    level_diff, xor_id, i1, i2, level, j = declare("int", 6)
    level_diff = cast(abs(level1[i] - level2[i]), "int")

    if level1[i] - level2[i] > 0:
        i1 = sfc1[i]
        i2 = sfc2[i] << 3 * level_diff
        level = level1[i]
    elif level1[i] - level2[i] < 0:
        i1 = sfc1[i] << 3 * level_diff
        i2 = sfc2[i]
        level = level2[i]
    else:
        i1 = sfc1[i]
        i2 = sfc2[i]
        level = level1[i]

    xor_id = i1 ^ i2

    if xor_id == 0:
        lca_sfc[i] = i1 >> 3 * level_diff
        lca_level[i] = level - level_diff
        lca_idx[i] = -1
        temp_idx[i] = all_idx[i]
        return

    for j in range(level + 1, 0, -1):
        if xor_id > ((1 << (j - 1) * 3) - 1):
            lca_sfc[i] = i1 >> 3 * j
            lca_level[i] = level - j
            lca_idx[i] = -1
            temp_idx[i] = all_idx[i]
            return

    lca_sfc[i] = 0
    lca_level[i] = 0
    lca_idx[i] = -1
    temp_idx[i] = all_idx[i]
    return


@annotate(int="i, max_level", gintp="sfc, level")
def sfc_same(i, sfc, level, max_level):
    sfc[i] = ((sfc[i] + 1) << 3 * (max_level - level[i])) - 1


@annotate(int="i, max_level", gintp="sfc, level")
def sfc_real(i, sfc, level, max_level):
    sfc[i] = ((sfc[i] + 1) >> 3 * (max_level - level[i])) - 1


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


@annotate(i="int", x="gintp")
def map_sum(i, x):
    return x[i]


@annotate(i="int",
          gintp="pc_sfc, pc_level, temp_idx, rel_idx, "
                "parent_idx, child_idx"
          )
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


@annotate(i="int",
          gintp="level_diff, sfc, level"
          )
def complete_tree(i, level_diff, sfc, level):
    if level_diff[i] < 1:
        return
    else:
        sfc[i] = sfc[i] >> 3 * level_diff[i]
        level[i] = level[i] - level_diff[i]


@annotate(i="int", lev_nr="gintp", return_="int")
def input_expr(i, lev_nr):
    if i == 0:
        return 0
    else:
        return lev_nr[i - 1]


@annotate(int="i, item", lev_cs="gintp")
def output_expr(i, item, lev_cs):
    lev_cs[i] = item


@annotate(x="int", return_="int")
def deinterleave(x):
    x = x & 0x49249249
    x = (x | (x >> 2)) & 0xC30C30C3
    x = (x | (x >> 4)) & 0xF00F00F
    x = (x | (x >> 8)) & 0xFF0000FF
    x = (x | (x >> 16)) & 0x0000FFFF
    return x


@annotate(i="int", gintp="sfc, level",
          gfloatp="cx, cy, cz",
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


@annotate(int="i, num_p2", level="gintp", length="double",
          gfloatp="cx, cy, cz, out_x, out_y, out_z, "
                  "in_x, in_y, in_z, sph_pts")
def setting_p2(i, out_x, out_y, out_z, in_x, in_y, in_z, 
               sph_pts, cx, cy, cz, length, level, num_p2):
    cid, sid = declare("int", 2)
    sz_cell = declare("double")
    cid = cast(floor(i*1.0/num_p2), "int")
    sid = i % num_p2
    sz_cell = length/(2.0**(level[cid]+1))
    out_x[i] = cx[cid] + 3*sz_cell*sph_pts[3*sid]
    out_y[i] = cy[cid] + 3*sz_cell*sph_pts[3*sid+1]
    out_z[i] = cz[cid] + 3*sz_cell*sph_pts[3*sid+2]
    in_x[i] = cx[cid] + 0.5*sz_cell*sph_pts[3*sid]
    in_y[i] = cy[cid] + 0.5*sz_cell*sph_pts[3*sid+1]
    in_z[i] = cz[cid] + 0.5*sz_cell*sph_pts[3*sid+2]


@annotate(int="i, max_depth", gintp="level, lev_n, idx")
def level_info(i, level, idx, lev_n, max_depth):
    ix = declare("int")
    if idx[i] == -1:
        ix = atomic_inc(lev_n[level[i]])
    else:
        ix = atomic_inc(lev_n[max_depth])


def build(N, max_depth, part_x, part_y, part_z, x_min,
          y_min, z_min, length, num_p2, backend):
    max_index = 2 ** max_depth
    part_x, part_y, part_z = wrap(part_x, part_y, part_z, backend=backend)

    leaf_sfc = ary.zeros(N, dtype=np.int32, backend=backend)
    leaf_idx = ary.arange(0, N, 1, dtype=np.int32, backend=backend)
    leaf_nodes_idx = ary.arange(0, 2*N-1, 1, dtype=np.int32, backend=backend)
    leaf_level = ary.empty(N, dtype=np.int32, backend=backend)
    leaf_level.fill(max_depth)

    leaf_sfc_sorted = ary.zeros(N, dtype=np.int32, backend=backend)
    leaf_idx_sorted = ary.zeros(N, dtype=np.int32, backend=backend)

    nodes_sfc = ary.zeros(N-1, dtype=np.int32, backend=backend)
    nodes_level = ary.zeros(N-1, dtype=np.int32, backend=backend)
    nodes_idx = ary.empty(N-1, dtype=np.int32, backend=backend)
    nodes_idx.fill(-1)

    nodes_sfc_sorted = ary.zeros(N-1, dtype=np.int32, backend=backend)
    nodes_level_sorted = ary.zeros(N-1, dtype=np.int32, backend=backend)
    nodes_idx_sorted = ary.empty(N-1, dtype=np.int32, backend=backend)
    nodes_idx_sorted.fill(-1)

    dp_idx = ary.zeros(N-1, dtype=np.int32, backend=backend)
    dp_idx_sorted = ary.zeros(N-1, dtype=np.int32, backend=backend)

    sfc = ary.zeros(2*N-1, dtype=np.int32, backend=backend)
    level = ary.zeros(2*N-1, dtype=np.int32, backend=backend)
    idx = ary.empty(2*N-1, dtype=np.int32, backend=backend)
    idx.fill(-1)

    sfc_sorted = ary.zeros(2*N-1, dtype=np.int32, backend=backend)
    level_sorted = ary.zeros(2*N-1, dtype=np.int32, backend=backend)
    idx_sorted = ary.empty(2*N-1, dtype=np.int32, backend=backend)
    idx_sorted.fill(-1)

    pc_sfc = ary.empty(4*N-2, dtype=np.int32, backend=backend)
    pc_level = ary.empty(4*N-2, dtype=np.int32, backend=backend)
    pc_idx = ary.empty(4*N-2, dtype=np.int32, backend=backend)
    temp_idx = ary.empty(4*N-2, dtype=np.int32, backend=backend)
    parent = ary.empty(2*N-1, dtype=np.int32, backend=backend)
    child = ary.empty(8*(2*N-1), dtype=np.int32, backend=backend)
    rel_idx = ary.empty(4*N-2, dtype=np.int32, backend=backend)
    pc_sfc.fill(-1)
    pc_level.fill(-1)
    pc_idx.fill(-1)
    temp_idx.fill(-1)
    parent.fill(-1)
    child.fill(-1)
    rel_idx.fill(-1)

    sort_pc_sfc = ary.zeros(4*N-2, dtype=np.int32, backend=backend)
    sort_pc_level = ary.zeros(4*N-2, dtype=np.int32, backend=backend)
    sort_pc_idx = ary.zeros(4*N-2, dtype=np.int32, backend=backend)
    sort_rel_idx = ary.zeros(4*N-2, dtype=np.int32, backend=backend)
    sort_temp_idx = ary.zeros(4*N-2, dtype=np.int32, backend=backend)

    level_diff = ary.zeros(2*N-1, dtype=np.int32, backend=backend)
    cumsum_diff = ary.zeros(2*N-1, dtype=np.int32, backend=backend)

    # with importlib.resources.open_text("fmm", "t_design.yaml") as file:
    #     data = yaml.load(file)[num_p2]
    data = yaml.load(open("t_design.yaml"), Loader=yaml.FullLoader)[num_p2]
    sph_pts = np.array(data['array'], dtype=np.float32)
    order = data['order']
    sph_pts = wrap(sph_pts, backend=backend)

    # different functions start from here
    eget_particle_index = Elementwise(get_particle_index, backend=backend)

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
    n_duplicates = Reduction('a+b', map_func=map_sum, backend=backend)

    efind_parents = Elementwise(find_parents, backend=backend)
    eget_relations = Elementwise(get_relations, backend=backend)

    efind_level_diff = Elementwise(find_level_diff, backend=backend)
    ecomplete_tree = Elementwise(complete_tree, backend=backend)

    ecalc_center = Elementwise(calc_center, backend=backend)
    esetting_p2 = Elementwise(setting_p2, backend=backend)
    elev_info = Elementwise(level_info, backend=backend)
    cumsum = Scan(input_expr, output_expr, 'a+b',
                  dtype=np.int32, backend=backend)

    eget_particle_index(leaf_sfc, part_x, part_y,
                        part_z, max_index, length,
                        x_min, y_min, z_min)

    [leaf_sfc_sorted, leaf_idx_sorted], _ = radix_sort(
        [leaf_sfc, leaf_idx], backend=backend)
    ecopy2(leaf_sfc_sorted, leaf_idx_sorted, leaf_sfc, leaf_idx)

    leaf_sfc_sorted.resize(0)
    leaf_idx_sorted.resize(0)

    einternal_nodes(leaf_sfc[:-1], leaf_sfc[1:], leaf_level[:-1],
                    leaf_level[1:], nodes_sfc, nodes_level,
                    nodes_idx)

    [nodes_level_sorted, nodes_sfc_sorted, nodes_idx_sorted], _ = radix_sort(
        [nodes_level, nodes_sfc, nodes_idx], backend=backend)

    ereverse3(nodes_level, nodes_sfc, nodes_idx,
              nodes_level_sorted, nodes_sfc_sorted,
              nodes_idx_sorted, N-1)

    esfc_same(nodes_sfc, nodes_level, max_depth)

    [nodes_sfc_sorted, nodes_level_sorted, nodes_idx_sorted], _ = radix_sort(
        [nodes_sfc, nodes_level, nodes_idx], backend=backend)

    ecopy3(nodes_sfc_sorted, nodes_level_sorted, nodes_idx_sorted,
           nodes_sfc, nodes_level, nodes_idx)

    eid_duplicates(nodes_sfc[:-1], nodes_level[:-1], dp_idx)

    eremove_duplicates(dp_idx, nodes_sfc, nodes_level)

    [dp_idx_sorted, nodes_sfc_sorted, nodes_level_sorted,
     nodes_idx_sorted], _ = radix_sort(
        [dp_idx, nodes_sfc, nodes_level, nodes_idx],
        backend=backend)

    ecopy3(nodes_sfc_sorted, nodes_level_sorted, nodes_idx_sorted,
           nodes_sfc, nodes_level, nodes_idx)

    nodes_sfc_sorted.resize(0)
    nodes_level_sorted.resize(0)
    nodes_idx_sorted.resize(0)

    # number of repeated internal nodes
    rep_cnt = int(n_duplicates(dp_idx_sorted))
    cells = 2*N-1 - rep_cnt

    # full sorted arrays (sfc, level, idx)

    ecopy6(leaf_sfc, nodes_sfc, leaf_level, nodes_level,
           leaf_idx, nodes_idx, sfc[:N], sfc[N:],
           level[:N], level[N:], idx[:N], idx[N:])

    [sfc_sorted, level_sorted, idx_sorted], _ = radix_sort(
        [sfc, level, idx], backend=backend)

    ecopy3(sfc_sorted, level_sorted, idx_sorted,
           sfc, level, idx)

    esfc_real(sfc, level, max_depth)

    sfc_sorted.resize(0)
    level_sorted.resize(0)
    idx_sorted.resize(0)

    sfc.resize(cells)
    level.resize(cells)
    idx.resize(cells)
    leaf_nodes_idx.resize(cells)
    parent.resize(cells)
    child.resize(8*cells)
    level_diff.resize(cells)

    # finding parent child relationships
    ecopy4(sfc, level, idx, leaf_nodes_idx,
           pc_sfc, pc_level, pc_idx, rel_idx)

    efind_parents(sfc[:-1], sfc[1:], level[:-1], level[1:], leaf_nodes_idx[:-1],
                  pc_sfc[2*N-1:], pc_level[2*N-1:], pc_idx[2*N-1:],
                  temp_idx[2*N-1:])

    [sort_pc_level, sort_pc_sfc, sort_pc_idx, sort_rel_idx,
     sort_temp_idx], _ = radix_sort([pc_level, pc_sfc,
                                     pc_idx, rel_idx, temp_idx],
                                    backend=backend)

    ereverse5(pc_level, pc_sfc, pc_idx, rel_idx,
              temp_idx, sort_pc_level, sort_pc_sfc,
              sort_pc_idx, sort_rel_idx, sort_temp_idx, 4*N-2)

    esfc_same(pc_sfc[2*rep_cnt+1:],
              pc_level[2*rep_cnt+1:], max_depth)

    [sort_pc_sfc, sort_pc_level, sort_pc_idx, sort_rel_idx,
     sort_temp_idx], _ = radix_sort([pc_sfc, pc_level,
                                     pc_idx, rel_idx, temp_idx],
                                    backend=backend)

    ecopy5(sort_pc_sfc, sort_pc_level, sort_pc_idx,
           sort_rel_idx, sort_temp_idx, pc_sfc, pc_level,
           pc_idx, rel_idx, temp_idx)

    sort_pc_sfc.resize(0)
    sort_pc_level.resize(0)
    sort_pc_idx.resize(0)
    sort_rel_idx.resize(0)
    sort_temp_idx.resize(0)
    esfc_real(pc_sfc[:-(2*rep_cnt+1)],
              pc_level[:-(2*rep_cnt+1)], max_depth)

    eget_relations(pc_sfc, pc_level, temp_idx, rel_idx,
                   parent, child)

    efind_level_diff(level, parent, level_diff)

    ecomplete_tree(level_diff, sfc, level)

    index = ary.arange(0, cells, 1, dtype=np.int32, backend=backend)

    s1_index = ary.zeros(cells, dtype=np.int32, backend=backend)
    s1r_index = ary.zeros(cells, dtype=np.int32, backend=backend)
    s2_index = ary.zeros(cells, dtype=np.int32, backend=backend)
    s1_lev = ary.zeros(cells, dtype=np.int32, backend=backend)
    s2_lev = ary.zeros(cells, dtype=np.int32, backend=backend)
    s1_idx = ary.zeros(cells, dtype=np.int32, backend=backend)
    s2_idx = ary.zeros(cells, dtype=np.int32, backend=backend)

    lev_n = ary.zeros(max_depth+1, dtype=np.int32, backend=backend)
    lev_nr = ary.zeros(max_depth+1, dtype=np.int32, backend=backend)
    lev_cs = ary.zeros(max_depth+1, dtype=np.int32, backend=backend)

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

    ecalc_center(sfc, level, cx, cy, cz,
                 x_min, y_min, z_min, length)
    esetting_p2(cx, cy, cz, out_x, out_y, out_z, in_x,
                in_y, in_z, sph_pts, length, level, num_p2)

    [s1_lev, s1_idx, s1_index], _ = radix_sort([level, idx, index],
                                               backend=backend)
    ereverse3(s2_idx, s2_lev, s2_index, s1_idx, s1_lev, s1_index, cells)
    [s1_idx, s1_lev, s1_index], _ = radix_sort([s2_idx, s2_lev, s2_index],
                                               backend=backend)

    [s2_index, s1r_index], _ = radix_sort([s1_index, index], backend=backend)

    s2_idx.resize(0)
    s2_lev.resize(0)
    s2_index.resize(0)

    s1_lev.resize(0)
    s1_idx.resize(0)
    index.resize(0)

    elev_info(level, idx, lev_n, max_depth)
    ereverse1(lev_nr, lev_n, max_depth+1)
    cumsum(lev_nr=lev_nr, lev_cs=lev_cs)
    ereverse1(lev_cs, lev_n, max_depth+1)

    return cells, sfc, level, idx, parent, child, lev_n, s1_index, s1r_index, \
        cx, cy, cz, out_x, out_y, out_z, in_x, in_y, in_z, out_vl, in_vl


if __name__ == "__main__":
    backend = 'opencl'
    N = 1000000
    max_depth = 5
    np.random.seed(4)
    part_x = np.random.random(N)
    part_y = np.random.random(N)
    part_z = np.random.random(N)
    x_min = 0
    y_min = 0
    z_min = 0
    length = 1
    num_p2 = 6
    build(N, max_depth, part_x, part_y, part_z, x_min,
          y_min, z_min, length, num_p2, backend)
