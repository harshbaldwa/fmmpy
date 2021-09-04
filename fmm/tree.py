from compyle.api import declare, Elementwise, \
    annotate, wrap, Reduction, get_config, Scan
from compyle.low_level import cast
from compyle.sort import radix_sort
import numpy as np
import compyle.array as ary
from math import floor
from compyle.template import Template
import argparse
import time

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
# compute lowest common ancestor (LCA) of two nodes
def internal_nodes(i, sfc1, sfc2, level1, level2, lca_sfc, lca_level,
                   lca_idx):
    level_diff, xor, i1, i2, level, j = declare("int", 6)
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

    xor = i1 ^ i2

    if xor == 0:
        lca_sfc[i] = i1 >> 3 * level_diff
        lca_level[i] = level - level_diff
        lca_idx[i] = -1
        return

    for j in range(level + 1, 0, -1):
        if xor > ((1 << (j - 1) * 3) - 1):
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
# compute lowest common ancestor (LCA) of two nodes
def find_parents(i, sfc1, sfc2, level1, level2, all_idx, lca_sfc,
                 lca_level, lca_idx, temp_idx):
    level_diff, xor, i1, i2, level, j = declare("int", 6)
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

    xor = i1 ^ i2

    if xor == 0:
        lca_sfc[i] = i1 >> 3 * level_diff
        lca_level[i] = level - level_diff
        lca_idx[i] = -1
        temp_idx[i] = all_idx[i]
        return

    for j in range(level + 1, 0, -1):
        if xor > ((1 << (j - 1) * 3) - 1):
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
# make the sfc of all elements of same length
def sfc_same(i, sfc, level, max_level):
    sfc[i] = ((sfc[i] + 1) << 3 * (max_level - level[i])) - 1


@annotate(int="i, max_level", gintp="sfc, level")
# make the sfc of all elements of their respective length
def sfc_real(i, sfc, level, max_level):
    sfc[i] = ((sfc[i] + 1) >> 3 * (max_level - level[i])) - 1


@annotate(i="int", gintp="sfc, level, dp_idx")
# find the duplicate items in the array
def id_duplicates(i, sfc, level, dp_idx):
    if i == 0:
        dp_idx[i] = 0

    if sfc[i] == sfc[i+1] and level[i] == level[i+1]:
        dp_idx[i+1] = 1


@annotate(i="int", gintp="dp_idx, sfc, level")
# removing the duplicate items from the array
def remove_duplicates(i, dp_idx, sfc, level):
    if dp_idx[i] == 1:
        sfc[i] = -1
        level[i] = -1


@annotate(i="int", x="gintp")
# mapping for reduction (calculating sum of all elements)
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
        if (pc_sfc[i] != pc_sfc[i-j-1] or pc_level[i] != pc_level[i-j-1]):
            return
        else:
            parent_idx[temp_idx[i-j-1]] = rel_idx[i]
            child_idx[8*rel_idx[i] + j] = temp_idx[i-j-1]


@annotate(i="int", gintp="level, parent_idx, level_diff")
def find_level_diff(i, level, parent_idx, level_diff):
    if parent_idx[i] != -1:
        level_diff[i] = level[i] - level[parent_idx[i]] - 1


@annotate(i="int", level="gintp", return_="int")
def input_expr(i, level):
    if i == 0:
        return 0
    else:
        return level[i - 1]


@annotate(int="i, item", cs_level="gintp")
def output_expr(i, item, cs_level):
    cs_level[i] = item


@annotate(i="int",
          gintp="level_diff, cumsum_diff, sfc, level, idx, parent, "
                "child, new_sfc, new_level, new_idx, new_parent, new_child"
          )
def complete_tree(i, level_diff, cumsum_diff, sfc, level, idx, parent,
                  child, new_sfc, new_level, new_idx, new_parent, new_child):
    offset, j, k, l = declare("int", 4)
    offset = i + cumsum_diff[i]
    new_sfc[offset] = sfc[i]
    new_level[offset] = level[i]
    new_idx[offset] = idx[i]
    # for j in range(8):
    #     if child[8*i + j] != -1:
    #         new_child[8*offset+j] = child[8*i+j] + cumsum_diff[child[8*i+j]]
    #     else:
    #         break
    if level_diff[i] == 0:
        if parent[i] != -1:
            new_parent[offset] = parent[i] + cumsum_diff[parent[i]]
        else:
            new_parent[offset] = -1
        return

    new_parent[offset] = offset + 1

    for k in range(1, level_diff[i]+1):
        l = offset + k
        new_sfc[l] = sfc[i] >> 3
        new_level[l] = level[i] - k
        new_idx[l] = -1
        new_parent[l] = l + 1
        new_child[8*l] = l - 1

    new_parent[offset+level_diff[i]] = parent[i] + cumsum_diff[parent[i]]
    for j in range(8):
        if child[8*parent[i]+j] == i:
            new_child[8*(parent[i] + cumsum_diff[parent[i]]) +
                      j] = offset + level_diff[i]
            break


def build(N, max_depth, part_x, part_y, part_z, x_min, y_min, z_min, length, backend):
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

    all_sfc = ary.zeros(2*N-1, dtype=np.int32, backend=backend)
    all_level = ary.zeros(2*N-1, dtype=np.int32, backend=backend)
    all_idx = ary.empty(2*N-1, dtype=np.int32, backend=backend)
    all_idx.fill(-1)

    all_sfc_sorted = ary.zeros(2*N-1, dtype=np.int32, backend=backend)
    all_level_sorted = ary.zeros(2*N-1, dtype=np.int32, backend=backend)
    all_idx_sorted = ary.empty(2*N-1, dtype=np.int32, backend=backend)
    all_idx_sorted.fill(-1)

    pc_sfc = ary.empty(4*N-2, dtype=np.int32, backend=backend)
    pc_level = ary.empty(4*N-2, dtype=np.int32, backend=backend)
    pc_idx = ary.empty(4*N-2, dtype=np.int32, backend=backend)
    temp_idx = ary.empty(4*N-2, dtype=np.int32, backend=backend)
    parent_idx = ary.empty(2*N-1, dtype=np.int32, backend=backend)
    child_idx = ary.empty(8*(2*N-1), dtype=np.int32, backend=backend)
    rel_idx = ary.empty(4*N-2, dtype=np.int32, backend=backend)
    pc_sfc.fill(-1)
    pc_level.fill(-1)
    pc_idx.fill(-1)
    temp_idx.fill(-1)
    parent_idx.fill(-1)
    child_idx.fill(-1)
    rel_idx.fill(-1)

    sort_pc_sfc = ary.zeros(4*N-2, dtype=np.int32, backend=backend)
    sort_pc_level = ary.zeros(4*N-2, dtype=np.int32, backend=backend)
    sort_pc_idx = ary.zeros(4*N-2, dtype=np.int32, backend=backend)
    sort_rel_idx = ary.zeros(4*N-2, dtype=np.int32, backend=backend)
    sort_temp_idx = ary.zeros(4*N-2, dtype=np.int32, backend=backend)

    level_diff = ary.zeros(2*N-1, dtype=np.int32, backend=backend)
    cumsum_diff = ary.zeros(2*N-1, dtype=np.int32, backend=backend)

    # different functions start from here
    eget_particle_index = Elementwise(get_particle_index, backend=backend)

    copy_arrs_2 = CopyArrays('copy_arrs_2', [
        'a1', 'a2',
        'b1', 'b2']).function
    copy_arrs_3 = CopyArrays('copy_arrs_3', [
        'a1', 'a2', 'a3',
        'b1', 'b2', 'b3']).function
    copy_arrs_4 = CopyArrays('copy_arrs_4', [
        'a1', 'a2', 'a3', 'a4',
        'b1', 'b2', 'b3', 'b4']).function
    copy_arrs_5 = CopyArrays('copy_arrs_5', [
        'a1', 'a2', 'a3', 'a4', 'a5',
        'b1', 'b2', 'b3', 'b4', 'b5']).function
    copy_arrs_6 = CopyArrays('copy_arrs_6', [
        'a1', 'a2', 'a3', 'a4', 'a5', 'a6',
        'b1', 'b2', 'b3', 'b4', 'b5', 'b6']).function

    ecopy_arrs_2 = Elementwise(copy_arrs_2, backend=backend)
    ecopy_arrs_3 = Elementwise(copy_arrs_3, backend=backend)
    ecopy_arrs_4 = Elementwise(copy_arrs_4, backend=backend)
    ecopy_arrs_5 = Elementwise(copy_arrs_5, backend=backend)
    ecopy_arrs_6 = Elementwise(copy_arrs_6, backend=backend)

    einternal_nodes = Elementwise(internal_nodes, backend=backend)

    reverse_arr_3 = ReverseArrays('reverse_arr_3', [
        'a1', 'a2', 'a3',
        'b1', 'b2', 'b3']).function
    reverse_arr_4 = ReverseArrays('reverse_arr_4', [
        'a1', 'a2', 'a3', 'a4',
        'b1', 'b2', 'b3', 'b4']).function
    reverse_arr_5 = ReverseArrays('reverse_arr_5', [
        'a1', 'a2', 'a3', 'a4', 'a5',
        'b1', 'b2', 'b3', 'b4', 'b5']).function

    ereverse_arrs_3 = Elementwise(reverse_arr_3, backend=backend)
    ereverse_arrs_4 = Elementwise(reverse_arr_4, backend=backend)
    ereverse_arrs_5 = Elementwise(reverse_arr_5, backend=backend)

    esfc_same = Elementwise(sfc_same, backend=backend)
    esfc_real = Elementwise(sfc_real, backend=backend)

    eid_duplicates = Elementwise(id_duplicates, backend=backend)
    eremove_duplicates = Elementwise(remove_duplicates, backend=backend)
    n_duplicates = Reduction('a+b', map_func=map_sum, backend=backend)

    efind_parents = Elementwise(find_parents, backend=backend)
    eget_relations = Elementwise(get_relations, backend=backend)

    efind_level_diff = Elementwise(find_level_diff, backend=backend)
    eget_cumsum_diff = Scan(input_expr, output_expr,
                            'a+b', dtype=np.int32, backend=backend)
    ecomplete_tree = Elementwise(complete_tree, backend=backend)

    # making the adaptive oct tree from bottom up
    # calculates sfc of all particles at the $max_depth level
    eget_particle_index(leaf_sfc, part_x, part_y,
                        part_z, max_index, length,
                        x_min, y_min, z_min)

    # sorts based on sfc array
    [leaf_sfc_sorted, leaf_idx_sorted], _ = radix_sort(
        [leaf_sfc, leaf_idx], backend=backend)
    ecopy_arrs_2(leaf_sfc_sorted, leaf_idx_sorted, leaf_sfc, leaf_idx)

    # finds the LCA of all particles

    # ecopy_arrs_3(leaf_sfc, leaf_level, leaf_idx,
    #              nodes_sfc, nodes_level, nodes_idx)
    einternal_nodes(leaf_sfc[:-1], leaf_sfc[1:], leaf_level[:-1],
                    leaf_level[1:], nodes_sfc, nodes_level,
                    nodes_idx)

    # sorts internal nodes array across level
    [nodes_level_sorted, nodes_sfc_sorted, nodes_idx_sorted], _ = radix_sort(
        [nodes_level, nodes_sfc, nodes_idx], backend=backend)

    ereverse_arrs_3(nodes_level, nodes_sfc, nodes_idx,
                    nodes_level_sorted, nodes_sfc_sorted,
                    nodes_idx_sorted, N-1)

    esfc_same(nodes_sfc, nodes_level, max_depth)

    [nodes_sfc_sorted, nodes_level_sorted, nodes_idx_sorted], _ = radix_sort(
        [nodes_sfc, nodes_level, nodes_idx], backend=backend)

    ecopy_arrs_3(nodes_sfc_sorted, nodes_level_sorted, nodes_idx_sorted,
                 nodes_sfc, nodes_level, nodes_idx)

    # deletes all duplicate nodes
    eid_duplicates(nodes_sfc[:-1], nodes_level[:-1], dp_idx)
    eremove_duplicates(dp_idx, nodes_sfc, nodes_level)

    [dp_idx_sorted, nodes_sfc_sorted, nodes_level_sorted,
     nodes_idx_sorted], _ = radix_sort(
        [dp_idx, nodes_sfc, nodes_level, nodes_idx],
        backend=backend)

    ecopy_arrs_3(nodes_sfc_sorted, nodes_level_sorted, nodes_idx_sorted,
                 nodes_sfc, nodes_level, nodes_idx)

    # number of repeated internal nodes
    count_repeated = int(n_duplicates(dp_idx_sorted))

    # full sorted arrays (sfc, level, idx)

    ecopy_arrs_6(leaf_sfc, nodes_sfc, leaf_level, nodes_level,
                 leaf_idx, nodes_idx, all_sfc[:N], all_sfc[N:],
                 all_level[:N], all_level[N:], all_idx[:N], all_idx[N:])

    [all_sfc_sorted, all_level_sorted, all_idx_sorted], _ = radix_sort(
        [all_sfc, all_level, all_idx], backend=backend)

    ecopy_arrs_3(all_sfc_sorted, all_level_sorted, all_idx_sorted,
                 all_sfc, all_level, all_idx)

    esfc_real(all_sfc, all_level, max_depth)

    # finding parent child relationships
    ecopy_arrs_4(all_sfc, all_level, all_idx, leaf_nodes_idx,
                 pc_sfc, pc_level, pc_idx, rel_idx)

    efind_parents(all_sfc[:-count_repeated-1],
                  all_sfc[1:-count_repeated],
                  all_level[:-count_repeated-1],
                  all_level[1:-count_repeated],
                  leaf_nodes_idx[:-count_repeated-1],
                  pc_sfc[2*N-1:], pc_level[2*N-1:],
                  pc_idx[2*N-1:], temp_idx[2*N-1:])

    [sort_pc_level, sort_pc_sfc, sort_pc_idx, sort_rel_idx,
     sort_temp_idx], _ = radix_sort([pc_level, pc_sfc,
                                     pc_idx, rel_idx, temp_idx],
                                    backend=backend)

    ereverse_arrs_5(pc_level, pc_sfc, pc_idx, rel_idx,
                    temp_idx, sort_pc_level, sort_pc_sfc,
                    sort_pc_idx, sort_rel_idx, sort_temp_idx, 4*N-2)

    esfc_same(pc_sfc[2*count_repeated+1:],
              pc_level[2*count_repeated+1:], max_depth)

    [sort_pc_sfc, sort_pc_level, sort_pc_idx, sort_rel_idx,
     sort_temp_idx], _ = radix_sort([pc_sfc, pc_level,
                                     pc_idx, rel_idx, temp_idx],
                                    backend=backend)

    ecopy_arrs_5(sort_pc_sfc, sort_pc_level, sort_pc_idx,
                 sort_rel_idx, sort_temp_idx, pc_sfc, pc_level,
                 pc_idx, rel_idx, temp_idx)

    esfc_real(pc_sfc[:-(2*count_repeated+1)],
              pc_level[:-(2*count_repeated+1)], max_depth)

    eget_relations(pc_sfc, pc_level, temp_idx, rel_idx,
                   parent_idx, child_idx)

    efind_level_diff(all_level[:-count_repeated],
                     parent_idx[:-count_repeated],
                     level_diff[:-count_repeated])

    eget_cumsum_diff(level=level_diff, cs_level=cumsum_diff)

    count = 2*N - 1 - count_repeated + cumsum_diff[-count_repeated-1]
    sfc = ary.zeros(count, dtype=np.int32, backend=backend)
    level = ary.zeros(count, dtype=np.int32, backend=backend)
    idx = ary.zeros(count, dtype=np.int32, backend=backend)
    parent = ary.zeros(count, dtype=np.int32, backend=backend)
    child = ary.zeros(8*count, dtype=np.int32, backend=backend)
    child.fill(-1)

    ecomplete_tree(level_diff[:-count_repeated], cumsum_diff[:-count_repeated],
                   all_sfc[:-count_repeated], all_level[:-count_repeated],
                   all_idx[:-count_repeated], parent_idx[:-count_repeated],
                   child_idx[:-8*count_repeated], sfc, level, idx, parent, child)

    return count, sfc, level, idx, parent, child


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="backend to use", default=10)
    parser.add_argument("-l", help="max depth for tree",
                        default=2)
    parser.add_argument("--seed", help="Random Seed", default=4)
    parser.add_argument("-b", "--backend", help="backend to use",
                        default='cython')
    parser.add_argument("-omp", "--openmp",
                        help="use openmp for calculations",
                        action="store_true")
    args = parser.parse_args()

    if args.openmp:
        get_config().use_openmp = True

    np.random.seed(int(args.seed))
    backend = args.backend
    N = int(args.n)
    max_depth = int(args.l)
    build(N, max_depth, backend)
