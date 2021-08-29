from compyle.api import declare, Elementwise, annotate, Scan, wrap, Reduction
from compyle.low_level import cast
from compyle.sort import radix_sort
import numpy as np
import compyle.array as ary
from math import floor, log
from compyle.template import Template

import pytest

np.set_printoptions(linewidth=np.inf)


@annotate(i="int", index="gintp", gdoublep="x, y, z",
          double="max_index, length"
          )
def get_particle_index(i, index, x, y, z, max_index, length):
    nx, ny, nz = declare("int", 3)
    nx = cast(floor((max_index * x[i]) / length), "int")
    ny = cast(floor((max_index * y[i]) / length), "int")
    nz = cast(floor((max_index * z[i]) / length), "int")

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


class CopyValue(Template):
    def __init__(self, name, arrays):
        super(CopyValue, self).__init__(name=name)
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


class ReverseArray(Template):
    def __init__(self, name, arrays):
        super(ReverseArray, self).__init__(name=name)
        self.arrays = arrays
        self.number = len(arrays)

    def extra_args(self):
        return self.arrays + ["length"], {"intp": ','.join(self.arrays),
                                          "length": "int"}

    @annotate(i='int')
    def template(self, i):
        '''
        % for t in range(obj.number//2):
        ${obj.arrays[t]}[length - i - 1] = ${obj.arrays[obj.number//2 + t]}[i]
        % endfor
        '''


@annotate(i="int",
          gintp="sfc1, sfc2, level1, level2, lca_sfc, lca_level, lca_idx"
          )
# compute lowest common ancestor (LCA) of two nodes
def internal_nodes(i, sfc1, sfc2, level1, level2, lca_sfc, lca_level, lca_idx):
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
          gintp="sfc1, sfc2, level1, level2, lca_sfc, lca_level, lca_idx, "
                "child_sfc, child_idx"
          )
# compute lowest common ancestor (LCA) of two nodes
def parent_child(i, sfc1, sfc2, level1, level2, lca_sfc,
                 lca_level, lca_idx, child_sfc, child_idx):
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
        child_sfc[i] = sfc1[i]
        child_idx[i] = i
        return

    for j in range(level + 1, 0, -1):
        if xor > ((1 << (j - 1) * 3) - 1):
            lca_sfc[i] = i1 >> 3 * j
            lca_level[i] = level - j
            lca_idx[i] = -1
            child_sfc[i] = sfc1[i]
            child_idx[i] = i
            return

    lca_sfc[i] = 0
    lca_level[i] = 0
    lca_idx[i] = -1
    child_sfc[i] = sfc1[i]
    child_idx[i] = i
    return


@annotate(int="i, max_level", gintp="sfc, level")
# make the sfc of all elements of same length
def sfc_same(i, sfc, level, max_level):
    sfc[i] = ((sfc[i] + 1) << 3 * (max_level - level[i])) - 1


@annotate(int="i, max_level", gintp="sfc, level")
# make the sfc of all elements of their respective length
def sfc_real(i, sfc, level, max_level):
    sfc[i] = ((sfc[i] + 1) >> 3 * (max_level - level[i])) - 1


@annotate(i="int", gintp="sfc, level, duplicate_idx")
# find the duplicate items in the array
def id_duplicates(i, sfc, level, duplicate_idx):
    if i == 0:
        duplicate_idx[i] = 0

    if sfc[i] == sfc[i+1] and level[i] == level[i+1]:
        duplicate_idx[i+1] = 1


@annotate(i="int", gintp="duplicate_idx, sfc, level")
# removing the duplicate items from the array
def remove_duplicates(i, duplicate_idx, sfc, level):
    if duplicate_idx[i] == 1:
        sfc[i] = 0
        level[i] = -1


@annotate(i="int", x="gintp")
# mapping for reduction (calculating sum of all elements)
def map_sum(i, x):
    return x[i]


if __name__ == "__main__":

    backend = "cython"
    N = 10
    max_depth = 2
    length = 1

    np.random.seed(4)
    particle_pos = np.random.random((3, N))
    idx = np.arange(N, dtype=np.int32)
    max_index = 2 ** max_depth

    sfc = ary.zeros(N, dtype=np.int32, backend=backend)
    level = ary.ones(N, dtype=np.int32, backend=backend) * max_depth

    sort_sfc = ary.zeros(N, dtype=np.int32, backend=backend)
    sort_idx = ary.zeros(N, dtype=np.int32, backend=backend)

    sfc_nodes = ary.zeros(2*N-1, dtype=np.int32, backend=backend)
    level_nodes = ary.zeros(2*N-1, dtype=np.int32, backend=backend)
    idx_nodes = ary.ones(2*N-1, dtype=np.int32, backend=backend) * -1

    sort_sfc_nodes = ary.zeros(N-1, dtype=np.int32, backend=backend)
    sort_level_nodes = ary.zeros(N-1, dtype=np.int32, backend=backend)
    sort_idx_nodes = ary.ones(N-1, dtype=np.int32, backend=backend) * -1

    sort_full_sfc = ary.zeros(2*N-1, dtype=np.int32, backend=backend)
    sort_full_level = ary.zeros(2*N-1, dtype=np.int32, backend=backend)
    sort_full_idx = ary.ones(2*N-1, dtype=np.int32, backend=backend) * -1

    duplicate_idx = ary.zeros(N-1, dtype=np.int32, backend=backend)
    sort_duplicate_idx = ary.zeros(N-1, dtype=np.int32, backend=backend)

    full_pc_sfc = ary.ones(4*N-2, dtype=np.int32, backend=backend) * -1
    full_pc_level = ary.ones(4*N-2, dtype=np.int32, backend=backend) * -1
    full_pc_idx = ary.ones(4*N-2, dtype=np.int32, backend=backend) * -1
    child_sfc = ary.ones(4*N-2, dtype=np.int32, backend=backend) * -1
    child_idx_arr = ary.ones(4*N-2, dtype=np.int32, backend=backend) * -1

    sort_full_pc_sfc = ary.zeros(4*N-2, dtype=np.int32, backend=backend)
    sort_full_pc_level = ary.zeros(4*N-2, dtype=np.int32, backend=backend)
    sort_full_pc_idx = ary.zeros(4*N-2, dtype=np.int32, backend=backend)
    sort_child_sfc = ary.zeros(4*N-2, dtype=np.int32, backend=backend)
    sort_child_idx_arr = ary.zeros(4*N-2, dtype=np.int32, backend=backend)

    # different functions start from here
    eget_particle_index = Elementwise(get_particle_index, backend=backend)

    copy_value_2 = CopyValue('copy_value_2', [
                             'a1', 'a2',
                             'b1', 'b2']).function
    copy_value_3 = CopyValue('copy_value_3', [
                             'a1', 'a2', 'a3',
                             'b1', 'b2', 'b3']).function
    copy_value_5 = CopyValue('copy_value_5', [
                             'a1', 'a2', 'a3', 'a4', 'a5',
                             'b1', 'b2', 'b3', 'b4', 'b5']).function

    ecopy_value_2 = Elementwise(copy_value_2, backend=backend)
    ecopy_value_3 = Elementwise(copy_value_3, backend=backend)
    ecopy_value_5 = Elementwise(copy_value_5, backend=backend)

    einternal_nodes = Elementwise(internal_nodes, backend=backend)

    reverse_arr_3 = ReverseArray('reverse_arr_3', [
                                 'a1', 'a2', 'a3',
                                 'b1', 'b2', 'b3']).function
    reverse_arr_5 = ReverseArray('reverse_arr_5', [
                                 'a1', 'a2', 'a3', 'a4', 'a5',
                                 'b1', 'b2', 'b3', 'b4', 'b5']).function

    ereverse_arrs_3 = Elementwise(reverse_arr_3, backend=backend)
    ereverse_arrs_5 = Elementwise(reverse_arr_5, backend=backend)

    esfc_same = Elementwise(sfc_same, backend=backend)
    esfc_real = Elementwise(sfc_real, backend=backend)

    eid_duplicates = Elementwise(id_duplicates, backend=backend)
    eremove_duplicates = Elementwise(remove_duplicates, backend=backend)
    n_duplicates = Reduction('a+b', map_func=map_sum, backend=backend)

    eparent_child = Elementwise(parent_child, backend=backend)

    # making the adaptive oct tree from bottom up
    # calculates sfc of all particles at the $max_depth level
    eget_particle_index(sfc, particle_pos[0], particle_pos[1],
                        particle_pos[2], max_index, length)

    # sorts based on sfc array
    [sort_sfc, sort_idx], _ = radix_sort([sfc, idx], backend=backend)
    ecopy_value_2(sort_sfc, sort_idx, sfc, idx)

    # finds the LCA of all particles

    ecopy_value_3(sfc, level, idx, sfc_nodes, level_nodes, idx_nodes)
    einternal_nodes(sfc[:-1], sfc[1:], level[:-1], level[1:],
                    sfc_nodes[N:], level_nodes[N:], idx_nodes[N:])

    # sorts internal nodes array across level
    [sort_level_nodes, sort_sfc_nodes, sort_idx_nodes], _ = radix_sort(
        [level_nodes[N:], sfc_nodes[N:], idx_nodes[N:]], backend=backend)

    ereverse_arrs_3(level_nodes[N:], sfc_nodes[N:], idx_nodes[N:],
                    sort_level_nodes, sort_sfc_nodes, sort_idx_nodes, N-1)

    esfc_same(sfc_nodes[N:], level_nodes[N:], max_depth)

    [sort_sfc_nodes, sort_level_nodes, sort_idx_nodes], _ = radix_sort(
        [sfc_nodes[N:], level_nodes[N:], idx_nodes[N:]], backend=backend)

    ecopy_value_3(sort_sfc_nodes, sort_level_nodes, sort_idx_nodes,
                  sfc_nodes[N:], level_nodes[N:], idx_nodes[N:])

    # deletes all duplicate nodes
    eid_duplicates(sfc_nodes[N:-1], level_nodes[N:-1], duplicate_idx)
    eremove_duplicates(duplicate_idx, sfc_nodes[N:], level_nodes[N:])

    [sort_duplicate_idx, sort_sfc_nodes, sort_level_nodes,
     sort_idx_nodes], _ = radix_sort(
        [duplicate_idx, sfc_nodes[N:], level_nodes[N:], idx_nodes[N:]],
        backend=backend)

    ecopy_value_3(sort_sfc_nodes, sort_level_nodes, sort_idx_nodes,
                  sfc_nodes[N:], level_nodes[N:], idx_nodes[N:])

    # number of repeated internal nodes
    count_repeated = int(n_duplicates(sort_duplicate_idx))

    # full sorted arrays (sfc, level, idx)
    [sort_full_sfc, sort_full_level, sort_full_idx], _ = radix_sort(
        [sfc_nodes, level_nodes, idx_nodes], backend=backend)

    ecopy_value_3(sort_full_sfc, sort_full_level, sort_full_idx,
                  sfc_nodes, level_nodes, idx_nodes)

    esfc_real(sfc_nodes, level_nodes, max_depth)

    # finding parent child relationships
    ecopy_value_3(sfc_nodes, level_nodes, idx_nodes,
                  full_pc_sfc, full_pc_level, full_pc_idx)

    eparent_child(sfc_nodes[count_repeated:-1],
                  sfc_nodes[count_repeated+1:],
                  level_nodes[count_repeated:-1],
                  level_nodes[count_repeated+1:],
                  full_pc_sfc[2*N-1:], full_pc_level[2*N-1:],
                  full_pc_idx[2*N-1:], child_sfc[2*N-1:],
                  child_idx_arr[2*N-1:])

    [sort_full_pc_level, sort_full_pc_sfc, sort_full_pc_idx, sort_child_sfc,
     sort_child_idx_arr], _ = radix_sort([full_pc_level, full_pc_sfc,
                                          full_pc_idx, child_sfc,
                                          child_idx_arr], backend=backend)

    ereverse_arrs_5(full_pc_level, full_pc_sfc, full_pc_idx, child_sfc,
                    child_idx_arr, sort_full_pc_level, sort_full_pc_sfc,
                    sort_full_pc_idx, sort_child_sfc, sort_child_idx_arr,
                    4*N-2)

    esfc_same(full_pc_sfc[2*count_repeated+1:],
              full_pc_level[2*count_repeated+1:], max_depth)

    [sort_full_pc_sfc, sort_full_pc_level, sort_full_pc_idx, sort_child_sfc,
     sort_child_idx_arr], _ = radix_sort([full_pc_sfc, full_pc_level,
                                          full_pc_idx, child_sfc,
                                          child_idx_arr], backend=backend)

    ecopy_value_5(sort_full_pc_sfc, sort_full_pc_level, sort_full_pc_idx,
                  sort_child_sfc, sort_child_idx_arr, full_pc_sfc,
                  full_pc_level, full_pc_idx, child_sfc, child_idx_arr)

    esfc_real(full_pc_sfc[:-(2*count_repeated+1)],
              full_pc_level[:-(2*count_repeated+1)], max_depth)
