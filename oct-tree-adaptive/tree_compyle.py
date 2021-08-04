from compyle.api import declare, Elementwise, annotate, Scan, wrap, Reduction
from compyle.low_level import cast
from compyle.sort import radix_sort
import numpy as np
from math import floor, log


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


@annotate(i="int",
          gintp="sfc_idx, level, idx, full_sfc_idx, full_level, full_idx"
          )
def cpy_idx_tree(i, sfc_idx, level, idx, full_sfc_idx, full_level, full_idx):
    full_sfc_idx[i] = sfc_idx[i]
    full_level[i] = level[i]
    full_idx[i] = idx[i]


@annotate(i="int",
          gintp="sfc1, sfc2, level1, level2, lca_sfc, lca_level, lca_idx"
          )
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


@annotate(
    int="i, len_arr",
    gintp="arr, sort_arr, index, sort_index, level, sort_level"
)
def reverse_arrs(i, arr, index, level, sort_arr,
                 sort_index, sort_level, len_arr):
    arr[len_arr - i - 1] = sort_arr[i]
    index[len_arr - i - 1] = sort_index[i]
    level[len_arr - i - 1] = sort_level[i]


@annotate(
    i="int",
    gintp="arr, sort_arr, index, sort_index"
)
def swap_arrs_two(i, arr, index, sort_arr, sort_index):
    arr[i] = sort_arr[i]
    index[i] = sort_index[i]


@annotate(
    i="int",
    gintp="arr, sort_arr, index, sort_index, level, sort_level"
)
def swap_arrs_three(i, arr, index, level, sort_arr, sort_index, sort_level):
    arr[i] = sort_arr[i]
    index[i] = sort_index[i]
    level[i] = sort_level[i]


@annotate(int="i, max_level", gintp="sfc, level")
def sfc_same(i, sfc, level, max_level):
    sfc[i] = ((sfc[i] + 1) << 3 * (max_level - level[i])) - 1


@annotate(int="i, max_level", gintp="sfc, level")
def sfc_real(i, sfc, level, max_level):
    sfc[i] = ((sfc[i] + 1) >> 3 * (max_level - level[i])) - 1


@annotate(i="int", gintp="sfc, duplicate_idx")
def id_duplicates(i, sfc, duplicate_idx):
    if i == 0:
        duplicate_idx[i] = 0

    if sfc[i] == sfc[i+1]:
        duplicate_idx[i+1] = 1


@annotate(i="int", x="gintp")
def map(i, x):
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

    sfc = np.zeros(N, dtype=np.int32)
    level = np.ones(N, dtype=np.int32) * max_depth

    sort_sfc = np.zeros(N, dtype=np.int32)
    sort_idx = np.zeros(N, dtype=np.int32)

    sfc_nodes = np.zeros(2*N-1, dtype=np.int32)
    level_nodes = np.zeros(2*N-1, dtype=np.int32)
    idx_nodes = np.ones(2*N-1, dtype=np.int32) * -1

    sort_sfc_nodes = np.zeros(N-1, dtype=np.int32)
    sort_level_nodes = np.zeros(N-1, dtype=np.int32)
    sort_idx_nodes = np.ones(N-1, dtype=np.int32) * -1

    duplicate_idx = np.zeros(N-1, dtype=np.int32)
    sort_duplicate_idx = np.zeros(N-1, dtype=np.int32)

    idx, sfc, sort_idx, sort_sfc, level, sfc_nodes,\
        level_nodes, idx_nodes, sort_sfc_nodes, sort_level_nodes, \
        sort_idx_nodes, duplicate_idx = wrap(
            idx, sfc, sort_idx, sort_sfc, level, sfc_nodes, level_nodes,
            idx_nodes, sort_sfc_nodes, sort_level_nodes, sort_idx_nodes,
            duplicate_idx, backend=backend)

    eget_particle_index = Elementwise(get_particle_index, backend=backend)

    ecpy_idx_tree = Elementwise(cpy_idx_tree, backend=backend)
    einternal_nodes = Elementwise(internal_nodes, backend=backend)

    ereverse_arrs = Elementwise(reverse_arrs, backend=backend)

    eswap_arrs_two = Elementwise(swap_arrs_two, backend=backend)
    eswap_arrs_three = Elementwise(swap_arrs_three, backend=backend)

    esfc_same = Elementwise(sfc_same, backend=backend)
    esfc_real = Elementwise(sfc_real, backend=backend)

    eid_duplicates = Elementwise(id_duplicates, backend=backend)
    n_duplicates = Reduction('a+b', map_func=map, backend=backend)

    eget_particle_index(sfc, particle_pos[0], particle_pos[1],
                        particle_pos[2], max_index, length)

    [sort_sfc, sort_idx], _ = radix_sort([sfc, idx], backend=backend)
    eswap_arrs_two(sfc, idx, sort_sfc, sort_idx)

    ecpy_idx_tree(sfc, level, idx, sfc_nodes, level_nodes, idx_nodes)
    einternal_nodes(sfc[:-1], sfc[1:], level[:-1], level[1:],
                    sfc_nodes[N:], level_nodes[N:], idx_nodes[N:])

    [sort_level_nodes, sort_sfc_nodes, sort_idx_nodes], _ = radix_sort(
        [level_nodes[N:], sfc_nodes[N:], idx_nodes[N:]], backend=backend)

    ereverse_arrs(level_nodes[N:], sfc_nodes[N:], idx_nodes[N:],
                  sort_level_nodes, sort_sfc_nodes, sort_idx_nodes,
                  N-1)

    esfc_same(sfc_nodes[N:], level_nodes[N:], max_depth)

    [sort_sfc_nodes, sort_level_nodes, sort_idx_nodes], _ = radix_sort(
        [sfc_nodes[N:], level_nodes[N:], idx_nodes[N:]], backend=backend)

    eswap_arrs_three(sfc_nodes[N:], level_nodes[N:], idx_nodes[N:],
                     sort_sfc_nodes, sort_level_nodes, sort_idx_nodes)

    eid_duplicates(sfc_nodes[N:-1], duplicate_idx)

    [sort_duplicate_idx, sort_sfc_nodes, sort_level_nodes,
     sort_idx_nodes], _ = radix_sort(
        [duplicate_idx, sfc_nodes[N:], level_nodes[N:], idx_nodes[N:]],
        backend=backend)

    eswap_arrs_three(sfc_nodes[N:], level_nodes[N:], idx_nodes[N:],
                     sort_sfc_nodes, sort_level_nodes, sort_idx_nodes)

    count_repeated = int(n_duplicates(sort_duplicate_idx)) - 1

    esfc_real(sfc_nodes[N:], level_nodes[N:], max_depth)
