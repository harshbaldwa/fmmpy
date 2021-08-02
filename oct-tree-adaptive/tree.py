from compyle.api import declare, Elementwise, annotate, Scan
from compyle.low_level import cast, atomic_inc, atomic_dec
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


@annotate(i="int", gintp="bin_arr, cumsum_arr")
def reset_bin_arr(i, bin_arr, cumsum_arr):
    cumsum_arr[i] = 0
    bin_arr[i] = 0


@annotate(int="i, digit, radix", gintp="arr, bin_arr")
def counting_sort_one(i, arr, bin_arr, digit, radix):
    digit_arr_i, idx = declare("int", 2)
    digit_arr_i = cast(((arr[i] / radix ** digit) % radix), "int")
    idx = atomic_inc(bin_arr[digit_arr_i])


@annotate(i="int", bin_arr="gintp", return_="int")
def input_cumsum_arr(i, bin_arr):
    return bin_arr[i]


@annotate(int="i, item", cumsum_arr="gintp")
def output_cumsum_arr(i, item, cumsum_arr):
    cumsum_arr[i] = item


@annotate(int="i, radix, digit, len_arr",
          gintp="arr, cumsum_arr, sort_arr, index, sort_index"
          )
def counting_sort_two(
    i, arr, cumsum_arr, sort_arr, index, sort_index, radix, digit, len_arr
):
    digit_arr_i, j, idx = declare("int", 3)
    j = len_arr - i - 1
    digit_arr_i = cast(((arr[j] / radix ** digit) % radix), "int")
    sort_arr[cumsum_arr[digit_arr_i] - 1] = arr[j]
    sort_index[cumsum_arr[digit_arr_i] - 1] = index[j]
    idx = atomic_dec(cumsum_arr[digit_arr_i])


@annotate(
    int="i, radix, digit, len_arr",
    gintp="arr, cumsum_arr, sort_arr, index, sort_index, level, sort_level",
)
def counting_sort_three(
    i,
    arr,
    cumsum_arr,
    sort_arr,
    index,
    sort_index,
    level,
    sort_level,
    radix,
    digit,
    len_arr,
):
    digit_arr_i, j, idx = declare("int", 2)
    j = len_arr - i - 1
    digit_arr_i = cast(((arr[j] / radix ** digit) % radix), "int")
    sort_arr[cumsum_arr[digit_arr_i] - 1] = arr[j]
    sort_index[cumsum_arr[digit_arr_i] - 1] = index[j]
    sort_level[cumsum_arr[digit_arr_i] - 1] = level[j]
    idx = atomic_dec(cumsum_arr[digit_arr_i])


@annotate(
    i="int",
    gintp="arr, sort_arr, index, sort_index"
)
def swap_arrs(i, arr, sort_arr, index, sort_index):
    arr[i] = sort_arr[i]
    index[i] = sort_index[i]


@annotate(
    i="int",
    gintp="arr, sort_arr, index, sort_index, level, sort_level"
)
def swap_arrs_two(i, arr, sort_arr, index, sort_index, level, sort_level):
    arr[i] = sort_arr[i]
    index[i] = sort_index[i]
    level[i] = sort_level[i]


@annotate(
    int="i, len_arr",
    gintp="arr, sort_arr, index, sort_index, level, sort_level"
)
def reverse_arrs(i, sort_arr, arr, sort_index,
                 index, sort_level, level, len_arr):
    arr[len_arr - i - 1] = sort_arr[i]
    index[len_arr - i - 1] = sort_index[i]
    level[len_arr - i - 1] = sort_level[i]


@annotate(int="i, max_level", gintp="sfc, level")
def sfc_same(i, sfc, level, max_level):
    sfc[i] = ((sfc[i] + 1) << 3 * (max_level - level[i])) - 1


@annotate(int="i, max_level", gintp="sfc, level")
def sfc_real(i, sfc, level, max_level):
    sfc[i] = ((sfc[i] + 1) >> 3 * (max_level - level[i])) - 1


if __name__ == "__main__":

    backend = "cython"
    N = 10
    max_depth = 2
    length = 1

    # all parameters here

    np.random.seed(4)
    particle_pos = np.random.random((3, N))
    idx = np.arange(N, dtype=np.int32)
    max_index = 2 ** max_depth

    sfc = np.zeros_like(idx, dtype=np.int32)
    level = np.ones(N, dtype=np.int32) * max_depth

    radix = 10
    max_digits = int(floor(log((1 << 3*max_depth)-1, radix)) + 1)
    bin_arr = np.zeros(radix, dtype=np.int32)
    cumsum_arr = np.zeros(radix, dtype=np.int32)
    sort_sfc = np.zeros(N, dtype=np.int32)
    sort_idx = np.zeros(N, dtype=np.int32)

    full_sfc = np.zeros(2*N-1, dtype=np.int32)
    full_level = np.zeros(2*N-1, dtype=np.int32)
    full_idx = np.ones(2*N-1, dtype=np.int32) * -1

    sort_sfc_nodes = np.zeros(N-1, dtype=np.int32)
    sort_level_nodes = np.zeros(N-1, dtype=np.int32)
    sort_idx_nodes = np.zeros(N-1, dtype=np.int32)

    # defining all the parallel functions here

    eget_particle_index = Elementwise(get_particle_index, backend=backend)

    ereset_bin_arr = Elementwise(reset_bin_arr, backend=backend)
    ecounting_sort_one = Elementwise(counting_sort_one, backend=backend)
    cumsum_arr_calc = Scan(
        input_cumsum_arr, output_cumsum_arr, "a+b",
        dtype=np.int32, backend=backend
    )
    ecounting_sort_two = Elementwise(counting_sort_two, backend=backend)
    eswap_arrs = Elementwise(swap_arrs, backend=backend)

    ecpy_idx_tree = Elementwise(cpy_idx_tree, backend=backend)
    einternal_nodes = Elementwise(internal_nodes, backend=backend)

    ecounting_sort_three = Elementwise(counting_sort_three, backend=backend)
    ereverse_arrs = Elementwise(reverse_arrs, backend=backend)
    eswap_arrs_two = Elementwise(swap_arrs_two, backend=backend)
    esfc_same = Elementwise(sfc_same, backend=backend)
    esfc_real = Elementwise(sfc_real, backend=backend)

    # calculations after this, cant be in serial !!!!!

    # getting SFC index of each particle at finest level
    eget_particle_index(sfc, particle_pos[0], particle_pos[1],
                        particle_pos[2], max_index, length)

    # sorting these SFC indices using parallel radix sort
    for digit in range(max_digits):
        ereset_bin_arr(bin_arr, cumsum_arr)
        ecounting_sort_one(sfc, bin_arr, digit, radix)
        cumsum_arr_calc(bin_arr=bin_arr, cumsum_arr=cumsum_arr)
        ecounting_sort_two(sfc, cumsum_arr, sort_sfc,
                           idx, sort_idx, radix, digit, N)
        eswap_arrs(sfc, sort_sfc, idx, sort_idx)

    # now finding internal nodes
    ecpy_idx_tree(sfc, level, idx, full_sfc, full_level, full_idx)
    einternal_nodes(sfc[:-1], sfc[1:], level[:-1], level[1:],
                    full_sfc[N:], full_level[N:], full_idx[N:])

    # sorting the internal nodes using parallel radix sort

    # first we sort based on the level of the nodes
    # (assuming less than 10 levels)
    ereset_bin_arr(bin_arr, cumsum_arr)
    ecounting_sort_one(full_level[N:], bin_arr, 0, radix)
    cumsum_arr_calc(bin_arr=bin_arr, cumsum_arr=cumsum_arr)
    ecounting_sort_three(full_level[N:], cumsum_arr, sort_level_nodes,
                         full_sfc[N:], sort_sfc_nodes, full_idx[N:],
                         sort_idx_nodes, radix, 0, N-1)
    ereverse_arrs(sort_level_nodes, full_level[N:], sort_sfc_nodes,
                  full_sfc[N:], sort_idx_nodes, full_idx[N:], N-1)

    # now we make sfc indices of the nodes of same length
    esfc_same(full_sfc[N:], full_level[N:], max_depth)

    # now we sort based on the sfc index of the nodes
    for digit in range(max_digits):
        ereset_bin_arr(bin_arr, cumsum_arr)
        ecounting_sort_one(full_sfc[N:], bin_arr, digit, radix)
        cumsum_arr_calc(bin_arr=bin_arr, cumsum_arr=cumsum_arr)
        ecounting_sort_three(full_sfc[N:], cumsum_arr, sort_sfc_nodes,
                             full_idx[N:], sort_idx_nodes, full_level[N:],
                             sort_level_nodes, radix, digit, N-1)
        eswap_arrs_two(full_sfc[N:], sort_sfc_nodes, full_idx[N:],
                       sort_idx_nodes, full_level[N:], sort_level_nodes)

    # now we make sfc indices of the nodes of respective length
    esfc_real(full_sfc[N:], full_level[N:], max_depth)
