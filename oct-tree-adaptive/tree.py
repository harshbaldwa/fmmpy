from compyle.api import declare, Elementwise, annotate, Scan
from compyle.low_level import cast, atomic_inc
import numpy as np
from math import floor


@annotate(double="x, y, z, x_min, y_min, z_min, length, b_len", return_="int")
def get_particle_index(x, y, z, b_len, x_min=0.0, y_min=0.0, z_min=0.0, length=1):
    nx, ny, nz, id = declare("int", 4)
    nx = cast(floor((b_len * (x - x_min)) / length), "int")
    ny = cast(floor((b_len * (y - y_min)) / length), "int")
    nz = cast(floor((b_len * (z - z_min)) / length), "int")

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

    id = (nz << 2) | (ny << 1) | nx
    return id


@annotate(i="int", gintp="index1, index2, level1, level2, lca_index, lca_level")
def internal_nodes(i, index1, index2, level1, level2, lca_index, lca_level):
    level_diff, xor, i1, i2, l, j = declare("int", 6)
    level_diff = cast(abs(level1[i] - level2[i]), "int")

    if level1[i] - level2[i] > 0:
        i1 = index1[i]
        i2 = index2[i] << 3 * level_diff
        l = level1[i]
    elif level1[i] - level2[i] < 0:
        i1 = index1[i] << 3 * level_diff
        i2 = index2[i]
        l = level2[i]
    else:
        i1 = index1[i]
        i2 = index2[i]
        l = level1[i]

    xor = i1 ^ i2

    if xor == 0:
        lca_index[i] = i1 >> 3 * level_diff
        lca_level[i] = l - level_diff
        return

    for j in range(l + 1, 0, -1):
        if xor > ((1 << (j - 1) * 3) - 1):
            lca_index[i] = i1 >> 3 * j
            lca_level[i] = l - j
            return

    lca_index[i] = 0
    lca_level[i] = 0
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


@annotate(
    int="i, radix, digit, len_arr", gintp="arr, cumsum_arr, sort_arr, index, sort_index"
)
def counting_sort_two(
    i, arr, cumsum_arr, sort_arr, index, sort_index, radix, digit, len_arr
):
    digit_arr_i, j = declare("int", 2)
    j = len_arr - i - 1
    digit_arr_i = cast(((arr[j] / radix ** digit) % radix), "int")
    sort_arr[cumsum_arr[digit_arr_i] - 1] = arr[j]
    sort_index[cumsum_arr[digit_arr_i] - 1] = index[j]
    cumsum_arr[digit_arr_i] -= 1


@annotate(int="i, radix, digit, len_arr", gintp="arr, cumsum_arr, sort_arr, index, sort_index, level, sort_level")
def counting_sort_three(i, arr, cumsum_arr, sort_arr, index, sort_index, level, sort_level, radix, digit, len_arr):
    digit_arr_i, j = declare("int", 2)
    j = len_arr - i - 1
    digit_arr_i = cast(((arr[j] / radix ** digit) % radix), "int")
    sort_arr[cumsum_arr[digit_arr_i] - 1] = arr[j]
    printf("%d, %d\n", cumsum_arr[digit_arr_i] - 1, arr[j])
    sort_index[cumsum_arr[digit_arr_i] - 1] = index[j]
    sort_level[cumsum_arr[digit_arr_i] - 1] = level[j]
    cumsum_arr[digit_arr_i] -= 1


@annotate(i="int", gintp="arr, sort_arr, index, sort_index")
def shift_arrs(i, arr, sort_arr, index, sort_index):
    arr[i] = sort_arr[i]
    index[i] = sort_index[i]


@annotate(i="int", gintp="arr, sort_arr, index, sort_index, level, sort_level")
def shift_arrs_two(i, arr, sort_arr, index, sort_index, level, sort_level):
    arr[i] = sort_arr[i]
    index[i] = sort_index[i]
    level[i] = sort_level[i]

if __name__ == "__main__":

    backend = "cython"

    # a = np.array([377, 3016], dtype=np.int32)
    # level = np.array([3, 4], dtype=np.int32)
    # lca_index = np.zeros(1, dtype=np.int32)
    # lca_level = np.zeros(1, dtype=np.int32)

    # einternal_nodes = Elementwise(internal_nodes, backend=backend)
    # einternal_nodes(a[:-1], a[1:], level[:-1], level[1:], lca_index, lca_level)

    # print(lca_index)
    # print(lca_level)

    ereset_bin_arr = Elementwise(reset_bin_arr, backend=backend)
    ecounting_sort_one = Elementwise(counting_sort_one, backend=backend)
    cumsum_arr_calc = Scan(
        input_cumsum_arr, output_cumsum_arr, "a+b", dtype=np.int32, backend=backend
    )
    ecounting_sort_two = Elementwise(counting_sort_two, backend=backend)
    eshift_arrs = Elementwise(shift_arrs, backend=backend)

    # arr = np.array([54, 34, 21, 65, 23, 47, 12, 17, 54], dtype=np.int32)
    # len_arr = len(arr)
    # index = np.arange(len_arr, dtype=np.int32)
    # radix = 10
    # digits = 2
    # sort_arr = np.zeros(len_arr, dtype=np.int32)
    # sort_index = np.zeros(len_arr, dtype=np.int32)
    # bin_arr = np.zeros(radix, dtype=np.int32)
    # cumsum_arr = np.zeros(radix, dtype=np.int32)

    # for digit in range(digits):
    #     ereset_bin_arr(bin_arr, cumsum_arr)
    #     ecounting_sort_one(arr, bin_arr, digit, radix)
    #     cumsum_arr_calc(bin_arr=bin_arr, cumsum_arr=cumsum_arr)
    #     ecounting_sort_two(arr, cumsum_arr, sort_arr, index, sort_index, radix, digit, len_arr)
    #     eshift_arrs(arr, sort_arr, index, sort_index)

    # print(sort_arr)
    # print(sort_index)

    ecounting_sort_three = Elementwise(counting_sort_three, backend=backend)
    eshift_arrs_two = Elementwise(shift_arrs_two, backend=backend)

    index_arr = np.array([9, 2, 11, 3, 1], dtype=np.int32)
    level_arr = np.array([3, 2, 3, 1, 2], dtype=np.int32)
    u_index_arr = np.zeros_like(index_arr, dtype=np.int32)
    max_level = np.max(level_arr)

    for i in range(len(u_index_arr)):
        u_index_arr[i] = ((index_arr[i] + 1) << 3 * (max_level - level_arr[i])) - 1

    sort_index_arr = np.zeros_like(index_arr, dtype=np.int32)
    sort_level_arr = np.zeros_like(level_arr, dtype=np.int32)
    sort_u_index_arr = np.zeros_like(u_index_arr, dtype=np.int32)
    bin_arr = np.zeros_like(index_arr, dtype=np.int32)
    cumsum_arr = np.zeros_like(index_arr, dtype=np.int32)
    radix = 10
    len_arr = len(index_arr)

    ereset_bin_arr(bin_arr, cumsum_arr)
    ecounting_sort_one(level_arr, bin_arr, 0, radix)
    cumsum_arr_calc(bin_arr=bin_arr, cumsum_arr=cumsum_arr)
    ecounting_sort_three(level_arr, cumsum_arr, sort_level_arr[::-1], index_arr, sort_index_arr[::-1], u_index_arr, sort_u_index_arr[::-1], radix, 0, len_arr)
    eshift_arrs_two(level_arr, sort_level_arr, index_arr, sort_index_arr, u_index_arr, sort_u_index_arr)

    print(sort_level_arr)
    # print(sort_index_arr)
    # print(sort_u_index_arr)