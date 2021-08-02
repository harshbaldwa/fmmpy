from compyle.api import declare, Elementwise, annotate, Scan, get_config
from compyle.low_level import cast, atomic_inc, atomic_dec
import numpy as np
from math import floor, log
from time import time


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
          gintp="arr, cumsum_arr, sort_arr"
          )
def counting_sort_two(
    i, arr, cumsum_arr, sort_arr, radix, digit, len_arr
):
    digit_arr_i, j, idx = declare("int", 3)
    j = len_arr - i - 1
    digit_arr_i = cast(((arr[j] / radix ** digit) % radix), "int")
    sort_arr[cumsum_arr[digit_arr_i] - 1] = arr[j]
    idx = atomic_dec(cumsum_arr[digit_arr_i])


@annotate(
    i="int",
    gintp="arr, sort_arr"
)
def swap_arrs(i, arr, sort_arr):
    arr[i] = sort_arr[i]


backend = 'cython'
get_config().use_openmp = True

ereset_bin_arr = Elementwise(reset_bin_arr, backend=backend)
ecounting_sort_one = Elementwise(counting_sort_one, backend=backend)
cumsum_arr_calc = Scan(
    input_cumsum_arr, output_cumsum_arr, "a+b",
    dtype=np.int32, backend=backend
)
ecounting_sort_two = Elementwise(counting_sort_two, backend=backend)
eswap_arrs = Elementwise(swap_arrs, backend=backend)

radix = 10
N = 1000000
a = np.random.randint(24, 300, N, dtype=np.int32)
sort_a = np.zeros_like(a)
bin_arr = np.zeros(radix, dtype=np.int32)
cumsum_arr = np.zeros(radix, dtype=np.int32)

# print(a)
t1 = time()

for digit in range(3):
    ereset_bin_arr(bin_arr, cumsum_arr)
    ecounting_sort_one(a, bin_arr, digit, radix)
    cumsum_arr_calc(bin_arr=bin_arr, cumsum_arr=cumsum_arr)
    ecounting_sort_two(a, cumsum_arr, sort_a, radix, digit, N)
    eswap_arrs(a, sort_a)

t2 = time()

print(t2-t1)
# print(a)
