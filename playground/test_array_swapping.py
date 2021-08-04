from compyle.api import declare, Elementwise, annotate, Scan, wrap
from compyle.config import get_config
from compyle.low_level import cast
from compyle.sort import radix_sort
import numpy as np
from math import floor, log
import time


@annotate(
    i="int",
    gintp="arr, sort_arr"
)
def swap_arrs_three(i, arr, sort_arr):
    arr[i] = sort_arr[i]


backend = "cython"
get_config().use_openmp = True

eswap = Elementwise(swap_arrs_three, backend=backend)

a = np.random.randint(0, 100, 1000000000)
b = np.zeros_like(a)

a, b = wrap(a, b, backend=backend)

time1 = time.time()
b = a
time2 = time.time()

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
b = np.zeros_like(a)

a, b = wrap(a, b, backend=backend)

time3 = time.time()
eswap(b, a)
time4 = time.time()

print("Direct Swap - ", time2 - time1)
print("Eswap - ", time4 - time3)
