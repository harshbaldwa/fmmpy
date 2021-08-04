from compyle.api import declare, Elementwise, annotate, Scan, get_config, wrap
from compyle.low_level import cast, atomic_inc, atomic_dec
import numpy as np
from math import floor, log
from time import time
from compyle.sort import radix_sort


backend = 'cython'
# get_config().use_openmp = True

N = 10
a = np.random.randint(-9, 9, N, dtype=np.int32)
# b = np.arange(N, dtype=np.int32)
a = wrap(a, backend=backend)

print(a)

vicky = time()
t1, t2 = radix_sort([a], backend=backend)
harsh = time()

print(t1[0])
