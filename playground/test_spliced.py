from compyle.api import Elementwise, annotate
import numpy as np
from numpy.lib.function_base import bartlett


@annotate(int="i, b", a="gintp")
def func(i, a, b):
    a[i] = b


backend = 'cython'

a = np.zeros(10, dtype=np.int32)
b = 2
efunc = Elementwise(func, backend=backend)
efunc(a[5:], b)

print(a)
