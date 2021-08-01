from compyle.low_level import atomic_dec
from compyle.api import annotate, declare, Elementwise
import numpy as np

backend = 'cython'


@annotate(i="int", a="gintp")
def atomic_dec_i(i, a):
    idx = declare("int")
    idx = atomic_dec(a[i])


a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
eatomic = Elementwise(atomic_dec_i, backend="cython")
eatomic(a)

print(a)