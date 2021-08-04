from compyle.api import declare, Elementwise, annotate, Scan, wrap
from compyle.low_level import cast
from compyle.parallel import drop_duplicates
from compyle.sort import radix_sort
import numpy as np
from math import floor, log

backend = "cython"


@annotate(i="int", gintp="sfc, duplicate_idx")
def id_duplicates(i, sfc, duplicate_idx):
    if i == 0:
        duplicate_idx[i] = 0

    if sfc[i] == sfc[i+1]:
        duplicate_idx[i+1] = 1


sfc = np.array([1, 2, 3, 4, 4, 5, 5], dtype=np.int32)
idx = np.zeros_like(sfc)

eid_duplicates = Elementwise(id_duplicates, backend=backend)

eid_duplicates(sfc[:-1], idx)
print(idx)
