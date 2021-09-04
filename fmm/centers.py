from compyle.api import annotate, Elementwise, get_config
import compyle.array as ary
import numpy as np
import time


@annotate(x="int", return_="int")
def deinterleave(x):
    x = x & 0x49249249
    x = (x | (x >> 2)) & 0xC30C30C3
    x = (x | (x >> 4)) & 0xF00F00F
    x = (x | (x >> 8)) & 0xFF0000FF
    x = (x | (x >> 16)) & 0x0000FFFF
    return x


@annotate(int="i, offset, level", idx="gintp",
          gdoublep="cx, cy, cz",
          double="x_min, y_min, z_min, length")
def calc_center(i, idx, cx, cy, cz, level, offset,
                x_min, y_min, z_min, length):
    x, y, z = declare("int", 3)
    x = deinterleave(idx[i])
    y = deinterleave(idx[i] >> 1)
    z = deinterleave(idx[i] >> 2)
    cx[offset+idx[i]] = x_min + length*(x+0.5)/2**level
    cy[offset+idx[i]] = y_min + length*(y+0.5)/2**level
    cz[offset+idx[i]] = z_min + length*(z+0.5)/2**level


if __name__ == "__main__":
    backend = 'cython'
    LEVEL = 8

    cells = 8*(8**LEVEL-1)//7
    x_min = 0
    y_min = 0
    z_min = 0
    length = 1

    cx = ary.zeros(cells, dtype=np.float64, backend=backend)
    cy = ary.zeros(cells, dtype=np.float64, backend=backend)
    cz = ary.zeros(cells, dtype=np.float64, backend=backend)

    e = Elementwise(calc_center, backend=backend)

    offset = 0

    for level in range(LEVEL, 0, -1):
        time1 = time.time()
        idx = ary.arange(0, 8**level, 1,
                         dtype=np.int32, backend=backend)
        e(idx, cx, cy, cz, level, offset,
          x_min, y_min, z_min, length)
        time2 = time.time()
        print("time taken for {} level = {:9f}s".format(level, time2-time1))
        offset = (8**level) * (8**(LEVEL-level+1) - 1) // 7
