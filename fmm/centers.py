from compyle.api import annotate, Elementwise, get_config
import compyle.array as ary
import numpy as np
import time
from .tree import build


@annotate(x="int", return_="int")
def deinterleave(x):
    x = x & 0x49249249
    x = (x | (x >> 2)) & 0xC30C30C3
    x = (x | (x >> 4)) & 0xF00F00F
    x = (x | (x >> 8)) & 0xFF0000FF
    x = (x | (x >> 16)) & 0x0000FFFF
    return x


@annotate(i="int", gintp="sfc, level",
          gfloatp="cx, cy, cz",
          double="x_min, y_min, z_min, length")
def calc_center(i, sfc, level, cx, cy, cz,
                x_min, y_min, z_min, length):
    x, y, z = declare("int", 3)
    x = deinterleave(sfc[i])
    y = deinterleave(sfc[i] >> 1)
    z = deinterleave(sfc[i] >> 2)

    cx[i] = x_min + length*(x + 0.5)/(2.0 ** level[i])
    cy[i] = y_min + length*(y + 0.5)/(2.0 ** level[i])
    cz[i] = z_min + length*(z + 0.5)/(2.0 ** level[i])


if __name__ == "__main__":
    backend = 'cython'
    N = 10
    max_depth = 2
    x_min = 0
    y_min = 0
    z_min = 0
    length = 1
    np.random.seed(4)
    part_x = np.random.random(N)*length + x_min
    part_y = np.random.random(N)*length + y_min
    part_z = np.random.random(N)*length + z_min

    cells, sfc, level, idx, parent, child = build(
        N, max_depth, part_x, part_y, part_z, x_min,
        y_min, z_min, length, backend)

    cx = ary.zeros(cells, dtype=np.float64, backend=backend)
    cy = ary.zeros(cells, dtype=np.float64, backend=backend)
    cz = ary.zeros(cells, dtype=np.float64, backend=backend)

    e = Elementwise(calc_center, backend=backend)
    e(sfc, level, cx, cy, cz, x_min, y_min, z_min, length)
