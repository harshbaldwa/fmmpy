from compyle.api import annotate, Elementwise, get_config
import compyle.array as ary
import numpy as np
import time
from .tree import build
from .spherical_points import spherical_points


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


@annotate(int="i, num_p2", level="gintp",
          length="double",
          gfloatp="cx, cy, cz, out_x, out_y, out_z, "
                  "in_x, in_y, in_z, sph_pts")
def setting_p2(i, cx, cy, cz, out_x, out_y,
               out_z, in_x, in_y, in_z, sph_pts,
               length, level, num_p2):
    j = declare("int")
    sz_cell = declare("double")
    sz_cell = length/(2.0**(level[i]+1))
    for j in range(num_p2):
        out_x[i*num_p2+j] = cx[i]+sph_pts[3*j]*3*sz_cell
        out_y[i*num_p2+j] = cy[i]+sph_pts[3*j+1]*3*sz_cell
        out_z[i*num_p2+j] = cz[i]+sph_pts[3*j+2]*3*sz_cell
        in_x[i*num_p2+j] = cx[i]+sph_pts[3*j]*0.5*sz_cell
        in_y[i*num_p2+j] = cy[i]+sph_pts[3*j+1]*0.5*sz_cell
        in_z[i*num_p2+j] = cz[i]+sph_pts[3*j+2]*0.5*sz_cell


def set_prob(N, max_depth, part_x, part_y, part_z, x_min,
             y_min, z_min, length, num_p2, backend):

    # TODO: collect all the declarations in a single
    #       list in a single file
    cells, sfc, level, idx, parent, child = build(
        N, max_depth, part_x, part_y, part_z, x_min,
        y_min, z_min, length, backend)

    cx = ary.zeros(cells, dtype=np.float32, backend=backend)
    cy = ary.zeros(cells, dtype=np.float32, backend=backend)
    cz = ary.zeros(cells, dtype=np.float32, backend=backend)
    out_x = ary.zeros(cells*num_p2, dtype=np.float32,
                      backend=backend)
    out_y = ary.zeros(cells*num_p2, dtype=np.float32,
                      backend=backend)
    out_z = ary.zeros(cells*num_p2, dtype=np.float32,
                      backend=backend)
    out_vl = ary.zeros(cells*num_p2, dtype=np.float32,
                       backend=backend)
    in_x = ary.zeros(cells*num_p2, dtype=np.float32,
                     backend=backend)
    in_y = ary.zeros(cells*num_p2, dtype=np.float32,
                     backend=backend)
    in_z = ary.zeros(cells*num_p2, dtype=np.float32,
                     backend=backend)
    in_vl = ary.zeros(cells*num_p2, dtype=np.float32,
                      backend=backend)

    sph_pts, order = spherical_points(N)

    ecalc_center = Elementwise(calc_center, backend=backend)
    esetting_p2 = Elementwise(setting_p2, backend=backend)

    ecalc_center(sfc, level, cx, cy, cz,
                 x_min, y_min, z_min, length)
    esetting_p2(cx, cy, cz, out_x, out_y, out_z, in_x,
                in_y, in_z, sph_pts, length, level, num_p2)

    return sfc, level, idx, parent, child, cx, cy, cz, \
        out_x, out_y, out_z, out_vl, in_x, in_y, in_z, \
        in_vl, sph_pts, order


if __name__ == "__main__":
    backend = 'cython'
    N = 10
    max_depth = 2
    x_min = 0
    y_min = 0
    z_min = 0
    length = 1
    num_p2 = 6
    np.random.seed(4)
    part_x = np.random.random(N)*length + x_min
    part_y = np.random.random(N)*length + y_min
    part_z = np.random.random(N)*length + z_min

    sfc, level, idx, parent, child, cx, cy, cz, \
        out_x, out_y, out_z, out_vl, in_x, in_y, \
        in_z,  in_vl, sph_pts, order = set_prob(
            N, max_depth, part_x, part_y, part_z,
            x_min, y_min, z_min, length, num_p2,
            backend)