import pickle
import time

import compyle.array as ary
import numpy as np
from compyle.api import Elementwise, Scan, annotate, get_config
from compyle.low_level import atomic_inc
from compyle.sort import radix_sort

from .tree import ReverseArrays, build


@annotate(i="int", lev_nr="gintp", return_="int")
def input_expr(i, lev_nr):
    if i == 0:
        return 0
    else:
        return lev_nr[i - 1]


@annotate(int="i, item", lev_cs="gintp")
def output_expr(i, item, lev_cs):
    lev_cs[i] = item


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


@annotate(i="int", gintp="level, lev_n")
def level_info(i, level, lev_n):
    idx = declare("int")
    idx = atomic_inc(lev_n[level[i]])


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

    # TODO: Have to check everything in this function
    index = ary.arange(0, cells, 1, dtype=np.int32,
                       backend=backend)
    index_t2 = ary.empty(cells, dtype=np.int32,
                         backend=backend)
    index_r = ary.zeros(cells, dtype=np.int32,
                        backend=backend)
    index_t1 = ary.arange(0, cells, 1, dtype=np.int32,
                          backend=backend)

    level_ls = ary.zeros(cells, dtype=np.int32, backend=backend)
    lev_n = ary.zeros(max_depth+1, dtype=np.int32,
                      backend=backend)
    lev_nr = ary.zeros(max_depth+1, dtype=np.int32,
                       backend=backend)
    lev_cs = ary.zeros(max_depth+1, dtype=np.int32,
                       backend=backend)

    temp_d = pickle.load(resources.open_binary(
        "fmm", "t_design.pickle"))[num_p2]
    sph_pts = temp_d['array']
    order = temp_d['order']
    sph_pts = wrap(sph_pts, backend=backend)

    ecalc_center = Elementwise(calc_center, backend=backend)
    esetting_p2 = Elementwise(setting_p2, backend=backend)
    elevel_info = Elementwise(level_info, backend=backend)
    cumsum = Scan(input_expr, output_expr, 'a+b',
                  dtype=np.int32, backend=backend)
    reverse = ReverseArrays('reverse', ['a', 'b']).function
    ereverse = Elementwise(reverse, backend=backend)

    ecalc_center(sfc, level, cx, cy, cz,
                 x_min, y_min, z_min, length)
    esetting_p2(cx, cy, cz, out_x, out_y, out_z, in_x,
                in_y, in_z, sph_pts, length, level, num_p2)

    # TODO: Can do this in tree file as well, would be better
    [level_ls, index], _ = radix_sort(
        [level, index], backend=backend)

    [index_t2, index_r], _ = radix_sort(
        [index, index_t1], backend=backend)

    elevel_info(level_ls, lev_n)
    ereverse(lev_nr, lev_n, max_depth)
    cumsum(lev_nr=lev_nr, lev_cs=lev_cs)

    # TODO: Is level_ls needed for further implementation ?
    return index, level_ls, sfc, level, idx, index_r, \
        parent, child,  cx, cy, cz, out_x, out_y, out_z, \
        out_vl, in_x, in_y, in_z, in_vl, sph_pts, order, lev_cs
