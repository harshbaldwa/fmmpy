import pytest
from fmm.fmm import *

check_all_backends = pytest.mark.parametrize('backend',
                                             ['cython', 'opencl'])


def check_import(backend):
    if backend == 'opencl':
        pytest.importorskip('pyopencl')


@check_all_backends
def test_lgndre(backend):
    check_import(backend)
    lst = legendre(2)
    cos_g = 0.5
    lst_len = 3
    result = lgndre(lst, cos_g, lst_len, 0)

    assert result == -0.125


@check_all_backends
def test_calc_p2_fine(backend):
    check_import(backend)
    max_depth = 1
    out_x = np.array([1, -0.5, 0.25, 0.25, 0.25, 0.25, 1.5, 0,
                     0.75, 0.75, 0.75, 0.75], dtype=np.float32)
    out_y = np.array([0.25, 0.25, 1, -0.5, 0.25, 0.25, 0.75,
                     0.75, 1.5, 0, 0.75, 0.75], dtype=np.float32)
    out_z = np.array([0.25, 0.25, 0.25, 0.25, 1, -0.5, 0.75,
                     0.75, 0.75, 0.75, 1.5, 0], dtype=np.float32)
    part_val = np.array([1.0, 3.0], dtype=np.float32)
    part_x = np.array([0.3, 0.8], dtype=np.float32)
    part_y = np.array([0.25, 0.75], dtype=np.float32)
    part_z = np.array([0.25, 0.75], dtype=np.float32)
    cx = np.array([0.25, 0.75], dtype=np.float32)
    cy = np.array([0.25, 0.75], dtype=np.float32)
    cz = np.array([0.25, 0.75], dtype=np.float32)
    num_p2 = 6
    length = 1.0
    index = np.array([0, 1], dtype=np.int32)
    leg_lim = 2
    # TODO: Fix this, use legendre function
    leg_lst = np.array([0, 1], dtype=np.float32)
    idx = np.array([0, 1], dtype=np.int32)

    r_out_val = np.array([3/15, 2/15, 1/6, 1/6, 1/6, 1/6,
                         0.6, 0.4, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    out_x, out_y, out_z, part_val, part_x, part_y, part_z, r_out_val, \
        cx, cy, cz, index, leg_lst, idx = wrap(
            out_x, out_y, out_z, part_val, part_x, part_y, part_z, r_out_val,
            cx, cy, cz, index, leg_lst, idx, backend=backend)

    out_val = ary.zeros(2*num_p2, dtype=np.float32, backend=backend)

    e = Elementwise(calc_p2_fine, backend=backend)
    e(out_val, out_x, out_y, out_z, part_val, part_x, part_y, part_z,
      cx, cy, cz, num_p2, length, index, leg_lim, leg_lst, max_depth, idx)

    np.testing.assert_array_almost_equal(r_out_val, out_val)


@check_all_backends
def test_calc_p2(backend):
    check_import(backend)
    out_x = np.array([1, -0.5, 0.25, 0.25, 0.25, 0.25, 1.5, 0,
                     0.75, 0.75, 0.75, 0.75], dtype=np.float32)
    out_y = np.array([0.25, 0.25, 1, -0.5, 0.25, 0.25, 0.75,
                     0.75, 1.5, 0, 0.75, 0.75], dtype=np.float32)
    out_z = np.array([0.25, 0.25, 0.25, 0.25, 1, -0.5, 0.75,
                     0.75, 0.75, 0.75, 1.5, 0], dtype=np.float32)
    outc_val = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        dtype=np.float32)
    outc_x = np.array([0.5, -0.25, 0.125, 0.125, 0.125, 0.125,
                      1.25, 0.5, 0.875, 0.875, 0.875, 0.875],
                      dtype=np.float32)
    outc_y = np.array([0.125, 0.125, 0.5, -0.25, 0.125, 0.125,
                      0.875, 0.875, 1.25, 0.5, 0.875, 0.875],
                      dtype=np.float32)
    outc_z = np.array([0.125, 0.125, 0.125, 0.125, 0.5, -0.25,
                      0.875, 0.875, 0.875, 0.875, 1.25, 0.5],
                      dtype=np.float32)
    cx = np.array([0.125, 0.25, 0.875, 0.75, 0.5],
                  dtype=np.float32)
    cy = np.array([0.125, 0.25, 0.875, 0.75, 0.5],
                  dtype=np.float32)
    cz = np.array([0.125, 0.25, 0.875, 0.75, 0.5],
                  dtype=np.float32)
    num_p2 = 6
    length = 1.0
    index = np.array([0, 2, 1, 3, 4], dtype=np.int32)
    index_r = np.array([0, 2, 1, 3, 4], dtype=np.int32)
    leg_lim = 2
    # TODO: Fix this, use legendre function
    leg_lst = np.array([0, 1], dtype=np.float32)
    child = np.ones(40, dtype=np.int32) * -1
    child[8] = 0
    child[24] = 2
    child[32] = 1
    child[33] = 3
    level = 1
    level_cs = np.array([4, 2, 0], dtype=np.int32)
    r_out_val = np.array([0.5, 1.5, 0.5, 1.5, 0.5, 1.5,
                         1.5, 0.5, 1.5, 0.5, 1.5, 0.5, ],
                         dtype=np.float32)
    out_val = np.zeros(12, dtype=np.float32)

    out_x, out_y, out_z, outc_val, outc_x, outc_y, outc_z, \
        r_out_val, cx, cy, cz, index, index_r, child, out_val, \
        leg_lst, level_cs = wrap(
            out_x, out_y, out_z, outc_val, outc_x, outc_y, outc_z,
            r_out_val, cx, cy, cz, index, index_r, child, out_val,
            leg_lst, level_cs, backend=backend)

    e = Elementwise(calc_p2, backend=backend)
    e(out_val, out_x, out_y, out_z, outc_val, outc_x, outc_y, outc_z,
      cx, cy, cz, num_p2, length, index, index_r, leg_lim, leg_lst,
      child, level, level_cs)

    np.testing.assert_array_almost_equal(r_out_val, out_val)


@check_all_backends
def test_is_adj(backend):
    check_import(backend)
    cx = np.array([0.375, 0.75, 0.25, 0.625], dtype=np.float32)
    cy = np.array([0.375, 0.75, 0.25, 0.125], dtype=np.float32)
    cz = np.array([0.375, 0.75, 0.25, 0.125], dtype=np.float32)
    r = np.array([0.125, 0.25, 0.25, 0.125], dtype=np.float32)

    adj1 = is_adj(cx[0], cy[0], cz[0], r[0],
                  cx[1], cy[1], cz[1], r[1])

    adj2 = is_adj(cx[3], cy[3], cz[3], r[3],
                  cx[2], cy[2], cz[2], r[2])

    adj3 = is_adj(cx[1], cy[1], cz[1], r[1],
                  cx[3], cy[3], cz[3], r[3])

    adj4 = is_adj(cx[2], cy[2], cz[2], r[2],
                  cx[1], cy[1], cz[1], r[1])

    assert adj1 == 1 and adj2 == 1 and adj3 == 0 and adj4 == 1


@check_all_backends
def test_assoc_coarse(backend):
    check_import(backend)
    sfc = np.array([1, 8, 0, 1, 0], dtype=np.int32)
    # offset = level_cs[1]
    offset = 2
    index_r = np.array([0, 2, 1, 3, 4], dtype=np.int32)
    parent = np.array([1, 4, 3, 4, -1], dtype=np.int32)
    child = np.ones(40, dtype=np.int32) * -1
    child[8] = 0
    child[24] = 2
    child[32] = 1
    child[33] = 3
    r_assoc = np.ones(108, dtype=np.int32) * -1
    r_collg = np.ones(108, dtype=np.int32) * -1
    r_assoc[55] = 3
    r_collg[55] = 1
    r_assoc[81] = 1
    r_collg[81] = 1
    sfc, index_r, parent, child, r_assoc, r_collg = wrap(
        sfc, index_r, parent, child, r_assoc, r_collg,
        backend=backend)

    assoc = ary.empty(108, dtype=np.int32, backend=backend)
    assoc.fill(-1)
    collg = ary.empty(108, dtype=np.int32, backend=backend)
    collg.fill(-1)

    eassoc_coarse = Elementwise(assoc_coarse, backend=backend)
    eassoc_coarse(sfc[2:4], parent, child, index_r, assoc, collg, offset)

    np.testing.assert_array_equal(r_assoc, assoc)
    np.testing.assert_array_equal(r_collg, collg)
