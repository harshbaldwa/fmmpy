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
    level_cs = np.array([0, 2, 4], dtype=np.int32)
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
