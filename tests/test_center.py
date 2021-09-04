import pytest
from compyle.api import wrap
from fmm.centers import *
from fmm.spherical_points import *

check_all_backends = pytest.mark.parametrize('backend',
                                             ['cython', 'opencl'])


def check_import(backend):
    if backend == 'opencl':
        pytest.importorskip('pyopencl')


@check_all_backends
def test_deinterleave(backend):
    check_import(backend)
    idx = 38
    x = deinterleave(idx)
    y = deinterleave(idx >> 1)
    z = deinterleave(idx >> 2)
    print(x)
    print(y)
    print(z)
    assert (x == 0) and (y == 1) and (z == 3)


@check_all_backends
def test_calc_center(backend):
    check_import(backend)
    x_min = 0
    y_min = 0
    z_min = 0
    length = 1
    sfc = np.array([0, 0], dtype=np.int32)
    level = np.array([1, 0], dtype=np.int32)
    r_cx = np.array([0.25, 0.5], dtype=np.float32)
    r_cy = np.array([0.25, 0.5], dtype=np.float32)
    r_cz = np.array([0.25, 0.5], dtype=np.float32)
    sfc, level, r_cx, r_cy, r_cz = wrap(
        sfc, level, r_cx, r_cy, r_cz, backend=backend)
    cx = ary.zeros(2, dtype=np.float32, backend=backend)
    cy = ary.zeros(2, dtype=np.float32, backend=backend)
    cz = ary.zeros(2, dtype=np.float32, backend=backend)

    e = Elementwise(calc_center, backend=backend)
    e(sfc, level, cx, cy, cz, x_min, y_min, z_min, length)
    print(cx)
    np.testing.assert_array_almost_equal(r_cx, cx)
    np.testing.assert_array_almost_equal(r_cy, cy)
    np.testing.assert_array_almost_equal(r_cz, cz)


@check_all_backends
def test_setting_p2(backend):
    check_import(backend)
    level = np.ones(1, dtype=np.int32)
    cx = np.array([0.25], dtype=np.float32)
    cy = np.array([0.25], dtype=np.float32)
    cz = np.array([0.25], dtype=np.float32)
    num_p2 = 6
    length = 1
    sph_pts, _ = spherical_points(num_p2)
    r_out_x = np.array([1, -0.5, 0.25, 0.25, 0.25, 0.25],
                       dtype=np.float32)
    r_out_y = np.array([0.25, 0.25, 1, -0.5, 0.25, 0.25],
                       dtype=np.float32)
    r_out_z = np.array([0.25, 0.25, 0.25, 0.25, 1, -0.5],
                       dtype=np.float32)
    r_in_x = np.array([0.375, 0.125, 0.25, 0.25, 0.25, 0.25],
                      dtype=np.float32)
    r_in_y = np.array([0.25, 0.25, 0.375, 0.125, 0.25, 0.25],
                      dtype=np.float32)
    r_in_z = np.array([0.25, 0.25, 0.25, 0.25, 0.375, 0.125],
                      dtype=np.float32)

    cx, cy, cz, r_out_x, r_out_y, r_out_z, r_in_x, r_in_y, \
        r_in_z, level, sph_pts = wrap(
            cx, cy, cz, r_out_x, r_out_y, r_out_z,
            r_in_x, r_in_y, r_in_z, level, sph_pts,
            backend=backend)

    out_x = ary.zeros(num_p2, dtype=np.float32, backend=backend)
    out_y = ary.zeros(num_p2, dtype=np.float32, backend=backend)
    out_z = ary.zeros(num_p2, dtype=np.float32, backend=backend)
    in_x = ary.zeros(num_p2, dtype=np.float32, backend=backend)
    in_y = ary.zeros(num_p2, dtype=np.float32, backend=backend)
    in_z = ary.zeros(num_p2, dtype=np.float32, backend=backend)

    e = Elementwise(setting_p2, backend=backend)
    e(cx, cy, cz, out_x, out_y, out_z, in_x,
      in_y, in_z, sph_pts, length, level, num_p2)

    np.testing.assert_array_almost_equal(r_out_x, out_x)
    np.testing.assert_array_almost_equal(r_out_y, out_y)
    np.testing.assert_array_almost_equal(r_out_z, out_z)
    np.testing.assert_array_almost_equal(r_in_x, in_x)
    np.testing.assert_array_almost_equal(r_in_y, in_y)
    np.testing.assert_array_almost_equal(r_in_z, in_z)
