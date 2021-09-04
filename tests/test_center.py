import pytest
from compyle.api import wrap
from tree.centers import *


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
    np.testing.assert_array_equal(r_cx, cx)
    np.testing.assert_array_equal(r_cy, cy)
    np.testing.assert_array_equal(r_cz, cz)
