import pytest

from ..tree import *
from compyle.api import wrap


backend = 'cython'


def test_copy_arr():
    arr = ary.ones(10, dtype=np.int32, backend=backend)
    arr2 = ary.empty(10, dtype=np.int32, backend=backend)

    copy_value = CopyValue('copy_value', ['a', 'b']).function
    e = Elementwise(copy_value, backend=backend)
    e(arr, arr2)
    np.testing.assert_array_equal(arr, arr2)


def test_reverse_arr():
    arr_org = np.array([1, 2, 4], dtype=np.int32)
    arr_result = np.array([4, 2, 1], dtype=np.int32)
    arr_org, arr_result = wrap(arr_org, arr_result, backend=backend)
    arr_empty = ary.empty(3, dtype=np.int32, backend=backend)

    reverse = ReverseArray('reverse', ['a', 'b']).function
    e = Elementwise(reverse, backend=backend)

    e(arr_empty, arr_org, 3)
    np.testing.assert_array_equal(arr_result, arr_empty)


def test_internal_nodes():
    sfc = np.array([52, 53], dtype=np.int32)
    level = np.array([2, 2], dtype=np.int32)
    r_lca_sfc = np.array([6], dtype=np.int32)
    r_lca_level = np.array([1], dtype=np.int32)
    r_lca_idx = np.array([-1], dtype=np.int32)
    (sfc, level, r_lca_sfc, r_lca_level,
     r_lca_idx) = wrap(sfc, level, r_lca_sfc,
                       r_lca_level, r_lca_idx,
                       backend=backend)

    lca_sfc = ary.empty(1, dtype=np.int32, backend=backend)
    lca_level = ary.empty(1, dtype=np.int32, backend=backend)
    lca_idx = ary.empty(1, dtype=np.int32, backend=backend)

    e = Elementwise(internal_nodes, backend=backend)
    e(sfc[:-1], sfc[1:], level[:-1], level[1:], lca_sfc, lca_level, lca_idx)

    (np.testing.assert_array_equal(r_lca_sfc, lca_sfc) and
     np.testing.assert_array_equal(r_lca_level, lca_level) and
     np.testing.assert_array_equal(r_lca_idx, lca_idx))


def test_parent_child():
    sfc = np.array([52, 53], dtype=np.int32)
    level = np.array([2, 2], dtype=np.int32)
    r_lca_sfc = np.array([6], dtype=np.int32)
    r_lca_level = np.array([1], dtype=np.int32)
    r_lca_idx = np.array([-1], dtype=np.int32)
    r_child_sfc = np.array([52], dtype=np.int32)
    r_child_idx = np.array([0], dtype=np.int32)
    (sfc, level, r_lca_sfc, r_lca_level, r_lca_idx, r_child_sfc,
     r_child_idx) = wrap(sfc, level, r_lca_sfc, r_lca_level,
                         r_lca_idx, r_child_sfc, r_child_idx,
                         backend=backend)

    lca_sfc = ary.empty(1, dtype=np.int32, backend=backend)
    lca_level = ary.empty(1, dtype=np.int32, backend=backend)
    lca_idx = ary.empty(1, dtype=np.int32, backend=backend)
    child_sfc = ary.empty(1, dtype=np.int32, backend=backend)
    child_idx = ary.empty(1, dtype=np.int32, backend=backend)

    e = Elementwise(parent_child, backend=backend)
    e(sfc[:-1], sfc[1:], level[:-1], level[1:],
      lca_sfc, lca_level, lca_idx, child_sfc, child_idx)

    np.testing.assert_array_equal(r_lca_sfc, lca_sfc)
    np.testing.assert_array_equal(r_lca_level, lca_level)
    np.testing.assert_array_equal(r_lca_idx, lca_idx)
    np.testing.assert_array_equal(r_child_sfc, child_sfc)
    np.testing.assert_array_equal(r_child_idx, child_idx)


def test_sfc_same():
    sfc = np.array([52, 6], dtype=np.int32)
    level = np.array([2, 1], dtype=np.int32)
    r_sfc = np.array([52, 55], dtype=np.int32)
    sfc, level, r_sfc = wrap(
        sfc, level, r_sfc, backend=backend)
    max_level = 2

    e = Elementwise(sfc_same, backend=backend)
    e(sfc, level, max_level)

    np.testing.assert_array_equal(r_sfc, sfc)


def test_sfc_real():
    sfc = np.array([52, 55], dtype=np.int32)
    level = np.array([2, 1], dtype=np.int32)
    cpy_sfc = np.array([52, 6], dtype=np.int32)
    sfc, level, cpy_sfc = wrap(
        sfc, level, cpy_sfc, backend=backend)
    max_level = 2

    e = Elementwise(sfc_real, backend=backend)
    e(sfc, level, max_level)

    np.testing.assert_array_equal(cpy_sfc, sfc)


def test_id_duplicates():
    sfc = np.array([52, 52, 53], dtype=np.int32)
    level = np.array([2, 2, 2], dtype=np.int32)
    r_duplicate_idx = np.array([0, 1], dtype=np.int32)

    sfc, level, r_duplicate_idx = wrap(
        sfc, level, r_duplicate_idx, backend=backend)

    duplicate_idx = ary.empty(2, dtype=np.int32, backend=backend)

    e = Elementwise(id_duplicates, backend=backend)
    e(sfc, level, duplicate_idx)

    np.testing.assert_array_equal(r_duplicate_idx, duplicate_idx)


def test_remove_duplicates():
    sfc = np.array([52, 52, 53], dtype=np.int32)
    level = np.array([2, 2, 2], dtype=np.int32)
    duplicate_idx = np.array([0, 1], dtype=np.int32)
    r_sfc = np.array([52, 0, 53], dtype=np.int32)
    r_level = np.array([2, -1, 2], dtype=np.int32)

    sfc, level, duplicate_idx, r_sfc, r_level = wrap(
        sfc, level, duplicate_idx, r_sfc, r_level, backend=backend)

    e = Elementwise(remove_duplicates, backend=backend)
    e(duplicate_idx, sfc, level)

    np.testing.assert_array_equal(r_sfc, sfc)
    np.testing.assert_array_equal(r_level, level)
