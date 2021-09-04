import pytest

from fmm.tree import *


check_all_backends = pytest.mark.parametrize('backend',
                                             ['cython', 'opencl'])


def check_import(backend):
    if backend == 'opencl':
        pytest.importorskip('pyopencl')


@check_all_backends
def test_get_particle_index(backend):
    check_import(backend)
    max_depth = 1
    length = 1
    x_min = 0
    y_min = 0
    z_min = 0
    x = np.array([0.125, 0.125, 0.875, 0.875])
    y = np.array([0.125, 0.875, 0.875, 0.875])
    z = np.array([0.875, 0.125, 0.125, 0.875])
    r_index = np.array([4, 2, 3, 7], dtype=np.int32)
    max_index = 2 ** max_depth
    x, y, z, r_index = wrap(x, y, z, r_index, backend=backend)
    index = ary.zeros(4, dtype=np.int32, backend=backend)

    e = Elementwise(get_particle_index, backend=backend)
    e(index, x, y, z, max_index, length, x_min, y_min, z_min)
    np.testing.assert_array_equal(r_index, index)


@check_all_backends
def test_copy_arr(backend):
    check_import(backend)
    arr = ary.ones(10, dtype=np.int32, backend=backend)
    arr2 = ary.empty(10, dtype=np.int32, backend=backend)

    copy_arrays = CopyArrays('copy_arrays', ['a', 'b']).function
    e = Elementwise(copy_arrays, backend=backend)
    e(arr, arr2)
    np.testing.assert_array_equal(arr, arr2)


@check_all_backends
def test_reverse_arr(backend):
    check_import(backend)
    arr_org = np.array([1, 2, 4], dtype=np.int32)
    arr_result = np.array([4, 2, 1], dtype=np.int32)
    arr_org, arr_result = wrap(arr_org, arr_result, backend=backend)
    arr_empty = ary.empty(3, dtype=np.int32, backend=backend)

    reverse = ReverseArrays('reverse', ['a', 'b']).function
    e = Elementwise(reverse, backend=backend)

    e(arr_empty, arr_org, 3)
    np.testing.assert_array_equal(arr_result, arr_empty)


@check_all_backends
def test_internal_nodes(backend):
    check_import(backend)
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


@check_all_backends
def test_find_parents(backend):
    check_import(backend)
    sfc = np.array([52, 53], dtype=np.int32)
    level = np.array([2, 2], dtype=np.int32)
    all_idx = np.arange(2, dtype=np.int32)
    r_lca_sfc = np.array([6], dtype=np.int32)
    r_lca_level = np.array([1], dtype=np.int32)
    r_lca_idx = np.array([-1], dtype=np.int32)
    r_child_idx = np.array([0], dtype=np.int32)
    (sfc, level, r_lca_sfc, r_lca_level, r_lca_idx, all_idx,
     r_child_idx) = wrap(sfc, level, r_lca_sfc, r_lca_level,
                         r_lca_idx, all_idx, r_child_idx,
                         backend=backend)

    lca_sfc = ary.empty(1, dtype=np.int32, backend=backend)
    lca_level = ary.empty(1, dtype=np.int32, backend=backend)
    lca_idx = ary.empty(1, dtype=np.int32, backend=backend)
    child_idx = ary.empty(1, dtype=np.int32, backend=backend)

    e = Elementwise(find_parents, backend=backend)
    e(sfc[:-1], sfc[1:], level[:-1], level[1:], all_idx,
      lca_sfc, lca_level, lca_idx, child_idx)

    np.testing.assert_array_equal(r_lca_sfc, lca_sfc)
    np.testing.assert_array_equal(r_lca_level, lca_level)
    np.testing.assert_array_equal(r_lca_idx, lca_idx)
    np.testing.assert_array_equal(r_child_idx, child_idx)


@check_all_backends
def test_get_relations(backend):
    check_import(backend)
    pc_sfc = np.array([2, 0, 0], dtype=np.int32)
    pc_level = np.array([1, 0, 0], dtype=np.int32)
    temp_idx = np.array([-1, 0, -1], dtype=np.int32)
    rel_idx = np.array([0, -1, 1], dtype=np.int32)
    r_parent_idx = np.array([1, -1], dtype=np.int32)
    r_child_idx = np.ones(16, dtype=np.int32) * -1
    r_child_idx[8] = 0

    (pc_sfc, pc_level, temp_idx, rel_idx, r_parent_idx,
     r_child_idx) = wrap(pc_sfc, pc_level, temp_idx, rel_idx,
                         r_parent_idx, r_child_idx, backend=backend)

    parent_idx = ary.empty(2, dtype=np.int32, backend=backend)
    child_idx = ary.empty(16, dtype=np.int32, backend=backend)
    parent_idx.fill(-1)
    child_idx.fill(-1)

    e = Elementwise(get_relations, backend=backend)
    e(pc_sfc, pc_level, temp_idx, rel_idx, parent_idx, child_idx)

    np.testing.assert_array_equal(r_parent_idx, parent_idx)
    np.testing.assert_array_equal(r_child_idx, child_idx)


@check_all_backends
def test_sfc_same(backend):
    check_import(backend)
    sfc = np.array([52, 6], dtype=np.int32)
    level = np.array([2, 1], dtype=np.int32)
    r_sfc = np.array([52, 55], dtype=np.int32)
    sfc, level, r_sfc = wrap(
        sfc, level, r_sfc, backend=backend)
    max_level = 2

    e = Elementwise(sfc_same, backend=backend)
    e(sfc, level, max_level)

    np.testing.assert_array_equal(r_sfc, sfc)


@check_all_backends
def test_sfc_real(backend):
    check_import(backend)
    sfc = np.array([52, 55], dtype=np.int32)
    level = np.array([2, 1], dtype=np.int32)
    cpy_sfc = np.array([52, 6], dtype=np.int32)
    sfc, level, cpy_sfc = wrap(
        sfc, level, cpy_sfc, backend=backend)
    max_level = 2

    e = Elementwise(sfc_real, backend=backend)
    e(sfc, level, max_level)

    np.testing.assert_array_equal(cpy_sfc, sfc)


@check_all_backends
def test_id_duplicates(backend):
    check_import(backend)
    sfc = np.array([52, 52, 53], dtype=np.int32)
    level = np.array([2, 2, 2], dtype=np.int32)
    r_duplicate_idx = np.array([0, 1], dtype=np.int32)

    sfc, level, r_duplicate_idx = wrap(
        sfc, level, r_duplicate_idx, backend=backend)

    duplicate_idx = ary.empty(2, dtype=np.int32, backend=backend)

    e = Elementwise(id_duplicates, backend=backend)
    e(sfc, level, duplicate_idx)

    np.testing.assert_array_equal(r_duplicate_idx, duplicate_idx)


@check_all_backends
def test_remove_duplicates(backend):
    check_import(backend)
    sfc = np.array([52, 52, 53], dtype=np.int32)
    level = np.array([2, 2, 2], dtype=np.int32)
    duplicate_idx = np.array([0, 1], dtype=np.int32)
    r_sfc = np.array([52, -1, 53], dtype=np.int32)
    r_level = np.array([2, -1, 2], dtype=np.int32)

    sfc, level, duplicate_idx, r_sfc, r_level = wrap(
        sfc, level, duplicate_idx, r_sfc, r_level, backend=backend)

    e = Elementwise(remove_duplicates, backend=backend)
    e(duplicate_idx, sfc, level)

    np.testing.assert_array_equal(r_sfc, sfc)
    np.testing.assert_array_equal(r_level, level)


@check_all_backends
def test_find_level_diff(backend):
    check_import(backend)
    level = np.array([2, 2, 0], dtype=np.int32)
    parent_idx = np.array([2, 2, -1], dtype=np.int32)
    r_level_diff = np.array([1, 1, 0], dtype=np.int32)
    level, parent_idx, r_level_diff = wrap(
        level, parent_idx, r_level_diff, backend=backend)
    level_diff = ary.zeros(3, dtype=np.int32, backend=backend)

    e = Elementwise(find_level_diff, backend=backend)
    e(level, parent_idx, level_diff)

    np.testing.assert_array_equal(r_level_diff, level_diff)


@check_all_backends
def test_cumsum_level_diff(backend):
    check_import(backend)
    level_diff = np.array([1, 1, 0], dtype=np.int32)
    r_cumsum_level_diff = np.array([0, 1, 2], dtype=np.int32)
    level_diff, r_cumsum_level_diff = wrap(
        level_diff, r_cumsum_level_diff, backend=backend)
    cumsum_level_diff = ary.zeros(3, dtype=np.int32, backend=backend)
    scan = Scan(input_expr, output_expr, 'a+b',
                dtype=np.int32, backend=backend)

    scan(level=level_diff, cs_level=cumsum_level_diff)

    np.testing.assert_array_equal(
        r_cumsum_level_diff, cumsum_level_diff)


@check_all_backends
def test_complete_tree(backend):
    check_import(backend)
    sfc = np.array([1, 8, 0], dtype=np.int32)
    level = np.array([2, 2, 0], dtype=np.int32)
    idx = np.array([0, 1, -1], dtype=np.int32)
    parent = np.array([2, 2, -1], dtype=np.int32)
    child = np.ones(24, dtype=np.int32) * -1
    child[16] = 0
    child[17] = 1
    level_diff = np.array([1, 1, 0], dtype=np.int32)
    cum_level_diff = np.array([0, 1, 2], dtype=np.int32)
    r_new_sfc = np.array([1, 0, 8, 1, 0], dtype=np.int32)
    r_new_level = np.array([2, 1, 2, 1, 0], dtype=np.int32)
    r_new_idx = np.array([0, -1, 1, -1, -1], dtype=np.int32)
    r_new_parent = np.array([1, 4, 3, 4, -1], dtype=np.int32)
    r_new_child = np.ones(40, dtype=np.int32)*-1
    r_new_child[8] = 0
    r_new_child[24] = 2
    r_new_child[32] = 1
    r_new_child[33] = 3

    sfc, level, idx, parent, child, level_diff, cum_level_diff, \
        r_new_sfc, r_new_level, r_new_idx, r_new_parent, \
        r_new_child = wrap(
            sfc, level, idx, parent, child, level_diff,
            cum_level_diff, r_new_sfc, r_new_level, r_new_idx,
            r_new_parent, r_new_child, backend=backend)

    new_sfc = ary.zeros(5, dtype=np.int32, backend=backend)
    new_level = ary.zeros(5, dtype=np.int32, backend=backend)
    new_idx = ary.zeros(5, dtype=np.int32, backend=backend)
    new_parent = ary.zeros(5, dtype=np.int32, backend=backend)
    new_child = ary.empty(40, dtype=np.int32, backend=backend)
    new_child.fill(-1)

    e = Elementwise(complete_tree, backend=backend)
    e(level_diff, cum_level_diff, sfc, level, idx, parent,
      child, new_sfc, new_level, new_idx, new_parent, new_child)

    np.testing.assert_array_equal(r_new_sfc, new_sfc)
    np.testing.assert_array_equal(r_new_level, new_level)
    np.testing.assert_array_equal(r_new_idx, new_idx)
    np.testing.assert_array_equal(r_new_parent, new_parent)
    np.testing.assert_array_equal(r_new_child, new_child)
