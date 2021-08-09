from tree_compyle import *

backend='cython'

def test_copy_arr():
    arr = ary.ones(10, dtype=np.int32, backend=backend)
    arr2 = ary.empty(10, dtype=np.int32, backend=backend)

    copy_value = CopyValue('copy_value', ['a', 'b']).function
    e = Elementwise(copy_value, backend=backend)
    e(arr, arr2)
    np.testing.assert_array_equal(arr, arr2)


def test_reverse_arr():
    arr_org = np.arange(10, dtype=np.int32)
    arr_result = arr_org[::-1]
    arr_org, arr_result = wrap(arr_org, arr_result, backend=backend)
    arr_empty = ary.empty(10, dtype=np.int32, backend=backend)

    reverse = ReverseArray('reverse', ['a', 'b']).function
    e = Elementwise(reverse, backend=backend)

    e(arr_empty, arr_org, 10)
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
