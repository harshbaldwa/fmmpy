import pytest
from fmm.tree import *

check_all_backends = pytest.mark.parametrize('backend',
                                             ['cython', 'opencl', 'cuda'])


def check_import(backend):
    if backend == 'opencl':
        pytest.importorskip('pyopencl')
    elif backend == 'cuda':
        pytest.importorskip('pycuda')


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
def test_single_node(backend):
    check_import(backend)
    n = 9
    leaf_sfc = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3], dtype=np.int32)
    bin_count = np.ones(n, dtype=np.int32)
    bin_idx = np.zeros(n, dtype=np.int32)
    
    r_bin_count = np.array([2, 1, 2, 1, 2, 1, 3, 1, 1], dtype=np.int32)
    r_bin_idx = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1], dtype=np.int32)
    
    leaf_sfc, bin_count, bin_idx, r_bin_count, r_bin_idx = wrap(
        leaf_sfc, bin_count, bin_idx, r_bin_count, r_bin_idx, backend=backend)
    
    e = Elementwise(single_node, backend=backend)
    e(leaf_sfc[:-1], leaf_sfc[1:], bin_count, bin_idx)
    
    np.testing.assert_array_equal(r_bin_count, bin_count)
    np.testing.assert_array_equal(r_bin_idx, bin_idx)
    

@check_all_backends
def test_internal_nodes(backend):
    check_import(backend)
    dimension = 3
    sfc = np.array([52, 53], dtype=np.int32)
    level = np.array([2, 2], dtype=np.int32)
    r_lca_sfc = np.array([6], dtype=np.int32)
    r_lca_level = np.array([1], dtype=np.int32)
    sfc, level, r_lca_sfc, r_lca_level = wrap(
        sfc, level, r_lca_sfc, r_lca_level, backend=backend)

    lca_sfc = ary.empty(1, dtype=np.int32, backend=backend)
    lca_level = ary.empty(1, dtype=np.int32, backend=backend)

    e = Elementwise(internal_nodes, backend=backend)
    e(sfc[:-1], sfc[1:], level[:-1], level[1:], lca_sfc, lca_level, dimension)

    (np.testing.assert_array_equal(r_lca_sfc, lca_sfc) and
     np.testing.assert_array_equal(r_lca_level, lca_level))


@check_all_backends
def test_find_parents(backend):
    check_import(backend)
    dimension = 3
    sfc = np.array([52, 53], dtype=np.int32)
    level = np.array([2, 2], dtype=np.int32)
    all_idx = np.arange(2, dtype=np.int32)
    r_lca_sfc = np.array([6], dtype=np.int32)
    r_lca_level = np.array([1], dtype=np.int32)
    r_temp_idx = np.array([0], dtype=np.int32)
    sfc, level, r_lca_sfc, r_lca_level, all_idx, r_temp_idx = wrap(
        sfc, level, r_lca_sfc, r_lca_level, all_idx, r_temp_idx, 
        backend=backend)

    lca_sfc = ary.empty(1, dtype=np.int32, backend=backend)
    lca_level = ary.empty(1, dtype=np.int32, backend=backend)
    temp_idx = ary.empty(1, dtype=np.int32, backend=backend)

    e = Elementwise(find_parents, backend=backend)
    e(sfc[:-1], sfc[1:], level[:-1], level[1:], all_idx,
      lca_sfc, lca_level, temp_idx, dimension)

    np.testing.assert_array_equal(r_lca_sfc, lca_sfc)
    np.testing.assert_array_equal(r_lca_level, lca_level)
    np.testing.assert_array_equal(r_temp_idx, temp_idx)


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
    dimension = 3

    e = Elementwise(sfc_same, backend=backend)
    e(sfc, level, max_level, dimension)

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
    dimension = 3

    e = Elementwise(sfc_real, backend=backend)
    e(sfc, level, max_level, dimension)

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
def test_complete_tree(backend):
    check_import(backend)
    dimension = 3
    sfc = np.array([1, 8, 0], dtype=np.int32)
    level = np.array([2, 2, 0], dtype=np.int32)
    level_diff = np.array([1, 1, 0], dtype=np.int32)
    r_sfc = np.array([0, 1, 0], dtype=np.int32)
    r_level = np.array([1, 1, 0], dtype=np.int32)

    sfc, level, level_diff, r_sfc, r_level = wrap(
        sfc, level, level_diff, r_sfc, r_level, backend=backend)

    e = Elementwise(complete_tree, backend=backend)
    e(level_diff, sfc, level, dimension)

    np.testing.assert_array_equal(r_sfc, sfc)
    np.testing.assert_array_equal(r_level, level)


@check_all_backends
def test_p2bin(backend):
    check_import(backend)
    idx = np.array([0, -1, 1, -1, 2, -1], dtype=np.int32)
    bin_count = np.array([3, 2, 1], dtype=np.int32)
    start_idx = np.array([0, 3, 5], dtype=np.int32)
    r_part2bin = np.array([0, 0, 0, 2, 2, 4], dtype=np.int32)
    part2bin = np.zeros(6, dtype=np.int32)
    idx, bin_count, start_idx, r_part2bin, part2bin = wrap(
        idx, bin_count, start_idx, r_part2bin, part2bin, backend=backend)
    
    e = Elementwise(p2bin, backend=backend)
    e(idx, bin_count, start_idx, part2bin)
    
    np.testing.assert_array_equal(r_part2bin, part2bin)


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
    out_r = sqrt(3)
    in_r = 0.75
    num_p2 = 6
    length = 1
    sz_cell = sqrt(3.0)*length/4
    with importlib.resources.open_text("fmm", "t_design.yaml") as file:
        data = yaml.load(file)[num_p2]
    sph_pts = np.array(data['array'], dtype=np.float32)
    r_out = sph_pts*out_r*sz_cell + 0.25
    r_in = sph_pts*in_r*sz_cell + 0.25
    r_out_x = r_out[::3]
    r_out_y = r_out[1::3]
    r_out_z = r_out[2::3]
    r_in_x = r_in[::3]
    r_in_y = r_in[1::3]
    r_in_z = r_in[2::3]

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
    e(out_x, out_y, out_z, in_x, in_y, in_z, sph_pts, cx, cy, cz,
      out_r, in_r, length, level, num_p2)

    np.testing.assert_array_almost_equal(r_out_x, out_x)
    np.testing.assert_array_almost_equal(r_out_y, out_y)
    np.testing.assert_array_almost_equal(r_out_z, out_z)
    np.testing.assert_array_almost_equal(r_in_x, in_x)
    np.testing.assert_array_almost_equal(r_in_y, in_y)
    np.testing.assert_array_almost_equal(r_in_z, in_z)


@check_all_backends
def test_level_info(backend):
    check_import(backend)
    max_depth = 3
    level = np.array([3, 3, 2, 2, 1, 1, 1, 0], dtype=np.int32)
    idx = np.array([0, 1, -1, 2, -1, -1, -1, -1], dtype=np.int32)
    r_lev_n = np.array([1, 3, 1, 3], dtype=np.int32)
    level, idx, r_lev_n = wrap(level, idx, r_lev_n, backend=backend)
    lev_n = ary.zeros(4, dtype=np.int32, backend=backend)

    e = Elementwise(level_info, backend=backend)
    e(level, idx, lev_n, max_depth)

    np.testing.assert_array_equal(r_lev_n, lev_n)


@check_all_backends
def test_lev_cumsum(backend):
    check_import(backend)
    max_depth = 3
    lev_nr = np.array([2, 2, 3, 1], dtype=np.int32)
    r_lev_csr = np.array([7, 4, 2, 0], dtype=np.int32)
    lev_nr, r_lev_csr = wrap(lev_nr, r_lev_csr, backend=backend)
    cumsum = Scan(input_expr, output_expr, 'a+b',
                  dtype=np.int32, backend=backend)
    reverse = ReverseArrays('reverse', ['a', 'b']).function
    ereverse = Elementwise(reverse, backend=backend)
    lev_csr = ary.zeros(max_depth+1, dtype=np.int32, backend=backend)
    lev_cs = ary.zeros(max_depth+1, dtype=np.int32, backend=backend)

    cumsum(in_arr=lev_nr, out_arr=lev_cs)
    ereverse(lev_csr, lev_cs, max_depth+1)

    np.testing.assert_array_equal(r_lev_csr, lev_csr)
