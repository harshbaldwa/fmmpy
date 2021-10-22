import pytest
from fmm.force2 import *
from fmm.tree import build
import importlib.resources
import yaml

check_all_backends = pytest.mark.parametrize('backend',
                                             ['cython', 'opencl', 'cuda'])


def check_import(backend):
    if backend == 'opencl':
        pytest.importorskip('pyopencl')
    elif backend == 'cuda':
        pytest.importorskip('pycuda')


@check_all_backends
def test_calc_inner_p2(backend):
    check_import(backend)

    N = 4
    max_depth = 2
    part_val = np.ones(N, dtype=np.float32)
    part_x = np.array([0.12, 0.12, 0.62, 0.62], dtype=np.float32)
    part_y = np.array([0.12, 0.37, 0.12, 0.37], dtype=np.float32)
    part_z = np.array([0.14, 0.12, 0.12, 0.12], dtype=np.float32)
    
    part_val, part_x, part_y, part_z = wrap(part_val, part_x, part_y, part_z, 
                                            backend=backend)
    
    dimension = 3
    x_min = 0
    y_min = 0
    z_min = 0
    length = 1
    
    out_r = 1.1
    in_r = 6
    num_p2 = 48
    
    (cells, sfc, level, idx, bin_count, start_idx, leaf_idx,
     parent, child, part2bin, p2b_offset, lev_cs, levwise_cs, index, index_r, 
     lev_index, lev_index_r, cx, cy, cz, out_x, out_y, out_z,
     in_x, in_y, in_z, out_val, in_val, order, time_tree) = build(
         N, max_depth, part_val, part_x, part_y, part_z, x_min, y_min, z_min, 
         out_r, in_r, length, num_p2, backend, dimension)
    
    leg_lim = order//2 + 1
    siz_leg = leg_lim*(leg_lim+1)//2 - 1
    leg_lst = np.zeros(siz_leg, dtype=np.float32)
    count = 0
    for i in range(1, leg_lim):
        temp_lst = np.array(legendre(i)).astype(np.float32)
        leg_lst[count:count+i+1] = temp_lst[::-1]
        count += i+1
    
    leg_lst = wrap(leg_lst, backend=backend)
    
    res_x = ary.zeros(N, dtype=np.float32, backend=backend)
    res_y = ary.zeros(N, dtype=np.float32, backend=backend)
    res_z = ary.zeros(N, dtype=np.float32, backend=backend)
    res_dir_x = ary.zeros(N, dtype=np.float32, backend=backend)
    res_dir_y = ary.zeros(N, dtype=np.float32, backend=backend)
    res_dir_z = ary.zeros(N, dtype=np.float32, backend=backend)
    assoc = ary.empty(26*cells, dtype=np.int32, backend=backend)
    assoc.fill(-1)
    
    ecalc_p2_fine = Elementwise(calc_p2_fine, backend=backend)
    eassoc_coarse = Elementwise(assoc_coarse, backend=backend)
    efind_assoc = Elementwise(find_assoc, backend=backend)
    eloc_coeff = Elementwise(loc_coeff, backend=backend)
    
    ecalc_p2_fine(out_val[:lev_cs[max_depth-1]*num_p2], out_x, out_y, out_z,
                  part_val, part_x, part_y, part_z, cx, cy, cz, num_p2,
                  length, index, leg_lim, leg_lst, level, idx, out_r*sqrt(3),
                  bin_count, start_idx, leaf_idx)
    
    eassoc_coarse(sfc[levwise_cs[1]:levwise_cs[0]], parent, child, lev_index,
                  assoc, levwise_cs[1])

    for lev in range(2, max_depth+1):
        lev_offset = levwise_cs[lev-1] - levwise_cs[lev]
        if lev_offset == 0:
            continue
        efind_assoc(idx[:lev_offset], cx, cy, cz, level,
                    assoc, child, parent, levwise_cs[lev], lev_index,
                    lev_index_r, length)

    eloc_coeff(in_val[:lev_cs[1]*num_p2], in_x, in_y, in_z, out_val, out_x,
               out_y, out_z, part_val, part_x, part_y, part_z, cx, cy, cz,
               assoc, child, parent, num_p2, level, index, index_r,
               lev_index_r, idx, leaf_idx, start_idx, bin_count, length, in_r,
               leg_lst, leg_lim)
    
    loc_exp(in_val, in_x, in_y, in_z, index_r[0]*num_p2, part_x, part_y, 
            part_z, 0, num_p2, res_x, res_y, res_z)
    
    for i in range(2, len(part_val)):
        direct_force_comp(
            part_val[i], part_x[i], part_y[i], part_z[i], part_x[0], part_y[0],
            part_z[0], res_dir_x, res_dir_y, res_dir_z, 0)

    print(res_x, res_dir_x)
    print(res_y, res_dir_y)
    print(res_z, res_dir_z)
    
    print(abs(res_x[0] - res_dir_x[0])/abs(res_dir_x[0]))
    print(abs(res_y[0] - res_dir_y[0])/abs(res_dir_y[0]))
    print(abs(res_z[0] - res_dir_z[0])/abs(res_dir_z[0]))
    
    assert abs(res_x[0] - res_dir_x[0])/abs(res_dir_x[0]) < 2e-3
    assert abs(res_y[0] - res_dir_y[0])/abs(res_dir_y[0]) < 2e-3
    assert abs(res_z[0] - res_dir_z[0])/abs(res_dir_z[0]) < 2e-3


@check_all_backends
def test_calc_inner_p2_fine(backend):
    check_import(backend)
    
    N = 3
    max_depth = 2
    part_val = np.ones(N, dtype=np.float32)
    part_x = np.array([0.12, 0.12, 0.7], dtype=np.float32)
    part_y = np.array([0.12, 0.37, 0.2], dtype=np.float32)
    part_z = np.array([0.12, 0.12, 0.2], dtype=np.float32)
    
    part_val, part_x, part_y, part_z = wrap(part_val, part_x, part_y, part_z, 
                                            backend=backend)
    
    dimension = 3
    x_min = 0
    y_min = 0
    z_min = 0
    length = 1
    
    out_r = 1.1
    in_r = 6
    num_p2 = 12
    
    (cells, sfc, level, idx, bin_count, start_idx, leaf_idx,
     parent, child, part2bin, p2b_offset, lev_cs, levwise_cs, index, index_r, 
     lev_index, lev_index_r, cx, cy, cz, out_x, out_y, out_z,
     in_x, in_y, in_z, out_val, in_val, order, time_tree) = build(
         N, max_depth, part_val, part_x, part_y, part_z, x_min, y_min, z_min, 
         out_r, in_r, length, num_p2, backend, dimension)
    
    leg_lim = order//2 + 1
    siz_leg = leg_lim*(leg_lim+1)//2 - 1
    leg_lst = np.zeros(siz_leg, dtype=np.float32)
    count = 0
    for i in range(1, leg_lim):
        temp_lst = np.array(legendre(i)).astype(np.float32)
        leg_lst[count:count+i+1] = temp_lst[::-1]
        count += i+1
    
    leg_lst = wrap(leg_lst, backend=backend)
    
    res_x = ary.zeros(N, dtype=np.float32, backend=backend)
    res_y = ary.zeros(N, dtype=np.float32, backend=backend)
    res_z = ary.zeros(N, dtype=np.float32, backend=backend)
    res_dir_x = ary.zeros(N, dtype=np.float32, backend=backend)
    res_dir_y = ary.zeros(N, dtype=np.float32, backend=backend)
    res_dir_z = ary.zeros(N, dtype=np.float32, backend=backend)
    assoc = ary.empty(26*cells, dtype=np.int32, backend=backend)
    assoc.fill(-1)
    
    ecalc_p2_fine = Elementwise(calc_p2_fine, backend=backend)
    eassoc_coarse = Elementwise(assoc_coarse, backend=backend)
    efind_assoc = Elementwise(find_assoc, backend=backend)
    eloc_coeff = Elementwise(loc_coeff, backend=backend)
    
    ecalc_p2_fine(out_val[:lev_cs[max_depth-1]*num_p2], out_x, out_y, out_z,
                  part_val, part_x, part_y, part_z, cx, cy, cz, num_p2,
                  length, index, leg_lim, leg_lst, level, idx, out_r*sqrt(3),
                  bin_count, start_idx, leaf_idx)
    
    eassoc_coarse(sfc[levwise_cs[1]:levwise_cs[0]], parent, child, lev_index,
                  assoc, levwise_cs[1])

    for lev in range(2, max_depth+1):
        lev_offset = levwise_cs[lev-1] - levwise_cs[lev]
        if lev_offset == 0:
            continue
        efind_assoc(idx[:lev_offset], cx, cy, cz, level,
                    assoc, child, parent, levwise_cs[lev], lev_index,
                    lev_index_r, length)

    eloc_coeff(in_val[:lev_cs[1]*num_p2], in_x, in_y, in_z, out_val, out_x,
               out_y, out_z, part_val, part_x, part_y, part_z, cx, cy, cz,
               assoc, child, parent, num_p2, level, index, index_r,
               lev_index_r, idx, leaf_idx, start_idx, bin_count, length, in_r,
               leg_lst, leg_lim)
    
    loc_exp(in_val, in_x, in_y, in_z, index_r[0]*num_p2, part_x, part_y, 
            part_z, 0, num_p2, res_x, res_y, res_z)
    
    for i in range(2, len(part_val)):
        direct_force_comp(
            part_val[i], part_x[i], part_y[i], part_z[i], part_x[0], part_y[0],
            part_z[0], res_dir_x, res_dir_y, res_dir_z, 0)
    
    assert abs(res_x[0] - res_dir_x[0])/abs(res_dir_x[0]) < 2e-3
    assert abs(res_y[0] - res_dir_y[0])/abs(res_dir_y[0]) < 2e-3
    assert abs(res_z[0] - res_dir_z[0])/abs(res_dir_z[0]) < 2e-3
    

@check_all_backends
def test_trans_inner_p2(backend):
    check_import(backend)
    
    N = 4
    max_depth = 3
    part_val = np.ones(N, dtype=np.float32)
    part_x = np.array([0.062, 0.187, 0.62, 0.62], dtype=np.float32)
    part_y = np.array([0.062, 0.187, 0.12, 0.37], dtype=np.float32)
    part_z = np.array([0.062, 0.187, 0.12, 0.12], dtype=np.float32)
    
    part_val, part_x, part_y, part_z = wrap(part_val, part_x, part_y, part_z, 
                                            backend=backend)
    
    dimension = 3
    x_min = 0
    y_min = 0
    z_min = 0
    length = 1
    
    out_r = 1.1
    in_r = 6
    num_p2 = 48
    
    (cells, sfc, level, idx, bin_count, start_idx, leaf_idx,
     parent, child, part2bin, p2b_offset, lev_cs, levwise_cs, index, index_r, 
     lev_index, lev_index_r, cx, cy, cz, out_x, out_y, out_z,
     in_x, in_y, in_z, out_val, in_val, order, time_tree) = build(
         N, max_depth, part_val, part_x, part_y, part_z, x_min, y_min, z_min, 
         out_r, in_r, length, num_p2, backend, dimension)
    
    leg_lim = order//2 + 1
    siz_leg = leg_lim*(leg_lim+1)//2 - 1
    leg_lst = np.zeros(siz_leg, dtype=np.float32)
    count = 0
    for i in range(1, leg_lim):
        temp_lst = np.array(legendre(i)).astype(np.float32)
        leg_lst[count:count+i+1] = temp_lst[::-1]
        count += i+1
    
    leg_lst = wrap(leg_lst, backend=backend)
    
    res_x = ary.zeros(N, dtype=np.float32, backend=backend)
    res_y = ary.zeros(N, dtype=np.float32, backend=backend)
    res_z = ary.zeros(N, dtype=np.float32, backend=backend)
    res_dir_x = ary.zeros(N, dtype=np.float32, backend=backend)
    res_dir_y = ary.zeros(N, dtype=np.float32, backend=backend)
    res_dir_z = ary.zeros(N, dtype=np.float32, backend=backend)
    assoc = ary.empty(26*cells, dtype=np.int32, backend=backend)
    assoc.fill(-1)
    
    ecalc_p2_fine = Elementwise(calc_p2_fine, backend=backend)
    eassoc_coarse = Elementwise(assoc_coarse, backend=backend)
    efind_assoc = Elementwise(find_assoc, backend=backend)
    eloc_coeff = Elementwise(loc_coeff, backend=backend)
    etrans_loc = Elementwise(trans_loc, backend=backend)
    
    ecalc_p2_fine(out_val[:lev_cs[max_depth-1]*num_p2], out_x, out_y, out_z,
                  part_val, part_x, part_y, part_z, cx, cy, cz, num_p2,
                  length, index, leg_lim, leg_lst, level, idx, out_r*sqrt(3),
                  bin_count, start_idx, leaf_idx)
    
    eassoc_coarse(sfc[levwise_cs[1]:levwise_cs[0]], parent, child, lev_index,
                  assoc, levwise_cs[1])

    for lev in range(2, max_depth+1):
        lev_offset = levwise_cs[lev-1] - levwise_cs[lev]
        if lev_offset == 0:
            continue
        efind_assoc(idx[:lev_offset], cx, cy, cz, level,
                    assoc, child, parent, levwise_cs[lev], lev_index,
                    lev_index_r, length)

    eloc_coeff(in_val[:lev_cs[1]*num_p2], in_x, in_y, in_z, out_val, out_x,
               out_y, out_z, part_val, part_x, part_y, part_z, cx, cy, cz,
               assoc, child, parent, num_p2, level, index, index_r,
               lev_index_r, idx, leaf_idx, start_idx, bin_count, length, in_r,
               leg_lst, leg_lim)

    for lev in range(3, max_depth+1):
        lev_offset = levwise_cs[lev-1] - levwise_cs[lev]
        if lev_offset == 0:
            continue
        etrans_loc(in_val[:lev_offset*num_p2], in_val, in_x, in_y, in_z, cx,
                   cy, cz, num_p2, leg_lst, leg_lim, index_r, lev_index,
                   parent, levwise_cs[lev], in_r, level, length)
    
    loc_exp(in_val, in_x, in_y, in_z, index_r[0]*num_p2, part_x, part_y, 
            part_z, 0, num_p2, res_x, res_y, res_z)
    
    for i in range(2, len(part_val)):
        direct_force_comp(
            part_val[i], part_x[i], part_y[i], part_z[i], part_x[0], part_y[0],
            part_z[0], res_dir_x, res_dir_y, res_dir_z, 0)
    
    assert abs(res_x[0] - res_dir_x[0])/abs(res_dir_x[0]) < 2e-2
    assert abs(res_y[0] - res_dir_y[0])/abs(res_dir_y[0]) < 2e-2
    assert abs(res_z[0] - res_dir_z[0])/abs(res_dir_z[0]) < 2e-2