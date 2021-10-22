import pytest
from fmm.fmm import *
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
    part_val = np.array([1, 2, 3, 4], dtype=np.float32)
    part_x = np.array([0.6, 0.4, 0.7, 0.3], dtype=np.float32)
    part_y = np.array([0.6, 0.4, 0.7, 0.3], dtype=np.float32)
    part_z = np.array([0.6, 0.4, 0.7, 0.3], dtype=np.float32)
    cx = np.array([0.5], dtype=np.float32)
    cy = np.array([0.5], dtype=np.float32)
    cz = np.array([0.5], dtype=np.float32)
    num_p2 = 6
    length = 1
    index = np.array([0], dtype=np.int32)
    level = np.array([0], dtype=np.int32)
    idx = np.array([0], dtype=np.int32)
    out_r = 1.1
    bin_count = np.array([4], dtype=np.int32)
    start_idx = np.array([0], dtype=np.int32)
    leaf_idx = np.array([0, 1, 2, 3], dtype=np.int32)

    px = 10
    py = 0
    pz = 0
    res_multipole = 0
    res_direct = 0

    with importlib.resources.open_text("fmm", "t_design.yaml") as file:
        data = yaml.load(file)[num_p2]
    sph_pts = np.array(data['array'], dtype=np.float32)
    order = data['order']

    out = sph_pts*out_r*sqrt(3)*length/2 + 0.5
    out_val = np.zeros(num_p2, dtype=np.float32)
    out_x = np.array(out[0::3])
    out_y = np.array(out[1::3])
    out_z = np.array(out[2::3])

    leg_lim = order//2 + 1
    siz_leg = leg_lim*(leg_lim+1)//2 - 1
    leg_lst = np.zeros(siz_leg, dtype=np.float32)
    count = 0
    for i in range(1, leg_lim):
        temp_lst = np.array(legendre(i)).astype(np.float32)
        leg_lst[count:count+i+1] = temp_lst[::-1]
        count += i+1

    (out_val, out_x, out_y, out_z, part_val, part_x, part_y, part_z, cx, cy,
     cz, index, leg_lst, level, idx, bin_count, start_idx, leaf_idx) = wrap(
         out_val, out_x, out_y, out_z, part_val, part_x, part_y, part_z, cx,
         cy, cz, index, leg_lst, level, idx, bin_count, start_idx, leaf_idx,
         backend=backend)

    e = Elementwise(calc_p2_fine, backend=backend)

    e(out_val, out_x, out_y, out_z, part_val, part_x, part_y, part_z, cx, cy,
      cz, num_p2, length, index, leg_lim, leg_lst, level, idx, out_r*sqrt(3),
      bin_count, start_idx, leaf_idx)

    for i in range(len(part_val)):
        res_direct += direct_comp(part_val[i], part_x[i], part_y[i], part_z[i],
                                  px, py, pz)

    for i in range(len(out_val)):
        res_multipole += direct_comp(out_val[i],
                                     out_x[i], out_y[i], out_z[i], px, py, pz)

    assert abs(res_multipole-res_direct) < 1e-3


@check_all_backends
def test_calc_p2(backend):
    check_import(backend)

    part_val = np.array([1, 1, 1, 1], dtype=np.float32)
    part_x = np.array([0.25, 0.75, 0.25, 0.75], dtype=np.float32)
    part_y = np.array([0.25, 0.25, 0.75, 0.75], dtype=np.float32)
    part_z = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    px = 10
    py = 0
    pz = 0
    res_direct = 0
    res_multipole = 0

    num_p2 = 12
    length = 1
    index = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    index_r = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    out_r = 1.1
    offset = 4
    m2c_l = out_r*sqrt(3)*length/2
    cx = np.array([0.25, 0.75, 0.25, 0.75, 0.5], dtype=np.float32)
    cy = np.array([0.25, 0.25, 0.75, 0.75, 0.5], dtype=np.float32)
    cz = np.array([0.25, 0.25, 0.25, 0.25, 0.5], dtype=np.float32)
    child = np.ones(40, dtype=np.int32)*-1
    child[32] = 0
    child[33] = 1
    child[34] = 2
    child[35] = 3

    with importlib.resources.open_text("fmm", "t_design.yaml") as file:
        data = yaml.load(file)[num_p2]
    sph_pts = np.array(data['array'], dtype=np.float32)
    order = data['order']

    out = sph_pts*out_r*sqrt(3)*length
    out_val = np.zeros(5*num_p2, dtype=np.float32)
    out_val[:4*num_p2] = 1/num_p2
    out_x = np.zeros(5*num_p2, dtype=np.float32)
    out_y = np.zeros(5*num_p2, dtype=np.float32)
    out_z = np.zeros(5*num_p2, dtype=np.float32)

    out_x[:num_p2] = np.array(out[0::3])/4 + 0.25
    out_x[num_p2:2*num_p2] = np.array(out[0::3])/4 + 0.75
    out_x[2*num_p2:3*num_p2] = np.array(out[0::3])/4 + 0.25
    out_x[3*num_p2:4*num_p2] = np.array(out[0::3])/4 + 0.75
    out_x[4*num_p2:5*num_p2] = np.array(out[0::3])/2 + 0.5

    out_y[:num_p2] = np.array(out[1::3])/4 + 0.25
    out_y[num_p2:2*num_p2] = np.array(out[1::3])/4 + 0.25
    out_y[2*num_p2:3*num_p2] = np.array(out[1::3])/4 + 0.75
    out_y[3*num_p2:4*num_p2] = np.array(out[1::3])/4 + 0.75
    out_y[4*num_p2:5*num_p2] = np.array(out[1::3])/2 + 0.5

    out_z[:num_p2] = np.array(out[2::3])/4 + 0.25
    out_z[num_p2:2*num_p2] = np.array(out[2::3])/4 + 0.25
    out_z[2*num_p2:3*num_p2] = np.array(out[2::3])/4 + 0.25
    out_z[3*num_p2:4*num_p2] = np.array(out[2::3])/4 + 0.25
    out_z[4*num_p2:5*num_p2] = np.array(out[2::3])/2 + 0.5

    leg_lim = order//2 + 1
    siz_leg = leg_lim*(leg_lim+1)//2 - 1
    leg_lst = np.zeros(siz_leg, dtype=np.float32)
    count = 0
    for i in range(1, leg_lim):
        temp_lst = np.array(legendre(i)).astype(np.float32)
        leg_lst[count:count+i+1] = temp_lst[::-1]
        count += i+1

    (part_val, part_x, part_y, part_z, index, index_r, child, out_val, out_x,
     out_y, out_z, cx, cy, cz, leg_lst) = wrap(
         part_val, part_x, part_y, part_z, index, index_r, child, out_val,
         out_x, out_y, out_z, cx, cy, cz, leg_lst, backend=backend)

    e = Elementwise(calc_p2, backend=backend)

    e(out_val[4*num_p2:5*num_p2], out_x[4*num_p2:5*num_p2],
      out_y[4*num_p2:5*num_p2], out_z[4*num_p2:5*num_p2], out_val[:4*num_p2],
      out_x[:4*num_p2], out_y[:4*num_p2], out_z[:4*num_p2], cx, cy, cz,
      num_p2, index, index_r, leg_lim, leg_lst, child, offset, m2c_l)

    for i in range(len(part_val)):
        res_direct += direct_comp(part_val[i], part_x[i], part_y[i], part_z[i],
                                  px, py, pz)

    for i in range(4*num_p2, 5*num_p2):
        res_multipole += direct_comp(out_val[i],
                                     out_x[i], out_y[i], out_z[i], px, py, pz)

    assert abs(res_multipole - res_direct) < 1e-6


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
def test_well_sep(backend):
    check_import(backend)
    cx = np.array([0.125, 0.625, 0.375, 0.625], dtype=np.float32)
    cy = np.array([0.125, 0.125, 0.375, 0.625], dtype=np.float32)
    cz = np.array([0.125, 0.125, 0.375, 0.625], dtype=np.float32)
    cr = np.array([0.125, 0.125, 0.125, 0.125], dtype=np.float32)

    well1 = well_sep(cx[0], cy[0], cz[0], sqrt(3)*cr[0], cx[1], cy[1], cz[1])
    well2 = well_sep(cx[0], cy[0], cz[0], sqrt(3)*cr[0], cx[2], cy[2], cz[2])
    well3 = well_sep(cx[0], cy[0], cz[0], sqrt(3)*cr[0], cx[3], cy[3], cz[3])

    assert well1 == 0 and well2 == 0 and well3 == 1


@check_all_backends
def test_assoc_coarse(backend):
    check_import(backend)
    sfc = np.array([1, 8, 0, 1, 0], dtype=np.int32)
    # offset = level_cs[1]
    offset = 2
    index = np.array([0, 2, 1, 3, 4], dtype=np.int32)
    parent = np.array([1, 4, 3, 4, -1], dtype=np.int32)
    child = np.ones(40, dtype=np.int32) * -1
    child[8] = 0
    child[24] = 2
    child[32] = 1
    child[33] = 3
    r_assoc = np.ones(320, dtype=np.int32) * -1
    r_assoc[160] = 3
    r_assoc[240] = 1
    sfc, index, parent, child, r_assoc = wrap(
        sfc, index, parent, child, r_assoc,
        backend=backend)

    assoc = ary.empty(320, dtype=np.int32, backend=backend)
    assoc.fill(-1)

    eassoc_coarse = Elementwise(assoc_coarse, backend=backend)
    eassoc_coarse(sfc[2:4], parent, child, index, assoc, offset)

    np.testing.assert_array_equal(r_assoc, assoc)


# TEST: Add multiple siblings and check their index in associates
@check_all_backends
def test_find_assoc(backend):
    check_import(backend)
    sfc = np.array([3, 0, 10, 11, 1, 3, 0], dtype=np.int32)
    level = np.array([2, 1, 2, 2, 1, 1, 0], dtype=np.int32)
    idx = np.array([0, -1, 1, 2, -1, 3, -1], dtype=np.int32)
    level_cs = [6, 4, 0]
    offset = level_cs[2]
    length = 1.0
    cx = np.array([0.375, 0.25, 0.625, 0.825, 0.75, 0.75, 0.5],
                  dtype=np.float32)
    cy = np.array([0.375, 0.25, 0.375, 0.375, 0.25, 0.75, 0.5],
                  dtype=np.float32)
    cz = np.array([0.125, 0.25, 0.125, 0.125, 0.25, 0.25, 0.5],
                  dtype=np.float32)
    index = np.array([0, 2, 3, 1, 4, 5, 6], dtype=np.int32)
    index_r = np.array([0, 3, 1, 2, 4, 5, 6], dtype=np.int32)
    parent = np.array([1, 6, 4, 4, 6, 6, -1], dtype=np.int32)
    child = np.ones(56, dtype=np.int32) * -1
    child[8] = 0
    child[32] = 2
    child[33] = 3
    child[48] = 1
    child[49] = 4
    child[50] = 5
    assoc = np.ones(480, dtype=np.int32) * -1
    assoc[240] = 4
    assoc[241] = 5
    assoc[320] = 1
    assoc[321] = 5
    assoc[400] = 1
    assoc[401] = 4

    sfc, level, cx, cy, cz, index, index_r, parent, child, assoc, \
        idx = wrap(sfc, level, cx, cy, cz, index, index_r, parent, child,
                   assoc, idx, backend=backend)

    efind_assoc = Elementwise(find_assoc, backend=backend)
    efind_assoc(idx[0:3], cx, cy, cz, level, assoc, child,
                parent, offset, index, index_r, length)

    assert (assoc[0] == 2 and assoc[1] == 3 and assoc[2] == 5 and
            assoc[80] == 0 and assoc[81] == 5 and assoc[82] == 3 and
            assoc[160] == 0 and assoc[161] == 5 and assoc[162] == 2)


@check_all_backends
def test_direct_comp(backend):
    check_import(backend)
    part_val = np.array([1, 1])
    part_x = np.array([0.25, 0.75])
    part_y = np.array([0.25, 0.25])
    part_z = np.array([0.25, 0.25])

    res = direct_comp(part_val[0], part_x[0], part_y[0], part_z[0],
                      part_x[1], part_y[1], part_z[1])

    assert res == 2


@check_all_backends
def test_loc_coeff(backend):
    check_import(backend)
    sfc = np.array([0, 2, 0, 9, 11, 1, 3, 0], dtype=np.int32)
    level = np.array([2, 2, 1, 2, 2, 1, 1, 0], dtype=np.int32)
    idx = np.array([0, 1, -1, 2, 3, -1, 4, -1], dtype=np.int32)
    bin_count = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    start_idx = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    leaf_idx = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    index = np.array([0, 1, 3, 4, 6, 5, 2, 7], dtype=np.int32)
    index_r = np.array([0, 1, 6, 2, 3, 5, 4, 7], dtype=np.int32)
    lev_index_r = np.array([3, 2, 6, 1, 0, 5, 4, 7], dtype=np.int32)
    parent = np.array([2, 2, 7, 5, 5, 7, 7, -1], dtype=np.int32)
    child = np.ones(64, dtype=np.int32) * -1
    assoc = np.ones(640, dtype=np.int32) * -1
    child[16] = 0
    child[17] = 1
    child[40] = 3
    child[41] = 4
    child[56] = 2
    child[57] = 5

    assoc[0] = 6
    assoc[1] = 3
    assoc[80] = 4
    assoc[160] = 6
    assoc[161] = 0
    assoc[240] = 1
    assoc[320] = 2
    assoc[321] = 5
    assoc[400] = 2
    assoc[401] = 6
    assoc[480] = 5
    assoc[481] = 6

    out_r = 1.1
    in_r = 1.05
    num_p2 = 12
    length = 1

    with importlib.resources.open_text("fmm", "t_design.yaml") as file:
        data = yaml.load(file)[num_p2]
    sph_pts = np.array(data['array'], dtype=np.float32)
    order = data['order']

    out = sph_pts*out_r*sqrt(3)*length
    in_all = sph_pts*in_r*sqrt(3)*length
    out_val = np.zeros(8*num_p2, dtype=np.float32)
    in_val = np.zeros(8*num_p2, dtype=np.float32)
    out_x = np.zeros(8*num_p2, dtype=np.float32)
    out_y = np.zeros(8*num_p2, dtype=np.float32)
    out_z = np.zeros(8*num_p2, dtype=np.float32)
    in_x = np.zeros(8*num_p2, dtype=np.float32)
    in_y = np.zeros(8*num_p2, dtype=np.float32)
    in_z = np.zeros(8*num_p2, dtype=np.float32)

    part_val = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    part_x = np.array([0.12, 0.12, 0.88, 0.88, 0.8], dtype=np.float32)
    part_y = np.array([0.12, 0.37, 0.13, 0.38, 0.8], dtype=np.float32)
    part_z = np.array([0.12, 0.12, 0.13, 0.13, 0.3], dtype=np.float32)

    cx = np.array([0.125, 0.125, 0.875, 0.875, 0.75, 0.75, 0.25, 0.5],
                  dtype=np.float32)
    cy = np.array([0.125, 0.375, 0.125, 0.375, 0.75, 0.25, 0.25, 0.5],
                  dtype=np.float32)
    cz = np.array([0.125, 0.125, 0.125, 0.125, 0.25, 0.25, 0.25, 0.5],
                  dtype=np.float32)

    l = np.array([2, 2, 2, 2, 1, 1, 1, 0], dtype=np.int32) + 1

    for i in range(len(cx)):
        out_x[i*num_p2:(i+1)*num_p2] = np.array(out[0::3])/2**l[i] + cx[i]
        out_y[i*num_p2:(i+1)*num_p2] = np.array(out[1::3])/2**l[i] + cy[i]
        out_z[i*num_p2:(i+1)*num_p2] = np.array(out[2::3])/2**l[i] + cz[i]

        in_x[i*num_p2:(i+1)*num_p2] = np.array(in_all[0::3])/2**l[i] + cx[i]
        in_y[i*num_p2:(i+1)*num_p2] = np.array(in_all[1::3])/2**l[i] + cy[i]
        in_z[i*num_p2:(i+1)*num_p2] = np.array(in_all[2::3])/2**l[i] + cz[i]

    cx = np.array([0.125, 0.125, 0.25, 0.875, 0.875, 0.75, 0.75, 0.5],
                  dtype=np.float32)
    cy = np.array([0.125, 0.375, 0.25, 0.125, 0.375, 0.25, 0.75, 0.5],
                  dtype=np.float32)
    cz = np.array([0.125, 0.125, 0.25, 0.125, 0.125, 0.25, 0.25, 0.5],
                  dtype=np.float32)

    leg_lim = order//2 + 1
    siz_leg = leg_lim*(leg_lim+1)//2 - 1
    leg_lst = np.zeros(siz_leg, dtype=np.float32)
    count = 0
    for i in range(1, leg_lim):
        temp_lst = np.array(legendre(i)).astype(np.float32)
        leg_lst[count:count+i+1] = temp_lst[::-1]
        count += i+1

    (level, idx, bin_count, start_idx, leaf_idx, index, index_r, out_x, out_y,
     out_z, in_x, in_y, in_z, out_val, in_val, cx, cy, cz, leg_lst, part_val,
     part_x, part_y, part_z, assoc, parent, child, lev_index_r) = wrap(
         level, idx, bin_count, start_idx, leaf_idx, index, index_r, out_x,
         out_y, out_z, in_x, in_y, in_z, out_val, in_val, cx, cy, cz, leg_lst,
         part_val, part_x, part_y, part_z, assoc, parent, child, lev_index_r,
         backend=backend)

    ecalc_p2_fine = Elementwise(calc_p2_fine, backend=backend)
    eloc_coeff = Elementwise(loc_coeff, backend=backend)

    ecalc_p2_fine(out_val[:5*num_p2], out_x, out_y, out_z, part_val, part_x,
                  part_y, part_z, cx, cy, cz, num_p2, length, index, leg_lim,
                  leg_lst, level, idx, out_r*sqrt(3), bin_count, start_idx,
                  leaf_idx)

    eloc_coeff(in_val[:5*num_p2], in_x, in_y, in_z, out_val, out_x, out_y,
               out_z, part_val, part_x, part_y, part_z, cx, cy, cz, assoc,
               child, parent, num_p2, level, index, index_r, lev_index_r, idx,
               leaf_idx, start_idx, bin_count, length)

    i2c_l = in_r*sqrt(3)*length/8

    res_loc = loc_exp(in_val, in_x, in_y, in_z, cx[0], cy[0], cz[0], part_x[0],
                      part_y[0], part_z[0], num_p2, i2c_l, 0, leg_lst, leg_lim)

    res_direct = 0
    for i in range(2, len(part_val)):
        res_direct += direct_comp(part_val[i], part_x[i], part_y[i], part_z[i],
                                  part_x[0], part_y[0], part_z[0])

    assert abs(res_loc - res_direct)/res_direct < 2e-4
