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

    num_p2 = 6
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

    out = sph_pts*out_r*sqrt(3)*length/2
    out_val = np.zeros(5*num_p2, dtype=np.float32)
    out_val[:24] = 1/6
    out_x = np.array(
        [0.72631395, -0.22631398, 0.25, 0.25, 0.25, 0.25, 1.226314, 0.27368602,
         0.75, 0.75, 0.75, 0.75, 0.72631395, -0.22631398, 0.25, 0.25, 0.25,
         0.25, 1.226314, 0.27368602, 0.75, 0.75, 0.75, 0.75, 1.4526279,
         -0.45262796, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    out_y = np.array(
        [0.25, 0.25, 0.72631395, -0.22631398, 0.25, 0.25, 0.25, 0.25,
         0.72631395, -0.22631398, 0.25, 0.25, 0.75, 0.75, 1.226314, 0.27368602,
         0.75, 0.75, 0.75, 0.75, 1.226314, 0.27368602, 0.75, 0.75, 0.5, 0.5,
         1.4526279, -0.45262796, 0.5, 0.5], dtype=np.float32)
    out_z = np.array(
        [0.25, 0.25, 0.25, 0.25, 0.72631395, -0.22631398, 0.25, 0.25, 0.25,
         0.25, 0.72631395, -0.22631398, 0.25, 0.25, 0.25, 0.25, 0.72631395,
         -0.22631398, 0.25, 0.25, 0.25, 0.25, 0.72631395, -0.22631398, 0.5,
         0.5, 0.5, 0.5, 1.4526279, -0.45262796], dtype=np.float32)

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

    e(out_val[24:30], out_x[24:30], out_y[24:30], out_z[24:30], out_val[:24],
      out_x[:24], out_y[:24], out_z[:24], cx, cy, cz, num_p2, index, index_r,
      leg_lim, leg_lst, child, offset, m2c_l)

    for i in range(len(part_val)):
        res_direct += direct_comp(part_val[i], part_x[i], part_y[i], part_z[i],
                                  px, py, pz)

    for i in range(24, 30):
        res_multipole += direct_comp(out_val[i],
                                     out_x[i], out_y[i], out_z[i], px, py, pz)

    assert abs(res_multipole - res_direct) < 1e-4


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

    well1 = well_sep(cx[0], cy[0], cz[0], cr[0],
                     cx[1], cy[1], cz[1], cr[1])
    well2 = well_sep(cx[0], cy[0], cz[0], cr[0],
                     cx[2], cy[2], cz[2], cr[2])
    well3 = well_sep(cx[0], cy[0], cz[0], cr[0],
                     cx[3], cy[3], cz[3], cr[3])

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
    r_assoc = np.ones(104, dtype=np.int32) * -1
    r_assoc[52] = 3
    r_assoc[78] = 1
    sfc, index, parent, child, r_assoc = wrap(
        sfc, index, parent, child, r_assoc,
        backend=backend)

    assoc = ary.empty(104, dtype=np.int32, backend=backend)
    assoc.fill(-1)

    eassoc_coarse = Elementwise(assoc_coarse, backend=backend)
    eassoc_coarse(sfc[2:4], parent, child, index, assoc, offset)

    np.testing.assert_array_equal(r_assoc, assoc)


# TEST: Add multiple siblings and check their index in associates
@check_all_backends
def test_find_assoc(backend):
    check_import(backend)
    sfc = np.array([3, 0, 10, 1, 3, 0], dtype=np.int32)
    level = np.array([2, 1, 2, 1, 1, 0], dtype=np.int32)
    idx = np.array([0, -1, 1, -1, 2, -1], dtype=np.int32)
    level_cs = [5, 3, 0]
    offset = level_cs[2]
    length = 1.0
    cx = np.array([0.375, 0.25, 0.625, 0.75, 0.75, 0.5], dtype=np.float32)
    cy = np.array([0.375, 0.25, 0.375, 0.25, 0.75, 0.5], dtype=np.float32)
    cz = np.array([0.125, 0.25, 0.125, 0.25, 0.25, 0.5], dtype=np.float32)
    index = np.array([0, 2, 1, 3, 4, 5], dtype=np.int32)
    index_r = np.array([0, 2, 1, 3, 4, 5], dtype=np.int32)
    parent = np.array([1, 5, 3, 5, 5, -1], dtype=np.int32)
    child = np.ones(48, dtype=np.int32) * -1
    child[8] = 0
    child[24] = 2
    child[40] = 1
    child[41] = 3
    child[42] = 4
    assoc = np.ones(130, dtype=np.int32) * -1
    assoc[52] = 3
    assoc[53] = 4
    assoc[78] = 1
    assoc[79] = 4
    assoc[104] = 1
    assoc[105] = 3

    sfc, level, cx, cy, cz, index, index_r, parent, child, assoc, \
        idx = wrap(sfc, level, cx, cy, cz, index, index_r, parent, child,
                   assoc, idx, backend=backend)

    efind_assoc = Elementwise(find_assoc, backend=backend)
    efind_assoc(idx[0:2], cx, cy, cz, level, assoc, child,
                parent, offset, index, index_r, length)

    assert (assoc[0] == 2 and assoc[1] == 4 and
            assoc[26] == 0 and assoc[27] == 4)


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
