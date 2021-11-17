import pytest
from fmmpy.fmm.hybrid_force import *
from fmmpy.tree.tree import build
import pkg_resources
import yaml

check_all_backends = pytest.mark.parametrize('backend',
                                             ['cython', 'opencl', 'cuda'])

T_DESIGN = pkg_resources.resource_filename('fmmpy', 'data/t_design.yaml')


def check_import(backend):
    if backend == 'opencl':
        pytest.importorskip('pyopencl')
    elif backend == 'cuda':
        pytest.importorskip('pycuda')


def test_lgndre():
    backend = "cython"
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

    with open(T_DESIGN, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)[num_p2]
    sph_pts = np.array(data['array'], dtype=np.float32)
    order = data['order']

    out = sph_pts*out_r*sqrt(3)*length/2 + 0.5
    out_val = np.zeros(num_p2, dtype=np.float32)
    out_x = np.array(out[0::3])
    out_y = np.array(out[1::3])
    out_z = np.array(out[2::3])

    leg_lim = order//2 + 1
    siz_leg = leg_lim*(leg_lim+1)//2 - 2
    leg_lst = np.zeros(siz_leg, dtype=np.float32)
    count = 1
    for i in range(2, leg_lim):
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

    assert pytest.approx(res_multipole, abs=1e-3) == res_direct


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

    with open(T_DESIGN, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)[num_p2]
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
    siz_leg = leg_lim*(leg_lim+1)//2 - 2
    leg_lst = np.zeros(siz_leg, dtype=np.float32)
    count = 1
    for i in range(2, leg_lim):
        temp_lst = np.array(legendre(i)).astype(np.float32)
        leg_lst[count:count+i+1] = temp_lst[::-1]
        count += i+1

    (part_val, part_x, part_y, part_z, index, index_r, child, out_val, out_x,
     out_y, out_z, cx, cy, cz, leg_lst) = wrap(
         part_val, part_x, part_y, part_z, index, index_r, child, out_val,
         out_x, out_y, out_z, cx, cy, cz, leg_lst, backend=backend)

    e = Elementwise(calc_p2, backend=backend)

    e(out_val[4*num_p2:5*num_p2], out_val, out_x, out_y, out_z, cx, cy, cz,
      num_p2, index, index_r, leg_lim, leg_lst, child, offset, m2c_l)

    for i in range(len(part_val)):
        res_direct += direct_comp(part_val[i], part_x[i], part_y[i], part_z[i],
                                  px, py, pz)

    for i in range(4*num_p2, 5*num_p2):
        res_multipole += direct_comp(out_val[i],
                                     out_x[i], out_y[i], out_z[i], px, py, pz)

    assert pytest.approx(res_multipole, abs=1e-6) == res_direct


def test_direct_comp():
    backend = "cython"
    part_val = np.array([1, 1])
    part_x = np.array([0.25, 0.75])
    part_y = np.array([0.25, 0.25])
    part_z = np.array([0.25, 0.25])

    res = direct_comp(part_val[0], part_x[0], part_y[0], part_z[0],
                      part_x[1], part_y[1], part_z[1])

    assert res == 2


def test_direct_comp_force():
    backend = "cython"
    part_val = np.array([1, 1])
    part_x = np.array([0.25, 0.75])
    part_y = np.array([0.25, 0.75])
    part_z = np.array([0.25, 0.75])

    res_x = np.zeros(2, dtype=np.float32)
    res_y = np.zeros(2, dtype=np.float32)
    res_z = np.zeros(2, dtype=np.float32)

    direct_comp_force(part_val[1], part_x[1], part_y[1], part_z[1], part_x[0],
                      part_y[0], part_z[0], res_x, res_y, res_z, 0)

    assert pytest.approx(res_x[0], 1e-6) == 1 / (0.75 * sqrt(3))
    assert pytest.approx(res_y[0], 1e-6) == 1 / (0.75 * sqrt(3))
    assert pytest.approx(res_z[0], 1e-6) == 1 / (0.75 * sqrt(3))


@check_all_backends
def test_direct_solv_force(backend):
    check_import(backend)
    part_val = np.array([1, 1], dtype=np.float32)
    part_x = np.array([0.25, 0.75], dtype=np.float32)
    part_y = np.array([0.25, 0.75], dtype=np.float32)
    part_z = np.array([0.25, 0.75], dtype=np.float32)
    r_res_x = np.array([1, -1], dtype=np.float32) * 1 / (0.75 * sqrt(3))
    r_res_y = np.array([1, -1], dtype=np.float32) * 1 / (0.75 * sqrt(3))
    r_res_z = np.array([1, -1], dtype=np.float32) * 1 / (0.75 * sqrt(3))
    res_x = ary.zeros(2, dtype=np.float32, backend=backend)
    res_y = ary.zeros(2, dtype=np.float32, backend=backend)
    res_z = ary.zeros(2, dtype=np.float32, backend=backend)

    part_val, part_x, part_y, part_z, r_res_x, r_res_y, r_res_z = wrap(
        part_val, part_x, part_y, part_z, r_res_x, r_res_y, r_res_z,
        backend=backend)

    e = Elementwise(direct_solv_force, backend=backend)
    e(part_val, part_x, part_y, part_z, res_x, res_y, res_z, 2)

    np.testing.assert_array_almost_equal(res_x, r_res_x)
    np.testing.assert_array_almost_equal(res_y, r_res_y)
    np.testing.assert_array_almost_equal(res_z, r_res_z)


def test_is_adj():
    backend = "cython"
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

    assert adj1 == 1
    assert adj2 == 1
    assert adj3 == 0
    assert adj4 == 1


def test_well_sep():
    backend = "cython"
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
    sfc = np.array([1, 0, 8, 1, 0], dtype=np.int32)
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


@check_all_backends
def test_find_assoc(backend):
    check_import(backend)
    sfc = np.array([0, 3, 0, 10, 11, 1, 3, 0], dtype=np.int32)
    level = np.array([2, 2, 1, 2, 2, 1, 1, 0], dtype=np.int32)
    idx = np.array([0, 1, -1, 2, 3, -1, 4, -1], dtype=np.int32)
    level_cs = [7, 4, 0]
    offset = level_cs[2]
    length = 1.0
    cx = np.array([0.125, 0.375, 0.25, 0.625, 0.875, 0.75, 0.75, 0.5],
                  dtype=np.float32)
    cy = np.array([0.125, 0.375, 0.25, 0.375, 0.375, 0.25, 0.75, 0.5],
                  dtype=np.float32)
    cz = np.array([0.125, 0.125, 0.25, 0.125, 0.125, 0.25, 0.25, 0.5],
                  dtype=np.float32)
    index = np.array([4, 3, 1, 0, 6, 5, 2, 7], dtype=np.int32)
    index_r = np.array([3, 2, 6, 1, 0, 5, 4, 7], dtype=np.int32)
    parent = np.array([2, 2, 7, 5, 5, 7, 7, -1], dtype=np.int32)
    child = np.ones(64, dtype=np.int32) * -1
    child[16] = 0
    child[17] = 1
    child[40] = 3
    child[41] = 4
    child[56] = 2
    child[57] = 5
    child[58] = 6

    assoc = np.ones(560, dtype=np.int32) * -1

    sfc, level, cx, cy, cz, index, index_r, parent, child, assoc, \
        idx = wrap(sfc, level, cx, cy, cz, index, index_r, parent, child,
                   assoc, idx, backend=backend)

    eassoc_coarse = Elementwise(assoc_coarse, backend=backend)
    efind_assoc = Elementwise(find_assoc, backend=backend)

    eassoc_coarse(sfc[4:7], parent, child, index, assoc, 4)
    efind_assoc(idx[0:4], cx, cy, cz, level, assoc, child,
                parent, offset, index, index_r, length)

    assert assoc[0] == 1
    assert assoc[1] == 6
    assert assoc[2] == 3
    assert assoc[3] == -1
    assert assoc[80] == 0
    assert assoc[81] == 1
    assert assoc[82] == 6
    assert assoc[83] == 4
    assert assoc[84] == -1
    assert assoc[160] == 3
    assert assoc[161] == 4
    assert assoc[162] == 6
    assert assoc[163] == 0
    assert assoc[164] == -1
    assert assoc[240] == 3
    assert assoc[241] == 1
    assert assoc[242] == -1


def test_loc_exp():
    backend = "cython"
    num_p2 = 6
    length = 1
    in_r = 1.05
    with open(T_DESIGN, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)[num_p2]
    sph_pts = np.array(data['array'], dtype=np.float32)
    order = 4
    offset = 0
    cx = 0.25
    cy = 0.25
    cz = 0.25

    leg_lim = order//2+1
    siz_leg = leg_lim*(leg_lim+1)//2 - 2
    leg_lst = np.zeros(siz_leg, dtype=np.float32)
    count = 1
    for i in range(2, leg_lim):
        temp_lst = np.array(legendre(i)).astype(np.float32)
        leg_lst[count:count+i+1] = temp_lst[::-1]
        count += i+1

    sz_cell = sqrt(3.0)*length/4
    i2c_l = in_r*sz_cell
    r_in = sph_pts*i2c_l + 0.25
    in_val = np.arange(num_p2, dtype=np.float32)
    in_x = np.array(r_in[0::3])
    in_y = np.array(r_in[1::3])
    in_z = np.array(r_in[2::3])
    px = 0.35
    py = 0.25
    pz = 0.15
    cos_x = sph_pts[0::3]
    cos_z = -sph_pts[2::3]
    cos_f = (cos_x + cos_z) / sqrt(2)
    rr = 0.1 * sqrt(2) / i2c_l
    r_res = np.sum((in_val/num_p2)*(1+3*rr*cos_f+5*rr**2*(3*cos_f**2-1)/2))

    res = loc_exp(in_val, in_x, in_y, in_z, cx, cy, cz, px, py, pz, num_p2,
                  i2c_l, offset, leg_lst, leg_lim)

    assert pytest.approx(res, 1e-6) == r_res


def test_loc_exp_force():
    backend = "cython"
    num_p2 = 12
    length = 1
    in_r = 1.05
    with open(T_DESIGN, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)[num_p2]
    sph_pts = np.array(data['array'], dtype=np.float32)
    order = data['order']
    offset = 0
    cx = 0.25
    cy = 0.25
    cz = 0.25

    leg_lim = order//2+1
    siz_leg = leg_lim*(leg_lim+1)//2 - 2
    leg_lst = np.zeros(siz_leg, dtype=np.float32)
    dleg_lst = np.zeros(siz_leg, dtype=np.float32)
    count = 1
    for i in range(2, leg_lim):
        temp_lst = legendre(i)
        dtemp_lst = temp_lst.deriv()
        leg_lst[count:count+i+1] = np.array(temp_lst)[::-1]
        dleg_lst[count:count+i] = np.array(dtemp_lst)[::-1]
        count += i+1

    sz_cell = sqrt(3.0)*length/4
    i2c_l = in_r*sz_cell
    r_in = sph_pts*i2c_l + 0.25
    in_val = np.arange(num_p2, dtype=np.float32)
    r_in_x = np.array(sph_pts[0::3])
    r_in_y = np.array(sph_pts[1::3])
    r_in_z = np.array(sph_pts[2::3])
    in_x = np.array(r_in[0::3])
    in_y = np.array(r_in[1::3])
    in_z = np.array(r_in[2::3])
    px = 0.35
    py = 0.35
    pz = 0.15
    r = 0.1 * sqrt(3)
    cos_g = (sph_pts[0::3] + sph_pts[1::3] - sph_pts[2::3]) / sqrt(3)
    r_res_x = np.sum(in_val/num_p2*((1/i2c_l**2)*((3*i2c_l + 15*cos_g*r)*r_in_x
                                                  - 5*(px-cx))))
    r_res_y = np.sum(in_val/num_p2*((1/i2c_l**2)*((3*i2c_l + 15*cos_g*r)*r_in_y
                                                  - 5*(py-cy))))
    r_res_z = np.sum(in_val/num_p2*((1/i2c_l**2)*((3*i2c_l + 15*cos_g*r)*r_in_z
                                                  - 5*(pz-cz))))

    res_x = np.zeros(1, dtype=np.float32)
    res_y = np.zeros(1, dtype=np.float32)
    res_z = np.zeros(1, dtype=np.float32)

    loc_exp_force(
        in_val, in_x, in_y, in_z, cx, cy, cz, px, py, pz, num_p2, i2c_l,
        offset, leg_lst, dleg_lst, leg_lim, res_x, res_y, res_z, 0)

    assert pytest.approx(res_x[0], 1e-6) == r_res_x
    assert pytest.approx(res_y[0], 1e-6) == r_res_y
    assert pytest.approx(res_z[0], 1e-6) == r_res_z


@check_all_backends
def test_loc_coeff(backend):
    check_import(backend)
    dimension = 3
    N = 5
    max_depth = 2
    num_p2 = 12
    x_min = 0
    y_min = 0
    z_min = 0
    length = 1
    out_r = 1.1
    in_r = 1.35
    part_val = np.ones(N, dtype=np.float32)
    part_x = np.array([0.13, 0.12, 0.12, 0.12, 0.65], dtype=np.float32)
    part_y = np.array([0.12, 0.38, 0.62, 0.88, 0.3], dtype=np.float32)
    part_z = np.array([0.13, 0.12, 0.12, 0.12, 0.3], dtype=np.float32)
    with open(T_DESIGN, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)[num_p2]
    sph_pts = np.array(data['array'], dtype=np.float32)
    order = data['order']
    deleave_coeff = np.array([0x49249249, 0xC30C30C3, 0xF00F00F, 0xFF0000FF,
                              0x0000FFFF], dtype=np.int32)

    part_val, part_x, part_y, part_z, sph_pts, deleave_coeff = wrap(
        part_val, part_x, part_y, part_z, sph_pts, deleave_coeff,
        backend=backend)

    (cells, sfc, level, idx, bin_count, start_idx, leaf_idx, parent, child,
     part2bin, p2b_offset, lev_cs, levwise_cs, index, index_r, lev_index,
     lev_index_r, cx, cy, cz, out_x, out_y, out_z, in_x, in_y, in_z, out_val,
     in_val) = tree.build(
         N, max_depth, part_val, part_x, part_y, part_z, x_min, y_min, z_min,
         out_r, in_r, length, num_p2, backend, dimension, sph_pts, order,
         deleave_coeff)

    assoc = ary.empty(80*cells, dtype=np.int32, backend=backend)
    assoc.fill(-1)

    leg_lim = order//2+1
    siz_leg = leg_lim*(leg_lim+1)//2 - 2
    leg_lst = np.zeros(siz_leg, dtype=np.float32)
    count = 1
    for i in range(2, leg_lim):
        temp_lst = np.array(legendre(i)).astype(np.float32)
        leg_lst[count:count+i+1] = temp_lst[::-1]
        count += i+1

    leg_lst = wrap(leg_lst, backend=backend)

    ecalc_p2_fine = Elementwise(calc_p2_fine, backend=backend)
    ecalc_p2 = Elementwise(calc_p2, backend=backend)
    eassoc_coarse = Elementwise(assoc_coarse, backend=backend)
    efind_assoc = Elementwise(find_assoc, backend=backend)
    eloc_coeff = Elementwise(loc_coeff, backend=backend)

    ecalc_p2_fine(out_val[:lev_cs[max_depth-1]*num_p2], out_x, out_y, out_z,
                  part_val, part_x, part_y, part_z, cx, cy, cz, num_p2,
                  length, index, leg_lim, leg_lst, level, idx, out_r*sqrt(3),
                  bin_count, start_idx, leaf_idx)

    for lev in range(max_depth-1, 0, -1):
        m2c_l = out_r*sqrt(3)*length/(2**(lev+1))
        lev_offset = lev_cs[lev-1]-lev_cs[lev]
        if lev_offset == 0:
            continue
        ecalc_p2(out_val[:lev_offset*num_p2], out_val, out_x, out_y, out_z,
                 cx, cy, cz, num_p2, index, index_r, leg_lim, leg_lst, child,
                 lev_cs[lev], m2c_l)

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
               lev_index_r, idx, leaf_idx, start_idx, bin_count, length)

    res = loc_exp(in_val[:num_p2], in_x, in_y, in_z, cx[0], cy[0], cz[0],
                  part_x[0], part_y[0], part_z[0], num_p2,
                  sqrt(3)*length*in_r/8, 0, leg_lst, leg_lim)

    res_dir = 0
    for i in range(3, N):
        res_dir += direct_comp(part_val[i], part_x[i], part_y[i], part_z[i],
                               part_x[0], part_y[0], part_z[0])

    assert pytest.approx(res, abs=1e-3) == res_dir


@check_all_backends
def test_loc_coeff_force(backend):
    check_import(backend)
    dimension = 3
    N = 5
    max_depth = 2
    num_p2 = 24
    x_min = 0
    y_min = 0
    z_min = 0
    length = 1
    out_r = 1.1
    in_r = 1.05
    part_val = np.ones(N, dtype=np.float32)
    part_x = np.array([0.13, 0.12, 0.12, 0.12, 0.65], dtype=np.float32)
    part_y = np.array([0.12, 0.38, 0.62, 0.88, 0.3], dtype=np.float32)
    part_z = np.array([0.13, 0.12, 0.12, 0.12, 0.3], dtype=np.float32)
    with open(T_DESIGN, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)[num_p2]
    sph_pts = np.array(data['array'], dtype=np.float32)
    order = data['order']
    deleave_coeff = np.array([0x49249249, 0xC30C30C3, 0xF00F00F, 0xFF0000FF,
                              0x0000FFFF], dtype=np.int32)

    part_val, part_x, part_y, part_z, sph_pts, deleave_coeff = wrap(
        part_val, part_x, part_y, part_z, sph_pts, deleave_coeff,
        backend=backend)

    (cells, sfc, level, idx, bin_count, start_idx, leaf_idx, parent, child,
     part2bin, p2b_offset, lev_cs, levwise_cs, index, index_r, lev_index,
     lev_index_r, cx, cy, cz, out_x, out_y, out_z, in_x, in_y, in_z, out_val,
     in_val) = tree.build(
         N, max_depth, part_val, part_x, part_y, part_z, x_min, y_min, z_min,
         out_r, in_r, length, num_p2, backend, dimension, sph_pts, order,
         deleave_coeff)

    res_x = ary.zeros(1, dtype=np.float32, backend=backend)
    res_y = ary.zeros(1, dtype=np.float32, backend=backend)
    res_z = ary.zeros(1, dtype=np.float32, backend=backend)
    res_dir_x = ary.zeros(1, dtype=np.float32, backend=backend)
    res_dir_y = ary.zeros(1, dtype=np.float32, backend=backend)
    res_dir_z = ary.zeros(1, dtype=np.float32, backend=backend)
    assoc = ary.empty(80*cells, dtype=np.int32, backend=backend)
    assoc.fill(-1)

    leg_lim = order//2+1
    siz_leg = leg_lim*(leg_lim+1)//2 - 2
    leg_lst = np.zeros(siz_leg, dtype=np.float32)
    dleg_lst = np.zeros(siz_leg, dtype=np.float32)
    count = 1
    for i in range(2, leg_lim):
        temp_lst = legendre(i)
        dtemp_lst = temp_lst.deriv()
        leg_lst[count:count+i+1] = np.array(temp_lst)[::-1]
        dleg_lst[count:count+i] = np.array(dtemp_lst)[::-1]
        count += i+1

    leg_lst, dleg_lst = wrap(leg_lst, dleg_lst, backend=backend)

    ecalc_p2_fine = Elementwise(calc_p2_fine, backend=backend)
    ecalc_p2 = Elementwise(calc_p2, backend=backend)
    eassoc_coarse = Elementwise(assoc_coarse, backend=backend)
    efind_assoc = Elementwise(find_assoc, backend=backend)
    eloc_coeff = Elementwise(loc_coeff, backend=backend)

    ecalc_p2_fine(out_val[:lev_cs[max_depth-1]*num_p2], out_x, out_y, out_z,
                  part_val, part_x, part_y, part_z, cx, cy, cz, num_p2,
                  length, index, leg_lim, leg_lst, level, idx, out_r*sqrt(3),
                  bin_count, start_idx, leaf_idx)

    for lev in range(max_depth-1, 0, -1):
        m2c_l = out_r*sqrt(3)*length/(2**(lev+1))
        lev_offset = lev_cs[lev-1]-lev_cs[lev]
        if lev_offset == 0:
            continue
        ecalc_p2(out_val[:lev_offset*num_p2], out_val, out_x, out_y, out_z,
                 cx, cy, cz, num_p2, index, index_r, leg_lim, leg_lst, child,
                 lev_cs[lev], m2c_l)

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
               lev_index_r, idx, leaf_idx, start_idx, bin_count, length)

    loc_exp_force(in_val[:num_p2], in_x, in_y, in_z, cx[0], cy[0], cz[0],
                  part_x[0], part_y[0], part_z[0], num_p2,
                  sqrt(3)*length*in_r/8, 0, leg_lst, dleg_lst, leg_lim, res_x,
                  res_y, res_z, 0)

    for i in range(3, N):
        direct_comp_force(part_val[i], part_x[i], part_y[i], part_z[i],
                          part_x[0], part_y[0], part_z[0], res_dir_x,
                          res_dir_y, res_dir_z, 0)

    assert pytest.approx(res_x[0], abs=3e-3) == res_dir_x[0]
    assert pytest.approx(res_y[0], abs=3e-3) == res_dir_y[0]
    assert pytest.approx(res_z[0], abs=3e-3) == res_dir_z[0]


@check_all_backends
def test_trans_loc(backend):
    check_import(backend)
    dimension = 3
    N = 4
    max_depth = 3
    num_p2 = 24
    x_min = 0
    y_min = 0
    z_min = 0
    length = 1
    out_r = 1.1
    in_r = 1.35
    part_val = np.ones(N, dtype=np.float32)
    part_x = np.array([0.0625, 0.0625, 0.9375, 0.9375], dtype=np.float32)
    part_y = np.array([0.0625, 0.1875, 0.0625, 0.1875], dtype=np.float32)
    part_z = np.array([0.0625, 0.0625, 0.0625, 0.0625], dtype=np.float32)

    with open(T_DESIGN, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)[num_p2]
    sph_pts = np.array(data['array'], dtype=np.float32)
    order = data['order']
    deleave_coeff = np.array([0x49249249, 0xC30C30C3, 0xF00F00F, 0xFF0000FF,
                              0x0000FFFF], dtype=np.int32)

    part_val, part_x, part_y, part_z, sph_pts, deleave_coeff = wrap(
        part_val, part_x, part_y, part_z, sph_pts, deleave_coeff,
        backend=backend)

    (cells, sfc, level, idx, bin_count, start_idx, leaf_idx, parent, child,
     part2bin, p2b_offset, lev_cs, levwise_cs, index, index_r, lev_index,
     lev_index_r, cx, cy, cz, out_x, out_y, out_z, in_x, in_y, in_z, out_val,
     in_val) = tree.build(
         N, max_depth, part_val, part_x, part_y, part_z, x_min, y_min, z_min,
         out_r, in_r, length, num_p2, backend, dimension, sph_pts, order,
         deleave_coeff)

    assoc = ary.empty(80*cells, dtype=np.int32, backend=backend)
    assoc.fill(-1)

    leg_lim = order//2+1
    siz_leg = leg_lim*(leg_lim+1)//2 - 2
    leg_lst = np.zeros(siz_leg, dtype=np.float32)
    count = 1
    for i in range(2, leg_lim):
        temp_lst = np.array(legendre(i)).astype(np.float32)
        leg_lst[count:count+i+1] = temp_lst[::-1]
        count += i+1

    leg_lst = wrap(leg_lst, backend=backend)

    ecalc_p2_fine = Elementwise(calc_p2_fine, backend=backend)
    ecalc_p2 = Elementwise(calc_p2, backend=backend)
    eassoc_coarse = Elementwise(assoc_coarse, backend=backend)
    efind_assoc = Elementwise(find_assoc, backend=backend)
    eloc_coeff = Elementwise(loc_coeff, backend=backend)
    etrans_loc = Elementwise(trans_loc, backend=backend)

    ecalc_p2_fine(out_val[:lev_cs[max_depth-1]*num_p2], out_x, out_y, out_z,
                  part_val, part_x, part_y, part_z, cx, cy, cz, num_p2,
                  length, index, leg_lim, leg_lst, level, idx, out_r*sqrt(3),
                  bin_count, start_idx, leaf_idx)

    for lev in range(max_depth-1, 0, -1):
        m2c_l = out_r*sqrt(3)*length/(2**(lev+1))
        lev_offset = lev_cs[lev-1]-lev_cs[lev]
        if lev_offset == 0:
            continue
        ecalc_p2(out_val[:lev_offset*num_p2], out_val, out_x, out_y, out_z,
                 cx, cy, cz, num_p2, index, index_r, leg_lim, leg_lst, child,
                 lev_cs[lev], m2c_l)

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
               lev_index_r, idx, leaf_idx, start_idx, bin_count, length)

    for lev in range(3, max_depth+1):
        i2c_l = in_r*sqrt(3)*length/(2**(lev))
        lev_offset = levwise_cs[lev-1] - levwise_cs[lev]
        if lev_offset == 0:
            continue
        etrans_loc(in_val[:lev_offset*num_p2], in_val, in_x, in_y, in_z, cx,
                   cy, cz, i2c_l, num_p2, leg_lst, leg_lim, index_r, lev_index,
                   parent, levwise_cs[lev])

    res = loc_exp(in_val[:num_p2], in_x, in_y, in_z, cx[0], cy[0], cz[0],
                  part_x[0], part_y[0], part_z[0], num_p2,
                  sqrt(3)*length*in_r/16, 0, leg_lst, leg_lim)

    res_dir = 0
    for i in range(2, N):
        res_dir += direct_comp(part_val[i], part_x[i], part_y[i], part_z[i],
                               part_x[0], part_y[0], part_z[0])

    assert pytest.approx(res, abs=1e-3) == res_dir


@check_all_backends
def test_trans_loc_force(backend):
    check_import(backend)
    dimension = 3
    N = 4
    max_depth = 3
    num_p2 = 24
    x_min = 0
    y_min = 0
    z_min = 0
    length = 1
    out_r = 1.1
    in_r = 1.05
    part_val = np.ones(N, dtype=np.float32)
    part_x = np.array([0.062, 0.062, 0.937, 0.937], dtype=np.float32)
    part_y = np.array([0.062, 0.187, 0.312, 0.437], dtype=np.float32)
    part_z = np.array([0.062, 0.062, 0.187, 0.187], dtype=np.float32)
    with open(T_DESIGN, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)[num_p2]
    sph_pts = np.array(data['array'], dtype=np.float32)
    order = data['order']
    deleave_coeff = np.array([0x49249249, 0xC30C30C3, 0xF00F00F, 0xFF0000FF,
                              0x0000FFFF], dtype=np.int32)

    part_val, part_x, part_y, part_z, sph_pts, deleave_coeff = wrap(
        part_val, part_x, part_y, part_z, sph_pts, deleave_coeff,
        backend=backend)

    (cells, sfc, level, idx, bin_count, start_idx, leaf_idx, parent, child,
     part2bin, p2b_offset, lev_cs, levwise_cs, index, index_r, lev_index,
     lev_index_r, cx, cy, cz, out_x, out_y, out_z, in_x, in_y, in_z, out_val,
     in_val) = tree.build(
         N, max_depth, part_val, part_x, part_y, part_z, x_min, y_min, z_min,
         out_r, in_r, length, num_p2, backend, dimension, sph_pts, order,
         deleave_coeff)

    res_x = ary.zeros(1, dtype=np.float32, backend=backend)
    res_y = ary.zeros(1, dtype=np.float32, backend=backend)
    res_z = ary.zeros(1, dtype=np.float32, backend=backend)
    res_dir_x = ary.zeros(1, dtype=np.float32, backend=backend)
    res_dir_y = ary.zeros(1, dtype=np.float32, backend=backend)
    res_dir_z = ary.zeros(1, dtype=np.float32, backend=backend)
    assoc = ary.empty(80*cells, dtype=np.int32, backend=backend)
    assoc.fill(-1)

    leg_lim = order//2+1
    siz_leg = leg_lim*(leg_lim+1)//2 - 2
    leg_lst = np.zeros(siz_leg, dtype=np.float32)
    dleg_lst = np.zeros(siz_leg, dtype=np.float32)
    count = 1
    for i in range(2, leg_lim):
        temp_lst = legendre(i)
        dtemp_lst = temp_lst.deriv()
        leg_lst[count:count+i+1] = np.array(temp_lst)[::-1]
        dleg_lst[count:count+i] = np.array(dtemp_lst)[::-1]
        count += i+1

    leg_lst, dleg_lst = wrap(leg_lst, dleg_lst, backend=backend)

    ecalc_p2_fine = Elementwise(calc_p2_fine, backend=backend)
    ecalc_p2 = Elementwise(calc_p2, backend=backend)
    eassoc_coarse = Elementwise(assoc_coarse, backend=backend)
    efind_assoc = Elementwise(find_assoc, backend=backend)
    eloc_coeff = Elementwise(loc_coeff, backend=backend)
    etrans_loc = Elementwise(trans_loc, backend=backend)

    ecalc_p2_fine(out_val[:lev_cs[max_depth-1]*num_p2], out_x, out_y, out_z,
                  part_val, part_x, part_y, part_z, cx, cy, cz, num_p2,
                  length, index, leg_lim, leg_lst, level, idx, out_r*sqrt(3),
                  bin_count, start_idx, leaf_idx)

    for lev in range(max_depth-1, 0, -1):
        m2c_l = out_r*sqrt(3)*length/(2**(lev+1))
        lev_offset = lev_cs[lev-1]-lev_cs[lev]
        if lev_offset == 0:
            continue
        ecalc_p2(out_val[:lev_offset*num_p2], out_val, out_x, out_y, out_z,
                 cx, cy, cz, num_p2, index, index_r, leg_lim, leg_lst, child,
                 lev_cs[lev], m2c_l)

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
               lev_index_r, idx, leaf_idx, start_idx, bin_count, length)

    for lev in range(3, max_depth+1):
        i2c_l = in_r*sqrt(3)*length/(2**(lev))
        lev_offset = levwise_cs[lev-1] - levwise_cs[lev]
        if lev_offset == 0:
            continue
        etrans_loc(in_val[:lev_offset*num_p2], in_val, in_x, in_y, in_z, cx,
                   cy, cz, i2c_l, num_p2, leg_lst, leg_lim, index_r, lev_index,
                   parent, levwise_cs[lev])

    loc_exp_force(in_val[:num_p2], in_x, in_y, in_z, cx[0], cy[0], cz[0],
                  part_x[0], part_y[0], part_z[0], num_p2,
                  sqrt(3)*length*in_r/16, 0, leg_lst, dleg_lst, leg_lim, res_x,
                  res_y, res_z, 0)

    for i in range(2, N):
        direct_comp_force(part_val[i], part_x[i], part_y[i], part_z[i],
                          part_x[0], part_y[0], part_z[0], res_dir_x,
                          res_dir_y, res_dir_z, 0)

    assert pytest.approx(res_x[0], abs=1e-2) == res_dir_x[0]
    assert pytest.approx(res_y[0], abs=1e-2) == res_dir_y[0]
    assert pytest.approx(res_z[0], abs=1e-2) == res_dir_z[0]


@check_all_backends
def test_compute_force(backend):
    check_import(backend)
    dimension = 3
    N = 5
    max_depth = 3
    num_p2 = 24
    x_min = 0
    y_min = 0
    z_min = 0
    length = 1
    out_r = 1.1
    in_r = 1.05
    part_val = ary.ones(N, dtype=np.float32, backend=backend)
    part_x = np.array([0.12, 0.37, 0.062, 0.062, 0.37], dtype=np.float32)
    part_y = np.array([0.37, 0.37, 0.562, 0.687, 0.62], dtype=np.float32)
    part_z = np.array([0.12, 0.12, 0.062, 0.062, 0.12], dtype=np.float32)
    with open(T_DESIGN, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)[num_p2]
    sph_pts = np.array(data['array'], dtype=np.float32)
    order = data['order']
    deleave_coeff = np.array([0x49249249, 0xC30C30C3, 0xF00F00F, 0xFF0000FF,
                              0x0000FFFF], dtype=np.int32)

    part_x, part_y, part_z, sph_pts, deleave_coeff = wrap(
        part_x, part_y, part_z, sph_pts, deleave_coeff, backend=backend)

    (cells, sfc, level, idx, bin_count, start_idx, leaf_idx, parent, child,
     part2bin, p2b_offset, lev_cs, levwise_cs, index, index_r, lev_index,
     lev_index_r, cx, cy, cz, out_x, out_y, out_z, in_x, in_y, in_z, out_val,
     in_val) = tree.build(
         N, max_depth, part_val, part_x, part_y, part_z, x_min, y_min, z_min,
         out_r, in_r, length, num_p2, backend, dimension, sph_pts, order,
         deleave_coeff)

    res_x = ary.zeros(N, dtype=np.float32, backend=backend)
    res_y = ary.zeros(N, dtype=np.float32, backend=backend)
    res_z = ary.zeros(N, dtype=np.float32, backend=backend)
    res_dir_x = ary.zeros(N, dtype=np.float32, backend=backend)
    res_dir_y = ary.zeros(N, dtype=np.float32, backend=backend)
    res_dir_z = ary.zeros(N, dtype=np.float32, backend=backend)
    assoc = ary.empty(80*cells, dtype=np.int32, backend=backend)
    assoc.fill(-1)

    leg_lim = order//2+1
    siz_leg = leg_lim*(leg_lim+1)//2 - 2
    leg_lst = np.zeros(siz_leg, dtype=np.float32)
    dleg_lst = np.zeros(siz_leg, dtype=np.float32)
    count = 1
    for i in range(2, leg_lim):
        temp_lst = legendre(i)
        dtemp_lst = temp_lst.deriv()
        leg_lst[count:count+i+1] = np.array(temp_lst)[::-1]
        dleg_lst[count:count+i] = np.array(dtemp_lst)[::-1]
        count += i+1

    leg_lst, dleg_lst = wrap(leg_lst, dleg_lst, backend=backend)

    ecalc_p2_fine = Elementwise(calc_p2_fine, backend=backend)
    ecalc_p2 = Elementwise(calc_p2, backend=backend)
    eassoc_coarse = Elementwise(assoc_coarse, backend=backend)
    efind_assoc = Elementwise(find_assoc, backend=backend)
    eloc_coeff = Elementwise(loc_coeff, backend=backend)
    etrans_loc = Elementwise(trans_loc, backend=backend)
    ecompute = Elementwise(compute_force, backend=backend)
    edirect = Elementwise(direct_solv_force, backend=backend)

    ecalc_p2_fine(out_val[:lev_cs[max_depth-1]*num_p2], out_x, out_y, out_z,
                  part_val, part_x, part_y, part_z, cx, cy, cz, num_p2,
                  length, index, leg_lim, leg_lst, level, idx, out_r*sqrt(3),
                  bin_count, start_idx, leaf_idx)

    for lev in range(max_depth-1, 0, -1):
        m2c_l = out_r*sqrt(3)*length/(2**(lev+1))
        lev_offset = lev_cs[lev-1]-lev_cs[lev]
        if lev_offset == 0:
            continue
        ecalc_p2(out_val[:lev_offset*num_p2], out_val, out_x, out_y, out_z,
                 cx, cy, cz, num_p2, index, index_r, leg_lim, leg_lst, child,
                 lev_cs[lev], m2c_l)

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
               lev_index_r, idx, leaf_idx, start_idx, bin_count, length)

    for lev in range(3, max_depth+1):
        i2c_l = in_r*sqrt(3)*length/(2**(lev))
        lev_offset = levwise_cs[lev-1] - levwise_cs[lev]
        if lev_offset == 0:
            continue
        etrans_loc(in_val[:lev_offset*num_p2], in_val, in_x, in_y, in_z, cx,
                   cy, cz, i2c_l, num_p2, leg_lst, leg_lim, index_r, lev_index,
                   parent, levwise_cs[lev])

    ecompute(part2bin, p2b_offset, part_val, part_x, part_y, part_z, level,
             idx, parent, child, assoc, index_r, lev_index_r, leaf_idx,
             bin_count, start_idx, out_val, out_x, out_y, out_z, in_val, in_x,
             in_y, in_z, cx, cy, cz, res_x, res_y, res_z, leg_lst, dleg_lst,
             num_p2, leg_lim, in_r, length)

    edirect(part_val, part_x, part_y, part_z, res_dir_x, res_dir_y, res_dir_z,
            N)

    assert pytest.approx(res_x[0], rel=5e-4) == res_dir_x[0]
    assert pytest.approx(res_y[0], rel=5e-4) == res_dir_y[0]
    assert pytest.approx(res_z[0], rel=5e-4) == res_dir_z[0]
