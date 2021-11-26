from math import fabs, floor, sqrt

import compyle.array as ary
import numpy as np
from compyle.api import Elementwise, annotate, declare, wrap
from compyle.low_level import cast
from scipy.special import legendre

from ..api import find_span, read_initial_state, save_sim, timestep
from ..tree import build


@annotate(int="lst_len, idx", cos_g="float", lst="gfloatp", return_="float")
def lgndre(lst, cos_g, lst_len, idx):
    i = declare("int")
    result = declare("float")
    result = 0
    for i in range(lst_len):
        result += lst[idx + i] * (cos_g**i)

    return result


@annotate(int="i, num_p2, leg_lim", float="length, out_r",
          gintp="index, idx, level, bin_count, start_idx, leaf_idx",
          gfloatp="out_val, out_x, out_y, out_z, part_val, part_x, part_y, "
                  "part_z, cx, cy, cz, leg_lst")
def calc_p2_fine(i, out_val, out_x, out_y, out_z, part_val, part_x, part_y,
                 part_z, cx, cy, cz, num_p2, length, index, leg_lim, leg_lst,
                 level, idx, out_r, bin_count, start_idx, leaf_idx):
    leg, cid, bid, paid, sid, j = declare("int", 6)
    p2c, m2c = declare("matrix(3)", 2)
    m2c_l, p2c_l, cos_g, rr, leg_res, out_res = declare("float", 6)
    cid = cast(floor(i * 1.0 / num_p2), "int")
    cid = index[cid]
    m2c[0] = out_x[i] - cx[cid]
    m2c[1] = out_y[i] - cy[cid]
    m2c[2] = out_z[i] - cz[cid]

    m2c_l = out_r * length / (2.0**(level[cid] + 1))
    out_res = 0
    out_val[i] = 0
    bid = idx[cid]
    for j in range(bin_count[bid]):
        paid = leaf_idx[start_idx[bid] + j]
        p2c[0] = part_x[paid] - cx[cid]
        p2c[1] = part_y[paid] - cy[cid]
        p2c[2] = part_z[paid] - cz[cid]
        p2c_l = sqrt(p2c[0]**2 + p2c[1]**2 + p2c[2]**2)
        out_res += part_val[paid]
        if p2c_l != 0:
            rr = p2c_l / m2c_l
            cos_g = (m2c[0] * p2c[0] + m2c[1] * p2c[1] +
                     m2c[2] * p2c[2]) / (p2c_l * m2c_l)
            out_res += 3 * part_val[paid] * cos_g * rr
            sid = 1
            for leg in range(2, leg_lim):
                leg_res = lgndre(leg_lst, cos_g, leg + 1, sid)
                out_res += leg_res * (2 * leg + 1) * (rr**leg) * part_val[paid]
                sid += leg + 1

    out_val[i] = out_res / num_p2


@annotate(int="i, num_p2, leg_lim, offset", gintp="index, index_r, child",
          gfloatp="out_temp, out_val, out_x, out_y, out_z, cx, cy, cz, "
                  "leg_lst", m2c_l="float")
def calc_p2(i, out_temp, out_val, out_x, out_y, out_z, cx, cy, cz, num_p2,
            index, index_r, leg_lim, leg_lst, child, offset, m2c_l):
    j, k, leg, tid, cid, pid, sid, outid = declare("int", 8)
    p2c, m2c = declare("matrix(3)", 2)
    p2c_l, cos_g, rr, leg_res, out_res = declare("float", 5)
    cid = cast(floor(i * 1.0 / num_p2), "int")
    cid = index[offset + cid]
    outid = index_r[cid] * num_p2 + i % num_p2
    m2c[0] = out_x[outid] - cx[cid]
    m2c[1] = out_y[outid] - cy[cid]
    m2c[2] = out_z[outid] - cz[cid]
    out_res = 0
    out_val[outid] = 0
    pid = 8 * cid
    for j in range(8):
        if child[pid + j] == -1:
            break
        else:
            for k in range(num_p2):
                tid = index_r[child[pid + j]] * num_p2 + k
                p2c[0] = out_x[tid] - cx[cid]
                p2c[1] = out_y[tid] - cy[cid]
                p2c[2] = out_z[tid] - cz[cid]
                p2c_l = sqrt(p2c[0]**2 + p2c[1]**2 + p2c[2]**2)
                out_res += out_val[tid]
                if p2c_l != 0:
                    rr = p2c_l / m2c_l
                    cos_g = (m2c[0] * p2c[0] + m2c[1] * p2c[1] +
                             m2c[2] * p2c[2]) / (p2c_l * m2c_l)
                    out_res += 3 * out_val[tid] * cos_g * rr
                    sid = 1
                    for leg in range(2, leg_lim):
                        leg_res = lgndre(leg_lst, cos_g, leg + 1, sid)
                        out_res += (leg_res * (2 * leg + 1) * (rr**leg) *
                                    out_val[tid])
                        sid += leg + 1

    out_val[outid] = out_res / num_p2


@annotate(float="part_val, part_x, part_y, part_z, px, py, pz",
          return_="float")
def direct_comp(part_val, part_x, part_y, part_z, px, py, pz):
    res, dist = declare("float", 2)
    res = 0
    dist = sqrt((part_x - px)**2 + (part_y - py)**2 + (part_z - pz)**2)
    res = part_val / dist
    return res


@annotate(float="part_val, part_x, part_y, part_z, px, py, pz",
          gfloatp="res_x, res_y, res_z", pid="int")
def direct_comp_force(part_val, part_x, part_y, part_z, px, py, pz, res_x,
                      res_y, res_z, pid):
    dist = declare("float")
    dist = sqrt(((part_x - px)**2 + (part_y - py)**2 + (part_z - pz)**2 +
                 0.01**2)**3)
    res_x[pid] += part_val * (part_x - px) / dist
    res_y[pid] += part_val * (part_y - py) / dist
    res_z[pid] += part_val * (part_z - pz) / dist


@annotate(int="i, num_part", gfloatp="val, x, y, z, res_x, res_y, res_z")
def direct_solv_force(i, val, x, y, z, res_x, res_y, res_z, num_part):
    j = declare("int")
    for j in range(num_part):
        if j != i:
            direct_comp_force(val[j], x[j], y[j], z[j], x[i], y[i], z[i],
                              res_x, res_y, res_z, i)


@annotate(float="cx, cy, cz, cr, ax, ay, az, ar", return_="int")
def is_adj(cx, cy, cz, cr, ax, ay, az, ar):
    dis_x, dis_y, dis_z, rr = declare("float", 4)
    dis_x = fabs(cx - ax)
    dis_y = fabs(cy - ay)
    dis_z = fabs(cz - az)
    rr = cr + ar

    if dis_x > rr or dis_y > rr or dis_z > rr:
        return 0
    else:
        return 1


@annotate(float="cx, cy, cz, cr, ax, ay, az", return_="int")
def well_sep(cx, cy, cz, cr, ax, ay, az):
    dis = declare("float")
    dis = sqrt((cx - ax)**2 + (cy - ay)**2 + (cz - az)**2)

    if dis >= 3 * cr:
        return 1
    else:
        return 0


@annotate(int="i, offset", gintp="sfc, parent, child, index, assoc")
def assoc_coarse(i, sfc, parent, child, index, assoc, offset):
    cid, rid, pid, j, count = declare("int", 5)
    cid = i + offset
    rid = index[cid]
    pid = parent[rid] * 8
    count = 0
    for j in range(8):
        if child[pid + j] == -1:
            break
        elif child[pid + j] != rid:
            assoc[80 * cid + count] = child[pid + j]
            count += 1
        else:
            continue


@annotate(int="i, offset", length="float", gfloatp="cx, cy, cz",
          gintp="sfc, level, assoc, child, parent, index, index_r, idx")
def find_assoc(i, idx, cx, cy, cz, level, assoc, child, parent, offset,
               index, index_r, length):

    bid, rid, paid, pid, aid, cid = declare("int", 6)
    count, lev, j, k, adj, well = declare("int", 6)
    cr, cR = declare("float", 2)
    bid = i + offset
    rid = index[bid]
    paid = parent[rid]
    pid = index_r[paid]
    count = 0
    lev = level[rid]
    cr = length / (2.0**(lev + 1))
    cR = cr * sqrt(3.0)
    for j in range(80):
        aid = assoc[80 * pid + j]

        if aid == -1:
            break

        if idx[aid] != -1:
            adj = is_adj(cx[aid], cy[aid], cz[aid],
                         length / (2.0**(level[aid] + 1)), cx[rid], cy[rid],
                         cz[rid], cr)
            if adj == 0:
                continue
            else:
                assoc[80 * bid + count] = aid
                count += 1
        else:
            for k in range(8):
                cid = child[8 * aid + k]
                if cid == -1:
                    break

                well = well_sep(cx[cid], cy[cid], cz[cid], cR,
                                cx[rid], cy[rid], cz[rid])
                if well == 1:
                    continue
                else:
                    assoc[80 * bid + count] = cid
                    count += 1

    for j in range(8):
        cid = child[8 * paid + j]
        if cid == -1:
            break
        if cid != rid:
            assoc[80 * bid + count] = cid
            count += 1


@annotate(gfloatp="part_val, part_x, part_y, part_z, res_x, res_y, res_z",
          leaf_idx="gintp", int="num_own, sid, pid")
def own_cell_force(part_val, part_x, part_y, part_z, leaf_idx, num_own, sid,
                   pid, res_x, res_y, res_z):
    n, oid = declare("int", 2)
    for n in range(num_own):
        oid = leaf_idx[sid + n]
        if oid != pid:
            direct_comp_force(
                part_val[oid], part_x[oid], part_y[oid], part_z[oid],
                part_x[pid], part_y[pid], part_z[pid], res_x, res_y, res_z,
                pid)


@annotate(gfloatp="part_val, part_x, part_y, part_z, res_x, res_y, res_z",
          leaf_idx="gintp", int="num_u, sid, pid")
def u_list_force(part_val, part_x, part_y, part_z, leaf_idx, num_u, sid, pid,
                 res_x, res_y, res_z):
    n, uid = declare("int", 2)
    for n in range(num_u):
        uid = leaf_idx[sid + n]
        direct_comp_force(
            part_val[uid], part_x[uid], part_y[uid], part_z[uid], part_x[pid],
            part_y[pid], part_z[pid], res_x, res_y, res_z, pid)


@annotate(gfloatp="out_val, out_x, out_y, out_z, part_x, part_y, part_z, "
                  "res_x, res_y, res_z", int="num_p2, wid, pid")
def w_list_force(out_val, out_x, out_y, out_z, part_x, part_y, part_z, wid,
                 num_p2, pid, res_x, res_y, res_z):
    wnid, n = declare("int", 2)
    for n in range(num_p2):
        wnid = wid + n
        direct_comp_force(
            out_val[wnid], out_x[wnid], out_y[wnid], out_z[wnid], part_x[pid],
            part_y[pid], part_z[pid], res_x, res_y, res_z, pid)


@annotate(int="chid, num_p2", float="in_x, in_y, in_z",
          gfloatp="out_val, out_x, out_y, out_z", return_="float")
def v_list(in_x, in_y, in_z, out_val, out_x, out_y, out_z, num_p2, chid):
    n, vid = declare("int", 2)
    res = declare("float")
    res = 0
    for n in range(num_p2):
        vid = chid + n
        res += direct_comp(out_val[vid], out_x[vid], out_y[vid], out_z[vid],
                           in_x, in_y, in_z)

    return res


@annotate(float="in_x, in_y, in_z", gfloatp="part_val, part_x, part_y, part_z",
          int="sid, num_part", leaf_idx="gintp", return_="float")
def z_list(in_x, in_y, in_z, part_val, part_x, part_y, part_z, sid,
           leaf_idx, num_part):
    n, cid = declare("int", 2)
    res = declare("float")
    res = 0
    for n in range(num_part):
        cid = leaf_idx[sid + n]
        res += direct_comp(part_val[cid], part_x[cid], part_y[cid],
                           part_z[cid], in_x, in_y, in_z)
    return res


@annotate(gintp="assoc, child, parent, index, index_r, lev_index_r, idx, "
                "leaf_idx, start_idx, bin_count, level",
          gfloatp="in_val, in_x, in_y, in_z, out_val, out_x, out_y, out_z, "
                  "part_val, part_x, part_y, part_z, cx, cy, cz",
          length="float", int="i, num_p2")
def loc_coeff(i, in_val, in_x, in_y, in_z, out_val, out_x, out_y, out_z,
              part_val, part_x, part_y, part_z, cx, cy, cz, assoc, child,
              parent, num_p2, level, index, index_r, lev_index_r, idx,
              leaf_idx, start_idx, bin_count, length):
    j, k, cid, pid, aid, chid, well, adj, paid, inid = declare("int", 10)
    cr, cR = declare("float", 2)
    cid = cast(floor(i * 1.0 / num_p2), "int")
    cid = index[cid]
    cr = length / (2.0**(level[cid] + 1))
    cR = sqrt(3.0) * cr
    pid = lev_index_r[parent[cid]]
    inid = index_r[cid] * num_p2 + i % num_p2
    in_val[inid] = 0
    for j in range(80):
        aid = assoc[80 * pid + j]
        if aid != -1:
            paid = idx[aid]
            if paid == -1:
                for k in range(8):
                    chid = child[8 * aid + k]
                    if chid == -1:
                        break

                    # HACK: cells which are neither well seperated
                    # nor adjacent are considered v list cells
                    else:
                        well = well_sep(cx[cid], cy[cid], cz[cid], cR,
                                        cx[chid], cy[chid], cz[chid])
                        if well == 1:
                            in_val[inid] += v_list(
                                in_x[inid], in_y[inid], in_z[inid], out_val,
                                out_x, out_y, out_z, num_p2,
                                num_p2 * index_r[chid])

            else:
                adj = is_adj(cx[cid], cy[cid], cz[cid], cr, cx[aid], cy[aid],
                             cz[aid], length / (2.0**(level[aid] + 1)))
                if adj != 1:
                    in_val[inid] += z_list(
                        in_x[inid], in_y[inid], in_z[inid], part_val, part_x,
                        part_y, part_z, start_idx[idx[aid]], leaf_idx,
                        bin_count[idx[aid]])
        else:
            break


@annotate(int="offset, leg_lim, num_p2", float="cx, cy, cz, px, py, pz, i2c_l",
          gfloatp="in_val, in_x, in_y, in_z, leg_lst", return_="float")
def loc_exp(in_val, in_x, in_y, in_z, cx, cy, cz, px, py, pz, num_p2, i2c_l,
            offset, leg_lst, leg_lim):
    j, leg, s1id, sid = declare("int", 4)
    p2c, i2c = declare("matrix(3)", 2)
    res, p2c_l, cos_g, rr, leg_res = declare("float", 5)
    p2c[0] = px - cx
    p2c[1] = py - cy
    p2c[2] = pz - cz
    p2c_l = sqrt(p2c[0]**2 + p2c[1]**2 + p2c[2]**2)
    res = 0
    for j in range(num_p2):
        s1id = offset + j
        i2c[0] = in_x[s1id] - cx
        i2c[1] = in_y[s1id] - cy
        i2c[2] = in_z[s1id] - cz
        res += in_val[s1id]
        if p2c_l != 0:
            cos_g = (i2c[0] * p2c[0] + i2c[1] * p2c[1] +
                     i2c[2] * p2c[2]) / (p2c_l * i2c_l)
            rr = p2c_l / i2c_l
            res += 3 * rr * cos_g * in_val[s1id]
            sid = 1
            for leg in range(2, leg_lim):
                leg_res = lgndre(leg_lst, cos_g, leg + 1, sid)
                res += leg_res * (2 * leg + 1) * (rr**leg) * in_val[s1id]
                sid += leg + 1
    res = res / num_p2
    return res


@annotate(gfloatp="in_val, in_x, in_y, in_z, leg_lst, dleg_lst, res_x, res_y, "
          "res_z", float="cx, cy, cz, px, py, pz, i2c_l",
          int="offset, leg_lim, num_p2, pid")
def loc_exp_force(in_val, in_x, in_y, in_z, cx, cy, cz, px, py, pz, num_p2,
                  i2c_l, offset, leg_lst, dleg_lst, leg_lim, res_x, res_y,
                  res_z, pid):
    j, s1id, sid, leg = declare("int", 4)
    p2c, i2c = declare("matrix(3)", 2)
    p2c_l, leg_res, cos_g, rr, pre_f, dcgx = declare("float", 6)
    dcgy, dcgz, x_fac, y_fac, z_fac, cur_fac = declare("float", 6)
    rx, ry, rz = declare("float", 3)
    rx = 0
    ry = 0
    rz = 0
    p2c[0] = px - cx
    p2c[1] = py - cy
    p2c[2] = pz - cz
    p2c_l = sqrt(p2c[0]**2 + p2c[1]**2 + p2c[2]**2)
    for j in range(num_p2):
        s1id = offset + j
        i2c[0] = in_x[s1id] - cx
        i2c[1] = in_y[s1id] - cy
        i2c[2] = in_z[s1id] - cz
        rx += 3 * i2c[0] * in_val[s1id] / (i2c_l**2)
        ry += 3 * i2c[1] * in_val[s1id] / (i2c_l**2)
        rz += 3 * i2c[2] * in_val[s1id] / (i2c_l**2)
        if p2c_l != 0:
            cg = (i2c[0] * p2c[0] + i2c[1] * p2c[1] +
                  i2c[2] * p2c[2]) / (p2c_l * i2c_l)
            rr = p2c_l / i2c_l
            dcgx = (i2c[0] * rr - cg * p2c[0])
            dcgy = (i2c[1] * rr - cg * p2c[1])
            dcgz = (i2c[2] * rr - cg * p2c[2])
            sid = 1
            for leg in range(2, leg_lim):
                pre_f = ((2 * leg + 1) * (rr**(leg - 2)) *
                         in_val[s1id] / (i2c_l**2))
                leg_res = lgndre(leg_lst, cg, leg + 1, sid)
                dleg_res = lgndre(dleg_lst, cg, leg + 1, sid)
                sid += leg + 1
                x_fac = leg * p2c[0] * leg_res + dleg_res * dcgx
                y_fac = leg * p2c[1] * leg_res + dleg_res * dcgy
                z_fac = leg * p2c[2] * leg_res + dleg_res * dcgz

                rx += pre_f * x_fac
                ry += pre_f * y_fac
                rz += pre_f * z_fac

    res_x[pid] += rx / num_p2
    res_y[pid] += ry / num_p2
    res_z[pid] += rz / num_p2


@annotate(int="i, num_p2, leg_lim, offset", i2c_l="float",
          gfloatp="in_temp, in_val, cx, cy, cz, in_x, in_y, in_z, leg_lst",
          gintp="index_r, lev_index, parent")
def trans_loc(i, in_temp, in_val, in_x, in_y, in_z, cx, cy, cz, i2c_l, num_p2,
              leg_lst, leg_lim, index_r, lev_index, parent, offset):
    pid, cid, tid, inid = declare("int", 4)
    cid = cast(floor(i * 1.0 / num_p2), "int")
    cid = lev_index[cid + offset]
    inid = index_r[cid] * num_p2 + i % num_p2
    pid = parent[cid]
    tid = index_r[pid] * num_p2
    in_val[inid] += loc_exp(in_val, in_x, in_y, in_z, cx[pid], cy[pid],
                            cz[pid], in_x[inid], in_y[inid], in_z[inid],
                            num_p2, i2c_l, tid, leg_lst, leg_lim)


@annotate(gintp="part2bin, p2b_offset, level, idx, parent, child, assoc, "
                "index_r, lev_index_r, leaf_idx, bin_count, start_idx",
          float="in_r, length", int="i, num_p2, leg_lim",
          gfloatp="part_val, part_x, part_y, part_z, out_val, out_x, out_y, "
                  "out_z, in_val, in_x, in_y, in_z, cx, cy, cz, res_x, res_y, "
                  "res_z, leg_lst, dleg_lst")
def compute_force(i, part2bin, p2b_offset, part_val, part_x, part_y, part_z,
                  level, idx, parent, child, assoc, index_r, lev_index_r,
                  leaf_idx, bin_count, start_idx, out_val, out_x, out_y, out_z,
                  in_val, in_x, in_y, in_z, cx, cy, cz, res_x, res_y, res_z,
                  leg_lst, dleg_lst, num_p2, leg_lim, in_r, length):
    h = declare('matrix(10, "int")')
    j, n, bid, baid, brid, lev, pid, aid, chid, t, adj = declare("int", 11)
    i2c_l = declare("float")

    for t in range(10):
        h[t] = 0

    bid = part2bin[i]
    brid = index_r[bid]
    baid = lev_index_r[bid]
    lev = level[bid]
    pid = leaf_idx[start_idx[idx[bid]] + p2b_offset[i]]
    i2c_l = in_r * sqrt(3.0) * length / (2.0**(lev + 1))
    cr_bid = length / (2.0**(lev + 1))

    own_cell_force(
        part_val, part_x, part_y, part_z, leaf_idx, bin_count[idx[bid]],
        start_idx[idx[bid]], pid, res_x, res_y, res_z)

    # calculation of potential using U and W interaction lists
    for j in range(80):
        aid = assoc[80 * baid + j]
        if aid == -1:
            break

        if level[aid] < lev:
            u_list_force(
                part_val, part_x, part_y, part_z, leaf_idx,
                bin_count[idx[aid]], start_idx[idx[aid]],
                pid, res_x, res_y, res_z)
        else:
            while True:
                if idx[aid] == -1:
                    for n in range(h[level[aid]], 8):
                        chid = child[8 * aid + n]
                        if chid == -1:
                            h[level[aid]] = -1
                            break
                        adj = is_adj(cx[chid], cy[chid], cz[chid],
                                     length / (2.0**(level[chid] + 1)),
                                     cx[bid], cy[bid], cz[bid], cr_bid)
                        if adj == 0:
                            w_list_force(
                                out_val, out_x, out_y, out_z, part_x, part_y,
                                part_z, index_r[chid] * num_p2, num_p2, pid,
                                res_x, res_y, res_z)
                        else:
                            h[level[aid]] = n + 1
                            aid = chid
                            break

                    if h[level[aid]] == -1 or child[8 * aid + 7] != -1:
                        h[level[aid]] = 0
                        if level[aid] == lev:
                            break
                        aid = parent[aid]

                else:
                    u_list_force(
                        part_val, part_x, part_y, part_z, leaf_idx,
                        bin_count[idx[aid]], start_idx[idx[aid]], pid,
                        res_x, res_y, res_z)
                    if level[aid] == lev:
                        break
                    else:
                        aid = parent[aid]

    # calculation using the local expansions
    loc_exp_force(in_val, in_x, in_y, in_z, cx[bid], cy[bid], cz[bid],
                  part_x[pid], part_y[pid], part_z[pid], num_p2, i2c_l,
                  brid * num_p2, leg_lst, dleg_lst, leg_lim, res_x, res_y,
                  res_z, pid)


def solver_force(part_val, part_x, part_y, part_z, vel_x, vel_y, vel_z,
                 backend, direct_call=False):

    data = read_initial_state()
    N = data["N"]
    max_depth = data["max_depth"]
    n_max = data["n_max"]
    out_r = data["out_r"]
    in_r = data["in_r"]
    num_p2 = data["num_p2"]
    dimension = data["dimension"]
    sph_pts = np.array(data['array'], dtype=np.float32)
    order = data['order']
    dt = data['dt']
    Ns = data['Ns']
    step = data['step']
    deleave_coeff = np.array([0x49249249, 0xC30C30C3, 0xF00F00F, 0xFF0000FF,
                              0x0000FFFF], dtype=np.int32)

    p_count = 0

    (part_val, part_x, part_y, part_z, vel_x, vel_y, vel_z, sph_pts,
     deleave_coeff) = wrap(
        part_val, part_x, part_y, part_z, vel_x, vel_y, vel_z, sph_pts,
        deleave_coeff, backend=backend)

    edirect = Elementwise(direct_solv_force, backend=backend)
    etimestep = Elementwise(timestep, backend=backend)

    leg_lim = order // 2 + 1
    siz_leg = leg_lim * (leg_lim + 1) // 2 - 2
    leg_lst = np.zeros(siz_leg, dtype=np.float32)
    dleg_lst = np.zeros(siz_leg, dtype=np.float32)
    count = 1
    for i in range(2, leg_lim):
        temp_lst = legendre(i)
        dtemp_lst = temp_lst.deriv()
        leg_lst[count:count + i + 1] = np.array(temp_lst)[::-1]
        dleg_lst[count:count + i] = np.array(dtemp_lst)[::-1]
        count += i + 1

    leg_lst, dleg_lst = wrap(leg_lst, dleg_lst, backend=backend)

    save_sim(part_x, part_y, part_z, -1, backend)

    for sim in range(Ns):
        res_x = ary.zeros(N, dtype=np.float32, backend=backend)
        res_y = ary.zeros(N, dtype=np.float32, backend=backend)
        res_z = ary.zeros(N, dtype=np.float32, backend=backend)

        if direct_call:
            edirect(part_val, part_x, part_y, part_z, res_x, res_y,
                    res_z, N)

        else:
            length, x_min, y_min, z_min = find_span(part_x, part_y, part_z)
            (cells, sfc, level, idx, bin_count, start_idx, leaf_idx, parent,
             child, part2bin, p2b_offset, lev_cs, levwise_cs, index, index_r,
             lev_index, lev_index_r, cx, cy, cz, out_x, out_y, out_z, in_x,
             in_y, in_z, out_val, in_val) = build(
                 N, max_depth, n_max, part_x, part_y, part_z, x_min, y_min,
                 z_min, out_r, in_r, length, num_p2, backend, dimension,
                 sph_pts, deleave_coeff)

            assoc = ary.empty(80 * cells, dtype=np.int32, backend=backend)
            assoc.fill(-1)

            ecalc_p2_fine = Elementwise(calc_p2_fine, backend=backend)
            ecalc_p2 = Elementwise(calc_p2, backend=backend)
            eassoc_coarse = Elementwise(assoc_coarse, backend=backend)
            efind_assoc = Elementwise(find_assoc, backend=backend)
            eloc_coeff = Elementwise(loc_coeff, backend=backend)
            etrans_loc = Elementwise(trans_loc, backend=backend)
            ecompute = Elementwise(compute_force, backend=backend)

            ecalc_p2_fine(
                out_val[:lev_cs[max_depth - 1] * num_p2], out_x, out_y, out_z,
                part_val, part_x, part_y, part_z, cx, cy, cz, num_p2, length,
                index, leg_lim, leg_lst, level, idx, out_r * sqrt(3),
                bin_count, start_idx, leaf_idx)

            for lev in range(max_depth - 1, 0, -1):
                m2c_l = out_r * sqrt(3) * length / (2**(lev + 1))
                lev_offset = lev_cs[lev - 1] - lev_cs[lev]
                if lev_offset == 0:
                    continue
                ecalc_p2(
                    out_val[:lev_offset * num_p2], out_val, out_x, out_y,
                    out_z, cx, cy, cz, num_p2, index, index_r, leg_lim,
                    leg_lst, child, lev_cs[lev], m2c_l)

            eassoc_coarse(
                sfc[levwise_cs[1]:levwise_cs[0]], parent, child, lev_index,
                assoc, levwise_cs[1])

            for lev in range(2, max_depth + 1):
                lev_offset = levwise_cs[lev - 1] - levwise_cs[lev]
                if lev_offset == 0:
                    continue
                efind_assoc(
                    idx[:lev_offset], cx, cy, cz, level, assoc, child, parent,
                    levwise_cs[lev], lev_index, lev_index_r, length)

            eloc_coeff(
                in_val[:lev_cs[1] * num_p2], in_x, in_y, in_z, out_val, out_x,
                out_y, out_z, part_val, part_x, part_y, part_z, cx, cy, cz,
                assoc, child, parent, num_p2, level, index, index_r,
                lev_index_r, idx, leaf_idx, start_idx, bin_count, length)

            for lev in range(3, max_depth + 1):
                i2c_l = in_r * sqrt(3) * length / (2**(lev))
                lev_offset = levwise_cs[lev - 1] - levwise_cs[lev]
                if lev_offset == 0:
                    continue
                etrans_loc(
                    in_val[:lev_offset * num_p2], in_val, in_x, in_y, in_z, cx,
                    cy, cz, i2c_l, num_p2, leg_lst, leg_lim, index_r,
                    lev_index, parent, levwise_cs[lev])

            ecompute(
                part2bin, p2b_offset, part_val, part_x, part_y, part_z, level,
                idx, parent, child, assoc, index_r, lev_index_r, leaf_idx,
                bin_count, start_idx, out_val, out_x, out_y, out_z, in_val,
                in_x, in_y, in_z, cx, cy, cz, res_x, res_y, res_z, leg_lst,
                dleg_lst, num_p2, leg_lim, in_r, length)

            sfc.resize(0)
            level.resize(0)
            idx.resize(0)
            bin_count.resize(0)
            start_idx.resize(0)
            leaf_idx.resize(0)
            parent.resize(0)
            child.resize(0)
            part2bin.resize(0)
            p2b_offset.resize(0)
            lev_cs.resize(0)
            levwise_cs.resize(0)
            index.resize(0)
            index_r.resize(0)
            lev_index.resize(0)
            lev_index_r.resize(0)
            cx.resize(0)
            cy.resize(0)
            cz.resize(0)
            out_x.resize(0)
            out_y.resize(0)
            out_z.resize(0)
            in_x.resize(0)
            in_y.resize(0)
            in_z.resize(0)
            out_val.resize(0)
            in_val.resize(0)

        etimestep(part_x, part_y, part_z, vel_x, vel_y, vel_z, res_x, res_y,
                  res_z, dt)

        res_x.resize(0)
        res_y.resize(0)
        res_z.resize(0)

        if sim % step == 0:
            save_sim(part_x, part_y, part_z, p_count, backend)
            p_count += 1
