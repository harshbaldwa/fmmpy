# TODO: Rearrange all the function variables

from math import fabs, floor, sqrt

import compyle.array as ary
import numpy as np
from compyle.api import Elementwise, annotate, declare, wrap
from compyle.low_level import cast
from scipy.special import legendre


@annotate(int="lst_len, idx", cos_g="float",
          lst="gfloatp", return_="float")
def lgndre(lst, cos_g, lst_len, idx):
    i = declare("int")
    result = declare("float")
    result = 0
    for i in range(lst_len):
        result += lst[idx+i]*(cos_g**i)

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
    cid = cast(floor(i*1.0/num_p2), "int")
    cid = index[cid]
    m2c[0] = out_x[i] - cx[cid]
    m2c[1] = out_y[i] - cy[cid]
    m2c[2] = out_z[i] - cz[cid]

    m2c_l = out_r*length/(2.0**(level[cid]+1))
    out_res = 0
    out_val[i] = 0
    bid = idx[cid]
    for j in range(bin_count[bid]):
        paid = leaf_idx[start_idx[bid]+j]
        p2c[0] = part_x[paid] - cx[cid]
        p2c[1] = part_y[paid] - cy[cid]
        p2c[2] = part_z[paid] - cz[cid]
        p2c_l = sqrt(p2c[0]**2 + p2c[1]**2 + p2c[2]**2)
        out_res += part_val[paid]
        if p2c_l != 0:
            rr = p2c_l/m2c_l
            cos_g = (m2c[0]*p2c[0] + m2c[1]*p2c[1] +
                     m2c[2]*p2c[2]) / (p2c_l * m2c_l)
            sid = 0
            for leg in range(1, leg_lim):
                leg_res = lgndre(leg_lst, cos_g, leg+1, sid)
                out_res += leg_res*(2*leg+1)*(rr**leg)*part_val[paid]
                sid += leg+1

    out_val[i] = out_res/num_p2


@annotate(int="i, num_p2, leg_lim, offset", gintp="index, index_r, child",
          gfloatp="out_val, out_x, out_y, out_z, outc_val, outc_x, outc_y, "
                  "outc_z, cx, cy, cz, leg_lst", m2c_l="float")
def calc_p2(i, out_val, out_x, out_y, out_z, outc_val, outc_x, outc_y, outc_z,
            cx, cy, cz, num_p2, index, index_r, leg_lim, leg_lst,
            child, offset, m2c_l):
    j, k, leg, tid, cid, pid, sid = declare("int", 7)
    p2c, m2c = declare("matrix(3)", 2)
    p2c_l, cos_g, rr, leg_res, out_res = declare("float", 5)
    cid = cast(floor(i*1.0/num_p2), "int")
    # REM: offset here is level_cs[level]
    cid = index[offset+cid]
    m2c[0] = out_x[i] - cx[cid]
    m2c[1] = out_y[i] - cy[cid]
    m2c[2] = out_z[i] - cz[cid]
    # REM: precomputed and passed as a variable m2c_l
    # m2c_l = out_r*sqrt(3.0)*length/(2.0**(level+1))
    out_res = 0
    out_val[i] = 0
    pid = 8*cid
    for j in range(8):
        if child[pid+j] == -1:
            break
        else:
            for k in range(num_p2):
                tid = index_r[child[pid+j]]*num_p2+k
                p2c[0] = outc_x[tid] - cx[cid]
                p2c[1] = outc_y[tid] - cy[cid]
                p2c[2] = outc_z[tid] - cz[cid]
                p2c_l = sqrt(p2c[0]**2 + p2c[1]**2 + p2c[2]**2)
                out_res += outc_val[tid]
                if p2c_l != 0:
                    rr = p2c_l/m2c_l
                    cos_g = (m2c[0]*p2c[0] + m2c[1]*p2c[1] +
                             m2c[2]*p2c[2]) / (p2c_l * m2c_l)
                    sid = 0
                    for leg in range(1, leg_lim):
                        leg_res = lgndre(leg_lst, cos_g, leg+1, sid)
                        out_res += leg_res*(2*leg+1)*(rr**leg)*outc_val[tid]
                        sid += leg+1

    out_val[i] = out_res/num_p2


@annotate(float="part_val, part_x, part_y, part_z, px, py, pz",
          return_="float")
def direct_comp(part_val, part_x, part_y, part_z, px, py, pz):
    value, dist = declare("float", 2)
    value = 0
    dist = sqrt((part_x-px)**2 + (part_y-py)**2 + (part_z-pz)**2)
    value += part_val/dist

    return value


@annotate(int="i, N", gfloatp="val, x, y, z, res")
def direct_solv(i, val, x, y, z, res, N):
    j = declare("int")
    res[i] = 0
    for j in range(N):
        if j != i:
            res[i] += direct_comp(val[j], x[j], y[j],
                                  z[j], x[i], y[i], z[i])


@annotate(float="cx, cy, cz, cr, ax, ay, az, ar",
          return_="int")
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


@annotate(float="cx, cy, cz, cr, ax, ay, az, ar",
          return_="int")
def well_sep(cx, cy, cz, cr, ax, ay, az, ar):
    dis = declare("float")
    dis = sqrt((cx-ax)**2 + (cy-ay)**2 + (cz-az)**2)

    if dis >= 2*sqrt(3)*cr + ar:
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
        if child[pid+j] == -1:
            break
        elif child[pid+j] != rid:
            assoc[26*cid+count] = child[pid+j]
            count += 1
        else:
            continue


@annotate(int="i, offset", length="float", gfloatp="cx, cy, cz",
          gintp="sfc, level, assoc, child, parent, index, index_r, idx")
def find_assoc(i, idx, cx, cy, cz, level, assoc, child, parent, offset,
               index, index_r, length):

    bid, rid, paid, pid, aid, cid, count, lev, j, k, adj = declare("int", 11)
    cr = declare("float")
    bid = i + offset
    rid = index[bid]
    paid = parent[rid]
    pid = index_r[paid]
    count = 0
    lev = level[rid]
    cr = length/(2.0**(lev+1))
    for j in range(26):
        aid = assoc[26*pid+j]

        if aid == -1:
            break

        if idx[aid] != -1:
            adj = is_adj(cx[aid], cy[aid], cz[aid],
                         length/(2.0**(level[aid]+1)), cx[rid], cy[rid],
                         cz[rid], cr)
            if adj == 0:
                continue
            else:
                assoc[26*bid+count] = aid
                count += 1
        else:
            for k in range(8):
                cid = child[8*aid+k]
                if cid == -1:
                    break
                adj = is_adj(cx[cid], cy[cid], cz[cid],
                             length/(2.0**(level[cid]+1)),
                             cx[rid], cy[rid], cz[rid],
                             length/(2.0**(lev+1)))
                if adj == 0:
                    continue
                else:
                    assoc[26*bid+count] = cid
                    count += 1

    for j in range(8):
        cid = child[8*paid+j]
        if cid == -1:
            break
        if cid != rid:
            assoc[26*bid+count] = cid
            count += 1


@annotate(gfloatp="part_val, part_x, part_y, part_z", leaf_idx="gintp",
          int="num_own, sid, pid", return_="float")
def own_cell(part_val, part_x, part_y, part_z, leaf_idx, num_own, sid, pid):
    n, oid = declare("int", 2)
    res = declare("float")
    res = 0
    for n in range(num_own):
        oid = leaf_idx[sid + n]
        if oid != pid:
            res += direct_comp(part_val[oid], part_x[oid], part_y[oid],
                               part_z[oid], part_x[pid], part_y[pid],
                               part_z[pid])

    return res


@annotate(gfloatp="result, part_val, part_x, part_y, part_z", int="oid, uid",
          gintp="leaf_idx, bin_count, start_idx, idx")
def not_well_adj(result, part_val, part_x, part_y, part_z, leaf_idx, idx, oid,
                 uid, bin_count, start_idx):
    n, k, pid, p1id = declare("int", 4)
    for n in range(bin_count[idx[oid]]):
        pid = leaf_idx[start_idx[idx[oid]] + n]
        for k in range(bin_count[idx[uid]]):
            p1id = leaf_idx[start_idx[idx[uid]] + k]
            result[pid] += direct_comp(
                part_val[p1id], part_x[p1id], part_y[p1id], part_z[p1id],
                part_x[pid], part_y[pid], part_z[pid])


@annotate(gfloatp="part_val, part_x, part_y, part_z", leaf_idx="gintp",
          int="num_u, sid, pid", return_="float")
def u_list(part_val, part_x, part_y, part_z, leaf_idx, num_u, sid, pid):
    n, uid = declare("int", 2)
    res = declare("float")
    res = 0
    for n in range(num_u):
        uid = leaf_idx[sid + n]
        res += direct_comp(part_val[uid], part_x[uid], part_y[uid],
                           part_z[uid], part_x[pid], part_y[pid], part_z[pid])

    return res


@annotate(gfloatp="out_val, out_x, out_y, out_z, part_x, part_y, part_z",
          int="num_p2, wid, pid", return_="float")
def w_list(out_val, out_x, out_y, out_z, part_x, part_y, part_z,
           wid, num_p2, pid):
    n = declare("int", 1)
    res = declare("float")
    res = 0
    for n in range(num_p2):
        res += direct_comp(out_val[wid+n], out_x[wid+n], out_y[wid+n],
                           out_z[wid+n], part_x[pid], part_y[pid], part_z[pid])

    return res


@annotate(int="chid, num_p2", float="in_x, in_y, in_z",
          gfloatp="out_val, out_x, out_y, out_z", return_="float")
def v_list(in_x, in_y, in_z, out_val, out_x, out_y, out_z, num_p2, chid):
    n, cid = declare("int", 2)
    res = declare("float")
    res = 0
    for n in range(num_p2):
        cid = chid + n
        res += direct_comp(out_val[cid], out_x[cid], out_y[cid], out_z[cid],
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


# TEST: loc_coeff
@annotate(gintp="assoc, child, parent, index, index_r, lev_index_r, idx, "
                "leaf_idx, start_idx, bin_count, level",
          gfloatp="in_val, in_x, in_y, in_z, out_val, out_x, out_y, out_z, "
                  "part_val, part_x, part_y, part_z, cx, cy, cz, result",
          length="float", int="i, num_p2")
def loc_coeff(i, in_val, in_x, in_y, in_z, out_val, out_x, out_y, out_z,
              part_val, part_x, part_y, part_z, cx, cy, cz, assoc, child,
              parent, num_p2, level, index, index_r, lev_index_r, idx, result,
              leaf_idx, start_idx, bin_count, length):
    j, k, cid, pid, aid, chid, well, adj, paid = declare("int", 9)
    cr = declare("float")
    cid = cast(floor(i*1.0/num_p2), "int")
    cid = index[cid]
    cr = length/(2.0**(level[cid]+1))
    pid = lev_index_r[parent[cid]]
    in_val[i] = 0
    for j in range(26):
        aid = assoc[26*pid+j]
        if aid != -1:
            paid = idx[aid]
            if paid == -1:
                for k in range(8):
                    chid = child[8*aid+k]
                    if chid == -1:
                        break

                    # HACK: put cells which are neither well seperated
                    # nor adjacent are calculated directly
                    else:
                        # well = well_sep(cx[cid], cy[cid], cz[cid], cr,
                        #                 cx[chid], cy[chid], cz[chid], cr)
                        adj = is_adj(cx[cid], cy[cid], cz[cid], cr,
                                     cx[chid], cy[chid], cz[chid], cr)
                        if adj != 1:
                            in_val[i] += v_list(
                                in_x[i], in_y[i], in_z[i], out_val, out_x,
                                out_y, out_z, num_p2, num_p2*index_r[chid])

                        # if well == 1:
                        #     in_val[i] += v_list(
                        #     in_x[i], in_y[i], in_z[i], out_val, out_x,
                        #     out_y, out_z, num_p2, num_p2*index_r[chid])
                        # elif adj != 1 and i % num_p2 == 0:
                        #     not_well_adj(result, part_val, part_x, part_y,
                        #                  part_z, leaf_idx, idx, cid, chid,
                        #                  bin_count,  start_idx)

            else:
                adj = is_adj(cx[cid], cy[cid], cz[cid], cr, cx[aid], cy[aid],
                             cz[aid], length/(2.0**(level[aid]+1)))
                if adj != 1:
                    in_val[i] += z_list(
                        in_x[i], in_y[i], in_z[i], part_val, part_x, part_y,
                        part_z, start_idx[idx[aid]], leaf_idx,
                        bin_count[idx[aid]])
        else:
            break


# TEST: loc_exp
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
    # REM: store length in this form -> in_r*sqrt(3.0)length/(2.0**(level+1))
    # i2c_l = in_r*sqrt(3.0)*length/(2.0**(level+1))
    for j in range(num_p2):
        s1id = offset + j
        i2c[0] = in_x[s1id] - cx
        i2c[1] = in_y[s1id] - cy
        i2c[2] = in_z[s1id] - cz
        res += in_val[s1id]
        if p2c_l != 0:
            cos_g = (i2c[0]*p2c[0] + i2c[1]*p2c[1] +
                     i2c[2]*p2c[2]) / (p2c_l * i2c_l)
            rr = p2c_l / i2c_l
            sid = 0
            for leg in range(1, leg_lim):
                leg_res = lgndre(leg_lst, cos_g, leg+1, sid)
                res += leg_res*(2*leg+1)*(rr**leg)*in_val[s1id]
                sid += leg+1

    return res / num_p2


# TEST: trans_loc
@annotate(int="i, num_p2, leg_lim, offset", i2c_l="float",
          gfloatp="inc_val, inc_x, inc_y, inc_z, in_val, cx, cy, cz, "
                  "in_x, in_y, in_z, leg_lst", gintp="index, index_r, parent")
def trans_loc(i, inc_val, inc_x, inc_y, inc_z, in_val, in_x, in_y, in_z,
              cx, cy, cz, i2c_l, num_p2, leg_lst, leg_lim,
              index, index_r, parent, offset):
    pid, cid, tid = declare("int", 3)
    cid = cast(floor(i*1.0/num_p2), "int")
    cid = index[cid+offset]
    pid = parent[cid]
    tid = index_r[pid]*num_p2
    inc_val[i] += loc_exp(in_val, in_x, in_y, in_z, cx[pid], cy[pid], cz[pid],
                          inc_x[i], inc_y[i], inc_z[i], num_p2, i2c_l,
                          tid, leg_lst, leg_lim)


# TEST: compute
@annotate(gintp="part2bin, level, idx, parent, child, assoc, index_r, "
                "lev_index_r, leaf_idx, bin_count, start_idx",
          gfloatp="part_val, part_x, part_y, part_z, out_val, out_x, out_y, "
                  "out_z, in_val, in_x, in_y, in_z, cx, cy, cz, result, "
                  "leg_lst", int="i, num_p2, leg_lim", float="in_r, length")
def compute(i, part2bin, part_val, part_x, part_y, part_z, level, idx, parent,
            child, assoc, index_r, lev_index_r, leaf_idx, bin_count,
            start_idx, out_val, out_x, out_y, out_z, in_val, in_x, in_y, in_z,
            cx, cy, cz, result, leg_lst, num_p2, leg_lim, in_r, length):
    h = declare('matrix(10, "int")')
    j, n, bid, baid, brid, lev, pid, aid, chid, t = declare("int", 10)
    i2c_l = declare("float")

    for t in range(10):
        h[t] = 0

    bid = part2bin[i]
    brid = index_r[bid]
    baid = lev_index_r[bid]
    lev = level[bid]
    pid = leaf_idx[i]
    i2c_l = in_r*sqrt(3.0)*length/(2.0**(lev+1))
    cr_bid = length/(2.0**(lev+1))

    result[pid] += own_cell(part_val, part_x, part_y, part_z, leaf_idx,
                            bin_count[idx[bid]], start_idx[idx[bid]], pid)

    # calculation of potential using U and W interaction lists
    for j in range(26):
        aid = assoc[26*baid+j]

        if aid == -1:
            break

        if level[aid] < lev:
            result[pid] += u_list(
                part_val, part_x, part_y, part_z, leaf_idx,
                bin_count[idx[aid]], start_idx[idx[aid]], pid)
        else:
            while True:
                if idx[aid] == -1:
                    for n in range(h[level[aid]], 8):
                        chid = child[8*aid + n]
                        if chid == -1:
                            h[level[aid]] = -1
                            break
                        adj = is_adj(cx[chid], cy[chid], cz[chid],
                                     length/(2.0**(level[chid]+1)),
                                     cx[bid], cy[bid], cz[bid], cr_bid)
                        if adj == 0:
                            result[pid] += w_list(
                                out_val, out_x, out_y, out_z, part_x, part_y,
                                part_z, index_r[chid]*num_p2, num_p2, pid)
                        else:
                            h[level[aid]] = n+1
                            aid = chid
                            break

                    if h[level[aid]] == -1:
                        h[level[aid]] = 0
                        if level[aid] == lev:
                            break
                        aid = parent[aid]

                else:
                    adj = is_adj(cx[aid], cy[aid], cz[aid],
                                 length/(2.0**(level[aid]+1)),
                                 cx[bid], cy[bid], cz[bid], cr_bid)
                    if adj == 1:
                        result[pid] += u_list(
                            part_val, part_x, part_y, part_z, leaf_idx,
                            bin_count[idx[aid]], start_idx[idx[aid]], pid)
                    else:
                        result[pid] += w_list(
                            out_val, out_x, out_y, out_z, part_x, part_y,
                            part_z, index_r[aid]*num_p2, num_p2, pid)
                    if level[aid] <= lev:
                        break
                    else:
                        aid = parent[aid]

    # calculation using the local expansions
    result[pid] += loc_exp(in_val, in_x, in_y, in_z, cx[bid], cy[bid],
                           cz[bid], part_x[pid], part_y[pid], part_z[pid],
                           num_p2, i2c_l, brid*num_p2, leg_lst, leg_lim)


if __name__ == "__main__":

    import tree

    backend = "cython"
    # part_x = np.array([0.125, 0.12, 0.375, 0.125, 0.125, 0.375, 0.375, 0.125, 0.375])
    # part_y = np.array([0.125, 0.12, 0.125, 0.375, 0.125, 0.375, 0.125, 0.375, 0.375])
    # part_z = np.array([0.125, 0.12, 0.125, 0.125, 0.375, 0.125, 0.375, 0.375, 0.375])
    # part_x = np.array([0.125, 0.375, 0.125, 0.125, 0.375, 0.375, 0.125, 0.375, 0.625, 0.875, 0.625, 0.625, 0.875, 0.875, 0.625, 0.875])
    # part_y = np.array([0.125, 0.125, 0.375, 0.125, 0.375, 0.125, 0.375, 0.375, 0.125, 0.125, 0.375, 0.125, 0.375, 0.125, 0.375, 0.375])
    # part_z = np.array([0.125, 0.125, 0.125, 0.375, 0.125, 0.375, 0.375, 0.375, 0.125, 0.125, 0.125, 0.375, 0.125, 0.375, 0.375, 0.375])
    # part_x = np.array([0.375, 0.375, 0.625, 0.875])
    # part_y = np.array([0.125, 0.375, 0.125, 0.125])
    # part_z = np.array([0.125, 0.125, 0.125, 0.125])
    N = 15
    np.random.seed(4)
    part_x = np.random.random(N)
    part_y = np.random.random(N)
    part_z = np.random.random(N)
    max_depth = 2
    part_val = np.ones(N)

    part_val = part_val.astype(np.float32)
    part_x = part_x.astype(np.float32)
    part_y = part_y.astype(np.float32)
    part_z = part_z.astype(np.float32)
    x_min = 0
    y_min = 0
    z_min = 0
    length = 1
    num_p2 = 48
    dimension = 3
    out_r = 1.2
    in_r = 1.06

    (cells, sfc, level, idx, bin_count, start_idx, leaf_idx, parent, child,
     part2bin, lev_cs, levwise_cs, index, index_r, lev_index, lev_index_r,
     cx, cy, cz, out_x, out_y, out_z, in_x, in_y, in_z, out_val, in_val,
     order, part_val, part_x, part_y, part_z) = tree.build(
         N, max_depth, part_val, part_x, part_y, part_z, x_min, y_min, z_min,
         out_r, in_r, length, num_p2, backend, dimension)

    result = ary.zeros(N, dtype=np.float32, backend=backend)
    res_direct = ary.zeros(N, dtype=np.float32, backend=backend)
    assoc = ary.empty(26*cells, dtype=np.int32, backend=backend)
    assoc.fill(-1)

    leg_lim = order//2+1
    siz_leg = leg_lim*(leg_lim+1)//2 - 1
    leg_lst = np.zeros(siz_leg, dtype=np.float32)
    count = 0
    for i in range(1, leg_lim):
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
    ecompute = Elementwise(compute, backend=backend)
    edirect = Elementwise(direct_solv, backend=backend)

    ecalc_p2_fine(out_val[:lev_cs[max_depth-1]*num_p2], out_x, out_y, out_z,
                  part_val, part_x, part_y, part_z, cx, cy, cz, num_p2,
                  length, index, leg_lim, leg_lst, level, idx, out_r*sqrt(3),
                  bin_count, start_idx, leaf_idx)

    for lev in range(max_depth-1, 0, -1):
        m2c_l = out_r*sqrt(3)*length/(2**(lev+1))
        ecalc_p2(out_val[lev_cs[lev]*num_p2:lev_cs[lev-1]*num_p2],
                 out_x, out_y, out_z, out_val, out_x, out_y, out_z,
                 cx, cy, cz, num_p2, index, index_r, leg_lim, leg_lst,
                 child, lev_cs[lev], m2c_l)

    eassoc_coarse(sfc[levwise_cs[1]:levwise_cs[0]], parent, child, lev_index,
                  assoc, levwise_cs[1])

    for lev in range(2, max_depth+1):
        efind_assoc(idx[levwise_cs[lev]:levwise_cs[lev-1]], cx, cy, cz, level,
                    assoc, child, parent, levwise_cs[lev], lev_index,
                    lev_index_r, length)

    eloc_coeff(in_val[:lev_cs[1]*num_p2], in_x, in_y, in_z, out_val, out_x, out_y,
               out_z, part_val, part_x, part_y, part_z, cx, cy, cz, assoc,
               child, parent, num_p2, level, index, index_r, lev_index_r, idx,
               result, leaf_idx, start_idx, bin_count, length)

    for lev in range(2, max_depth):
        i2c_l = in_r*sqrt(3)*length/(2**(lev))
        etrans_loc(in_val[levwise_cs[lev]:levwise_cs[lev-1]],
                   in_x[levwise_cs[lev]:levwise_cs[lev-1]],
                   in_y[levwise_cs[lev]:levwise_cs[lev-1]],
                   in_z[levwise_cs[lev]:levwise_cs[lev-1]],
                   in_val, in_x, in_y, in_z, cx, cy, cz,
                   i2c_l, num_p2, leg_lst, leg_lim,
                   index, index_r, parent, lev_cs[lev])

    ecompute(part2bin, part_val, part_x, part_y, part_z, level, idx, parent,
             child, assoc, index_r, lev_index_r, leaf_idx, bin_count,
             start_idx, out_val, out_x, out_y, out_z, in_val, in_x, in_y,
             in_z, cx, cy, cz, result, leg_lst, num_p2, leg_lim, in_r, length)

    edirect(part_val, part_x, part_y, part_z, res_direct, N)

    # print(sfc)

    # print(part2bin)
    print(result)
    print(res_direct)
    # print(np.arange(18))
    # print(sfc)
    # print(idx)

    print(np.mean(np.abs(result-res_direct)/res_direct))
