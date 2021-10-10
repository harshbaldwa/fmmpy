# TODO: Rearrange all the function variables
# TODO: Convert all to either double or float

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


# TEST: Add test for lower level childless cells
@annotate(int="i, num_p2, leg_lim", double="length, out_r",
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
    
    m2c_l = out_r*sqrt(3.0)*length/(2.0**(level[cid]+1))
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
                  "outc_z, cx, cy, cz, leg_lst", m2c_l="double")
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
    # this is same for all the cells,
    # REM: precomputed and passed as a variable m2c_l
    # m2c_l = out_r*sqrt(3.0)*length/(2.0**(level+1))*3
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


@annotate(double="part_val, part_x, part_y, part_z, px, py, pz",
          return_="double")
def direct_comp(part_val, part_x, part_y, part_z, px, py, pz):
    value, dist = declare("double", 2)
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


@annotate(double="cx, cy, cz, cr, ax, ay, az, ar",
          return_="int")
def is_adj(cx, cy, cz, cr, ax, ay, az, ar):
    dis_x, dis_y, dis_z, rr = declare("double", 4)
    dis_x = fabs(cx - ax)
    dis_y = fabs(cy - ay)
    dis_z = fabs(cz - az)
    rr = cr + ar

    if dis_x > rr or dis_y > rr or dis_z > rr:
        return 0
    else:
        return 1


@annotate(double="cx, cy, cz, cr, ax, ay, az, ar",
          return_="int")
def well_sep(cx, cy, cz, cr, ax, ay, az, ar):
    dis = declare("double")
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


@annotate(int="i, offset", length="double", gfloatp="cx, cy, cz",
          gintp="sfc, level, assoc, child, parent, index, index_r, idx")
def find_assoc(i, idx, cx, cy, cz, level, assoc, child, parent, offset,
               index, index_r, length):

    bid, rid, pid, aid, cid, count, lev, j, k, adj = declare("int", 10)
    bid = i + offset
    rid = index[bid]
    pid = index_r[parent[rid]]
    count = 0
    lev = level[rid]
    for j in range(26):
        aid = assoc[26*pid+j]

        if aid == -1:
            break

        if idx[aid] != -1:
            adj = is_adj(cx[aid], cy[aid], cz[aid],
                            length/(2.0**(level[aid]+1)), cx[rid], cy[rid],
                            cz[rid], length/(2.0**(lev+1)))
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
                           part_z[oid], part_x[pid], part_y[pid], part_z[pid])
    
    return res


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
    n, wid = declare("int", 2)
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
@annotate(gintp="assoc, child, parent, index, index_r, idx, leaf_idx, "
                "start_idx, bin_count",
          gfloatp="in_val, in_x, in_y, in_z, out_val, out_x, out_y, out_z, "
                  "part_val, part_x, part_y, part_z, cx, cy, cz, result",
          cr="float", int="i, num_p2")
def loc_coeff(i, in_val, in_x, in_y, in_z, out_vl, out_x, out_y, out_z,
              part_val, part_x, part_y, part_z, cx, cy, cz, assoc, child,
              parent, num_p2, cr, index, index_r, idx, result, leaf_idx,
              start_idx, bin_count):
    j, k, cid, pid, aid, chid, well, adj, paid = declare("int", 9)
    # REM: pass cr as this (no need to pass two variables for this)
    # cr = declare("double")
    # cr = length/(2.0**(level[i]+1))
    cid = cast(floor(i*1.0/num_p2), "int")
    cid = index[cid]
    pid = parent[cid]
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
                    # nor adjacent in v_list
                    else:
                        # well = well_sep(cx[cid], cy[cid], cz[cid], cr,
                        #                 cx[chid], cy[chid], cz[chid], cr)
                        adj = is_adj(cx[cid], cy[cid], cz[cid], cr,
                                     cx[chid], cy[chid], cz[chid], cr)
                        if adj != 1:
                            in_val[i] += v_list(in_x[i], in_y[i], in_z[i],
                                                out_val, out_x, out_y, out_z,
                                                num_p2, 8*index_r[chid])
                        # elif adj != 1:
                        #     # write the direct function (u_list)
                        #     # something wrong here!!

            else:
                adj = is_adj(cx[cid], cy[cid], cz[cid], cr,
                             cx[aid], cy[aid], cz[aid], cr)
                if adj != 1:
                    in_val[i] += z_list(in_x[i], in_y[i], in_z[i], part_val,
                                        part_x, part_y, part_z, start_idx[aid],
                                        leaf_idx, bin_count[idx[aid]])
        else:
            break


# TEST: loc_exp
@annotate(int="offset, leg_lim, num_p2", float="cx, cy, cz, px, py, pz, i2c_l",
          gfloatp="in_val, in_x, in_y, in_z, leg_lst", return_="float")
def loc_exp(in_val, in_x, in_y, in_z, cx, cy, cz, px, py, pz, num_p2, i2c_l,
            offset, leg_lst, leg_lim):
    j, leg, s1id, sid = declare("int", 4)
    p2c, i2c = declare("matrix(3)", 2)
    res, p2c_l, cos_g, rr, leg_res = declare("double", 5)
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
@annotate(int="i, level, num_p2, leg_lim", gintp="index, index_r, parent",
          gfloatp="inc_val, inc_x, inc_y, inc_z, in_val, cx, cy, cz"
                  "in_x, in_y, in_z, leg_lst")
def trans_loc(i, inc_val, inc_x, inc_y, inc_z, in_val, in_x, in_y, in_z,
              cx, cy, cz, level, length, num_p2, leg_lst, leg_lim,
              index, index_r, parent):
    pid, cid, tid = declare("int", 3)
    cid = cast(floor(i*1.0/num_p2), "int")
    cid = index[cid]
    pid = parent[cid]
    tid = index_r[pid]*num_p2
    inc_val[i] += loc_exp(in_val, in_x, in_y, in_z, cx[pid], cy[pid], cz[pid],
                          inc_x[i], inc_y[i], inc_z[i], num_p2, level, length,
                          tid, leg_lst, leg_lim)


@annotate(gintp="part2bin, level, idx, parent, child, assoc, index_r, "
                "leaf_idx, bin_count, start_idx, level_cs",
          gfloatp="part_val, part_x, part_y, part_z, out_val, out_x, out_y, "
                  "out_z, in_val, in_x, in_y, in_z, cx, cy, cz, result, "
                  "leg_lst", int="i, num_p2, leg_lim", float="in_r, length")
def compute(i, part2bin, part_val, part_x, part_y, part_z, level, idx, parent, 
            child, assoc, index_r, leaf_idx, bin_count, start_idx, level_cs, 
            out_val, out_x, out_y, out_z, in_val, in_x, in_y, in_z, cx, cy, cz, 
            result, leg_lst, num_p2, leg_lim, in_r, length):
    h = declare('matrix(10, "int")')
    bid, brid, lev, pid = declare("int", 4)
    i2c_l = declare("float")
    
    bid = part2bin[i]
    brid = index_r[bid]
    lev = level[bid]
    pid = leaf_idx[i]
    i2c_l = in_r*sqrt(3.0)*length/(2.0**(lev+1))

    own_cell(part_val, part_x, part_y, part_z, leaf_idx, bin_count[idx[bid]], 
             start_idx[idx[bid]], pid)

    # calculation of potential using U and W interaction lists
    for j in range(26):
        aid = assoc[26*brid+j]

        if aid == -1:
            break

        if level[aid] < lev:
            result[pid] += u_list(
                result, part_val, part_x, part_y, part_z, leaf_idx, 
                bin_count[idx[aid]], start_idx[idx[aid]], pid)
        else:
            while True:
                if idx[aid] == -1:
                    for n in range(h[level[aid]], 8):
                        chid = child[8*aid + n]
                        if chid == -1:
                            h[level[aid]] == -1
                            break

                        adj = is_adj(chid, bid)
                        if adj == 0:
                            result[pid] += w_list(
                                out_val, out_x, out_y, out_z, part_x, part_y, 
                                part_z, index_r[num_p2*chid], num_p2, pid)
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
                    adj = is_adj(aid, bid)
                    if adj == 1:
                        result[pid] += u_list(
                            result, part_val, part_x, part_y, part_z, leaf_idx, 
                            bin_count[idx[aid]], start_idx[idx[aid]], pid)
                    else:
                        result[pid] += w_list(
                            out_val, out_x, out_y, out_z, part_x, part_y, 
                            part_z, index_r[num_p2*aid], num_p2, pid)
                    if level[aid] >= lev:
                        break
                    else:
                        aid = parent[aid]
    
    # calculation using the local expansions
    result[pid] += loc_exp(in_val, in_x, in_y, in_z, cx[bid], cy[bid], 
                           cz[bid], part_x[pid], part_y[pid], part_z[pid], 
                           num_p2, i2c_l, brid*num_p2, leg_lst, leg_lim)
