from compyle.api import annotate, declare, wrap, Elementwise
from compyle.low_level import cast
import compyle.array as ary
from math import floor, sqrt, fabs
from scipy.special import legendre
from .centers import set_prob
import numpy as np


@annotate(int="lst_len, id", cos_g="float",
          lst="gfloatp", return_="float")
def lgndre(lst, cos_g, lst_len, id):
    i = declare("int")
    result = declare("float")
    result = 0
    for i in range(lst_len):
        result += lst[id+i]*(cos_g**i)

    return result


@annotate(int="i, num_p2, max_depth, leg_lim",
          gintp="index, idx", length="float",
          gfloatp="out_val, out_x, out_y, out_z, part_val, "
                  "part_x, part_y, part_z, cx, cy, cz, leg_lst")
def calc_p2_fine(i, out_val, out_x, out_y, out_z, part_val, part_x,
                 part_y, part_z, cx, cy, cz, num_p2, length, index,
                 leg_lim, leg_lst, max_depth, idx):
    l, cid, pid, sid = declare("int", 4)
    p2c, m2c = declare("matrix(3)", 2)
    m2c_l, p2c_l, cos_g, rr, leg_res, out_res = declare("float", 6)
    cid = cast(floor(i*1.0/num_p2), "int")
    cid = index[cid]
    m2c[0] = out_x[i] - cx[cid]
    m2c[1] = out_y[i] - cy[cid]
    m2c[2] = out_z[i] - cz[cid]
    m2c_l = length/(2.0**(max_depth+1))*3
    out_res = 0
    out_val[i] = 0
    pid = idx[cid]
    p2c[0] = part_x[pid] - cx[cid]
    p2c[1] = part_y[pid] - cy[cid]
    p2c[2] = part_z[pid] - cz[cid]
    p2c_l = sqrt(p2c[0]**2 + p2c[1]**2 + p2c[2]**2)
    out_res += part_val[pid]
    if p2c_l != 0:
        rr = p2c_l/m2c_l
        cos_g = (m2c[0]*p2c[0] + m2c[1]*p2c[1] +
                 m2c[2]*p2c[2]) / (p2c_l * m2c_l)
        sid = 0
        for l in range(1, leg_lim):
            leg_res = lgndre(leg_lst, cos_g, l+1, sid)
            out_res += leg_res*(2*l+1)*(rr**l)*part_val[pid]
            sid += l+1

    out_val[i] += out_res/num_p2


@annotate(int="i, num_p2, level, leg_lim", length="double",
          gintp="index, index_r, child, level_cs",
          gfloatp="out_val, out_x, out_y, out_z, outc_val, "
                  "outc_x, outc_y, outc_z, cx, cy, cz, leg_lst")
def calc_p2(i, out_val, out_x, out_y, out_z, outc_val, outc_x,
            outc_y, outc_z, cx, cy, cz, num_p2, length, index,
            index_r, leg_lim, leg_lst, child, level, level_cs):
    j, k, l, tid, cid, pid, sid = declare("int", 7)
    p2c, m2c = declare("matrix(3)", 2)
    m2c_l, p2c_l, cos_g, rr, leg_res, out_res = declare("float", 6)
    cid = cast(floor(i*1.0/num_p2), "int")
    cid = index[level_cs[level]+cid]
    m2c[0] = out_x[i] - cx[cid]
    m2c[1] = out_y[i] - cy[cid]
    m2c[2] = out_z[i] - cz[cid]
    m2c_l = length/(2.0**(level+1))*3
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
                    for l in range(1, leg_lim):
                        leg_res = lgndre(leg_lst, cos_g, l+1, sid)
                        out_res += leg_res*(2*l+1)*(rr**l)*outc_val[tid]
                        sid += l+1

    out_val[i] = out_res/num_p2


# TODO: add tests for these two functions as well
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


# TODO: complete rigrous mathematical solution for is_adj and is_well_sep
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


@annotate(int="i, offset",
          gintp="sfc, parent, child, index_r, assoc, collg")
def assoc_coarse(i, sfc, parent, child, index_r, assoc, collg,
                 offset):
    cid, rid, pid, j = declare("int", 4)
    cid = i + offset
    rid = index_r[cid]
    pid = parent[rid] * 8
    for j in range(8):
        if child[pid+j] == -1:
            break
        elif child[pid+j] != rid:
            assoc[27*cid+j] = child[pid+j]
            collg[27*cid+j] = 1
        else:
            continue


# TODO: try to parallelize this whole thing, each associate parallel
@annotate(int="i, offset", length="double",
          gintp="sfc, level, assoc, collg, child, "
                "parent, index, index_r",
          gfloatp="cx, cy, cz")
def find_assoc(i, sfc, cx, cy, cz, level, assoc,
               collg, child, parent, offset,
               index, index_r, length):

    bid, rid, pid, aid, cid, count, lev, j, k, adj = declare("int", 10)
    bid = i + offset
    rid = index_r[bid]
    pid = index[parent[rid]]
    count = 0
    lev = level[rid]
    for j in range(27):
        aid = assoc[27*pid+j]

        if aid == -1:
            continue

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
                assoc[27*bid+count] = cid
                # TODO: this makes no sense unless single
                #       child parents are made nodes
                if level[cid] <= lev:
                    collg[27*bid+count] = 1
                else:
                    collg[27*bid+count] = 0
                count += 1

# TODO: interaction lists for all the cells
