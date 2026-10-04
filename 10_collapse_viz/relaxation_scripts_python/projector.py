"""Port of subdiv_sample_tracker/subdiv_relax_projector.h: projection of a point
onto its own MAT structure (sheet faces / seam and boundary edges).

The data lives in flat arrays (CSR lists) bundled in one tuple `PJ` that the
numba kernels take; the arithmetic of every kernel is the C++ one, operation
by operation:
  seg_t, tri_bary (Ericson RTCD 5.1.5), interp, PrimBVH (same tree: same
  split rule; nth_element is replaced by a full sort, which yields the same
  node sets), Projector::project, ::project_local, ::finish, ::record.

PJ layout (see Projector.pj):
  0 VO (nV,3) f8        1 FO (nF,3) i8          2 E origEdges (nE,2) i8
  3 edgeFace (nE)       4 vf_off 5 vf           vertFaces CSR
  6 ve_off 7 ve         vertEdges CSR           8 fs_off 9 fs   faceSheets CSR
  10 ec_off 11 ec       edgeCurves CSR          12 ss_off 13 ss S.sheet per set
  14 sc_off 15 sc       S.curve per set         16 isCurveSet (P) u1
  17 tg_off 18 tg       targets per set (tree ids)
  19 tr_prim_off 20 pr_corner (n,3) 21 pr_gid 22 tr_order (local prim index)
  23 tr_node_off 24 nd_min (m,3) 25 nd_max (m,3) 26 nd_left 27 nd_right
  28 nd_begin 29 nd_end (node children / ranges are tree-local)
"""
import sys

import numpy as np
from numba import njit

from matstruct import matstruct_edge_key
from struct_ids import RELAX_CURVE, RELAX_SHEET, intersects
from subdiv_mesh import subdiv_find_edges

INF = np.inf
NO_TRACE = np.zeros(0, dtype=np.int64)  # empty checkpoint buffer
INT_MAX = 2 ** 31 - 1


def _csr(lists, dtype=np.int64):
    off = np.zeros(len(lists) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(x) for x in lists])
    flat = np.array([v for x in lists for v in x], dtype=dtype)
    return off, flat


# ------------------------------------------------------------------ closest points

@njit(cache=True, inline='always')
def _sqn(x, y, z):
    return (x * x + y * y) + z * z


@njit(cache=True)
def seg_t(px, py, pz, ax, ay, az, bx, by, bz):
    abx = bx - ax; aby = by - ay; abz = bz - az
    l2 = _sqn(abx, aby, abz)
    if not (l2 > 0.0):
        return 0.0
    x = ((px - ax) * abx + (py - ay) * aby) + (pz - az) * abz
    x = x / l2
    y = x if 0.0 < x else 0.0       # std::max(0.0, x)
    return y if y < 1.0 else 1.0    # std::min(1.0, y)


@njit(cache=True)
def tri_bary(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
    abx = bx - ax; aby = by - ay; abz = bz - az
    acx = cx - ax; acy = cy - ay; acz = cz - az
    l0 = _sqn(abx, aby, abz)
    l1 = _sqn(acx, acy, acz)
    l2 = _sqn(cx - bx, cy - by, cz - bz)
    lmax = l0
    if lmax < l1:
        lmax = l1
    if lmax < l2:
        lmax = l2
    crx = aby * acz - abz * acy
    cry = abz * acx - abx * acz
    crz = abx * acy - aby * acx
    if not (_sqn(crx, cry, crz) > 1e-24 * lmax * lmax):
        b0 = 1.0; b1 = 0.0; b2 = 0.0
        bd = INF
        for e in range(3):
            if e == 0:
                ix, iy, iz, jx, jy, jz = ax, ay, az, bx, by, bz
            elif e == 1:
                ix, iy, iz, jx, jy, jz = bx, by, bz, cx, cy, cz
            else:
                ix, iy, iz, jx, jy, jz = cx, cy, cz, ax, ay, az
            t = seg_t(px, py, pz, ix, iy, iz, jx, jy, jz)
            qx = (1 - t) * ix + t * jx
            qy = (1 - t) * iy + t * jy
            qz = (1 - t) * iz + t * jz
            d = _sqn(px - qx, py - qy, pz - qz)
            if d < bd:
                bd = d
                b0 = 0.0; b1 = 0.0; b2 = 0.0
                if e == 0:
                    b0 = 1 - t; b1 = t
                elif e == 1:
                    b1 = 1 - t; b2 = t
                else:
                    b2 = 1 - t; b0 = t
        return b0, b1, b2
    apx = px - ax; apy = py - ay; apz = pz - az
    d1 = (abx * apx + aby * apy) + abz * apz
    d2 = (acx * apx + acy * apy) + acz * apz
    if d1 <= 0 and d2 <= 0:
        return 1.0, 0.0, 0.0
    bpx = px - bx; bpy = py - by; bpz = pz - bz
    d3 = (abx * bpx + aby * bpy) + abz * bpz
    d4 = (acx * bpx + acy * bpy) + acz * bpz
    if d3 >= 0 and d4 <= d3:
        return 0.0, 1.0, 0.0
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return 1 - v, v, 0.0
    cpx = px - cx; cpy = py - cy; cpz = pz - cz
    d5 = (abx * cpx + aby * cpy) + abz * cpz
    d6 = (acx * cpx + acy * cpy) + acz * cpz
    if d6 >= 0 and d5 <= d6:
        return 0.0, 0.0, 1.0
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return 1 - w, 0.0, w
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return 0.0, 1 - w, w
    denom = 1.0 / ((va + vb) + vc)
    v = vb * denom
    w = vc * denom
    return (1 - v) - w, v, w


@njit(cache=True)
def interp3(VO, FO, f, b0, b1, b2):
    i0 = FO[f, 0]; i1 = FO[f, 1]; i2 = FO[f, 2]
    x = (b0 * VO[i0, 0] + b1 * VO[i1, 0]) + b2 * VO[i2, 0]
    y = (b0 * VO[i0, 1] + b1 * VO[i1, 1]) + b2 * VO[i2, 1]
    z = (b0 * VO[i0, 2] + b1 * VO[i1, 2]) + b2 * VO[i2, 2]
    return x, y, z


@njit(cache=True)
def prim_closest(VO, c0, c1, c2, px, py, pz):
    """(squared distance, b0, b1, b2) of p to a segment (c2 < 0) or triangle,
    as PrimBVH::query / Projector::project_local compute it."""
    if c2 < 0:
        ax = VO[c0, 0]; ay = VO[c0, 1]; az = VO[c0, 2]
        bx = VO[c1, 0]; by = VO[c1, 1]; bz = VO[c1, 2]
        t = seg_t(px, py, pz, ax, ay, az, bx, by, bz)
        qx = (1 - t) * ax + t * bx
        qy = (1 - t) * ay + t * by
        qz = (1 - t) * az + t * bz
        return _sqn(px - qx, py - qy, pz - qz), 1 - t, t, 0.0
    ax = VO[c0, 0]; ay = VO[c0, 1]; az = VO[c0, 2]
    bx = VO[c1, 0]; by = VO[c1, 1]; bz = VO[c1, 2]
    cx = VO[c2, 0]; cy = VO[c2, 1]; cz = VO[c2, 2]
    b0, b1, b2 = tri_bary(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz)
    qx = (b0 * ax + b1 * bx) + b2 * cx
    qy = (b0 * ay + b1 * by) + b2 * cy
    qz = (b0 * az + b1 * bz) + b2 * cz
    return _sqn(px - qx, py - qy, pz - qz), b0, b1, b2


# ------------------------------------------------------------------ BVH query

@njit(cache=True)
def _box_sqdist(mn, mx, n, px, py, pz):
    """AlignedBox3d::squaredExteriorDistance."""
    d2 = 0.0
    for k in range(3):
        p = px if k == 0 else (py if k == 1 else pz)
        if mn[n, k] > p:
            aux = mn[n, k] - p
            d2 += aux * aux
        elif p > mx[n, k]:
            aux = p - mx[n, k]
            d2 += aux * aux
    return d2


@njit(cache=True)
def bvh_query(PJ, t, px, py, pz, bestD, bestGid, b0, b1, b2, vis, nvis):
    """PrimBVH::query on tree t; returns the updated incumbent, the tree-local
    leaf node holding a strictly better primitive (-1 if the incumbent stands)
    and the updated count of vis.
    Checkpoint: vis records (up to its size) every node the query enters
    (not pruned), as global node indices; pass an empty array to skip."""
    VO = PJ[0]
    prim0 = PJ[19][t]
    corner = PJ[20]; gid = PJ[21]; order = PJ[22]
    n0 = PJ[23][t]; n1 = PJ[23][t + 1]
    mn = PJ[24]; mx = PJ[25]; left = PJ[26]; right = PJ[27]; beg = PJ[28]; end = PJ[29]
    leaf = -1
    if n1 == n0:
        return bestD, bestGid, b0, b1, b2, leaf, nvis
    stack = np.empty(128, dtype=np.int64)
    sp = 0
    stack[sp] = 0
    sp += 1
    while sp:
        sp -= 1
        nd = n0 + stack[sp]
        if _box_sqdist(mn, mx, nd, px, py, pz) > bestD:
            continue
        if nvis < vis.shape[0]:
            vis[nvis] = nd
            nvis += 1
        if left[nd] < 0:
            for k in range(beg[nd], end[nd]):
                i = prim0 + order[prim0 + k]
                d, c0, c1, c2 = prim_closest(VO, corner[i, 0], corner[i, 1], corner[i, 2], px, py, pz)
                if d < bestD or (d == bestD and gid[i] < bestGid):
                    bestD = d; bestGid = gid[i]; b0 = c0; b1 = c1; b2 = c2
                    leaf = nd - n0
        else:
            if sp + 2 > 128:
                raise RuntimeError('[subdiv_relax] BVH too deep')
            dl = _box_sqdist(mn, mx, n0 + left[nd], px, py, pz)
            dr = _box_sqdist(mn, mx, n0 + right[nd], px, py, pz)
            if dl <= dr:
                stack[sp] = right[nd]; stack[sp + 1] = left[nd]
            else:
                stack[sp] = left[nd]; stack[sp + 1] = right[nd]
            sp += 2
    return bestD, bestGid, b0, b1, b2, leaf, nvis


# ------------------------------------------------------------------ membership

@njit(cache=True)
def _intersects(a, a0, a1, b, b0, b1):
    i = a0; j = b0
    while i < a1 and j < b1:
        if a[i] == b[j]:
            return True
        if a[i] < b[j]:
            i += 1
        else:
            j += 1
    return False


@njit(cache=True)
def face_in(PJ, k, f):
    ss_off = PJ[12]
    if ss_off[k + 1] == ss_off[k]:
        return True  # global tree
    fs_off = PJ[8]
    return _intersects(PJ[9], fs_off[f], fs_off[f + 1], PJ[13], ss_off[k], ss_off[k + 1])


@njit(cache=True)
def edge_in(PJ, k, e):
    ec_off = PJ[10]; sc_off = PJ[14]
    return _intersects(PJ[11], ec_off[e], ec_off[e + 1], PJ[15], sc_off[k], sc_off[k + 1])


# ------------------------------------------------------------------ finish / record

@njit(cache=True)
def finish(PJ, curve, bestGid, bb0, bb1, bb2):
    """Projector::finish -> (face, b0, b1, b2, edge, x, y, z, baryFix)."""
    VO = PJ[0]; FO = PJ[1]; E = PJ[2]
    if curve:
        e = bestGid
        f = PJ[3][e]
        edge = e
        face = f
        b = np.zeros(3)
        for c in range(3):
            if FO[f, c] == E[e, 0]:
                b[c] = bb0
            if FO[f, c] == E[e, 1]:
                b[c] = bb1
        b0 = b[0]; b1 = b[1]; b2 = b[2]
    else:
        edge = -1
        face = bestGid
        b0 = bb0; b1 = bb1; b2 = bb2
    mn = b0
    if b1 < mn:
        mn = b1
    if b2 < mn:
        mn = b2
    fix = 1 if (mn < -1e-12 or abs(((b0 + b1) + b2) - 1.0) > 1e-12) else 0
    if b0 < 0:
        b0 = 0.0
    if b1 < 0:
        b1 = 0.0
    if b2 < 0:
        b2 = 0.0
    s = (b0 + b1) + b2
    if abs(s - 1.0) > 1e-15:
        b0 = b0 / s; b1 = b1 / s; b2 = b2 / s
    x, y, z = interp3(VO, FO, face, b0, b1, b2)
    return face, b0, b1, b2, edge, x, y, z, fix


@njit(cache=True)
def _record(px, py, pz, b0, b1, b2, edge, x, y, z):
    """Projector::record contributions: (onBorder, distance)."""
    zeros = 0
    if b0 == 0.0:
        zeros += 1
    if b1 == 0.0:
        zeros += 1
    if b2 == 0.0:
        zeros += 1
    onb = 1 if (zeros >= 2 if edge >= 0 else zeros >= 1) else 0
    d = np.sqrt(_sqn(px - x, py - y, pz - z))
    return onb, d


# ------------------------------------------------------------------ project / project_local
# Both return (face, b0, b1, b2, edge, x, y, z, onBorder, baryFix, dist, grown, global,
#               winTree, winLeaf, nTrace)
# Checkpoint fields (not used by the relaxation): winTree / winLeaf = the tree and
# tree-local leaf node that gave the result (-1 when the incumbent location or the
# local region did); nTrace = entries written to vis (project) or the size of the
# final search region left in `region` (project_local).

@njit(cache=True)
def project(PJ, k, px, py, pz, curFace, curEdge, vis):
    VO = PJ[0]; FO = PJ[1]; E = PJ[2]
    bestD = INF
    bestGid = INT_MAX
    b0 = 1.0; b1 = 0.0; b2 = 0.0
    curve = PJ[16][k] != 0
    if curve:
        if curEdge >= 0 and edge_in(PJ, k, curEdge):
            d, c0, c1, c2 = prim_closest(VO, E[curEdge, 0], E[curEdge, 1], -1, px, py, pz)
            bestD = d; bestGid = curEdge; b0 = c0; b1 = c1; b2 = c2
    elif curFace >= 0 and face_in(PJ, k, curFace):
        d, c0, c1, c2 = prim_closest(VO, FO[curFace, 0], FO[curFace, 1], FO[curFace, 2], px, py, pz)
        bestD = d; bestGid = curFace; b0 = c0; b1 = c1; b2 = c2
    tg_off = PJ[17]; tg = PJ[18]
    winTree = -1
    winLeaf = -1
    nvis = 0
    for a in range(tg_off[k], tg_off[k + 1]):
        bestD, bestGid, b0, b1, b2, leaf, nvis = bvh_query(PJ, tg[a], px, py, pz, bestD, bestGid, b0, b1, b2,
                                                           vis, nvis)
        if leaf >= 0:
            winTree = tg[a]
            winLeaf = leaf
    if bestGid == INT_MAX:
        raise RuntimeError('[subdiv_relax] projection found no target')
    face, r0, r1, r2, edge, x, y, z, fix = finish(PJ, curve, bestGid, b0, b1, b2)
    onb, dist = _record(px, py, pz, r0, r1, r2, edge, x, y, z)
    return face, r0, r1, r2, edge, x, y, z, onb, fix, dist, 0, 0, winTree, winLeaf, nvis


@njit(cache=True)
def _in_region(region, n, x):
    for i in range(n):
        if region[i] == x:
            return True
    return False


@njit(cache=True)
def _add_corner(PJ, k, curve, v, region, n):
    if curve:
        off = PJ[6]; lst = PJ[7]
    else:
        off = PJ[4]; lst = PJ[5]
    for a in range(off[v], off[v + 1]):
        x = lst[a]
        if not _in_region(region, n, x):
            ins = edge_in(PJ, k, x) if curve else face_in(PJ, k, x)
            if ins:
                region[n] = x
                n += 1
    return n


@njit(cache=True)
def _add_ring(PJ, k, curve, x, region, n):
    if curve:
        E = PJ[2]
        n = _add_corner(PJ, k, curve, E[x, 0], region, n)
        n = _add_corner(PJ, k, curve, E[x, 1], region, n)
    else:
        FO = PJ[1]
        for c in range(3):
            n = _add_corner(PJ, k, curve, FO[x, c], region, n)
    return n


@njit(cache=True)
def project_local(PJ, k, px, py, pz, curFace, curEdge, region, vis):
    """region: scratch buffer of size >= max(#faces, #edges)."""
    VO = PJ[0]; FO = PJ[1]; E = PJ[2]
    curve = PJ[16][k] != 0
    valid = (curEdge >= 0 and edge_in(PJ, k, curEdge)) if curve else (curFace >= 0 and face_in(PJ, k, curFace))
    if not valid:
        r = project(PJ, k, px, py, pz, curFace, curEdge, vis)
        return r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], 0, 1, r[13], r[14], r[15]
    n = 0
    region[n] = curEdge if curve else curFace
    n += 1
    n = _add_ring(PJ, k, curve, region[0], region, n)
    grown = 0
    vf_off = PJ[4]; vf = PJ[5]; ve_off = PJ[6]; ve = PJ[7]
    for rnd in range(64):
        bestD = INF
        bestGid = INT_MAX
        b0 = 1.0; b1 = 0.0; b2 = 0.0
        for a in range(n):
            x = region[a]
            if curve:
                d, c0, c1, c2 = prim_closest(VO, E[x, 0], E[x, 1], -1, px, py, pz)
            else:
                d, c0, c1, c2 = prim_closest(VO, FO[x, 0], FO[x, 1], FO[x, 2], px, py, pz)
            if d < bestD or (d == bestD and x < bestGid):
                bestD = d; bestGid = x; b0 = c0; b1 = c1; b2 = c2
        closed = True
        if curve:
            if b1 == 0:
                v = E[bestGid, 0]
                for a in range(ve_off[v], ve_off[v + 1]):
                    e = ve[a]
                    if closed and edge_in(PJ, k, e) and not _in_region(region, n, e):
                        closed = False
            if b0 == 0:
                v = E[bestGid, 1]
                for a in range(ve_off[v], ve_off[v + 1]):
                    e = ve[a]
                    if closed and edge_in(PJ, k, e) and not _in_region(region, n, e):
                        closed = False
        else:
            nz = 0
            z0 = -1
            z1 = -1
            if b0 == 0:
                if nz == 0:
                    z0 = 0
                else:
                    z1 = 0
                nz += 1
            if b1 == 0:
                if nz == 0:
                    z0 = 1
                else:
                    z1 = 1
                nz += 1
            if b2 == 0:
                if nz == 0:
                    z0 = 2
                else:
                    z1 = 2
                nz += 1
            if nz >= 2:      # on corner v
                c = 3 - z0 - z1
                v = FO[bestGid, c]
                for a in range(vf_off[v], vf_off[v + 1]):
                    g = vf[a]
                    if closed and face_in(PJ, k, g) and not _in_region(region, n, g):
                        closed = False
            elif nz == 1:    # on the edge opposite corner z0
                v0 = FO[bestGid, (z0 + 1) % 3]
                v1 = FO[bestGid, (z0 + 2) % 3]
                for a in range(vf_off[v0], vf_off[v0 + 1]):
                    g = vf[a]
                    if FO[g, 0] == v1 or FO[g, 1] == v1 or FO[g, 2] == v1:
                        if closed and face_in(PJ, k, g) and not _in_region(region, n, g):
                            closed = False
        if closed:
            face, r0, r1, r2, edge, x, y, z, fix = finish(PJ, curve, bestGid, b0, b1, b2)
            onb, dist = _record(px, py, pz, r0, r1, r2, edge, x, y, z)
            return face, r0, r1, r2, edge, x, y, z, onb, fix, dist, grown, 0, -1, -1, n
        if rnd == 0:
            grown = 1
        before = n
        for r in range(before):
            n = _add_ring(PJ, k, curve, region[r], region, n)
        if n == before:  # whole component
            face, r0, r1, r2, edge, x, y, z, fix = finish(PJ, curve, bestGid, b0, b1, b2)
            onb, dist = _record(px, py, pz, r0, r1, r2, edge, x, y, z)
            return face, r0, r1, r2, edge, x, y, z, onb, fix, dist, grown, 0, -1, -1, n
    r = project(PJ, k, px, py, pz, curFace, curEdge, vis)
    return r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], grown, 1, r[13], r[14], r[15]


# ------------------------------------------------------------------ BVH build (Python, once)

class _Tree:
    def __init__(self, corner, gid, VO):
        self.corner = np.asarray(corner, dtype=np.int64).reshape(-1, 3)
        self.gid = np.asarray(gid, dtype=np.int64)
        n = len(self.gid)
        self.order = list(range(n))
        self.nodes = []  # [min(3), max(3), left, right, begin, end]
        if n == 0:
            return
        pmin = np.full((n, 3), np.finfo(np.float64).max)
        pmax = np.full((n, 3), -np.finfo(np.float64).max)
        for c in range(3):
            ok = self.corner[:, c] >= 0
            P = VO[self.corner[ok, c], :3]
            pmin[ok] = np.where(P < pmin[ok], P, pmin[ok])   # cwiseMin: (b < a) ? b : a
            pmax[ok] = np.where(pmax[ok] < P, P, pmax[ok])
        self.pmin, self.pmax = pmin, pmax
        self.cen = (pmin + pmax) / 2.0
        self._build(0, n)

    def _build(self, begin, end):
        nid = len(self.nodes)
        idx = self.order[begin:end]
        mn = self.pmin[idx].min(axis=0)
        mx = self.pmax[idx].max(axis=0)
        cen = self.cen[idx]
        cmn = cen.min(axis=0)
        cmx = cen.max(axis=0)
        self.nodes.append([mn, mx, -1, -1, begin, end])
        if end - begin <= 4:
            return nid
        sizes = cmx - cmn
        axis = 0
        for k in (1, 2):
            if sizes[k] > sizes[axis]:
                axis = k
        mid = (begin + end) // 2
        # nth_element with the strict total order (cen[axis], index): the first
        # mid-begin elements are the smallest ones
        self.order[begin:end] = sorted(idx, key=lambda a: (self.cen[a, axis], a))
        l = self._build(begin, mid)
        r = self._build(mid, end)
        self.nodes[nid][2] = l
        self.nodes[nid][3] = r
        return nid


class Projector:
    """Projector(VO, FO, origEdges, ms, pal, S, setRole)."""

    def __init__(self, VO, FO, E, ms, pal, S, setRole):
        VO = np.ascontiguousarray(VO[:, :3], dtype=np.float64)
        FO = np.ascontiguousarray(FO, dtype=np.int64)
        E = np.ascontiguousarray(E, dtype=np.int64)
        self.VO, self.FO, self.E = VO, FO, E
        nF, nE, nV = FO.shape[0], E.shape[0], VO.shape[0]
        a = FO.reshape(-1)
        b = FO[:, [1, 2, 0]].reshape(-1)
        fe = subdiv_find_edges(E, a, b)
        if (fe < 0).any():
            raise RuntimeError('[subdiv_relax] face edge missing from origEdges')
        edgeFace = np.full(nE, -1, dtype=np.int64)
        edgeFaces = [[] for _ in range(nE)]
        vertFaces = [[] for _ in range(nV)]
        FOl = FO.tolist()
        fel = fe.tolist()
        for f in range(nF):
            for c in range(3):
                e = fel[3 * f + c]
                if edgeFace[e] < 0:
                    edgeFace[e] = f
                edgeFaces[e].append(f)
                vertFaces[FOl[f][c]].append(f)
        vertEdges = [[] for _ in range(nV)]
        for e, (u, w) in enumerate(E.tolist()):
            vertEdges[u].append(e)
            vertEdges[w].append(e)
        self.edgeFace, self.edgeFaces, self.vertFaces, self.vertEdges = edgeFace, edgeFaces, vertFaces, vertEdges

        faceSheets = [[] for _ in range(nF)]
        edgeCurves = [[] for _ in range(nE)]
        if ms is not None:
            faceSheets = [list(ms.faceIds[f]) for f in range(nF)]
            El = E.tolist()
            for e in range(nE):
                edgeCurves[e] = list(ms.edgeIds.get(matstruct_edge_key(El[e][0], El[e][1]), []))
        self.faceSheets, self.edgeCurves = faceSheets, edgeCurves

        sheetFaces, curveEdges = {}, {}
        for f in range(nF):
            for i in faceSheets[f]:
                sheetFaces.setdefault(i, []).append(f)
        for e in range(nE):
            for i in edgeCurves[e]:
                curveEdges.setdefault(i, []).append(e)
        trees = []

        def add_tree(prims, seg):
            if seg:
                corner = [(int(E[p, 0]), int(E[p, 1]), -1) for p in prims]
            else:
                corner = [tuple(FOl[p]) for p in prims]
            trees.append(_Tree(corner, prims, VO))
            return len(trees) - 1
        self.sheetTree = {i: add_tree(sheetFaces[i], False) for i in sorted(sheetFaces)}
        self.curveTree = {i: add_tree(curveEdges[i], True) for i in sorted(curveEdges)}
        self.globalTree = -1
        P = pal.size()
        targets = [[] for _ in range(P)]
        isCurveSet = np.zeros(P, dtype=np.uint8)
        for k in range(P):
            if setRole[k] == RELAX_CURVE:
                isCurveSet[k] = 1
                for i in S.curve[k]:
                    if i in self.curveTree:
                        targets[k].append(self.curveTree[i])
            elif setRole[k] == RELAX_SHEET:
                for i in S.sheet[k]:
                    if i in self.sheetTree:
                        targets[k].append(self.sheetTree[i])
                if not S.sheet[k]:
                    if self.globalTree < 0:
                        self.globalTree = add_tree(list(range(nF)), False)
                    targets[k].append(self.globalTree)
        self.targets = targets
        self.trees = trees
        self.S = S

        # flatten
        vf_off, vf = _csr(vertFaces)
        ve_off, ve = _csr(vertEdges)
        fs_off, fs = _csr(faceSheets)
        ec_off, ec = _csr(edgeCurves)
        ss_off, ss = _csr(S.sheet)
        sc_off, sc = _csr(S.curve)
        tg_off, tg = _csr(targets)
        prim_off = np.zeros(len(trees) + 1, dtype=np.int64)
        node_off = np.zeros(len(trees) + 1, dtype=np.int64)
        for t, tr in enumerate(trees):
            prim_off[t + 1] = prim_off[t] + len(tr.gid)
            node_off[t + 1] = node_off[t] + len(tr.nodes)
        nP, nN = prim_off[-1], node_off[-1]
        pr_corner = np.zeros((nP, 3), dtype=np.int64)
        pr_gid = np.zeros(nP, dtype=np.int64)
        tr_order = np.zeros(nP, dtype=np.int64)
        nd_min = np.zeros((nN, 3)); nd_max = np.zeros((nN, 3))
        nd_left = np.zeros(nN, dtype=np.int64); nd_right = np.zeros(nN, dtype=np.int64)
        nd_begin = np.zeros(nN, dtype=np.int64); nd_end = np.zeros(nN, dtype=np.int64)
        for t, tr in enumerate(trees):
            p0, n0 = prim_off[t], node_off[t]
            m = len(tr.gid)
            pr_corner[p0:p0 + m] = tr.corner
            pr_gid[p0:p0 + m] = tr.gid
            tr_order[p0:p0 + m] = tr.order
            for j, nd in enumerate(tr.nodes):
                nd_min[n0 + j] = nd[0]; nd_max[n0 + j] = nd[1]
                nd_left[n0 + j] = nd[2]; nd_right[n0 + j] = nd[3]
                nd_begin[n0 + j] = nd[4]; nd_end[n0 + j] = nd[5]
        self.pj = (VO, FO, E, edgeFace, vf_off, vf, ve_off, ve, fs_off, fs, ec_off, ec, ss_off, ss, sc_off, sc,
                   isCurveSet, tg_off, tg, prim_off, pr_corner, pr_gid, tr_order, node_off,
                   nd_min, nd_max, nd_left, nd_right, nd_begin, nd_end)
        self.scratch = max(nF, nE) + 1

    # Python-side helpers (same answers as the kernels)
    def face_in(self, k, f):
        if not self.S.sheet[k]:
            return True
        return intersects(self.faceSheets[f], self.S.sheet[k])

    def edge_in(self, k, e):
        return intersects(self.edgeCurves[e], self.S.curve[k])

    def project(self, k, p, curFace, curEdge):
        return project(self.pj, int(k), float(p[0]), float(p[1]), float(p[2]), int(curFace), int(curEdge), NO_TRACE)
