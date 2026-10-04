"""Port of src/query_coarse_to_fine.cpp (+ compute_barycentric.cpp) and of
coarse_subdiv_c2f.cpp: subdivide the compact coarse mesh and map every
subdivided vertex to the fine mesh by walking the SSP collapses backwards.

Same arithmetic, in the same order, as the C++:
  * the sheet of a step is the one whose global_sheet_id equals
    faceSheetID[face] (single-sheet collapses fall back to sheet 0; a seam
    collapse with no matching sheet is skipped),
  * the chosen pre-collapse triangle is the first one whose
    -min(barycentric) < the running minimum, starting at 1.0 (row 0 when the
    point is outside every triangle by >= 1),
  * negatives clamped to 0 and the row divided by its sum (always).
(11_correspond_viz/c2f_query.query_coarse_to_fine differs in all three; see
verify_c2f.py.)
"""
import sys

import numpy as np

from log_util import log
from numba import njit, prange

from subdiv_mesh import build_subdiv_mesh


@njit(cache=True)
def _bary2d(px, py, ax, ay, bx, by, cx, cy):
    """compute_barycentric for one triangle: (u, v, w)."""
    v0x = bx - ax; v0y = by - ay
    v1x = cx - ax; v1y = cy - ay
    v2x = -ax + px; v2y = -ay + py
    d00 = v0x * v0x + v0y * v0y
    d01 = v0x * v1x + v0y * v1y
    d11 = v1x * v1x + v1y * v1y
    d20 = v2x * v0x + v2y * v0y
    d21 = v2x * v1x + v2y * v1y
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - (v + w)
    return u, v, w


@njit(cache=True)
def _min3(a, b, c):
    # Eigen minCoeff: running min with "<"
    m = a
    if b < m:
        m = b
    if c < m:
        m = c
    return m


@njit(parallel=True, cache=True)
def query_coarse_to_fine_kernel(decIM_off, decIM_idx, faceSheetID,
                                sh_off, sh_gid, sh_b, sh_sv_off, sv, sh_uv_off, UV_pre, UV_post,
                                sh_fpre_off, FUV_pre, FIdx_pre,
                                BC, BF, FIdx,
                                st_steps, st_clamped, st_far, st_maxNeg, st_sumNeg, st_maxSnap):
    nq = BF.shape[0]
    nDec = sh_off.shape[0] - 1
    nFS = faceSheetID.shape[0]
    for q in prange(nq):
        dIdx = nDec
        while True:
            queryF = FIdx[q]
            found = False
            for ii in range(decIM_off[queryF + 1] - 1, decIM_off[queryF] - 1, -1):
                if dIdx > decIM_idx[ii]:
                    dIdx = decIM_idx[ii]
                    found = True
                    break
            if not found:
                break
            sid = faceSheetID[queryF] if queryF < nFS else 0
            s = -1
            for k in range(sh_off[dIdx], sh_off[dIdx + 1]):
                if sh_gid[k] == sid:
                    s = k
                    break
            if s < 0 and sh_off[dIdx + 1] - sh_off[dIdx] == 1:
                s = sh_off[dIdx]
            if s < 0:
                continue
            pr0 = sh_fpre_off[s]
            pr1 = sh_fpre_off[s + 1]
            pre_row = -1
            for r in range(pr1 - pr0):
                if FIdx_pre[pr0 + r] == queryF:
                    pre_row = r
                    break
            if pre_row < 0:
                continue
            u0 = sh_uv_off[s]
            v0 = FUV_pre[pr0 + pre_row, 0]
            v1 = FUV_pre[pr0 + pre_row, 1]
            v2 = FUV_pre[pr0 + pre_row, 2]
            bs = sh_b[s, 0]
            bd = sh_b[s, 1]
            if bs >= 0 and bd >= 0:
                if v0 == bd:
                    v0 = bs
                if v1 == bd:
                    v1 = bs
                if v2 == bd:
                    v2 = bs
            # queryUV = BC0 * UV_post(v0) + BC1 * UV_post(v1) + BC2 * UV_post(v2)
            qx = (BC[q, 0] * UV_post[u0 + v0, 0] + BC[q, 1] * UV_post[u0 + v1, 0]) + BC[q, 2] * UV_post[u0 + v2, 0]
            qy = (BC[q, 0] * UV_post[u0 + v0, 1] + BC[q, 1] * UV_post[u0 + v1, 1]) + BC[q, 2] * UV_post[u0 + v2, 1]
            minD = 1.0
            best = 0
            bu = 0.0; bv = 0.0; bw = 0.0
            first = True
            for r in range(pr1 - pr0):
                a = u0 + FUV_pre[pr0 + r, 0]
                b = u0 + FUV_pre[pr0 + r, 1]
                c = u0 + FUV_pre[pr0 + r, 2]
                u, v, w = _bary2d(qx, qy, UV_pre[a, 0], UV_pre[a, 1], UV_pre[b, 0], UV_pre[b, 1],
                                  UV_pre[c, 0], UV_pre[c, 1])
                if first:  # row 0 is the default choice
                    bu = u; bv = v; bw = w
                    first = False
                dist = -_min3(u, v, w)
                if dist < minD:
                    minD = dist
                    best = r
                    bu = u; bv = v; bw = w
            distBest = -_min3(bu, bv, bw)
            negBary = distBest
            cu = bu if bu > 0.0 else 0.0  # max(0.0, x)
            cv = bv if bv > 0.0 else 0.0
            cw = bw if bw > 0.0 else 0.0
            ssum = (cu + cv) + cw
            cu = cu / ssum
            cv = cv / ssum
            cw = cw / ssum
            # statistics
            st_steps[q] += 1
            if distBest >= 1.0:
                st_far[q] += 1
            if negBary > 1e-12:
                st_clamped[q] += 1
                if negBary > st_maxNeg[q]:
                    st_maxNeg[q] = negBary
                st_sumNeg[q] += negBary
                a = u0 + FUV_pre[pr0 + best, 0]
                b = u0 + FUV_pre[pr0 + best, 1]
                c = u0 + FUV_pre[pr0 + best, 2]
                sx = ((0.0 + cu * UV_pre[a, 0]) + cv * UV_pre[b, 0]) + cw * UV_pre[c, 0]
                sy = ((0.0 + cu * UV_pre[a, 1]) + cv * UV_pre[b, 1]) + cw * UV_pre[c, 1]
                e0 = np.sqrt((UV_pre[b, 0] - UV_pre[a, 0]) ** 2 + (UV_pre[b, 1] - UV_pre[a, 1]) ** 2)
                e1 = np.sqrt((UV_pre[c, 0] - UV_pre[b, 0]) ** 2 + (UV_pre[c, 1] - UV_pre[b, 1]) ** 2)
                e2 = np.sqrt((UV_pre[a, 0] - UV_pre[c, 0]) ** 2 + (UV_pre[a, 1] - UV_pre[c, 1]) ** 2)
                edge = max(e0, max(e1, e2))
                if edge > 0:
                    rel = np.sqrt((qx - sx) ** 2 + (qy - sy) ** 2) / edge
                    if rel > st_maxSnap[q]:
                        st_maxSnap[q] = rel
            svo = sh_sv_off[s]
            BC[q, 0] = cu
            BC[q, 1] = cv
            BC[q, 2] = cw
            BF[q, 0] = sv[svo + FUV_pre[pr0 + best, 0]]
            BF[q, 1] = sv[svo + FUV_pre[pr0 + best, 1]]
            BF[q, 2] = sv[svo + FUV_pre[pr0 + best, 2]]
            FIdx[q] = FIdx_pre[pr0 + best]


def query_coarse_to_fine(B, BC, BF, FIdx):
    """query_coarse_to_fine with identity IM / IMF (SSP never renumbers).
    Modifies BC, BF, FIdx in place; returns the C2FQueryStats arrays."""
    n = BF.shape[0]
    st = dict(steps=np.zeros(n, np.int64), clamped=np.zeros(n, np.int64), farOutside=np.zeros(n, np.int64),
              maxNegBary=np.zeros(n), sumNegBary=np.zeros(n), maxSnapRel=np.zeros(n))
    query_coarse_to_fine_kernel(B.decIM_off, B.decIM_idx, B.faceSheetID,
                                B.sh_off, B.sh_gid, B.sh_b, B.sh_sv_off, B.sv, B.sh_uv_off, B.UV_pre, B.UV_post,
                                B.sh_fpre_off, B.FUV_pre, B.FIdx_pre,
                                BC, BF, FIdx,
                                st['steps'], st['clamped'], st['farOutside'],
                                st['maxNegBary'], st['sumNegBary'], st['maxSnapRel'])
    return st


class CoarseSubdivC2F:
    """S: subdivided coarse mesh (carriers on the coarse mesh); P: fine
    correspondences; fineFace / fineBary: location on the fine mesh (gFO)."""
    pass


def coarse_subdiv_c2f_build(B, nTarget, refined=None):
    """coarse_subdiv_c2f_build(cmc, nTarget) with cmc from the bundle.
    refined (Python-only): equal_area_refine.RefinedCoarse to subdivide from."""
    C = CoarseSubdivC2F()
    C.S = build_subdiv_mesh(B.coarseV, B.coarseF, nTarget, start=refined)
    S = C.S
    n = S.V.shape[0]
    nDec = B.nDec
    nDecIM = len(B.decIM_off) - 1
    nFS = len(B.faceSheetID)

    qVert = []
    BC = np.zeros((n, 3))
    BF = np.zeros((n, 3), dtype=np.int64)
    FIdx = np.zeros(n, dtype=np.int64)
    nUnmatched = 0
    # per coarse face: the initial (BF, column map) is the same for all its vertices
    faceInit = {}
    Sff = S.fineFace.tolist()
    for i in range(n):
        cf = Sff[i]
        if cf < 0:
            continue
        if cf not in faceInit:
            fi = int(B.faceMap[cf])
            cur = [int(B.vtxMap[B.coarseF[cf, c]]) for c in range(3)]
            dIdx = -1
            if fi < nDecIM:
                L = B.decIM_idx[B.decIM_off[fi]:B.decIM_off[fi + 1]].tolist()
                for k in range(len(L) - 1, -1, -1):
                    if L[k] < nDec:
                        dIdx = L[k]
                        break
            if dIdx < 0:
                init = (fi, cur, [0, 1, 2])
            else:
                sid = int(B.faceSheetID[fi]) if fi < nFS else 0
                s = -1
                for k in range(B.sh_off[dIdx], B.sh_off[dIdx + 1]):
                    if B.sh_gid[k] == sid:
                        s = k
                        break
                if s < 0 and B.sh_off[dIdx + 1] - B.sh_off[dIdx] == 1:
                    s = int(B.sh_off[dIdx])
                row = -1
                hasb = s >= 0 and B.sh_b[s, 0] >= 0 and B.sh_b[s, 1] >= 0
                if hasb:
                    p0, p1 = B.sh_fpre_off[s], B.sh_fpre_off[s + 1]
                    hit = np.nonzero(B.FIdx_pre[p0:p1] == fi)[0]
                    row = int(hit[0]) if len(hit) else -1
                ok = row >= 0
                bf, col = [0, 0, 0], [0, 0, 0]
                if ok:
                    svo = B.sh_sv_off[s]
                    gs = int(B.sv[svo + B.sh_b[s, 0]])
                    gd = int(B.sv[svo + B.sh_b[s, 1]])
                    for c in range(3):
                        g = int(B.sv[svo + B.FUV_pre[B.sh_fpre_off[s] + row, c]])
                        post = gs if g == gd else g
                        k = 0
                        while k < 3 and cur[k] != post:
                            k += 1
                        if k == 3:
                            ok = False
                            break
                        bf[c] = g
                        col[c] = k
                init = (fi, bf, col) if ok else None
            faceInit[cf] = init
        init = faceInit[cf]
        if init is None:
            nUnmatched += 1
            continue
        fi, bf, col = init
        q = len(qVert)
        BF[q] = bf
        BC[q] = S.fineBary[i, col]
        FIdx[q] = fi
        qVert.append(i)
    nq = len(qVert)
    BC = BC[:nq].copy()
    BF = BF[:nq].copy()
    FIdx = FIdx[:nq].copy()

    C.BC0, C.BF0, C.FIdx0 = BC.copy(), BF.copy(), FIdx.copy()  # query start (for verify_c2f)
    st = query_coarse_to_fine(B, BC, BF, FIdx)
    qv = np.array(qVert, dtype=np.int64)
    C.walkSteps = np.zeros(n, np.int64); C.clampedSteps = np.zeros(n, np.int64); C.farSteps = np.zeros(n, np.int64)
    C.maxNegBary = np.zeros(n); C.sumNegBary = np.zeros(n); C.maxSnapRel = np.zeros(n)
    C.walkSteps[qv] = st['steps']; C.clampedSteps[qv] = st['clamped']; C.farSteps[qv] = st['farOutside']
    C.maxNegBary[qv] = st['maxNegBary']; C.sumNegBary[qv] = st['sumNegBary']; C.maxSnapRel[qv] = st['maxSnapRel']

    steps, clamped, far = int(st['steps'].sum()), int(st['clamped'].sum()), int(st['farOutside'].sum())
    cm = st['clamped'] > 0
    qClamped = int(cm.sum())

    def pct(v, p):
        if len(v) == 0:
            return 0.0
        v = np.sort(v)
        return float(v[int(min(len(v) - 1.0, p * len(v)))])
    neg, snap = st['maxNegBary'][cm], st['maxSnapRel'][cm]
    log('[coarse_subdiv] c2f clamp: %d queries, %d walk steps (%.1f per query); %d steps clamped (%.2f%%), '
          '%d far outside; %d queries clamped at least once (%.1f%%); per clamped query, largest negative '
          'barycentric median %.3g p90 %.3g max %.3g, largest snap / triangle edge median %.3g p90 %.3g max %.3g'
          % (nq, steps, steps / nq if nq else 0.0, clamped, 100.0 * clamped / steps if steps else 0.0, far,
             qClamped, 100.0 * qClamped / nq if nq else 0.0, pct(neg, 0.5), pct(neg, 0.9), pct(neg, 1.0),
             pct(snap, 0.5), pct(snap, 0.9), pct(snap, 1.0)))

    VO, FO = B.fineV, B.fineF
    C.P = S.V.copy()
    C.fineFace = np.full(n, -1, dtype=np.int64)
    C.fineBary = np.zeros((n, 3))
    C.P[qv] = (BC[:, 0:1] * VO[BF[:, 0]] + BC[:, 1:2] * VO[BF[:, 1]]) + BC[:, 2:3] * VO[BF[:, 2]]
    inr = (FIdx >= 0) & (FIdx < FO.shape[0])
    fsafe = np.where(inr, FIdx, 0)
    onFace = inr & np.all(BF == FO[fsafe], axis=1)
    nOffFace = int((~onFace).sum())
    C.fineFace[qv[onFace]] = FIdx[onFace]
    C.fineBary[qv[onFace]] = BC[onFace]
    C.BC, C.BF, C.FIdx, C.qVert = BC, BF, FIdx, qv

    log('[coarse_subdiv] %d levels: |V| %d -> %d, |F| %d -> %d (target %d); '
          '%d mapped to fine, %d on no coarse face, %d unmatched, %d ended off their fine face'
          % (S.nLevels, B.coarseV.shape[0], n, B.coarseF.shape[0], S.F.shape[0], nTarget,
             nq, n - nq - nUnmatched, nUnmatched, nOffFace))
    return C
