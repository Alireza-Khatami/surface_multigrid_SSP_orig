"""Port of subdiv_sample_tracker/subdiv_mesh.h / .cpp: uniform midpoint (1->4,
no smoothing) subdivision with exact carriers.

Every subdivided vertex records its carrier, the lowest-dimensional element of
the input mesh containing it (vertex, edge, or face interior), with exact
(dyadic) coordinates on that carrier. Vertices 0..|VO|-1 are the input vertices;
each level appends its edge midpoints.
"""
import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np

from log_util import log

SUBDIV_CARRIER_VERTEX = 0  # carrierIndex = input vertex
SUBDIV_CARRIER_EDGE = 1    # carrierIndex = row of origEdges
SUBDIV_CARRIER_FACE = 2    # carrierIndex = input face

K_MAX_IDX = 2 ** 31 - 1


class SubdivError(RuntimeError):
    pass


def _fail(msg):
    raise SubdivError('[subdiv_mesh] ' + msg)


def edge_key(a, b):
    if a > b:
        a, b = b, a
    return ((a & 0xffffffff) << 32) | (b & 0xffffffff)


@dataclass
class SubdivMesh:
    V: np.ndarray = None             # Vs x 3
    F: np.ndarray = None             # Fs x 3 (int64)
    faceOrig: np.ndarray = None      # subdivided face -> input face
    origEdges: np.ndarray = None     # unique undirected input edges (min, max), sorted
    carrierType: np.ndarray = None   # uint8
    carrierIndex: np.ndarray = None  # int64
    carrierCoord: np.ndarray = None  # Vs x 3
    fineFace: np.ndarray = None      # int64, -1 when on no face
    fineBary: np.ndarray = None      # Vs x 3
    nOrigVerts: int = 0
    nLevels: int = 0
    levelVerts: List[int] = field(default_factory=list)

    def copy(self):
        return SubdivMesh(V=self.V.copy(), F=self.F.copy(), faceOrig=self.faceOrig.copy(),
                          origEdges=self.origEdges.copy(), carrierType=self.carrierType.copy(),
                          carrierIndex=self.carrierIndex.copy(), carrierCoord=self.carrierCoord.copy(),
                          fineFace=self.fineFace.copy(), fineBary=self.fineBary.copy(),
                          nOrigVerts=self.nOrigVerts, nLevels=self.nLevels, levelVerts=list(self.levelVerts))


def subdiv_unique_edges(F):
    """Unique undirected edges of F as (min, max), sorted lexicographically."""
    F = np.asarray(F, dtype=np.int64)
    a = F.reshape(-1)
    b = F[:, [1, 2, 0]].reshape(-1)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keys = np.unique((lo << 32) | hi)
    return np.stack([keys >> 32, keys & 0xffffffff], axis=1).astype(np.int64)


def subdiv_find_edge(edges, a, b):
    """Index of edge (a, b) in `edges` (from subdiv_unique_edges), -1 if absent."""
    if a > b:
        a, b = b, a
    lo, hi = 0, edges.shape[0]
    while lo < hi:
        mid = lo + (hi - lo) // 2
        ea, eb = edges[mid, 0], edges[mid, 1]
        if ea < a or (ea == a and eb < b):
            lo = mid + 1
        else:
            hi = mid
    return lo if (lo < edges.shape[0] and edges[lo, 0] == a and edges[lo, 1] == b) else -1


class _EdgeIndex:
    """subdiv_find_edge through a dict (same answers, O(1))."""

    def __init__(self, edges):
        self.d = {(int(a), int(b)): i for i, (a, b) in enumerate(edges.tolist())}

    def find(self, a, b):
        if a > b:
            a, b = b, a
        return self.d.get((a, b), -1)


def subdiv_find_edges(edges, a, b):
    """Vectorized subdiv_find_edge for arrays a, b (same result per pair)."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keys = (edges[:, 0].astype(np.int64) << 32) | edges[:, 1].astype(np.int64)
    q = (lo << 32) | hi
    idx = np.searchsorted(keys, q)
    ok = idx < len(keys)
    idx_c = np.where(ok, idx, 0)
    ok &= keys[idx_c] == q
    return np.where(ok, idx_c, -1)


def build_subdiv_mesh(VO, FO, nTarget):
    """Subdivides (VO, FO) uniformly until the vertex count is >= nTarget."""
    VO = np.asarray(VO, dtype=np.float64)
    FO = np.asarray(FO, dtype=np.int64)
    nVO, nFO = VO.shape[0], FO.shape[0]
    FOl = FO.tolist()
    for f in range(nFO):
        r = FOl[f]
        for c in range(3):
            if r[c] < 0 or r[c] >= nVO:
                _fail('face %d has out-of-range vertex' % f)
        if r[0] == r[1] or r[1] == r[2] or r[2] == r[0]:
            _fail('face %d has a repeated corner' % f)

    M = SubdivMesh()
    M.nOrigVerts = nVO
    M.levelVerts = [nVO]
    M.origEdges = subdiv_unique_edges(FO)
    E = _EdgeIndex(M.origEdges)
    oE = M.origEdges.tolist()

    # carriers (growable python lists)
    ctype = [SUBDIV_CARRIER_VERTEX] * nVO
    cidx = list(range(nVO))
    ccoord = [(1.0, 0.0, 0.0)] * nVO

    def corner_of(f, v):
        r = FOl[f]
        for c in range(3):
            if r[c] == v:
                return c
        return -1

    def to_face_bary(v, f):
        b = [0.0, 0.0, 0.0]
        idx = cidx[v]
        t = ctype[v]
        if t == SUBDIV_CARRIER_VERTEX:
            c = corner_of(f, idx)
            if c < 0:
                _fail('vertex %d (orig vertex %d) is not a corner of face %d' % (v, idx, f))
            b[c] = 1.0
        elif t == SUBDIV_CARRIER_EDGE:
            c0 = corner_of(f, oE[idx][0])
            c1 = corner_of(f, oE[idx][1])
            if c0 < 0 or c1 < 0:
                _fail('vertex %d (orig edge %d) is not on face %d' % (v, idx, f))
            b[c0] = ccoord[v][0]
            b[c1] = ccoord[v][1]
        elif t == SUBDIV_CARRIER_FACE:
            if idx != f:
                _fail('vertex %d lies inside face %d, not face %d' % (v, idx, f))
            b = list(ccoord[v])
        else:
            _fail('bad carrier type')
        return b

    def classify(b, fo):
        # dyadic entries: exact zero tests and sum check
        if (b[0] + b[1]) + b[2] != 1.0 or min(b) < 0.0:
            _fail('non-exact barycentric in face %d' % fo)
        nz = [c for c in range(3) if b[c] != 0.0]
        if len(nz) == 3:
            return SUBDIV_CARRIER_FACE, fo, (b[0], b[1], b[2])
        if len(nz) == 2:
            a, bb = FOl[fo][nz[0]], FOl[fo][nz[1]]
            e = E.find(a, bb)
            if e < 0:
                _fail('edge not found in face %d' % fo)
            wa, wb = b[nz[0]], b[nz[1]]
            coord = (wa, wb, 0.0) if oE[e][0] == a else (wb, wa, 0.0)
            return SUBDIV_CARRIER_EDGE, e, coord
        _fail('midpoint collapsed onto an original vertex in face %d' % fo)

    F = FO.copy()
    faceOrig = np.arange(nFO, dtype=np.int64)

    while len(ctype) < nTarget and F.shape[0] > 0:
        Fs = F.shape[0]
        if 4 * Fs > K_MAX_IDX:
            _fail('face count would overflow 32-bit indices')
        # half-edge h = 3f + c is F(f,c) -> F(f,(c+1)%3); sorted by (edge key, h)
        a = F.reshape(-1)
        b = F[:, [1, 2, 0]].reshape(-1)
        keys = (np.minimum(a, b) << 32) | np.maximum(a, b)
        hs = np.arange(3 * Fs, dtype=np.int64)
        order = np.lexsort((hs, keys))
        keys_s = keys[order].tolist()
        order = order.tolist()
        Fl = F.tolist()
        fol = faceOrig.tolist()
        mid = [-1] * (3 * Fs)
        n = len(order)
        i = 0
        while i < n:
            j = i
            ki = keys_s[i]
            while j < n and keys_s[j] == ki:
                j += 1
            edgeVertex = -1
            eIndex = -1
            eCoord = None
            faceMid = []
            for k in range(i, j):
                h = order[k]
                f, c = divmod(h, 3)
                u, w = Fl[f][c], Fl[f][(c + 1) % 3]
                fo = fol[f]
                bu = to_face_bary(u, fo)
                bw = to_face_bary(w, fo)
                bm = [0.5 * (bu[0] + bw[0]), 0.5 * (bu[1] + bw[1]), 0.5 * (bu[2] + bw[2])]
                typ, index, coord = classify(bm, fo)
                v = -1
                if typ == SUBDIV_CARRIER_EDGE:
                    if faceMid:
                        _fail('edge- and face-carried midpoints on one sub edge')
                    if edgeVertex < 0:
                        edgeVertex = len(ctype)
                        eIndex, eCoord = index, coord
                        ctype.append(typ); cidx.append(index); ccoord.append(coord)
                    elif index != eIndex or coord != eCoord:
                        _fail('faces around an original edge disagree on a midpoint (level %d)' % M.nLevels)
                    v = edgeVertex
                else:
                    if edgeVertex >= 0:
                        _fail('edge- and face-carried midpoints on one sub edge')
                    for fm in faceMid:
                        if fm[0] == fo:
                            v = fm[1]
                            break
                    if v < 0:
                        v = len(ctype)
                        faceMid.append((fo, v))
                        ctype.append(typ); cidx.append(index); ccoord.append(coord)
                if len(ctype) >= K_MAX_IDX:
                    _fail('vertex count would overflow 32-bit indices')
                mid[h] = v
            i = j

        midA = np.array(mid, dtype=np.int64).reshape(Fs, 3)
        m0, m1, m2 = midA[:, 0], midA[:, 1], midA[:, 2]  # a-b, b-c, c-a
        A, B, C = F[:, 0], F[:, 1], F[:, 2]
        F2 = np.empty((4 * Fs, 3), dtype=np.int64)
        F2[0::4] = np.stack([A, m0, m2], axis=1)
        F2[1::4] = np.stack([m0, B, m1], axis=1)
        F2[2::4] = np.stack([m2, m1, C], axis=1)
        F2[3::4] = np.stack([m0, m1, m2], axis=1)
        F = F2
        faceOrig = np.repeat(faceOrig, 4)
        M.nLevels += 1
        M.levelVerts.append(len(ctype))

    # ---- fine-mesh face + barycentric per vertex ----
    vtxFace = [-1] * nVO
    for f in range(nFO):
        for c in range(3):
            if vtxFace[FOl[f][c]] < 0:
                vtxFace[FOl[f][c]] = f
    edgeFace = [-1] * len(oE)
    for f in range(nFO):
        for c in range(3):
            e = E.find(FOl[f][c], FOl[f][(c + 1) % 3])
            if edgeFace[e] < 0:
                edgeFace[e] = f

    Vs = len(ctype)
    fineFace = np.full(Vs, -1, dtype=np.int64)
    fineBary = np.zeros((Vs, 3), dtype=np.float64)
    noFace = []
    for v in range(Vs):
        t = ctype[v]
        if t == SUBDIV_CARRIER_VERTEX:
            f = vtxFace[cidx[v]]
        elif t == SUBDIV_CARRIER_EDGE:
            f = edgeFace[cidx[v]]
        else:
            f = cidx[v]
        fineFace[v] = f
        if f < 0:
            noFace.append(v)
            continue
        fineBary[v] = to_face_bary(v, f)
    V = interp_rows(VO, FO, fineFace, fineBary)
    for v in noFace:
        V[v] = VO[cidx[v], :3]

    M.V = V
    M.F = F
    M.faceOrig = faceOrig
    M.carrierType = np.array(ctype, dtype=np.uint8)
    M.carrierIndex = np.array(cidx, dtype=np.int64)
    M.carrierCoord = np.array(ccoord, dtype=np.float64).reshape(Vs, 3)
    M.fineFace = fineFace
    M.fineBary = fineBary
    log('[subdiv_mesh] %d levels: |V| %d -> %d, |F| %d -> %d (target %d)'
          % (M.nLevels, nVO, Vs, nFO, F.shape[0], nTarget))
    return M


def interp_rows(VO, FO, face, bary):
    """Row-wise b0*VO[F0] + b1*VO[F1] + b2*VO[F2] (left to right, as Eigen
    evaluates it). Rows with face < 0 are 0."""
    face = np.asarray(face)
    ok = face >= 0
    fc = np.where(ok, face, 0)
    T = FO[fc]
    P = (bary[:, 0:1] * VO[T[:, 0], :3] + bary[:, 1:2] * VO[T[:, 1], :3]) + bary[:, 2:3] * VO[T[:, 2], :3]
    P[~ok] = 0.0
    return P
