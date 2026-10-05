"""Concave parts of the MAT's edge structures (seams and boundaries of the .ma_struct).

A sheet ends at its border curves: the seams (type 1, edges where 3+ faces meet) and the
boundaries (type 2, edges with one face). At a vertex v on the border of sheet s, the
interior angle of s is the sum of the angles of s's faces at v. More than 180 degrees
means s wraps around v: the border curve is concave there (seen from the sheet).

  corner (v, s) concave  <=>  interior angle of sheet s at v > 180 + margin

The concave parts are the concave corners and the seam / boundary edges at them. The
fine faces they touch are the faces of s at a concave corner (v, s); the step viewer
marks the subdivided vertices of the relaxation input whose fine face is one of these.
"""
from dataclasses import dataclass

import numpy as np

SHEET, SEAM, BOUNDARY = 0, 1, 2


@dataclass
class ConcaveParts:
    corners: np.ndarray      # (n, 2): vertex, sheet id
    angles: np.ndarray       # (n,) interior angle of the sheet at the corner, degrees
    edges: np.ndarray        # (m, 2): seam / boundary edges with a concave corner at an end (fine vertex ids)
    edgeType: np.ndarray     # (m,) 1 seam, 2 boundary
    faceHit: np.ndarray      # (|F|,) bool: face of sheet s at a concave corner (v, s)
    nBorderCorners: int = 0  # all (vertex, sheet) border corners


def concave_curve_parts(V, F, ms, margin_deg):
    V = np.asarray(V, dtype=np.float64)[:, :3]
    F = np.asarray(F, dtype=np.int64)
    nF = F.shape[0]
    sheet = np.array([ids[0] if ids else -1 for ids in ms.faceIds], dtype=np.int64)

    # interior angle per (vertex, sheet): sum of the face angles
    cv = F.reshape(-1)
    nx, pv = F[:, [1, 2, 0]].reshape(-1), F[:, [2, 0, 1]].reshape(-1)
    u, w = V[nx] - V[cv], V[pv] - V[cv]
    cosA = np.einsum('ij,ij->i', u, w) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-300)
    ang = np.degrees(np.arccos(np.clip(cosA, -1.0, 1.0)))
    fs = np.repeat(sheet, 3)
    key = cv * (int(sheet.max()) + 2) + (fs + 1)
    uk, inv = np.unique(key, return_inverse=True)
    angSum = np.bincount(inv, ang, minlength=len(uk))

    # border corners of each sheet: ends of edges that have exactly one face of that sheet
    a, b = cv, nx
    ek = (np.minimum(a, b) << 32) | np.maximum(a, b)
    esk = np.stack([ek, fs], axis=1)
    pairs, cnt = np.unique(esk, axis=0, return_counts=True)
    one = pairs[cnt == 1]
    ends = np.concatenate([one[:, 0] >> 32, one[:, 0] & 0xffffffff])
    endSheet = np.concatenate([one[:, 1], one[:, 1]])
    bk = np.unique(ends * (int(sheet.max()) + 2) + (endSheet + 1))
    isBorder = np.isin(uk, bk)

    conc = isBorder & (angSum > 180.0 + margin_deg) & (uk % (int(sheet.max()) + 2) > 0)
    cvert = uk[conc] // (int(sheet.max()) + 2)
    csheet = uk[conc] % (int(sheet.max()) + 2) - 1
    corners = np.stack([cvert, csheet], axis=1)

    # faces of sheet s at a concave corner (v, s)
    ckeys = set((uk[conc]).tolist())
    faceHit = np.zeros(nF, dtype=bool)
    hit = np.isin(key, list(ckeys)) if ckeys else np.zeros(len(key), dtype=bool)
    faceHit[np.nonzero(hit)[0] // 3] = True

    # seam / boundary edges with a concave corner at one end
    cset = set(cvert.tolist())
    E, T = [], []
    for st in ms.structs:
        if st.type not in (SEAM, BOUNDARY):
            continue
        for e in st.elements:
            p, q = ms.maEdges[e]
            if p in cset or q in cset:
                E.append((min(p, q), max(p, q)))
                T.append(st.type)
    E = np.array(E, dtype=np.int64).reshape(-1, 2)
    T = np.array(T, dtype=np.uint8)
    if len(E):
        E, idx = np.unique(E, axis=0, return_index=True)
        T = T[idx]
    return ConcaveParts(corners=corners, angles=angSum[conc], edges=E, edgeType=T, faceHit=faceHit,
                        nBorderCorners=int(isBorder.sum()))


def compact(V, E):
    """Curve network data with only the vertices the edges use: (nodes, edges, vertex ids)."""
    ids = np.unique(E.reshape(-1))
    remap = np.full(int(ids.max()) + 1 if len(ids) else 0, -1, dtype=np.int64)
    remap[ids] = np.arange(len(ids))
    return np.asarray(V)[ids, :3], remap[E], ids
