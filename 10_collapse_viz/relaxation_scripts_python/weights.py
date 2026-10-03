"""Ports of cotan_weights / meanvalue_weights (coarse_subdiv_relax.cpp): one
weight per graph entry (aligned with G.cols), from the triangles F at
positions P. Contributions are summed per key in face / corner order, as the
C++ unordered_map accumulation does.
"""
import sys

import numpy as np
from numba import njit


@njit(cache=True)
def _cotan_contrib(P, F):
    nF = F.shape[0]
    keys = np.empty(3 * nF, dtype=np.int64)
    vals = np.empty(3 * nF)
    n = 0
    for f in range(nF):
        for c in range(3):
            o = F[f, c]; a = F[f, (c + 1) % 3]; b = F[f, (c + 2) % 3]
            ux = P[a, 0] - P[o, 0]; uy = P[a, 1] - P[o, 1]; uz = P[a, 2] - P[o, 2]
            vx = P[b, 0] - P[o, 0]; vy = P[b, 1] - P[o, 1]; vz = P[b, 2] - P[o, 2]
            cx = uy * vz - uz * vy; cy = uz * vx - ux * vz; cz = ux * vy - uy * vx
            s = np.sqrt((cx * cx + cy * cy) + cz * cz)
            if not (s > 0):
                continue
            lo = a if a < b else b
            hi = b if a < b else a
            keys[n] = (lo << 32) | hi
            vals[n] = 0.5 * ((ux * vx + uy * vy) + uz * vz) / s
            n += 1
    return keys[:n], vals[:n]


@njit(cache=True)
def _meanvalue_contrib(P, F):
    nF = F.shape[0]
    keys = np.empty(6 * nF, dtype=np.int64)
    vals = np.empty(6 * nF)
    n = 0
    deg = 0
    for f in range(nF):
        for c in range(3):
            i = F[f, c]; a = F[f, (c + 1) % 3]; b = F[f, (c + 2) % 3]
            ux = P[a, 0] - P[i, 0]; uy = P[a, 1] - P[i, 1]; uz = P[a, 2] - P[i, 2]
            vx = P[b, 0] - P[i, 0]; vy = P[b, 1] - P[i, 1]; vz = P[b, 2] - P[i, 2]
            lu = np.sqrt((ux * ux + uy * uy) + uz * uz)
            lv = np.sqrt((vx * vx + vy * vy) + vz * vz)
            cx = uy * vz - uz * vy; cy = uz * vx - ux * vz; cz = ux * vy - uy * vx
            s = np.sqrt((cx * cx + cy * cy) + cz * cz)
            if not (s > 0) or not (lu > 0) or not (lv > 0):
                deg += 1
                continue
            t = (lu * lv - ((ux * vx + uy * vy) + uz * vz)) / s  # tan(theta / 2)
            keys[n] = (i << 32) | a
            vals[n] = t / lu
            n += 1
            keys[n] = (i << 32) | b
            vals[n] = t / lv
            n += 1
    return keys[:n], vals[:n], deg


@njit(cache=True)
def _group_sums(keys_sorted, vals_sorted):
    """Sequential sums of runs of equal keys (input stably sorted by key)."""
    n = keys_sorted.shape[0]
    uk = np.empty(n, dtype=np.int64)
    us = np.empty(n)
    m = 0
    i = 0
    while i < n:
        k = keys_sorted[i]
        s = 0.0
        while i < n and keys_sorted[i] == k:
            s += vals_sorted[i]
            i += 1
        uk[m] = k
        us[m] = s
        m += 1
    return uk[:m], us[:m]


def _lookup(keys, vals, G, directed):
    order = np.argsort(keys, kind='stable')
    uk, us = _group_sums(keys[order], vals[order])
    rows = np.repeat(np.arange(len(G.rowOffs) - 1, dtype=np.int64), np.diff(G.rowOffs))
    cols = G.cols
    if directed:
        q = (rows << 32) | cols
    else:
        q = (np.minimum(rows, cols) << 32) | np.maximum(rows, cols)
    idx = np.searchsorted(uk, q)
    ok = idx < len(uk)
    idc = np.where(ok, idx, 0)
    ok &= uk[idc] == q
    return np.where(ok, us[idc], 0.0), ok


def cotan_weights(P, F, G, log):
    """w_ij = 1/2 (cot alpha + cot beta); negatives clamped to 0; non-mesh-edge entries 0."""
    k, v = _cotan_contrib(np.ascontiguousarray(P), np.ascontiguousarray(F, dtype=np.int64))
    w, _ = _lookup(k, v, G, False)
    neg = int((w < 0).sum())
    w = np.where(w < 0, 0.0, w)
    log('[coarse_subdiv_relax] cotangent weights from the coarse positions: %d entries, %d negative '
        '(clamped to 0), range %.3g .. %.3g' % (len(w), neg, w.min() if len(w) else np.inf, max(0.0, w.max()) if len(w) else 0.0))
    return w


def meanvalue_weights(P, F, G, log):
    """w_ij = sum over triangles on edge ij of tan(theta_i / 2) / |x_i - x_j| (Floater 2003)."""
    k, v, deg = _meanvalue_contrib(np.ascontiguousarray(P), np.ascontiguousarray(F, dtype=np.int64))
    w, ok = _lookup(k, v, G, True)
    missing = int((~ok).sum())
    log('[coarse_subdiv_relax] mean-value weights from the coarse positions: %d entries (%d not a mesh '
        'edge), %d degenerate corners skipped, range %.3g .. %.3g'
        % (len(w), missing, deg, w.min() if len(w) else np.inf, max(0.0, w.max()) if len(w) else 0.0))
    return w
