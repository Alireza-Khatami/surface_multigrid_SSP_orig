"""Port of subdiv_mesh_quality (subdiv_sample_tracker/subdiv_relax.cpp)."""
from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass
class MeshQuality:
    edgeCV: float = 0.0
    minAngle: float = 0.0
    p1: float = 0.0
    p5: float = 0.0
    median: float = 0.0
    degenerate: int = 0
    flippedVsRef: int = 0


@njit(cache=True)
def _quality(V, F, Vref, haveRef, diag):
    nF = F.shape[0]
    s = 0.0
    s2 = 0.0
    n = 0
    minAng = np.empty(nF, dtype=np.float32)
    degen = 0
    flipped = 0
    for f in range(nF):
        ang = 180.0
        for c in range(3):
            i0 = F[f, c]; i1 = F[f, (c + 1) % 3]; i2 = F[f, (c + 2) % 3]
            ux = V[i1, 0] - V[i0, 0]; uy = V[i1, 1] - V[i0, 1]; uz = V[i1, 2] - V[i0, 2]
            wx = V[i2, 0] - V[i0, 0]; wy = V[i2, 1] - V[i0, 1]; wz = V[i2, 2] - V[i0, 2]
            l = np.sqrt((ux * ux + uy * uy) + uz * uz)
            s += l
            s2 += l * l
            n += 1
            nu = l
            nw = np.sqrt((wx * wx + wy * wy) + wz * wz)
            if nu > 0 and nw > 0:
                x = ((ux * wx + uy * wy) + uz * wz) / (nu * nw)
                y = x if x < 1.0 else 1.0
                y = y if -1.0 < y else -1.0
                a = np.arccos(y) * 180.0 / np.pi
            else:
                a = 0.0
            if a < ang:
                ang = a
        minAng[f] = ang
        a0 = F[f, 0]; a1 = F[f, 1]; a2 = F[f, 2]
        ux = V[a1, 0] - V[a0, 0]; uy = V[a1, 1] - V[a0, 1]; uz = V[a1, 2] - V[a0, 2]
        vx = V[a2, 0] - V[a0, 0]; vy = V[a2, 1] - V[a0, 1]; vz = V[a2, 2] - V[a0, 2]
        nx = uy * vz - uz * vy; ny = uz * vx - ux * vz; nz = ux * vy - uy * vx
        if 0.5 * np.sqrt((nx * nx + ny * ny) + nz * nz) <= 1e-14 * diag * diag:
            degen += 1
        if haveRef:
            ux = Vref[a1, 0] - Vref[a0, 0]; uy = Vref[a1, 1] - Vref[a0, 1]; uz = Vref[a1, 2] - Vref[a0, 2]
            vx = Vref[a2, 0] - Vref[a0, 0]; vy = Vref[a2, 1] - Vref[a0, 1]; vz = Vref[a2, 2] - Vref[a0, 2]
            rx = uy * vz - uz * vy; ry = uz * vx - ux * vz; rz = ux * vy - uy * vx
            if (nx * rx + ny * ry) + nz * rz < 0:
                flipped += 1
    return s, s2, n, minAng, degen, flipped


def subdiv_mesh_quality(V, F, Vref=None):
    Q = MeshQuality()
    V = np.ascontiguousarray(V[:, :3], dtype=np.float64)
    d = V.max(axis=0) - V.min(axis=0)
    diag = float(np.sqrt((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]))
    have = Vref is not None
    s, s2, n, minAng, Q.degenerate, Q.flippedVsRef = _quality(
        V, np.ascontiguousarray(F, dtype=np.int64), np.ascontiguousarray(Vref if have else V), have, diag)
    mean = s / n if n else 0.0
    Q.edgeCV = np.sqrt(max(0.0, s2 / n - mean * mean)) / mean if mean > 0 else 0.0
    if len(minAng):
        m = np.sort(minAng)
        pct = lambda q: float(m[int(min(len(m) - 1.0, q * len(m)))])
        Q.minAngle, Q.p1, Q.p5, Q.median = float(m[0]), pct(0.01), pct(0.05), pct(0.5)
    return Q
