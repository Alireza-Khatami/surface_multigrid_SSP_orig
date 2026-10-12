"""Port of the C++ Newton relaxation (subdiv_sample_tracker/subdiv_relax.cpp, subdiv_relax with
RelaxSolver::Newton; coarse_subdiv_relax method "newton", the C++ default).

Minimises E = 1/2 sum over the graph's edges of |x_i - x_j|^2 with every vertex kept on its own
structure, in two passes: the curve vertices first (their rows only hold curve / junction
neighbours, so they ignore the sheets), then the sheet vertices with the curves fixed. That is
the directed graph; each pass then has a real, symmetric energy. jointPass: one pass over all
free vertices (needs the symmetric graph).

One Newton iteration of a pass:
  1. a tangent frame T_i per vertex (3 x dim, dim 0..3): a curve vertex inside an edge -> the
     edge direction; a sheet vertex inside a face -> the face plane; on a kink / crease /
     polyline vertex -> the incident element its pull r_i = sum_j (x_j - x_i) points into
     (none: held, dim 0);
  2. the tangent displacements u_i (dim_i unknowns each) from the Gauss-Newton system
         deg_i u_i - sum_{j in the pass} T_i^T T_j u_j = T_i^T r_i
     (symmetric positive definite). C++: sparse LDLT. Here: conjugate gradients, matrix-free,
     Jacobi preconditioner (numba; no scipy). CG from 0 gives a descent direction at every
     iterate, so an inexact solve is still a valid step;
  3. backtracking line search on E (Armijo, up to 50 halvings) of y_i = x_i + alpha T_i u_i,
     projected onto the vertex's own structure, with the no-new-folds hold-back. No decrease
     even for tiny steps: the Newton round stops (a "stall").
Newton rounds alternate with plain projected steps x <- Pi(x + 0.5 (mean - x)), which carry a
vertex over a curve corner or a sheet crease that its frame cannot. A pass is done when a plain
step moves nothing (<= tol x diag), or after maxIter iterations (Newton + plain steps).

Set-up, projection, fold hold-back, commit and checks are the explicit relaxer's
(relax_explicit.ExplicitRelaxer); only the pinning (per role class, as in C++) and the
iteration differ. Not bit-exact with the C++ (CG instead of LDLT).
"""
import time
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from projector import _csr, edge_in, face_in
from relax_explicit import (ExplicitRelaxer, RelaxReport, _commit, _hold_back, _new_folds, count_degenerate,
                            step_kernel, _NO_TR)
from relax_momentum import project_targets
from struct_ids import RELAX_CURVE, RELAX_JUNCTION, RELAX_SHEET

K_ZERO = 1e-12


@dataclass
class NewtonReport(RelaxReport):
    itersCurve: int = 0
    deltaCurve: float = 0.0
    halvings: int = 0
    lineSearchStalls: int = 0
    plainSteps: int = 0
    nPinnedCurve: int = 0
    cgIters: int = 0
    cgWorst: float = 0.0
    residualSheet: float = 0.0
    residualCurve: float = 0.0
    nResidualSheet: int = 0
    nResidualCurve: int = 0
    jacobiStepMove: float = 0.0


# ------------------------------------------------------------------ frames

@njit(cache=True, inline='always')
def _face_frame(VO, FO, f, T, i):
    """Orthonormal tangent basis of fine face f into T[i, :, 0:2]; returns (ok, n)."""
    a0 = VO[FO[f, 0], 0]; a1 = VO[FO[f, 0], 1]; a2 = VO[FO[f, 0], 2]
    ux = VO[FO[f, 1], 0] - a0; uy = VO[FO[f, 1], 1] - a1; uz = VO[FO[f, 1], 2] - a2
    vx = VO[FO[f, 2], 0] - a0; vy = VO[FO[f, 2], 1] - a1; vz = VO[FO[f, 2], 2] - a2
    nx = uy * vz - uz * vy; ny = uz * vx - ux * vz; nz = ux * vy - uy * vx
    l = np.sqrt(nx * nx + ny * ny + nz * nz)
    wx = VO[FO[f, 2], 0] - VO[FO[f, 1], 0]; wy = VO[FO[f, 2], 1] - VO[FO[f, 1], 1]
    wz = VO[FO[f, 2], 2] - VO[FO[f, 1], 2]
    lmax = max(ux * ux + uy * uy + uz * uz, vx * vx + vy * vy + vz * vz, wx * wx + wy * wy + wz * wz)
    if not (l * l > 1e-24 * lmax * lmax):
        return False, 0.0, 0.0, 0.0
    nx /= l; ny /= l; nz /= l
    if abs(nx) < 0.9:   # UnitX x n
        tx, ty, tz = 0.0, -nz, ny
    else:               # UnitY x n
        tx, ty, tz = nz, 0.0, -nx
    tl = np.sqrt(tx * tx + ty * ty + tz * tz)
    tx /= tl; ty /= tl; tz /= tl
    T[i, 0, 0] = tx; T[i, 1, 0] = ty; T[i, 2, 0] = tz
    T[i, 0, 1] = ny * tz - nz * ty; T[i, 1, 1] = nz * tx - nx * tz; T[i, 2, 1] = nx * ty - ny * tx
    return True, nx, ny, nz


@njit(cache=True, inline='always')
def _line_frame(VO, v0, v1, T, i):
    dx = VO[v1, 0] - VO[v0, 0]; dy = VO[v1, 1] - VO[v0, 1]; dz = VO[v1, 2] - VO[v0, 2]
    l = np.sqrt(dx * dx + dy * dy + dz * dz)
    if not (l > 0):
        return False
    T[i, 0, 0] = dx / l; T[i, 1, 0] = dy / l; T[i, 2, 0] = dz / l
    return True


@njit(cache=True, inline='always')
def _identity(T, i):
    for a in range(3):
        for b in range(3):
            T[i, a, b] = 1.0 if a == b else 0.0


@njit(cache=True)
def _find_edge(E, p, q):
    a, b = (p, q) if p < q else (q, p)
    lo, hi = 0, E.shape[0] - 1
    while lo <= hi:
        m = (lo + hi) // 2
        if E[m, 0] < a or (E[m, 0] == a and E[m, 1] < b):
            lo = m + 1
        elif E[m, 0] == a and E[m, 1] == b:
            return m
        else:
            hi = m - 1
    return -1


@njit(cache=True)
def _best_edge(PJ, v, k, rx, ry, rz, mode, eF_off, eF, T, i):
    """Incident edge of fine vertex v the pull points along most, as a line frame. mode 0: the
    edges of the set's curves; mode 1: edges on one of the set's sheets."""
    VO, E, ve_off, ve = PJ[0], PJ[2], PJ[6], PJ[7]
    best = 0.0
    be = -1
    for t in range(ve_off[v], ve_off[v + 1]):
        e = ve[t]
        if mode == 0:
            if not edge_in(PJ, k, e):
                continue
        else:
            inSheet = False
            for s in range(eF_off[e], eF_off[e + 1]):
                if face_in(PJ, k, eF[s]):
                    inSheet = True
                    break
            if not inSheet:
                continue
        w = E[e, 1] if E[e, 0] == v else E[e, 0]
        dx = VO[w, 0] - VO[v, 0]; dy = VO[w, 1] - VO[v, 1]; dz = VO[w, 2] - VO[v, 2]
        l = np.sqrt(dx * dx + dy * dy + dz * dz)
        if not (l > 0):
            continue
        s_ = (dx * rx + dy * ry + dz * rz) / l
        if s_ > best or (s_ == best and be >= 0 and e < be):
            best = s_
            be = e
    return be >= 0 and _line_frame(VO, E[be, 0], E[be, 1], T, i)


@njit(cache=True)
def _frame(PJ, i, k, role, f, e, b0, b1, b2, rx, ry, rz, eF_off, eF, T):
    """Tangent frame of vertex i into T[i]; returns its dimension (0..3)."""
    VO, FO, E, vf_off, vf = PJ[0], PJ[1], PJ[2], PJ[4], PJ[5]
    bc = (b0, b1, b2)
    if role == RELAX_CURVE:
        if e < 0:
            return 0
        v0 = E[e, 0]; v1 = E[e, 1]
        w0 = 0.0; w1 = 0.0
        for c in range(3):
            if FO[f, c] == v0:
                w0 = bc[c]
            if FO[f, c] == v1:
                w1 = bc[c]
        if w0 > K_ZERO and w1 > K_ZERO:
            if _line_frame(VO, v0, v1, T, i):
                return 1
            _identity(T, i)
            return 3
        v = v0 if w0 >= w1 else v1   # on the polyline vertex v
        return 1 if _best_edge(PJ, v, k, rx, ry, rz, 0, eF_off, eF, T, i) else 0
    nz = 0
    zc = 0
    for c in range(3):
        if not (bc[c] > K_ZERO):
            if nz == 0:
                zc = c
            nz += 1
    if nz == 0:
        ok, _, _, _ = _face_frame(VO, FO, f, T, i)
        if ok:
            return 2
        _identity(T, i)
        return 3
    if nz == 1:   # on the edge opposite corner zc
        p = FO[f, (zc + 1) % 3]; q = FO[f, (zc + 2) % 3]
        ed = _find_edge(E, p, q)
        dx = VO[q, 0] - VO[p, 0]; dy = VO[q, 1] - VO[p, 1]; dz = VO[q, 2] - VO[p, 2]
        dl = np.sqrt(dx * dx + dy * dy + dz * dz)
        if dl > 0:
            dx /= dl; dy /= dl; dz /= dl
        best = 0.0
        bf = -1
        if ed >= 0:
            for s in range(eF_off[ed], eF_off[ed + 1]):
                g = eF[s]
                if not face_in(PJ, k, g):
                    continue
                ok, nx, ny, nzz = _face_frame(VO, FO, g, T, i)
                if not ok:
                    continue
                third = -1
                for c in range(3):
                    if FO[g, c] != p and FO[g, c] != q:
                        third = FO[g, c]
                wx = VO[third, 0] - VO[p, 0]; wy = VO[third, 1] - VO[p, 1]; wz = VO[third, 2] - VO[p, 2]
                t = wx * dx + wy * dy + wz * dz
                wx -= t * dx; wy -= t * dy; wz -= t * dz
                wl = np.sqrt(wx * wx + wy * wy + wz * wz)
                if not (wl > 0):
                    continue
                rn = rx * nx + ry * ny + rz * nzz
                s_ = ((rx - rn * nx) * wx + (ry - rn * ny) * wy + (rz - rn * nzz) * wz) / wl
                if s_ > best or (s_ == best and bf >= 0 and g < bf):
                    best = s_
                    bf = g
        if bf >= 0:
            ok, _, _, _ = _face_frame(VO, FO, bf, T, i)
            if ok:
                return 2
        if _line_frame(VO, p, q, T, i):
            return 1
        return 0
    # on fine vertex v
    v = FO[f, 0 if b0 > K_ZERO else (1 if b1 > K_ZERO else 2)]
    best = 0.0
    bf = -1
    for t in range(vf_off[v], vf_off[v + 1]):
        g = vf[t]
        if not face_in(PJ, k, g):
            continue
        ok, nx, ny, nzz = _face_frame(VO, FO, g, T, i)
        if not ok:
            continue
        cv = 0
        while FO[g, cv] != v:
            cv += 1
        g1 = FO[g, (cv + 1) % 3]; g2 = FO[g, (cv + 2) % 3]
        e1x = VO[g1, 0] - VO[v, 0]; e1y = VO[g1, 1] - VO[v, 1]; e1z = VO[g1, 2] - VO[v, 2]
        e2x = VO[g2, 0] - VO[v, 0]; e2y = VO[g2, 1] - VO[v, 1]; e2z = VO[g2, 2] - VO[v, 2]
        rn = rx * nx + ry * ny + rz * nzz
        gx = rx - rn * nx; gy = ry - rn * ny; gz = rz - rn * nzz
        a11 = e1x * e1x + e1y * e1y + e1z * e1z
        a12 = e1x * e2x + e1y * e2y + e1z * e2z
        a22 = e2x * e2x + e2y * e2y + e2z * e2z
        c1 = e1x * gx + e1y * gy + e1z * gz
        c2 = e2x * gx + e2y * gy + e2z * gz
        det = a11 * a22 - a12 * a12
        if not (det != 0.0):
            continue
        s1 = (a22 * c1 - a12 * c2) / det
        s2 = (a11 * c2 - a12 * c1) / det
        gn = np.sqrt(gx * gx + gy * gy + gz * gz)
        if s1 > 0 and s2 > 0 and gn > best:
            best = gn
            bf = g
    if bf >= 0:
        ok, _, _, _ = _face_frame(VO, FO, bf, T, i)
        if ok:
            return 2
    return 1 if _best_edge(PJ, v, k, rx, ry, rz, 1, eF_off, eF, T, i) else 0


@njit(parallel=True, cache=True)
def _frames(lst, X, face, edge, bary, role, setId, rowOffs, cols, PJ, eF_off, eF, T, dims, R):
    """Pull r_i = sum_j (x_j - x_i) (into R) and the frame of every listed vertex."""
    for a in prange(lst.shape[0]):
        i = lst[a]
        rx = 0.0; ry = 0.0; rz = 0.0
        for q in range(rowOffs[i], rowOffs[i + 1]):
            j = cols[q]
            rx += X[j, 0] - X[i, 0]; ry += X[j, 1] - X[i, 1]; rz += X[j, 2] - X[i, 2]
        R[i, 0] = rx; R[i, 1] = ry; R[i, 2] = rz
        for c in range(3):
            T[i, 0, c] = 0.0; T[i, 1, c] = 0.0; T[i, 2, c] = 0.0
        dims[i] = _frame(PJ, i, setId[i], role[i], face[i], edge[i], bary[i, 0], bary[i, 1], bary[i, 2],
                         rx, ry, rz, eF_off, eF, T)


# ------------------------------------------------------------------ tangent system (CG)

@njit(parallel=True, cache=True)
def _matvec(lst, rowOffs, cols, inPass, T, dims, dg, U, W, out):
    """out_i = dg_i u_i - T_i^T sum_{j in the pass} T_j u_j (W = T u, scratch)."""
    n = lst.shape[0]
    for a in prange(n):
        i = lst[a]
        for c in range(3):
            s = 0.0
            for d in range(dims[i]):
                s += T[i, c, d] * U[i, d]
            W[i, c] = s
    for a in prange(n):
        i = lst[a]
        sx = 0.0; sy = 0.0; sz = 0.0
        for q in range(rowOffs[i], rowOffs[i + 1]):
            j = cols[q]
            if inPass[j]:
                sx += W[j, 0]; sy += W[j, 1]; sz += W[j, 2]
        for d in range(3):
            if d < dims[i]:
                out[i, d] = dg[i] * U[i, d] - (T[i, 0, d] * sx + T[i, 1, d] * sy + T[i, 2, d] * sz)
            else:
                out[i, d] = 0.0


@njit(cache=True)
def _dot(lst, A, B):
    s = 0.0
    for a in range(lst.shape[0]):
        i = lst[a]
        s += (A[i, 0] * B[i, 0] + A[i, 1] * B[i, 1]) + A[i, 2] * B[i, 2]
    return s


@njit(cache=True)
def _cg(lst, rowOffs, cols, inPass, T, dims, dg, Rpull, U, tol, maxIter, r, z, p, Ap, W):
    """Solves the tangent system for U (from 0); rhs_i = T_i^T r_i. Returns
    (iterations, relative residual, slope = rhs . U)."""
    for a in range(lst.shape[0]):
        i = lst[a]
        for d in range(3):
            U[i, d] = 0.0
            if d < dims[i]:
                r[i, d] = T[i, 0, d] * Rpull[i, 0] + T[i, 1, d] * Rpull[i, 1] + T[i, 2, d] * Rpull[i, 2]
            else:
                r[i, d] = 0.0
            z[i, d] = r[i, d] / dg[i]
            p[i, d] = z[i, d]
    bnorm = np.sqrt(_dot(lst, r, r))
    rhsU = 0.0
    if not (bnorm > 0):
        return 0, 0.0, 0.0
    rz = _dot(lst, r, z)
    it = 0
    res = 1.0
    for it in range(1, maxIter + 1):
        _matvec(lst, rowOffs, cols, inPass, T, dims, dg, p, W, Ap)
        pAp = _dot(lst, p, Ap)
        if not (pAp > 0):
            break
        alpha = rz / pAp
        for a in range(lst.shape[0]):
            i = lst[a]
            for d in range(3):
                U[i, d] += alpha * p[i, d]
                r[i, d] -= alpha * Ap[i, d]
        res = np.sqrt(_dot(lst, r, r)) / bnorm
        if res <= tol:
            break
        for a in range(lst.shape[0]):
            i = lst[a]
            for d in range(3):
                z[i, d] = r[i, d] / dg[i]
        rzN = _dot(lst, r, z)
        beta = rzN / rz
        rz = rzN
        for a in range(lst.shape[0]):
            i = lst[a]
            for d in range(3):
                p[i, d] = z[i, d] + beta * p[i, d]
    # slope = rhs . U, with rhs = T^T r_pull
    for a in range(lst.shape[0]):
        i = lst[a]
        for d in range(dims[i]):
            rhsU += (T[i, 0, d] * Rpull[i, 0] + T[i, 1, d] * Rpull[i, 1] + T[i, 2, d] * Rpull[i, 2]) * U[i, d]
    return it, res, rhsU


@njit(parallel=True, cache=True)
def _targets(lst, X, T, dims, U, alpha, Y):
    for a in prange(lst.shape[0]):
        i = lst[a]
        for c in range(3):
            s = 0.0
            for d in range(dims[i]):
                s += T[i, c, d] * U[i, d]
            Y[i, c] = X[i, c] + alpha * s


@njit(parallel=True, cache=True)
def _energy_change(lst, rowOffs, cols, inPass, X, nX, dE):
    """E(nX) - E(X) over the pass's rows, per edge from differences (an edge between two
    listed vertices is in both rows, weight 1/2 each)."""
    for a in prange(lst.shape[0]):
        i = lst[a]
        s = 0.0
        for q in range(rowOffs[i], rowOffs[i + 1]):
            j = cols[q]
            both = inPass[j]
            for c in range(3):
                d = X[i, c] - X[j, c]
                dn = nX[i, c] - (nX[j, c] if both else X[j, c])
                s += (0.25 if both else 0.5) * (dn - d) * (dn + d)
        dE[a] = s
    t = 0.0
    for a in range(lst.shape[0]):
        t += dE[a]
    return t


@njit(cache=True)
def _pin_by_role(isFree, rowOffs, cols, role, joint):
    """C++ pinning: a group of free vertices of one role class is anchored when it reaches a
    fixed vertex or (two passes) a vertex of another class. Returns (pinned, pinned curves)."""
    Vs = isFree.shape[0]
    seen = np.zeros(Vs, dtype=np.uint8)
    st = np.empty(Vs, dtype=np.int64)
    pinned = 0
    pinnedCurve = 0
    for s0 in range(Vs):
        if not isFree[s0] or seen[s0]:
            continue
        cls = role[s0]
        sp = 0
        st[sp] = s0
        sp += 1
        seen[s0] = 1
        anchored = False
        lowest = s0
        while sp:
            sp -= 1
            v = st[sp]
            if v < lowest:
                lowest = v
            for q in range(rowOffs[v], rowOffs[v + 1]):
                j = cols[q]
                if not isFree[j] or (not joint and role[j] != cls):
                    anchored = True
                    continue
                if not seen[j]:
                    seen[j] = 1
                    st[sp] = j
                    sp += 1
        if not anchored:
            isFree[lowest] = 0
            pinned += 1
            if role[lowest] == RELAX_CURVE:
                pinnedCurve += 1
    return pinned, pinnedCurve


@njit(parallel=True, cache=True)
def _residuals(lst, X, face, edge, bary, role, setId, rowOffs, cols, PJ, eF_off, eF, T, dims, res, interior):
    """Fixed-point residual |T^T (mean - x)| / mean edge length of the free vertices strictly
    inside a face (sheet) or an edge (curve)."""
    for a in prange(lst.shape[0]):
        i = lst[a]
        res[a] = 0.0
        interior[a] = 0
        if role[i] == RELAX_CURVE:
            nzc = 0
            mn = 1.0
            for c in range(3):
                if bary[i, c] != 0.0:
                    nzc += 1
                    mn = min(mn, bary[i, c])
            inside = nzc == 2 and mn > 1e-9
        else:
            inside = min(bary[i, 0], min(bary[i, 1], bary[i, 2])) > 1e-9
        if not inside:
            continue
        mx = 0.0; my = 0.0; mz = 0.0
        ln = 0.0
        for q in range(rowOffs[i], rowOffs[i + 1]):
            j = cols[q]
            mx += X[j, 0]; my += X[j, 1]; mz += X[j, 2]
            dx = X[j, 0] - X[i, 0]; dy = X[j, 1] - X[i, 1]; dz = X[j, 2] - X[i, 2]
            ln += np.sqrt(dx * dx + dy * dy + dz * dz)
        deg = float(rowOffs[i + 1] - rowOffs[i])
        dim = _frame(PJ, i, setId[i], role[i], face[i], edge[i], bary[i, 0], bary[i, 1], bary[i, 2],
                     mx - deg * X[i, 0], my - deg * X[i, 1], mz - deg * X[i, 2], eF_off, eF, T)
        mx /= deg; my /= deg; mz /= deg
        ln /= deg
        if not (ln > 0):
            continue
        s = 0.0
        for d in range(dim):
            t = T[i, 0, d] * (mx - X[i, 0]) + T[i, 1, d] * (my - X[i, 1]) + T[i, 2, d] * (mz - X[i, 2])
            s += t * t
        res[a] = np.sqrt(s) / ln
        interior[a] = 1


# ------------------------------------------------------------------ relaxer

class NewtonRelaxer(ExplicitRelaxer):
    """The C++ Newton relaxation (see the module doc). opt: ExplicitRelaxOptions, of which
    maxIter (per pass), tol, localProjection, noNewFolds / foldRef, holdFixed, snapshots and
    logEvery are used (lam is the plain step's 0.5). Uniform weights only."""

    def __init__(self, *args, joint=False, cg_tol=1e-10, cg_max_iter=20000, **kw):
        self.joint = bool(joint)   # needed by _pin, which the base constructor calls
        self.nPinnedCurve = 0
        super().__init__(*args, **kw)
        self.R = NewtonReport(**{k: v for k, v in vars(self.R).items()})
        self.R.nPinnedCurve = self.nPinnedCurve
        self.cgTol, self.cgMaxIter = float(cg_tol), int(cg_max_iter)
        if self.opt.weights is not None and len(self.opt.weights):
            self.log('[relax_newton] WARNING: weights are explicit only; Newton uses uniform weights')
        Vs = self.Vs
        self.role = np.ascontiguousarray(self.G.role, dtype=np.int64)
        self.eF_off, self.eF = _csr(self.proj.edgeFaces)
        self.T = np.zeros((Vs, 3, 3))
        self.dims = np.zeros(Vs, np.int64)
        self.Rpull = np.zeros((Vs, 3))
        self.U = np.zeros((Vs, 3))
        self.work = tuple(np.zeros((Vs, 3)) for _ in range(5))
        self.Y = self.X.copy()
        self.dEa = np.zeros(max(1, len(self.lst)))
        self.uni = np.ones(Vs, dtype=np.uint8)
        # seed self-projection (C++): commit the projected seed of every free vertex (moves
        # <= 1e-12 x diag, checked by the base constructor); also fills edge[] of curve vertices
        lst = self.lst
        if len(lst):
            project_targets(lst, self.X, self.X, self.face, self.edge, self.setId, False, self.PJ,
                            self.proj.scratch, self.nX, self.nFace, self.nEdge, self.nBary, self.move,
                            self.s_onb, self.s_fix, self.s_dist, self.s_grown, self.s_global)
            self.X[lst] = self.nX[lst]; self.face[lst] = self.nFace[lst]
            self.edge[lst] = self.nEdge[lst]; self.bary[lst] = self.nBary[lst]
        self.log('[relax_newton] Newton (%s), tangent system by CG (tol %.1e, max %d iterations), tol %.3g x diag, '
                 'max %d iterations per pass, %d pinned (%d curve)'
                 % ('joint pass' if self.joint else 'curves, then sheets with the curves fixed', self.cgTol,
                    self.cgMaxIter, self.opt.tol, self.opt.maxIter, self.R.nPinned, self.nPinnedCurve))

    def _pin(self, isFree):
        n, nc = _pin_by_role(isFree, self.rowOffs, self.cols, np.ascontiguousarray(self.G.role, dtype=np.int64),
                             self.joint)
        self.nPinnedCurve = nc
        return n

    # -- helpers on a pass's vertex list
    def _stats(self, n):
        t = self.total
        t.calls += n
        t.onBorder += int(self.s_onb[:n].sum())
        t.baryFix += int(self.s_fix[:n].sum())
        t.distSum += float(self.s_dist[:n].sum())
        if n:
            t.distMax = max(t.distMax, float(self.s_dist[:n].max()))

    def _hold(self, lst):
        if not self.opt.noNewFolds:
            return 0
        held = 0
        while True:
            _new_folds(self.X, self.nX, self.Fi, self.ref, self.newFold)
            h = _hold_back(self.X, self.nX, self.Fi, self.newFold, self.isFree, self.face, self.edge, self.bary,
                           self.nFace, self.nEdge, self.nBary, self.move)
            if not h:
                return held
            held += h

    def _commit(self, lst):
        m, moves, jumps, jumpMax = _commit(lst, self.move, self.FOi, self.X, self.face, self.edge, self.bary,
                                           self.nX, self.nFace, self.nEdge, self.nBary, self.diag)
        R = self.R
        R.projMoves += moves
        R.projJumps += jumps
        R.projJumpMax = max(R.projJumpMax, jumpMax)
        return m

    def _plain_step(self, lst):
        n = len(lst)
        step_kernel(lst, self.rowOffs, self.cols, np.zeros(1), self.uni, self.X, self.face, self.edge, self.setId,
                    0.5, self.opt.localProjection, self.PJ, self.proj.scratch, self.nX, self.nFace, self.nEdge,
                    self.nBary, self.move, self.s_onb, self.s_fix, self.s_dist, self.s_grown, self.s_global, _NO_TR)
        self._stats(n)

    def _tick(self, passName, it, nVar, delta, alpha, dE, cgIt, cgRes, t0, plain=False):
        self.it += 1
        self.snapshot(self.it0 + self.it)
        if plain:
            if self.R.plainSteps % 100 == 1:
                self.log('[relax_newton] %s iter %d: plain step, max move %.3g (x diag)' % (passName, it, delta))
            return
        if it <= 10 or it % 10 == 0:
            self.log('[relax_newton] %s iter %d: %d unknowns, max move %.3g (x diag), step %.3g, dE %.3g, '
                     'CG %d iterations (residual %.1e), folded %d, degenerate %d (%.2f s)'
                     % (passName, it, nVar, delta, alpha, dE, cgIt, cgRes, self.count_folded(self.X),
                        count_degenerate(self.X, self.Fi, self.diag), time.perf_counter() - t0))

    def run(self):
        opt, R = self.opt, self.R
        X = self.X
        tolAbs = self.tolAbs
        maxIter = opt.maxIter if opt.maxIter > 0 else 100000
        isFree = self.isFree
        conv_all = True
        for ps in range(1 if self.joint else 2):
            cls = RELAX_CURVE if ps == 0 else RELAX_SHEET
            passName = 'joint' if self.joint else ('curve' if ps == 0 else 'sheet')
            sel = (isFree == 1) & ((self.role != RELAX_JUNCTION) if self.joint else (self.role == cls))
            lst = np.nonzero(sel)[0].astype(np.int64)
            inPass = sel.astype(np.bool_)
            n = len(lst)
            iters = 0
            delta = 0.0
            conv = n == 0
            t0 = time.perf_counter()
            while n:
                conv = False
                while not conv and iters < maxIter:
                    ti = time.perf_counter()
                    _frames(lst, X, self.face, self.edge, self.bary, self.role, self.setId, self.rowOffs, self.cols,
                            self.PJ, self.eF_off, self.eF, self.T, self.dims, self.Rpull)
                    deg = np.diff(self.rowOffs).astype(np.float64)
                    dg = deg * (1.0 + 1e-10)
                    nVar = int(self.dims[lst].sum())
                    r, z, p, Ap, W = self.work
                    cgIt, cgRes, slope = _cg(lst, self.rowOffs, self.cols, inPass, self.T, self.dims, dg, self.Rpull,
                                             self.U, self.cgTol, self.cgMaxIter, r, z, p, Ap, W)
                    R.cgIters += cgIt
                    R.cgWorst = max(R.cgWorst, cgRes)
                    alpha, m, dE = 1.0, 0.0, 0.0
                    accepted = False
                    halvings = 0
                    while halvings <= 50:
                        self.nX[:] = X
                        _targets(lst, X, self.T, self.dims, self.U, alpha, self.Y)
                        project_targets(lst, self.Y, X, self.face, self.edge, self.setId, opt.localProjection,
                                        self.PJ, self.proj.scratch, self.nX, self.nFace, self.nEdge, self.nBary,
                                        self.move, self.s_onb, self.s_fix, self.s_dist, self.s_grown, self.s_global)
                        self._stats(n)
                        held = self._hold(lst)
                        R.foldReverts += held
                        m = float(self.move[lst].max())
                        if halvings == 0 and m <= tolAbs:
                            accepted = True
                            break
                        dE = _energy_change(lst, self.rowOffs, self.cols, inPass, X, self.nX, self.dEa)
                        if dE <= -1e-4 * alpha * slope or (held > 0 and dE < 0):
                            accepted = True
                            break
                        halvings += 1
                        alpha *= 0.5
                    iters += 1
                    if not accepted:
                        delta = 0.0
                        conv = True
                        R.lineSearchStalls += 1
                        self.it += 1
                        break
                    self._commit(lst)
                    delta = m / self.diag
                    conv = halvings == 0 and m <= tolAbs
                    R.halvings += halvings
                    self._tick(passName, iters, nVar, delta, alpha, dE, cgIt, cgRes, ti)
                # plain projected step; done when it moves nothing
                self.nX[:] = X
                self._plain_step(lst)
                self._hold(lst)
                mp = float(self.move[lst].max())
                delta = mp / self.diag
                if mp <= tolAbs:
                    conv = True
                    break
                if iters >= maxIter:
                    conv = False
                    break
                self._commit(lst)
                iters += 1
                R.plainSteps += 1
                self._tick(passName, iters, 0, delta, 0.0, 0.0, 0, 0.0, t0, plain=True)
            if ps == 0 and not self.joint:
                R.itersCurve, R.deltaCurve = iters, delta
            else:
                R.itersSheet, R.deltaSheet = iters, delta
            self.log('[relax_newton] %s pass: %s after %d iterations (last max move %.3g x diag), %d vertices (%.1f s)'
                     % (passName, 'converged' if conv else 'NOT CONVERGED', iters, delta, n,
                        time.perf_counter() - t0))
            conv_all = conv_all and conv
        R.converged = conv_all
        self.lastMove = R.deltaSheet * self.diag
        self.done = True

    def step(self, trace=None):
        raise RuntimeError('[relax_newton] Newton runs whole passes (run()); stepping is not supported')

    def finish(self):
        R = self.R
        itC, itS, dS, conv = R.itersCurve, R.itersSheet, R.deltaSheet, R.converged
        super().finish()
        R.itersCurve, R.itersSheet, R.deltaSheet, R.converged = itC, itS, dS, conv
        # fixed-point residual and the "stopped changing" check: one plain step from the result
        lst = self.lst
        n = len(lst)
        if n:
            res = np.zeros(n)
            interior = np.zeros(n, np.int64)
            _residuals(lst, self.X, self.face, self.edge, self.bary, self.role, self.setId, self.rowOffs, self.cols,
                       self.PJ, self.eF_off, self.eF, self.T, self.dims, res, interior)
            cur = self.role[lst] == RELAX_CURVE
            ins = interior == 1
            R.residualCurve = float(res[cur & ins].max()) if (cur & ins).any() else 0.0
            R.residualSheet = float(res[~cur & ins].max()) if (~cur & ins).any() else 0.0
            R.nResidualCurve, R.nResidualSheet = int((cur & ins).sum()), int((~cur & ins).sum())
            self.nX[:] = self.X
            step_kernel(lst, self.rowOffs, self.cols, np.zeros(1), self.uni, self.X, self.face, self.edge, self.setId,
                        0.5, False, self.PJ, self.proj.scratch, self.nX, self.nFace, self.nEdge, self.nBary,
                        self.move, self.s_onb, self.s_fix, self.s_dist, self.s_grown, self.s_global, _NO_TR)
            R.jacobiStepMove = float(self.move[lst].max()) / self.diag
        self.log('[relax_newton] newton: %s | curve iters %d (last %.3g), sheet iters %d (last %.3g) | halvings %d, '
                 'stalls %d, plain steps %d, pinned %d (%d curve) | CG %d iterations in all, worst residual %.2g\n'
                 '[relax_newton]   one step x <- Pi(x + 0.5 L x) from the result: max move %.3g (x diag)\n'
                 '[relax_newton]   fixed-point residual |T^T L x| / edge: sheet %.3g (%d interior), curve %.3g '
                 '(%d interior)'
                 % ('converged' if R.converged else 'NOT CONVERGED', R.itersCurve, R.deltaCurve, R.itersSheet,
                    R.deltaSheet, R.halvings, R.lineSearchStalls, R.plainSteps, R.nPinned, R.nPinnedCurve,
                    R.cgIters, R.cgWorst, R.jacobiStepMove, R.residualSheet, R.nResidualSheet, R.residualCurve,
                    R.nResidualCurve))
        return R
