"""Implicit (backward-Euler) relaxation, Python-only.

The explicit step y = x + lambda (mean(x) - x) evaluates the neighbour mean at the OLD
positions, so a large-scale imbalance only spreads one ring per step. The implicit step
evaluates it at the NEW positions:

    y_i - x_i = tau (mean_i(y) - y_i)        for every free vertex i
    =>  (1 + tau) d_i y_i - tau sum_j w_ij y_j = d_i x_i      (d_i = sum_j w_ij)

with the fixed vertices' y = x moved to the right-hand side; then y_i is projected onto the
vertex's own structure, as in the explicit step. One linear solve per iteration damps every
spatial frequency at once (a mode with Laplacian eigenvalue mu by 1 / (1 + tau mu)), so the
large-scale drift that takes the explicit step thousands of iterations settles in a few.
tau -> infinity is the C++ solve_project (one solve of L y = 0, then project); a moderate tau
keeps the targets close to the surface, so the projection does not tear the mesh.

The system is symmetric positive definite for a symmetric graph with symmetric weights
(the two-sided graph, uniform or cotan weights) and is solved per coordinate with
conjugate gradients (numba, Jacobi preconditioner, warm start at x; no scipy). The directed
graph is not symmetric and is refused.

Everything else is the explicit relaxer's (graph, weights, free set, projection, no-new-folds
hold-back, commit, checks, report): ImplicitRelaxer only replaces how the candidate is made
(ExplicitRelaxer._propose). The projection is relax_momentum.project_targets.
"""
import numpy as np
from numba import njit, prange

from relax_explicit import ExplicitRelaxer
from relax_momentum import project_targets


@njit(parallel=True, cache=True)
def _row_weight_sums(lst, rowOffs, wts, uniformRow, d):
    for a in prange(lst.shape[0]):
        i = lst[a]
        s = 0.0
        for q in range(rowOffs[i], rowOffs[i + 1]):
            s += 1.0 if uniformRow[i] else wts[q]
        d[i] = s


@njit(parallel=True, cache=True)
def _matvec(lst, rowOffs, cols, wts, uniformRow, isFree, d, tau, V, out):
    """out_i = (1 + tau) d_i V_i - tau sum_{j free} w_ij V_j for the free vertices (3 columns)."""
    for a in prange(lst.shape[0]):
        i = lst[a]
        sx = 0.0; sy = 0.0; sz = 0.0
        for q in range(rowOffs[i], rowOffs[i + 1]):
            j = cols[q]
            if isFree[j]:
                w = 1.0 if uniformRow[i] else wts[q]
                sx += w * V[j, 0]; sy += w * V[j, 1]; sz += w * V[j, 2]
        c = (1.0 + tau) * d[i]
        out[i, 0] = c * V[i, 0] - tau * sx
        out[i, 1] = c * V[i, 1] - tau * sy
        out[i, 2] = c * V[i, 2] - tau * sz


@njit(parallel=True, cache=True)
def _rhs(lst, rowOffs, cols, wts, uniformRow, isFree, d, tau, X, b):
    """b_i = d_i x_i + tau sum_{j fixed} w_ij x_j."""
    for a in prange(lst.shape[0]):
        i = lst[a]
        sx = 0.0; sy = 0.0; sz = 0.0
        for q in range(rowOffs[i], rowOffs[i + 1]):
            j = cols[q]
            if not isFree[j]:
                w = 1.0 if uniformRow[i] else wts[q]
                sx += w * X[j, 0]; sy += w * X[j, 1]; sz += w * X[j, 2]
        b[i, 0] = d[i] * X[i, 0] + tau * sx
        b[i, 1] = d[i] * X[i, 1] + tau * sy
        b[i, 2] = d[i] * X[i, 2] + tau * sz


@njit(cache=True)
def _cg(lst, rowOffs, cols, wts, uniformRow, isFree, d, tau, X, Y, tol, maxIter, r, z, p, Ap, b):
    """Jacobi-preconditioned conjugate gradients, the 3 coordinates at once (each its own
    CG scalars); Y holds the warm start (x) and the result. Returns (iterations, largest
    relative residual)."""
    n = lst.shape[0]
    _rhs(lst, rowOffs, cols, wts, uniformRow, isFree, d, tau, X, b)
    _matvec(lst, rowOffs, cols, wts, uniformRow, isFree, d, tau, Y, Ap)
    rz = np.zeros(3)
    bb = np.zeros(3)
    for a in range(n):
        i = lst[a]
        pi = 1.0 / ((1.0 + tau) * d[i])
        for c in range(3):
            r[i, c] = b[i, c] - Ap[i, c]
            z[i, c] = r[i, c] * pi
            p[i, c] = z[i, c]
            rz[c] += r[i, c] * z[i, c]
            bb[c] += b[i, c] * b[i, c]
    for c in range(3):
        bb[c] = max(np.sqrt(bb[c]), 1e-300)
    rr = np.zeros(3)
    it = 0
    for it in range(1, maxIter + 1):
        _matvec(lst, rowOffs, cols, wts, uniformRow, isFree, d, tau, p, Ap)
        pAp = np.zeros(3)
        for a in range(n):
            i = lst[a]
            for c in range(3):
                pAp[c] += p[i, c] * Ap[i, c]
        alpha = np.zeros(3)
        for c in range(3):
            alpha[c] = rz[c] / pAp[c] if pAp[c] > 0 else 0.0
        rr[:] = 0.0
        for a in range(n):
            i = lst[a]
            for c in range(3):
                Y[i, c] += alpha[c] * p[i, c]
                r[i, c] -= alpha[c] * Ap[i, c]
                rr[c] += r[i, c] * r[i, c]
        done = True
        for c in range(3):
            if np.sqrt(rr[c]) > tol * bb[c]:
                done = False
        if done:
            break
        rzNew = np.zeros(3)
        for a in range(n):
            i = lst[a]
            pi = 1.0 / ((1.0 + tau) * d[i])
            for c in range(3):
                z[i, c] = r[i, c] * pi
                rzNew[c] += r[i, c] * z[i, c]
        beta = np.zeros(3)
        for c in range(3):
            beta[c] = rzNew[c] / rz[c] if rz[c] > 0 else 0.0
            rz[c] = rzNew[c]
        for a in range(n):
            i = lst[a]
            for c in range(3):
                p[i, c] = z[i, c] + beta[c] * p[i, c]
    worst = 0.0
    for c in range(3):
        worst = max(worst, np.sqrt(rr[c]) / bb[c])
    return it, worst


def solve_implicit(lst, rowOffs, cols, wts, uniformRow, isFree, d, tau, X, Y, tol, maxIter, work):
    """The implicit step's linear system, solved in place in Y (warm start: x)."""
    r, z, p, Ap, b = work
    return _cg(lst, rowOffs, cols, wts, uniformRow, isFree, d, tau, X, Y, float(tol), int(maxIter), r, z, p, Ap, b)


class ImplicitRelaxer(ExplicitRelaxer):
    """ExplicitRelaxer whose candidate is the implicit step's solution (see the module doc).
    Checkpoints (trace): only the step target Y is recorded."""

    def __init__(self, *args, tau=10.0, cg_tol=1e-8, cg_max_iter=1000, symmetric=True, **kw):
        super().__init__(*args, **kw)
        if not symmetric:
            raise RuntimeError('[relax_implicit] the implicit step needs a symmetric graph (not the directed one)')
        self.tau, self.cgTol, self.cgMaxIter = float(tau), float(cg_tol), int(cg_max_iter)
        Vs = self.Vs
        self.d = np.zeros(Vs)
        _row_weight_sums(self.lst, self.rowOffs, self.wts, self.uniformRow, self.d)
        self.isFreeB = self.isFree.astype(np.bool_)
        self.Ytarget = self.X.copy()
        self.work = tuple(np.zeros((Vs, 3)) for _ in range(5))
        self.cgIters = 0
        self.cgWorst = 0.0
        self.log('[relax_implicit] backward-Euler step, tau %.3g; CG tol %.1e, max %d iterations'
                 % (self.tau, self.cgTol, self.cgMaxIter))

    def _propose(self, trace):
        X, Y = self.X, self.Ytarget
        Y[:] = X                                     # warm start; fixed vertices stay at x
        it, res = solve_implicit(self.lst, self.rowOffs, self.cols, self.wts, self.uniformRow, self.isFreeB,
                                 self.d, self.tau, X, Y, self.cgTol, self.cgMaxIter, self.work)
        self.cgIters += it
        self.cgWorst = max(self.cgWorst, res)
        if res > 1e3 * self.cgTol:
            self.log('[relax_implicit] WARNING: CG stopped at relative residual %.3g after %d iterations' % (res, it))
        if trace is not None:
            trace.Y[self.lst] = Y[self.lst]
        project_targets(self.lst, Y, X, self.face, self.edge, self.setId, self.opt.localProjection, self.PJ,
                        self.proj.scratch, self.nX, self.nFace, self.nEdge, self.nBary, self.move, self.s_onb,
                        self.s_fix, self.s_dist, self.s_grown, self.s_global)

    def finish(self):
        self.log('[relax_implicit] %d iterations, %d CG iterations in all (%.1f per step), worst relative residual %.2g'
                 % (self.it, self.cgIters, self.cgIters / max(1, self.it), self.cgWorst))
        return super().finish()
