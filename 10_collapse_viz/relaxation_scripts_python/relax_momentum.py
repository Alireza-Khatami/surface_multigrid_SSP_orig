"""Explicit relaxation with momentum (heavy ball), Python-only.

The explicit step (relax_explicit.py) moves every free vertex a fraction lambda towards the
weighted mean of its graph neighbours and projects it onto its own structure:

    y_i = x_i + lambda (mean_i - x_i),          x_i <- Pi_i(y_i)

Fine wrinkles go fast, but a large-scale imbalance spreads one ring per step, so the number
of steps grows with the square of the mesh's width in vertices. Momentum adds the last move:

    y_i = x_i + lambda (mean_i - x_i) + beta (x_i - x_i^prev)

which for this kind of averaging cuts the steps needed for the large-scale part from about
width^2 to about width. Everything else is the explicit relaxer's (graph, weights, free set,
projection, no-new-folds hold-back, commit, checks, report): MomentumRelaxer only replaces how
the candidate is made (ExplicitRelaxer._propose).

The velocity is x - x^prev of the last committed step, so a vertex held back by the
no-new-folds rule (x = x^prev) starts again from rest. Adaptive restart (on by default):
when a step raises the energy, all velocities are set to 0 for the next step.

`project_targets` (project given targets y onto each vertex's structure) is also used by
relax_implicit.py.
"""
import numpy as np
from numba import njit, prange

from projector import project, project_local
from relax_explicit import ExplicitRelaxer


@njit(parallel=True, cache=True)
def project_targets(lst, Y, X, face, edge, setId, local, PJ, scratch, nX, nFace, nEdge, nBary, move,
                    s_onb, s_fix, s_dist, s_grown, s_global):
    """Every free vertex i (lst) gets the candidate Pi_i(Y[i]): its own structure's closest
    point (local: slid from its current face / edge), with the same outputs as the explicit
    step kernel (nX, nFace, nEdge, nBary, move = |candidate - X[i]|, projection stats)."""
    n = lst.shape[0]
    nchunks = 256
    noVis = np.zeros(0, dtype=np.int64)
    for ch in prange(nchunks):
        a0 = (n * ch) // nchunks
        a1 = (n * (ch + 1)) // nchunks
        region = np.empty(scratch, dtype=np.int64)
        for a in range(a0, a1):
            i = lst[a]
            if local:
                r = project_local(PJ, setId[i], Y[i, 0], Y[i, 1], Y[i, 2], face[i], edge[i], region, noVis)
            else:
                r = project(PJ, setId[i], Y[i, 0], Y[i, 1], Y[i, 2], face[i], edge[i], noVis)
            dx = r[5] - X[i, 0]; dy = r[6] - X[i, 1]; dz = r[7] - X[i, 2]
            move[i] = np.sqrt((dx * dx + dy * dy) + dz * dz)
            nX[i, 0] = r[5]; nX[i, 1] = r[6]; nX[i, 2] = r[7]
            nFace[i] = r[0]; nEdge[i] = r[4]
            nBary[i, 0] = r[1]; nBary[i, 1] = r[2]; nBary[i, 2] = r[3]
            s_onb[a] = r[8]; s_fix[a] = r[9]; s_dist[a] = r[10]; s_grown[a] = r[11]; s_global[a] = r[12]


@njit(parallel=True, cache=True)
def _momentum_targets(lst, rowOffs, cols, wts, uniformRow, X, Xprev, lam, beta, Y):
    """Y[i] = x_i + lam (mean_i - x_i) + beta (x_i - x_i^prev) for the free vertices."""
    for a in prange(lst.shape[0]):
        i = lst[a]
        mx = 0.0; my = 0.0; mz = 0.0
        ws = 0.0
        for q in range(rowOffs[i], rowOffs[i + 1]):
            wq = 1.0 if uniformRow[i] else wts[q]
            j = cols[q]
            mx += wq * X[j, 0]
            my += wq * X[j, 1]
            mz += wq * X[j, 2]
            ws += wq
        for c, m in ((0, mx), (1, my), (2, mz)):
            x = X[i, c]
            Y[i, c] = x + lam * (m / ws - x) + beta * (x - Xprev[i, c])


class MomentumRelaxer(ExplicitRelaxer):
    """ExplicitRelaxer with heavy-ball momentum (see the module doc). Checkpoints (trace):
    only the step target Y is recorded."""

    def __init__(self, *args, beta=0.9, restart=True, **kw):
        super().__init__(*args, **kw)
        self.beta = float(beta)
        self.restart = bool(restart)
        self.Xprev = self.X.copy()
        self.Ytarget = self.X.copy()
        self.lastEnergy = None
        self.restarts = 0
        self.log('[relax_momentum] momentum beta %.3g, adaptive restart %s' % (self.beta, 'on' if self.restart else 'off'))

    def _propose(self, trace):
        X = self.X
        if self.restart:
            e = self.energy(X)
            if self.lastEnergy is not None and e > self.lastEnergy:
                self.Xprev[:] = X           # energy went up: start again from rest
                self.restarts += 1
            self.lastEnergy = e
        _momentum_targets(self.lst, self.rowOffs, self.cols, self.wts, self.uniformRow, X, self.Xprev,
                          self.opt.lam, self.beta, self.Ytarget)
        self.Xprev[:] = X                   # the velocity of the next step is this step's move
        if trace is not None:
            trace.Y[self.lst] = self.Ytarget[self.lst]
        project_targets(self.lst, self.Ytarget, X, self.face, self.edge, self.setId, self.opt.localProjection,
                        self.PJ, self.proj.scratch, self.nX, self.nFace, self.nEdge, self.nBary, self.move,
                        self.s_onb, self.s_fix, self.s_dist, self.s_grown, self.s_global)

    def finish(self):
        self.log('[relax_momentum] %d adaptive restarts in %d iterations' % (self.restarts, self.it))
        return super().finish()
