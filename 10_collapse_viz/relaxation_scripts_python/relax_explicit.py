"""Port of coarse_subdiv_relax_explicit.h / .cpp: explicit (small-step)
Laplacian relaxation of the subdivided coarse mesh on the fine MAT.

Every iteration moves all free vertices together (Jacobi: all candidates from
the same X) by a step towards the (weighted) mean of their graph neighbours and
projects them back onto their own structure:

    x_i <- Pi_i( x_i + lambda * (mean_j x_j - x_i) ),  mean_j = sum w_ij x_j / sum w_ij

Junctions are fixed. Optional no-new-folds rule: the moved vertices of a
triangle that is unfolded at X and folded at the candidate keep their
position, repeated until no such triangle is left. Stops when no vertex moves
more than tol * bbox diagonal, or after maxIter iterations.
"""
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import log_util
from numba import njit, prange

from obj_io import write_obj
from projector import NO_TRACE, Projector, interp3, project, project_local
from struct_ids import RELAX_CURVE, RELAX_JUNCTION, RELAX_SHEET, role_of, split_palette
from subdiv_mesh import SUBDIV_CARRIER_EDGE


@dataclass
class ExplicitRelaxOptions:
    lam: float = 0.5              # step: fraction of the way to the neighbour mean
    maxIter: int = 20000
    tol: float = 1e-7             # stop when the largest move <= tol * bbox diagonal
    localProjection: bool = True  # project_local (slide) instead of the global closest point
    holdFixed: Optional[np.ndarray] = None  # per vertex: nonzero = held at its seed
    foldRef: Optional[np.ndarray] = None    # per face of M.F: reference normal; folded = n . ref <= 0
    noNewFolds: bool = False
    weights: Optional[np.ndarray] = None    # per graph entry (aligned with G.cols), >= 0; None: uniform
    logEvery: int = 100
    snapshotIters: List[int] = field(default_factory=lambda: [1, 10, 100, 1000, 10000])
    snapshotPrefix: str = ''      # <prefix>it<N>.obj, none if empty


@dataclass
class RelaxReport:
    itersSheet: int = 0
    deltaSheet: float = 0.0
    converged: bool = False
    nFree: int = 0
    nFixed: int = 0
    nPinned: int = 0
    foldReverts: int = 0
    foldedSeed: int = -1
    foldedResult: int = -1
    projMoves: int = 0
    projJumps: int = 0
    projJumpMax: float = 0.0
    localCalls: int = 0
    localGrown: int = 0
    localGlobal: int = 0
    seedOffStructure: int = 0
    maxMove: float = 0.0
    meanMove: float = 0.0
    fixedMoved: int = 0
    posMismatch: int = 0
    badBary: int = 0
    offStructure: int = 0


class ProjStats:
    def __init__(self, calls=0, onBorder=0, baryFix=0, distSum=0.0, distMax=0.0):
        self.calls, self.onBorder, self.baryFix, self.distSum, self.distMax = calls, onBorder, baryFix, distSum, distMax

    def copy(self):
        return ProjStats(self.calls, self.onBorder, self.baryFix, self.distSum, self.distMax)

    def __sub__(self, o):  # distMax is not differenced
        return ProjStats(self.calls - o.calls, self.onBorder - o.onBorder, self.baryFix - o.baryFix,
                         self.distSum - o.distSum, self.distMax)


# ------------------------------------------------------------------ kernels

@njit(cache=True, inline='always')
def _tri_normal(P, F, f):
    a0 = P[F[f, 0], 0]; a1 = P[F[f, 0], 1]; a2 = P[F[f, 0], 2]
    ux = P[F[f, 1], 0] - a0; uy = P[F[f, 1], 1] - a1; uz = P[F[f, 1], 2] - a2
    vx = P[F[f, 2], 0] - a0; vy = P[F[f, 2], 1] - a1; vz = P[F[f, 2], 2] - a2
    return uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx


@njit(cache=True)
def _dot_ref(nx, ny, nz, ref, f):
    return (nx * ref[f, 0] + ny * ref[f, 1]) + nz * ref[f, 2]


@njit(cache=True)
def count_folded(P, F, ref):
    n = 0
    for f in range(F.shape[0]):
        nx, ny, nz = _tri_normal(P, F, f)
        if _dot_ref(nx, ny, nz, ref, f) <= 0:
            n += 1
    return n


@njit(cache=True)
def count_degenerate(P, F, diag):
    n = 0
    for f in range(F.shape[0]):
        nx, ny, nz = _tri_normal(P, F, f)
        if 0.5 * np.sqrt((nx * nx + ny * ny) + nz * nz) <= 1e-14 * diag * diag:
            n += 1
    return n


@njit(cache=True)
def energy(P, rowOffs, cols, wts, uniformRow):
    """1/2 sum over graph edges (each once, j > i) of w_ij |x_i - x_j|^2."""
    e = 0.0
    for i in range(rowOffs.shape[0] - 1):
        for q in range(rowOffs[i], rowOffs[i + 1]):
            j = cols[q]
            if j > i:
                w = 1.0 if uniformRow[i] else wts[q]
                dx = P[i, 0] - P[j, 0]; dy = P[i, 1] - P[j, 1]; dz = P[i, 2] - P[j, 2]
                e += 0.5 * w * ((dx * dx + dy * dy) + dz * dz)
    return e


@njit(parallel=True, cache=True)
def step_kernel(lst, rowOffs, cols, wts, uniformRow, X, face, edge, setId, lam, local, PJ, scratch,
                nX, nFace, nEdge, nBary, move, s_onb, s_fix, s_dist, s_grown, s_global, TR):
    """Candidate step for every free vertex, all from the same X (Jacobi).

    TR = checkpoint buffers (trace, Y, regCnt, reg, winTree, winLeaf, watch, watchVis, watchN);
    with trace False nothing is written to them. Per vertex: Y = the step target
    x + lam (mean - x); regCnt / reg = the final local search region (regCnt = -1 when
    the projection fell back to / was the global one); winTree / winLeaf = the BVH tree and
    leaf that gave the global result. For the one vertex `watch`, watchVis gets every BVH
    node its query entered (count in watchN[0])."""
    trace, Y, regCnt, reg, winTree, winLeaf, watch, watchVis, watchN = TR
    noVis = watchVis[:0]
    n = lst.shape[0]
    nchunks = 256
    for ch in prange(nchunks):
        a0 = (n * ch) // nchunks
        a1 = (n * (ch + 1)) // nchunks
        region = np.empty(scratch, dtype=np.int64)
        for a in range(a0, a1):
            i = lst[a]
            mx = 0.0; my = 0.0; mz = 0.0
            ws = 0.0
            for q in range(rowOffs[i], rowOffs[i + 1]):
                wq = 1.0 if uniformRow[i] else wts[q]
                j = cols[q]
                mx = mx + wq * X[j, 0]
                my = my + wq * X[j, 1]
                mz = mz + wq * X[j, 2]
                ws += wq
            mx = mx / ws; my = my / ws; mz = mz / ws
            x0 = X[i, 0]; x1 = X[i, 1]; x2 = X[i, 2]
            y0 = x0 + lam * (mx - x0)
            y1 = x1 + lam * (my - x1)
            y2 = x2 + lam * (mz - x2)
            vis = watchVis[:] if (trace and i == watch) else noVis
            if local:
                r = project_local(PJ, setId[i], y0, y1, y2, face[i], edge[i], region, vis)
            else:
                r = project(PJ, setId[i], y0, y1, y2, face[i], edge[i], vis)
            dx = r[5] - x0; dy = r[6] - x1; dz = r[7] - x2
            move[i] = np.sqrt((dx * dx + dy * dy) + dz * dz)
            nX[i, 0] = r[5]; nX[i, 1] = r[6]; nX[i, 2] = r[7]
            nFace[i] = r[0]; nEdge[i] = r[4]
            nBary[i, 0] = r[1]; nBary[i, 1] = r[2]; nBary[i, 2] = r[3]
            s_onb[a] = r[8]; s_fix[a] = r[9]; s_dist[a] = r[10]; s_grown[a] = r[11]; s_global[a] = r[12]
            if trace:
                Y[i, 0] = y0; Y[i, 1] = y1; Y[i, 2] = y2
                winTree[i] = r[13]
                winLeaf[i] = r[14]
                if local and r[12] == 0:
                    c = r[15]
                    regCnt[i] = c
                    for t in range(min(c, reg.shape[1])):
                        reg[i, t] = region[t]
                else:
                    regCnt[i] = -1
                if i == watch:
                    watchN[0] = r[15] if not (local and r[12] == 0) else 0


@njit(parallel=True, cache=True)
def _new_folds(X, nX, F, ref, newFold):
    for f in prange(F.shape[0]):
        ax, ay, az = _tri_normal(X, F, f)
        bx, by, bz = _tri_normal(nX, F, f)
        newFold[f] = _dot_ref(ax, ay, az, ref, f) > 0 and _dot_ref(bx, by, bz, ref, f) <= 0


@njit(cache=True)
def _hold_back(X, nX, F, newFold, isFree, face, edge, bary, nFace, nEdge, nBary, move):
    held = 0
    for f in range(F.shape[0]):
        if not newFold[f]:
            continue
        for c in range(3):
            v = F[f, c]
            if not isFree[v] or (nX[v, 0] == X[v, 0] and nX[v, 1] == X[v, 1] and nX[v, 2] == X[v, 2]):
                continue
            nX[v, 0] = X[v, 0]; nX[v, 1] = X[v, 1]; nX[v, 2] = X[v, 2]
            nFace[v] = face[v]; nEdge[v] = edge[v]
            nBary[v, 0] = bary[v, 0]; nBary[v, 1] = bary[v, 1]; nBary[v, 2] = bary[v, 2]
            move[v] = 0.0
            held += 1
    return held


@njit(cache=True)
def _commit(lst, move, FO, X, face, edge, bary, nX, nFace, nEdge, nBary, diag):
    m = 0.0
    moves = 0
    jumps = 0
    jumpMax = 0.0
    for a in range(lst.shape[0]):
        i = lst[a]
        if m < move[i]:
            m = move[i]
        if move[i] > 0:
            moves += 1
            f = face[i]; g = nFace[i]
            share = False
            for p in range(3):
                for q in range(3):
                    if FO[f, p] == FO[g, q]:
                        share = True
            if not share:
                jumps += 1
                r = move[i] / diag
                if jumpMax < r:
                    jumpMax = r
        X[i, 0] = nX[i, 0]; X[i, 1] = nX[i, 1]; X[i, 2] = nX[i, 2]
        face[i] = nFace[i]; edge[i] = nEdge[i]
        bary[i, 0] = nBary[i, 0]; bary[i, 1] = nBary[i, 1]; bary[i, 2] = nBary[i, 2]
    return m, moves, jumps, jumpMax


# ------------------------------------------------------------------ main routine

REG_CAP = 64      # checkpoint: local search region entries kept per vertex
WATCH_CAP = 4096  # checkpoint: BVH nodes kept for the watched vertex


def _bbox_diag(VO):
    d = VO[:, :3].max(axis=0) - VO[:, :3].min(axis=0)
    return float(np.sqrt((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]))


class StepTrace:
    """Checkpoint of one iteration, filled by the relaxation itself (step_kernel,
    the fold hold-back and the commit):
      X0        positions at the start of the step
      Y         step targets x + lambda (mean - x) (free vertices; fixed ones = X0)
      P         projections Pi(Y), before the no-new-folds hold-back
      Pface, Pedge, Pbary   their fine face / edge / barycentrics
      X1        committed positions (= P, or X0 where held back)
      held      1 where the no-new-folds rule held the vertex back
      regCnt / reg   final local search region (-1: global projection)
      winTree / winLeaf   BVH tree / tree-local leaf that gave the result (-1: none)
      watch, watchVis / watchN   the watched vertex and the BVH nodes its query entered
      it, maxMove, local   iteration number (1-based), largest move, projection mode."""

    def __init__(self, Vs):
        self.Vs = Vs
        self.X0 = np.zeros((Vs, 3)); self.Y = np.zeros((Vs, 3)); self.P = np.zeros((Vs, 3))
        self.Pface = np.zeros(Vs, np.int64); self.Pedge = np.zeros(Vs, np.int64); self.Pbary = np.zeros((Vs, 3))
        self.X1 = np.zeros((Vs, 3)); self.held = np.zeros(Vs, np.uint8)
        self.regCnt = np.full(Vs, -1, np.int64); self.reg = np.zeros((Vs, REG_CAP), np.int64)
        self.winTree = np.full(Vs, -1, np.int64); self.winLeaf = np.full(Vs, -1, np.int64)
        self.watch = -1
        self.watchVis = np.zeros(WATCH_CAP, np.int64); self.watchN = np.zeros(1, np.int64)
        self.it = 0; self.maxMove = 0.0; self.local = True

    def buffers(self):
        return (True, self.Y, self.regCnt, self.reg, self.winTree, self.winLeaf, int(self.watch),
                self.watchVis, self.watchN)


_NO_TR = (False, np.zeros((1, 3)), np.zeros(1, np.int64), np.zeros((1, 1), np.int64), np.zeros(1, np.int64),
          np.zeros(1, np.int64), -1, np.zeros(1, np.int64), np.zeros(1, np.int64))


class ExplicitRelaxer:
    """subdiv_relax_explicit as a stepper: the constructor is its set-up, step() one
    iteration, run() the iteration loop, finish() the checks and the report.

    state = (X, face, edge, bary) to start from instead of M (the viewer continues a
    run under a new configuration this way; edge = current curve edge per vertex).
    proj  = a Projector built from the same inputs, reused instead of rebuilt.
    it0   = iterations done before (logged iteration numbers continue from it)."""

    def __init__(self, M, VO, FO, ms, pal, setId, G, opt, log=None, state=None, proj=None, it0=0):
        self.log = log or log_util.log
        log = self.log
        self.tStart = time.perf_counter()
        R = self.R = RelaxReport()
        self.M, self.G, self.opt, self.ms = M, G, opt, ms
        Vs, nF = M.V.shape[0], M.F.shape[0]
        self.Vs, self.nF = Vs, nF

        def fail(msg):
            raise RuntimeError('[relax_explicit] ' + msg)
        if G.role.shape[0] != Vs:
            fail('graph / mesh size mismatch')
        hold = opt.holdFixed if opt.holdFixed is not None and len(opt.holdFixed) else None
        if hold is not None and len(hold) != Vs:
            fail('holdFixed / mesh size mismatch')
        haveRef = opt.foldRef is not None and opt.foldRef.size > 0
        if haveRef and opt.foldRef.shape != (nF, 3):
            fail('foldRef / face count mismatch')
        if opt.noNewFolds and not haveRef:
            fail('noNewFolds needs foldRef')
        if not (opt.lam > 0 and opt.lam <= 1):
            fail('lambda must be in (0, 1]')
        weighted = opt.weights is not None and len(opt.weights) > 0
        if weighted and len(opt.weights) != len(G.cols):
            fail('weights / graph size mismatch')
        self.haveRef = haveRef
        self.rowOffs, self.cols = G.rowOffs, G.cols
        self.wts = np.asarray(opt.weights, dtype=np.float64) if weighted else np.zeros(1)
        self.uniformRow = np.ones(Vs, dtype=np.uint8)
        zeroRows = 0
        if weighted:
            # row sums, sequential per row
            s = _row_sums(self.rowOffs, self.wts)
            self.uniformRow = (~(s > 0)).astype(np.uint8)
            zeroRows = int(((self.uniformRow == 1) & (np.diff(self.rowOffs) > 0)).sum())
        self.diag = diag = _bbox_diag(VO)
        self.tolAbs = opt.tol * diag

        if proj is None:
            S = split_palette(pal, ms)
            setRole = [role_of(pal.typeMask[k]) if ms is not None else RELAX_SHEET for k in range(pal.size())]
            proj = Projector(VO, FO, M.origEdges, ms, pal, S, setRole)
        self.proj = proj
        PJ = self.PJ = proj.pj
        self.FOi = proj.FO
        self.Fi = np.ascontiguousarray(M.F, dtype=np.int64)
        self.setId = setId = np.asarray(setId, dtype=np.int64)

        if state is None:
            X = np.ascontiguousarray(M.V, dtype=np.float64).copy()
            bary = np.ascontiguousarray(M.fineBary, dtype=np.float64).copy()
            face = np.asarray(M.fineFace, dtype=np.int64).copy()
            edge = np.full(Vs, -1, dtype=np.int64)
            ce = (G.role == RELAX_CURVE) & (M.carrierType == SUBDIV_CARRIER_EDGE)
            edge[ce] = M.carrierIndex[ce]
        else:
            X, face, edge, bary = (np.ascontiguousarray(a).copy() for a in state)
        self.Vseed = X.copy()
        self.X, self.face, self.edge, self.bary = X, face, edge, bary

        # free vertices: not a junction, located, pulled by someone, has a target, not held;
        # and the seed already on its own structure
        hasTarget = np.array([len(t) > 0 for t in proj.targets], dtype=bool)
        isFree = ((G.role != RELAX_JUNCTION) & (face >= 0) & (np.diff(self.rowOffs) > 0)
                  & hasTarget[setId]).astype(np.uint8)
        if hold is not None:
            isFree &= (np.asarray(hold) == 0).astype(np.uint8)
        for i in np.nonzero(isFree)[0].tolist():
            r = project(PJ, setId[i], X[i, 0], X[i, 1], X[i, 2], face[i], edge[i], NO_TRACE)
            dx, dy, dz = r[5] - X[i, 0], r[6] - X[i, 1], r[7] - X[i, 2]
            if np.sqrt((dx * dx + dy * dy) + dz * dz) > 1e-12 * diag:
                isFree[i] = 0
                R.seedOffStructure += 1
        # a connected group of free vertices that nothing fixed pulls on: pin its lowest-index vertex
        R.nPinned = _pin_unanchored(isFree, self.rowOffs, self.cols)
        self.isFree = isFree
        self.lst = np.nonzero(isFree)[0].astype(np.int64)
        R.nFree = len(self.lst)
        R.nFixed = Vs - R.nFree

        self.ref = np.ascontiguousarray(opt.foldRef, dtype=np.float64) if haveRef else np.zeros((1, 3))
        if haveRef:
            R.foldedSeed = self.count_folded(X)
        self.total = ProjStats()
        self.lastLog = ProjStats()
        log('[relax_explicit] %d free, %d fixed (%d pinned, %d seeds off structure) | lambda %.3g, '
            'tol %.3g x diag, max %d iterations | %s projection | no new folds %s | %s weights%s | '
            'energy %.6g, folded %d'
            % (R.nFree, R.nFixed, R.nPinned, R.seedOffStructure, opt.lam, opt.tol, opt.maxIter,
               'local' if opt.localProjection else 'global', 'on' if opt.noNewFolds else 'off',
               'given' if weighted else 'uniform',
               (' (%d rows with zero weight sum -> uniform)' % zeroRows) if weighted else '',
               self.energy(X), R.foldedSeed))

        self.nX = X.copy(); self.nBary = bary.copy(); self.nFace = face.copy(); self.nEdge = edge.copy()
        self.move = np.zeros(Vs)
        self.newFold = np.zeros(nF, dtype=np.bool_)
        n = len(self.lst)
        self.s_onb = np.zeros(n, np.int64); self.s_fix = np.zeros(n, np.int64); self.s_dist = np.zeros(n)
        self.s_grown = np.zeros(n, np.int64); self.s_global = np.zeros(n, np.int64)
        self.it0 = it0
        self.it = 0
        self.lastMove = 0.0
        self.done = False

    # metrics
    def count_folded(self, P):
        return count_folded(P, self.Fi, self.ref) if self.haveRef else 0

    def energy(self, P):
        return energy(P, self.rowOffs, self.cols, self.wts, self.uniformRow)

    def step(self, trace=None):
        """One iteration. trace: a StepTrace to fill (checkpoints), or None.
        Returns the largest move of the step."""
        opt, R, X = self.opt, self.R, self.X
        face, edge, bary = self.face, self.edge, self.bary
        nX, nFace, nEdge, nBary, move = self.nX, self.nFace, self.nEdge, self.nBary, self.move
        lst = self.lst
        n = len(lst)
        if trace is not None:
            trace.X0[:] = X
            trace.Y[:] = X
            trace.regCnt[:] = -1
            trace.winTree[:] = -1
            trace.winLeaf[:] = -1
            trace.watchN[0] = 0
            trace.local = bool(opt.localProjection)
        step_kernel(lst, self.rowOffs, self.cols, self.wts, self.uniformRow, X, face, edge, self.setId,
                    opt.lam, opt.localProjection, self.PJ, self.proj.scratch, nX, nFace, nEdge, nBary, move,
                    self.s_onb, self.s_fix, self.s_dist, self.s_grown, self.s_global,
                    trace.buffers() if trace is not None else _NO_TR)
        if trace is not None:
            trace.P[:] = X
            trace.P[lst] = nX[lst]
            trace.Pface[:] = face
            trace.Pface[lst] = nFace[lst]
            trace.Pedge[:] = edge
            trace.Pedge[lst] = nEdge[lst]
            trace.Pbary[:] = bary
            trace.Pbary[lst] = nBary[lst]
        total = self.total
        total.calls += n
        total.onBorder += int(self.s_onb.sum())
        total.baryFix += int(self.s_fix.sum())
        total.distSum += float(self.s_dist.sum())
        if n:
            total.distMax = max(total.distMax, float(self.s_dist.max()))
        if opt.localProjection:
            R.localCalls += n
            R.localGrown += int(self.s_grown.sum())
            R.localGlobal += int(self.s_global.sum())

        if opt.noNewFolds:
            while True:
                _new_folds(X, nX, self.Fi, self.ref, self.newFold)
                held = _hold_back(X, nX, self.Fi, self.newFold, self.isFree, face, edge, bary, nFace, nEdge,
                                  nBary, move)
                if not held:
                    break
                R.foldReverts += held

        if trace is not None:
            # held back: the candidate was reset to X although it was projected elsewhere
            trace.held[:] = 0
            trace.held[lst] = np.any(nX[lst] != trace.P[lst], axis=1)
        m, moves, jumps, jumpMax = _commit(lst, move, self.FOi, X, face, edge, bary, nX, nFace, nEdge,
                                           nBary, self.diag)
        R.projMoves += moves
        R.projJumps += jumps
        R.projJumpMax = max(R.projJumpMax, jumpMax)
        self.lastMove = m
        self.it += 1
        it = self.it0 + self.it
        if trace is not None:
            trace.X1[:] = X
            trace.it = it
            trace.maxMove = m
        self.snapshot(it)
        if it % opt.logEvery == 0 or m <= self.tolAbs:
            now = total.copy()
            d = now - self.lastLog
            self.lastLog = now
            self.log('[relax_explicit] iter %d: max move %.3g (x diag), energy %.6g, folded %d, degenerate %d | '
                     'projection: %.2f%% on an edge/vertex, off-surface distance mean %.3g max %.3g (x diag) (%.1f s)'
                     % (it, m / self.diag, self.energy(X), self.count_folded(X),
                        count_degenerate(X, self.Fi, self.diag),
                        100.0 * d.onBorder / d.calls if d.calls else 0.0,
                        d.distSum / d.calls / self.diag if d.calls else 0.0,
                        now.distMax / self.diag, time.perf_counter() - self.tStart))
        if m <= self.tolAbs:
            R.converged = True
            self.done = True
        elif self.it >= opt.maxIter:
            self.done = True
        return m

    def snapshot(self, it):
        opt = self.opt
        if not opt.snapshotPrefix or it not in opt.snapshotIters:
            return
        path = opt.snapshotPrefix + 'it%d.obj' % it
        if not write_obj(path, self.X, self.M.F):
            self.log('[relax_explicit] writeOBJ failed: %s' % path)

    def run(self):
        while not self.done and self.it < self.opt.maxIter:
            self.step()

    def finish(self):
        """Writes the state back to M (V, fineFace, fineBary), runs the checks and
        logs the report. Returns the RelaxReport."""
        R, M, X = self.R, self.M, self.X
        face, edge, bary, isFree, setId, G, proj = (self.face, self.edge, self.bary, self.isFree, self.setId,
                                                    self.G, self.proj)
        Vs, diag, FOi = self.Vs, self.diag, self.FOi
        R.itersSheet = self.it
        R.deltaSheet = self.lastMove / diag
        M.V = X
        M.fineFace = face
        M.fineBary = bary
        if self.haveRef:
            R.foldedResult = self.count_folded(X)
        ps = self.total
        Vseed = self.Vseed

        # ---- checks
        mv = np.sqrt(((X[:, 0] - Vseed[:, 0]) ** 2 + (X[:, 1] - Vseed[:, 1]) ** 2) + (X[:, 2] - Vseed[:, 2]) ** 2)
        R.maxMove = float(mv.max()) if Vs else 0.0
        sumMove = seq_sum(mv)
        fixed = isFree == 0
        R.fixedMoved = int(np.any(X[fixed] != Vseed[fixed], axis=1).sum())
        fr = np.nonzero(isFree)[0]
        b = bary[fr]
        R.badBary = int(((b.min(axis=1) < 0) | (np.abs(((b[:, 0] + b[:, 1]) + b[:, 2]) - 1.0) > 1e-12)).sum())
        P = np.array([interp3(proj.VO, FOi, face[i], bary[i, 0], bary[i, 1], bary[i, 2]) for i in fr.tolist()]
                     ).reshape(-1, 3)
        R.posMismatch = int(np.any(P != X[fr], axis=1).sum())
        off = 0
        E = proj.E
        for i in fr.tolist():
            k = setId[i]
            if G.role[i] == RELAX_CURVE:
                e = edge[i]
                ok = e >= 0 and proj.edge_in(k, e)
                for c in range(3):
                    if not ok:
                        break
                    if bary[i, c] != 0.0 and FOi[face[i], c] != E[e, 0] and FOi[face[i], c] != E[e, 1]:
                        ok = False
                if not ok:
                    off += 1
            elif not proj.face_in(k, face[i]):
                off += 1
        R.offStructure = off
        R.maxMove /= diag
        R.meanMove = sumMove / Vs / diag if Vs else 0.0

        self.log('[relax_explicit] %s after %d iterations (last max move %.3g x diag) | move max %.3g mean %.3g '
                 '(x diag) | folded %d -> %d, moves held back %d (%.1f s)\n'
                 '[relax_explicit]   projection: %d vertex moves, %d jumps, largest jump %.3g (x diag); local calls '
                 '%d, grew past the first ring %d, fell back to global %d\n'
                 '[relax_explicit]   projection over all iterations: %d calls, %.2f%% on an edge/vertex of their '
                 'triangle (or an end of their edge), off-surface distance mean %.3g max %.3g (x diag), barycentric '
                 'fix-ups %d\n'
                 '[relax_explicit]   checks: seed off structure %d, fixed moved %d, pos != interp %d, bad bary %d, '
                 'off own structure %d'
                 % ('converged' if R.converged else 'NOT CONVERGED', R.itersSheet, R.deltaSheet, R.maxMove,
                    R.meanMove, R.foldedSeed, R.foldedResult, R.foldReverts, time.perf_counter() - self.tStart,
                    R.projMoves, R.projJumps, R.projJumpMax, R.localCalls, R.localGrown, R.localGlobal,
                    ps.calls, 100.0 * ps.onBorder / ps.calls if ps.calls else 0.0,
                    ps.distSum / ps.calls / diag if ps.calls else 0.0, ps.distMax / diag, ps.baryFix,
                    R.seedOffStructure, R.fixedMoved, R.posMismatch, R.badBary, R.offStructure))
        return R


def subdiv_relax_explicit(M, VO, FO, ms, pal, setId, G, opt, log=None):
    """Moves M.V and rewrites M.fineFace / M.fineBary. G must carry the
    structure roles. Returns a RelaxReport."""
    r = ExplicitRelaxer(M, VO, FO, ms, pal, setId, G, opt, log)
    r.run()
    return r.finish()


@njit(cache=True)
def seq_sum(a):
    """Left-to-right sum (numpy's sum is pairwise)."""
    s = 0.0
    for x in a:
        s += x
    return s


@njit(cache=True)
def _row_sums(rowOffs, w):
    s = np.zeros(rowOffs.shape[0] - 1)
    for i in range(rowOffs.shape[0] - 1):
        t = 0.0
        for q in range(rowOffs[i], rowOffs[i + 1]):
            t += w[q]
        s[i] = t
    return s


@njit(cache=True)
def _pin_unanchored(isFree, rowOffs, cols):
    Vs = isFree.shape[0]
    seen = np.zeros(Vs, dtype=np.uint8)
    st = np.empty(Vs, dtype=np.int64)
    pinned = 0
    for s0 in range(Vs):
        if not isFree[s0] or seen[s0]:
            continue
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
                if not isFree[j]:
                    anchored = True
                    continue
                if not seen[j]:
                    seen[j] = 1
                    st[sp] = j
                    sp += 1
        if not anchored:
            isFree[lowest] = 0
            pinned += 1
    return pinned
