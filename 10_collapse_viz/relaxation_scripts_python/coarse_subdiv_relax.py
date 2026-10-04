"""Port of coarse_subdiv_relax_export (coarse_subdiv_relax.cpp), explicit method:
structure-aware relaxation of the subdivided coarse mesh on the fine MAT.

  1. struct IDs of the subdivided vertices, from the coarse carriers
  2. the same vertices / faces located on the fine mesh (seeds = c2f query)
  3. seeds snapped onto their own fine structure (junctions onto the nearest
     fine vertex of their junction)
  4. relaxation graph (directed structure graph, or the symmetric mesh graph
     with the roles kept), fold reference normals, optional weights, relax
  5. quality before / after, consistency checks, OBJ export

Only the "explicit" method (coarse_subdiv_relax_explicit.cpp) is ported; the
Newton / solve_project methods are not.
"""
import sys
from dataclasses import dataclass

import numpy as np
import log_util
from numba import njit

from obj_io import write_obj
from projector import Projector
from quality import subdiv_mesh_quality
from relax_explicit import ExplicitRelaxOptions, ExplicitRelaxer
from relax_graph import build_relax_graph
from struct_ids import (RELAX_CURVE, RELAX_JUNCTION, RELAX_SHEET, StructError, build_struct_sets,
                        coarse_matstruct, role_of, split_palette)
from subdiv_mesh import SUBDIV_CARRIER_EDGE, SUBDIV_CARRIER_FACE, subdiv_unique_edges
from weights import cotan_weights, meanvalue_weights


@dataclass
class CoarseSubdivRelaxConfig:
    """CoarseSubdivRelaxConfig (coarse_subdiv_relax.h), same defaults."""
    method: str = 'newton'
    curveAnchorTol: float = 3e-3   # solve_project only (not ported)
    maxIter: int = -1              # newton only (not ported)
    perCoarseFace: bool = False    # hold vertices on coarse vertices / edges
    noNewFolds: bool = False
    localProjection: bool = False  # newton only (only changes the output name for explicit)
    jointPass: bool = False        # newton only; also makes the graph symmetric
    jointSolve: bool = False       # solve_project only
    explicitLambda: float = 0.5
    explicitMaxIter: int = 20000
    explicitTol: float = 1e-7
    explicitGlobalProj: bool = False
    explicitDirected: bool = False
    weights: str = 'uniform'       # uniform | cotan | meanvalue
    # Python-only extras (C++: fixed values)
    logEvery: int = 100
    snapshotIters: tuple = (1, 10, 100, 1000, 10000)


def relaxed_obj_prefix(c):
    """The relaxed OBJ name prefix main.cpp builds from the config."""
    return ('coarse_subdiv_at_fine_pos_relaxed_' + c.method
            + ('_perface' if c.perCoarseFace else '')
            + ('_nofold' if c.noNewFolds else '')
            + ('_local' if c.localProjection else '')
            + ('_joint' if c.jointPass else '')
            + ('_jointsolve' if c.jointSolve else '')
            + ('_' + c.weights if c.weights != 'uniform' else '')
            + ('_global' if c.method == 'explicit' and c.explicitGlobalProj else '')
            + ('_directed' if c.method == 'explicit' and c.explicitDirected else '') + '_')


class RelaxFailure(RuntimeError):
    pass


@njit(cache=True)
def _fold_ref(SV, MV, F, faceOrig, FC):
    """Reference normal per subdivided triangle: its coarse face's (summed)
    normal, signed to agree with the majority of that coarse face's seed triangles."""
    nF = F.shape[0]
    cn = np.zeros((FC, 3))
    vote = np.zeros(FC, dtype=np.int64)
    for f in range(nF):
        cf = faceOrig[f]
        if cf < 0 or cf >= FC:
            raise RuntimeError('[coarse_subdiv_relax] subdivided face without a coarse face')
        a = F[f, 0]; b = F[f, 1]; c = F[f, 2]
        ux = SV[b, 0] - SV[a, 0]; uy = SV[b, 1] - SV[a, 1]; uz = SV[b, 2] - SV[a, 2]
        vx = SV[c, 0] - SV[a, 0]; vy = SV[c, 1] - SV[a, 1]; vz = SV[c, 2] - SV[a, 2]
        nx = uy * vz - uz * vy; ny = uz * vx - ux * vz; nz = ux * vy - uy * vx
        cn[cf, 0] += nx; cn[cf, 1] += ny; cn[cf, 2] += nz
        ux = MV[b, 0] - MV[a, 0]; uy = MV[b, 1] - MV[a, 1]; uz = MV[b, 2] - MV[a, 2]
        vx = MV[c, 0] - MV[a, 0]; vy = MV[c, 1] - MV[a, 1]; vz = MV[c, 2] - MV[a, 2]
        mx = uy * vz - uz * vy; my = uz * vx - ux * vz; mz = ux * vy - uy * vx
        vote[cf] += 1 if (mx * nx + my * ny) + mz * nz >= 0 else -1
    ref = np.zeros((nF, 3))
    for f in range(nF):
        cf = faceOrig[f]
        x = cn[cf, 0]; y = cn[cf, 1]; z = cn[cf, 2]
        sq = (x * x + y * y) + z * z
        if sq > 0:  # Eigen normalized(): v / sqrt(|v|^2)
            r = np.sqrt(sq)
            x = x / r; y = y / r; z = z / r
        sgn = 1.0 if vote[cf] >= 0 else -1.0
        ref[f, 0] = sgn * x; ref[f, 1] = sgn * y; ref[f, 2] = sgn * z
    return ref


class CoarseRelaxSession:
    """Steps 1-3 of coarse_subdiv_relax_export (struct IDs, the vertices located on the
    fine mesh, seeds snapped onto their own structure, projector), done once, and
    make_relaxer() for step 4 under any configuration. The batch export and the
    step viewer both relax through this."""

    def __init__(self, B, C, ms, log=None, anc=None):
        from struct_ids import bundle_ancestors
        log = self.log = log or log_util.log
        self.B, self.C, self.ms = B, C, ms
        gVO, gFO = B.fineV, B.fineF
        Vs = self.Vs = C.S.V.shape[0]
        d = gVO.max(axis=0) - gVO.min(axis=0)
        diag = self.diag = float(np.sqrt((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]))

        # 1. struct IDs of the subdivided vertices, from the coarse carriers
        if anc is None:
            anc = bundle_ancestors(B)
        cms = coarse_matstruct(B, ms, anc)
        pal, setId = build_struct_sets(C.S, B.coarseF, cms)

        # 2. the same vertices and faces, located on the fine mesh
        M = C.S.copy()
        M.V = C.P.copy()
        M.fineFace = C.fineFace.copy()
        M.fineBary = C.fineBary.copy()
        M.origEdges = subdiv_unique_edges(gFO)
        M.carrierType[:] = SUBDIV_CARRIER_FACE
        M.carrierIndex = M.fineFace.copy()

        # 3. seeds onto their own fine structure
        S = split_palette(pal, ms)
        setRole = [role_of(pal.typeMask[k]) for k in range(pal.size())]
        proj = Projector(gVO, gFO, M.origEdges, ms, pal, S, setRole)
        junctionVerts = {}
        for st in ms.structs:
            if st.type == 3:
                junctionVerts[st.id] = st.elements
        snapMax = [0.0] * 3; snapSum = [0.0] * 3; snapN = [0] * 3
        noTarget = unmapped = 0
        VO = proj.VO
        for i in range(Vs):
            if M.fineFace[i] < 0:
                unmapped += 1
                continue
            k = int(setId[i])
            role = setRole[k]
            p = M.V[i].copy()
            if role == RELAX_JUNCTION:
                best, bd = -1, np.inf
                for a in range(pal.offsets[k], pal.offsets[k + 1]):
                    for v in junctionVerts.get(pal.ids[a], []):
                        dx, dy, dz = VO[v, 0] - p[0], VO[v, 1] - p[1], VO[v, 2] - p[2]
                        dd = (dx * dx + dy * dy) + dz * dz
                        if dd < bd:
                            bd, best = dd, v
                if best < 0 or not proj.vertFaces[best]:
                    noTarget += 1
                    continue
                f = proj.vertFaces[best][0]
                M.fineFace[i] = f
                M.fineBary[i] = 0.0
                for c in range(3):
                    if gFO[f, c] == best:
                        M.fineBary[i, c] = 1.0
                q = VO[best].copy()
            else:
                if not proj.targets[k]:
                    noTarget += 1
                    continue
                r = proj.project(k, p, M.fineFace[i], -1)
                M.fineFace[i] = r[0]
                M.fineBary[i] = (r[1], r[2], r[3])
                if role == RELAX_CURVE:
                    M.carrierType[i] = SUBDIV_CARRIER_EDGE
                    M.carrierIndex[i] = r[4]
                q = np.array((r[5], r[6], r[7]))
            M.V[i] = q
            dx, dy, dz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
            dd = float(np.sqrt((dx * dx + dy * dy) + dz * dz)) / diag
            snapMax[role] = max(snapMax[role], dd)
            snapSum[role] += dd
            snapN[role] += 1
        for r, name in enumerate(('sheet', 'curve', 'junction')):
            log('[coarse_subdiv_relax] seed snap onto own structure, %s: %d vertices, move max %.3g mean %.3g (x diag)'
                % (name, snapN[r], snapMax[r], snapSum[r] / snapN[r] if snapN[r] else 0.0))
        log('[coarse_subdiv_relax] %d vertices not mapped to fine, %d without a structure target (kept fixed)'
            % (unmapped, noTarget))

        self.cms, self.pal, self.setId, self.S, self.setRole = cms, pal, setId, S, setRole
        self.M, self.proj = M, proj
        self.Vseed = M.V.copy()
        self._graphs = {}
        self._foldRef = None

    def graph(self, symmetric):
        """build_relax_graph with the structure (directed), or the plain mesh graph with
        the roles kept (symmetric). Cached; the directed one is built first, as in C++."""
        if 'directed' not in self._graphs:
            self._graphs['directed'] = build_relax_graph(self.M.F, self.Vs, self.pal, self.setId, self.ms)
        if not symmetric:
            return self._graphs['directed']
        if 'symmetric' not in self._graphs:
            J = build_relax_graph(self.M.F, self.Vs, self.pal, self.setId, None)
            J.role = self._graphs['directed'].role
            self._graphs['symmetric'] = J
        return self._graphs['symmetric']

    def fold_ref(self):
        """Reference normal per subdivided triangle (from the seeds)."""
        if self._foldRef is None:
            C, M = self.C, self.M
            self._foldRef = _fold_ref(np.ascontiguousarray(C.S.V), np.ascontiguousarray(self.Vseed),
                                      np.ascontiguousarray(M.F, dtype=np.int64),
                                      np.ascontiguousarray(C.S.faceOrig, dtype=np.int64), self.B.coarseF.shape[0])
        return self._foldRef

    def make_relaxer(self, cfg, objPath='', state=None, it0=0, log_quality=None):
        """Step 4: graph, hold set, fold reference and weights for cfg, then the
        ExplicitRelaxer. state = (X, face, edge, bary) to continue from (default: the
        seeds); it0 = iterations already done."""
        log, C, M = self.log, self.C, self.M
        method = cfg.method
        isExplicit = method == 'explicit'
        if method not in ('newton', 'solve_project', 'explicit'):
            raise RelaxFailure('[coarse_subdiv_relax] unknown relax method ' + method)
        if not isExplicit:
            raise RelaxFailure('[coarse_subdiv_relax] only the explicit method is ported to Python')
        G = self.graph(cfg.jointPass or (isExplicit and not cfg.explicitDirected))
        if log_quality is not None:
            log_quality()
        if cfg.jointSolve and method != 'solve_project':
            log('[coarse_subdiv_relax] WARNING: the joint solve is solve_project only; ignored by %s' % method)
        holdFixed = None
        if cfg.perCoarseFace:
            holdFixed = (C.S.carrierType != SUBDIV_CARRIER_FACE).astype(np.uint8)
            log('[coarse_subdiv_relax] per coarse face: %d vertices on coarse vertices / edges held at their seeds'
                % int(holdFixed.sum()))
        foldRef = self.fold_ref() if (cfg.noNewFolds or isExplicit) else None
        if cfg.weights not in ('uniform', 'cotan', 'meanvalue'):
            raise RelaxFailure('[coarse_subdiv_relax] unknown weights ' + cfg.weights)
        eo = ExplicitRelaxOptions()
        eo.lam = cfg.explicitLambda
        eo.maxIter = cfg.explicitMaxIter
        eo.tol = cfg.explicitTol
        eo.localProjection = not cfg.explicitGlobalProj
        eo.holdFixed = holdFixed
        eo.noNewFolds = cfg.noNewFolds
        eo.foldRef = foldRef
        eo.logEvery = cfg.logEvery
        eo.snapshotIters = list(cfg.snapshotIters)
        eo.snapshotPrefix = objPath[:-4] + '_' if objPath else ''
        if cfg.weights == 'cotan':
            eo.weights = cotan_weights(C.S.V, M.F, G, log)
        if cfg.weights == 'meanvalue':
            eo.weights = meanvalue_weights(C.S.V, M.F, G, log)
        Mw = M.copy()  # the relaxer writes its result here
        return ExplicitRelaxer(Mw, self.B.fineV, self.B.fineF, self.ms, self.pal, self.setId, G, eo, log,
                               state=state, proj=self.proj, it0=it0)


def coarse_subdiv_relax_export(B, C, ms, cfg, maxObjVerts, objPath, log=None, anc=None, on_relaxer=None):
    """Relaxes C's vertices on the fine mesh (B.fineV / B.fineF = gVO / gFO) and
    writes the result to objPath. Returns (M, report, q0, q1).
    on_relaxer(sess, rel) (Python-only): called after the relaxer's initialization,
    before its first step (run_relax.py exports the relaxation input there)."""
    log = log or log_util.log
    sess = CoarseRelaxSession(B, C, ms, log, anc)
    method = cfg.method
    isExplicit = method == 'explicit'
    Vseed = sess.Vseed
    q = {}

    def quality_before():
        q['q0'] = subdiv_mesh_quality(sess.M.V, sess.M.F)
    rel = sess.make_relaxer(cfg, objPath, log_quality=quality_before)
    if on_relaxer is not None:
        on_relaxer(sess, rel)
    rel.run()
    R = rel.finish()
    M = rel.M
    q0 = q['q0']
    Vs = sess.Vs

    q1 = subdiv_mesh_quality(M.V, M.F, Vseed)
    tag = (method + (', per coarse face' if cfg.perCoarseFace else '') + (', no new folds' if cfg.noNewFolds else '')
           + (', local projection' if cfg.localProjection else '') + (', joint pass' if cfg.jointPass else '')
           + (', joint solve' if cfg.jointSolve else '')
           + (', ' + cfg.weights + ' weights' if cfg.weights != 'uniform' else '')
           + (', directed graph' if isExplicit and cfg.explicitDirected else ''))
    log('[coarse_subdiv_relax] relaxation (%s), before -> after: edge CV %.4f -> %.4f | min angle %.3f -> %.3f, '
        'p1 %.3f -> %.3f, p5 %.3f -> %.3f, median %.3f -> %.3f deg | degenerate %d -> %d | '
        'flipped vs seed %d | move max %.3g mean %.3g (x diag)'
        % (tag, q0.edgeCV, q1.edgeCV, q0.minAngle, q1.minAngle, q0.p1, q1.p1, q0.p5, q1.p5, q0.median, q1.median,
           q0.degenerate, q1.degenerate, q1.flippedVsRef, R.maxMove, R.meanMove))
    if R.seedOffStructure or R.fixedMoved or R.posMismatch or R.badBary or R.offStructure:
        raise RelaxFailure('[coarse_subdiv_relax] relaxation consistency checks failed')

    # 5. export
    if Vs > maxObjVerts:
        log('[coarse_subdiv_relax] skipping %s: %d vertices > %d' % (objPath, Vs, maxObjVerts))
    elif not write_obj(objPath, M.V, M.F):
        log('[coarse_subdiv_relax] writeOBJ failed: %s' % objPath)
    else:
        log('[coarse_subdiv_relax] relaxed subdivided coarse mesh on the fine MAT -> %s' % objPath)
    return M, R, q0, q1
