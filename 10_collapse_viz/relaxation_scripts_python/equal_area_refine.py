"""Equal-area refinement of the coarse mesh before the uniform subdivision
(Python-only mode, no C++ counterpart).

The coarse faces have very different areas (ABC 00040057: max / min about 2000),
and the uniform subdivision puts the same number of samples in each of them, so
the sample density varies by the same factor. The relaxation has to move samples
across coarse edges to even this out, which the explicit (Jacobi) steps do slowly.

This module splits the big coarse faces first, until no face is larger than a
target area A*:

- longest-edge bisection (Rivara, LEPP): a face is split at the midpoint of its
  longest edge; every face on that edge is split at the same point, after its own
  longest edge has been split if that is a longer edge. The mesh stays conforming
  (also on non-manifold edges, where all faces of the edge are split) and the
  angles do not degrade.
- the largest face is split first; the loop stops when the largest face has area
  <= A*. Faces already smaller than A* are only split when a neighbour's split
  reaches them. Splitting only: faces below A* keep their size.
- k = --equal_area_levels: A* is searched (bisection on log A*) so that the refined
  mesh has about |F| * 4^k faces, i.e. as many as k uniform levels would make. The
  uniform subdivision then needs k levels fewer for the same sample count, and the
  vertex count stays close to the run without refinement. --equal_area_target sets
  A* directly.
- areas are measured on the coarse geometry.

Every refined vertex keeps an exact carrier on the ORIGINAL coarse mesh (coarse
vertex, coarse edge with dyadic coordinates, or coarse face with dyadic
barycentrics), and every refined face keeps its coarse face. The uniform
subdivision (subdiv_mesh.build_subdiv_mesh with `start`) continues from there,
so the subdivided mesh has the same carriers, faceOrig, fineFace / fineBary as
without refinement and everything downstream (c2f walk, struct ids, roles,
projection, per-face holds, fold reference) is unchanged.

Standalone: area statistics before / after and the refined coarse mesh as OBJ:
  python equal_area_refine.py --bundle <run>/correspondence_<stem>.c2f --equal_area_levels 2 --out refined.obj
"""
import argparse
import heapq
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from log_util import log  # noqa: E402
from subdiv_mesh import (SUBDIV_CARRIER_EDGE, SUBDIV_CARRIER_FACE, SUBDIV_CARRIER_VERTEX,  # noqa: E402
                         subdiv_unique_edges)


class RefineError(RuntimeError):
    pass


def _fail(msg):
    raise RefineError('[equal_area_refine] ' + msg)


class RefinedCoarse:
    """F: refined faces (int64), faceOrig: coarse face of each, carrierType /
    carrierIndex / carrierCoord: carrier of each refined vertex on the coarse mesh
    (vertices 0..|VO|-1 are the coarse vertices), V: positions on the coarse geometry."""
    pass


def tri_areas(V, F):
    u = V[F[:, 1]] - V[F[:, 0]]
    w = V[F[:, 2]] - V[F[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(u, w), axis=1)


def refine_equal_area(VO, FO, Astar, maxFaces=50000000):
    """Longest-edge bisection of (VO, FO) until every face has area <= Astar."""
    VO = np.asarray(VO, dtype=np.float64)[:, :3]
    FO = np.asarray(FO, dtype=np.int64)
    nVO, nFO = VO.shape[0], FO.shape[0]
    FOl = FO.tolist()
    oEdges = subdiv_unique_edges(FO)
    oE = oEdges.tolist()
    eIdx = {(a, b): i for i, (a, b) in enumerate(oE)}

    ctype = [SUBDIV_CARRIER_VERTEX] * nVO
    cidx = list(range(nVO))
    ccoord = [(1.0, 0.0, 0.0)] * nVO
    P = [tuple(r) for r in VO.tolist()]

    faces = [list(r) for r in FOl]
    parent = list(range(nFO))
    alive = [True] * nFO
    edgeFaces = {}

    def ekey(a, b):
        return (a, b) if a < b else (b, a)

    def add_face(f):
        r = faces[f]
        for c in range(3):
            edgeFaces.setdefault(ekey(r[c], r[(c + 1) % 3]), []).append(f)

    for f in range(nFO):
        add_face(f)

    def area(f):
        a, b, c = faces[f]
        pa, pb, pc = P[a], P[b], P[c]
        u = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
        w = (pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2])
        x = u[1] * w[2] - u[2] * w[1]
        y = u[2] * w[0] - u[0] * w[2]
        z = u[0] * w[1] - u[1] * w[0]
        return 0.5 * (x * x + y * y + z * z) ** 0.5

    def longest(f):
        # strict total order (squared length, then the vertex pair): LEPP terminates
        r = faces[f]
        best = None
        for c in range(3):
            a, b = r[c], r[(c + 1) % 3]
            pa, pb = P[a], P[b]
            l2 = (pb[0] - pa[0]) ** 2 + (pb[1] - pa[1]) ** 2 + (pb[2] - pa[2]) ** 2
            k = (l2,) + ekey(a, b)
            if best is None or k > best:
                best = k
        return best[1], best[2]

    def corner_of(fo, v):
        r = FOl[fo]
        for c in range(3):
            if r[c] == v:
                return c
        return -1

    def to_face_bary(v, fo):
        # same rules as subdiv_mesh.build_subdiv_mesh
        b = [0.0, 0.0, 0.0]
        t, idx = ctype[v], cidx[v]
        if t == SUBDIV_CARRIER_VERTEX:
            c = corner_of(fo, idx)
            if c < 0:
                _fail('vertex %d is not a corner of coarse face %d' % (v, fo))
            b[c] = 1.0
        elif t == SUBDIV_CARRIER_EDGE:
            c0, c1 = corner_of(fo, oE[idx][0]), corner_of(fo, oE[idx][1])
            if c0 < 0 or c1 < 0:
                _fail('vertex %d (coarse edge %d) is not on coarse face %d' % (v, idx, fo))
            b[c0] = ccoord[v][0]
            b[c1] = ccoord[v][1]
        else:
            if idx != fo:
                _fail('vertex %d lies inside coarse face %d, not %d' % (v, idx, fo))
            b = list(ccoord[v])
        return b

    def classify(b, fo):
        if (b[0] + b[1]) + b[2] != 1.0 or min(b) < 0.0:
            _fail('non-exact barycentric in coarse face %d' % fo)
        nz = [c for c in range(3) if b[c] != 0.0]
        if len(nz) == 3:
            return SUBDIV_CARRIER_FACE, fo, (b[0], b[1], b[2])
        if len(nz) == 2:
            a, bb = FOl[fo][nz[0]], FOl[fo][nz[1]]
            e = eIdx.get(ekey(a, bb), -1)
            if e < 0:
                _fail('edge not found in coarse face %d' % fo)
            wa, wb = b[nz[0]], b[nz[1]]
            return SUBDIV_CARRIER_EDGE, e, ((wa, wb, 0.0) if oE[e][0] == a else (wb, wa, 0.0))
        _fail('midpoint on a coarse vertex in coarse face %d' % fo)

    def split_edge(a, b):
        fs = list(edgeFaces[ekey(a, b)])
        car = None
        for g in fs:
            fo = parent[g]
            ba, bb = to_face_bary(a, fo), to_face_bary(b, fo)
            c = classify([0.5 * (ba[0] + bb[0]), 0.5 * (ba[1] + bb[1]), 0.5 * (ba[2] + bb[2])], fo)
            if car is None:
                car = c
            elif c != car:
                _fail('faces around edge (%d, %d) disagree on its midpoint' % (a, b))
        m = len(ctype)
        ctype.append(car[0]); cidx.append(car[1]); ccoord.append(car[2])
        pa, pb = P[a], P[b]
        P.append((0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]), 0.5 * (pa[2] + pb[2])))
        new = []
        for g in fs:
            r = faces[g]
            c = 0
            while not ({r[c], r[(c + 1) % 3]} == {a, b}):
                c += 1
            u, w, o = r[c], r[(c + 1) % 3], r[(c + 2) % 3]
            alive[g] = False
            for k in range(3):
                lst = edgeFaces[ekey(r[k], r[(k + 1) % 3])]
                lst.remove(g)
            for tri in ((u, m, o), (m, w, o)):
                faces.append(list(tri)); parent.append(parent[g]); alive.append(True)
                add_face(len(faces) - 1)
                new.append(len(faces) - 1)
        del edgeFaces[ekey(a, b)]
        return new

    def refine(f0):
        new = []
        stack = [f0]
        while stack:
            f = stack[-1]
            if not alive[f]:
                stack.pop()
                continue
            e = longest(f)
            bad = [g for g in edgeFaces[ekey(*e)] if g != f and longest(g) != e]
            if bad:
                stack.append(bad[0])
                continue
            new += split_edge(*e)
            stack.pop()
        return new

    heap = [(-area(f), f) for f in range(nFO)]
    heapq.heapify(heap)
    while heap:
        na, f = heapq.heappop(heap)
        if not alive[f]:
            continue
        if -na <= Astar:
            break
        new = refine(f)
        for g in new:
            if alive[g]:
                heapq.heappush(heap, (-area(g), g))
        if len(faces) > maxFaces:
            _fail('more than %d faces (target area too small?)' % maxFaces)

    keep = [f for f in range(len(faces)) if alive[f]]
    R = RefinedCoarse()
    R.F = np.array([faces[f] for f in keep], dtype=np.int64).reshape(-1, 3)
    R.faceOrig = np.array([parent[f] for f in keep], dtype=np.int64)
    R.carrierType = ctype
    R.carrierIndex = cidx
    R.carrierCoord = ccoord
    R.V = np.array(P, dtype=np.float64)
    R.Astar = Astar

    # conformity: a refined edge inside a coarse face has 2 faces, one on a coarse
    # edge has as many faces as that coarse edge (no T-junctions)
    oCount = {}
    for r in FOl:
        for c in range(3):
            key = ekey(r[c], r[(c + 1) % 3])
            oCount[key] = oCount.get(key, 0) + 1
    edgeOne = {}
    for f in keep:
        r = faces[f]
        for c in range(3):
            key = ekey(r[c], r[(c + 1) % 3])
            n, g = edgeOne.get(key, (0, f))
            edgeOne[key] = (n + 1, g)
    for (a, b), (n, g) in edgeOne.items():
        fo = parent[g]
        ba, bb = to_face_bary(a, fo), to_face_bary(b, fo)
        typ, e, _ = classify([0.5 * (ba[0] + bb[0]), 0.5 * (ba[1] + bb[1]), 0.5 * (ba[2] + bb[2])], fo)
        want = 2 if typ == SUBDIV_CARRIER_FACE else oCount[tuple(oE[e])]
        if n != want:
            _fail('refined edge (%d, %d) has %d faces, expected %d (T-junction?)' % (a, b, n, want))
    check_cover(VO, FO, R)
    return R


def check_cover(VO, FO, R):
    """Per coarse face, the refined faces cover its area."""
    a0 = tri_areas(VO, FO)
    a1 = np.bincount(R.faceOrig, tri_areas(R.V, R.F), minlength=FO.shape[0])
    rel = np.abs(a1 - a0) / np.maximum(a0, 1e-300)
    if rel.max() > 1e-9:
        _fail('refined faces do not cover their coarse face (rel. area error %.3g)' % rel.max())


def area_stats(a):
    q = np.percentile(a, [0, 10, 50, 90, 100])
    return dict(n=len(a), min=q[0], p10=q[1], median=q[2], p90=q[3], max=q[4], max_over_min=q[4] / q[0],
                p90_over_p10=q[3] / q[1], cv=float(a.std() / a.mean()))


def stats_line(name, a):
    s = area_stats(a)
    return ('%s: %d faces, area min %.4g p10 %.4g median %.4g p90 %.4g max %.4g | max/min %.1f, p90/p10 %.2f, CV %.3f'
            % (name, s['n'], s['min'], s['p10'], s['median'], s['p90'], s['max'], s['max_over_min'],
               s['p90_over_p10'], s['cv']))


def refine_to_count(VO, FO, nFaces, iters=24):
    """Refinement whose face count is closest to nFaces (A* by bisection on log A*)."""
    total = float(tri_areas(np.asarray(VO)[:, :3], np.asarray(FO)).sum())
    lo, hi = np.log(total / nFaces / 16.0), np.log(total / nFaces * 4.0)  # many / few faces
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        R = refine_equal_area(VO, FO, float(np.exp(mid)))
        n = R.F.shape[0]
        if best is None or abs(n - nFaces) < abs(best.F.shape[0] - nFaces):
            best = R
        if n == nFaces:
            break
        if n > nFaces:
            lo = mid
        else:
            hi = mid
    return best


def refine_from_args(VO, FO, levels=None, target=None):
    """None when the mode is off (no levels, no target)."""
    if (levels is None or levels <= 0) and (target is None or target <= 0):
        return None
    VO = np.asarray(VO, dtype=np.float64)[:, :3]
    FO = np.asarray(FO, dtype=np.int64)
    if target is not None and target > 0:
        R = refine_equal_area(VO, FO, float(target))
        how = 'given'
    else:
        R = refine_to_count(VO, FO, FO.shape[0] * 4 ** levels)
        how = 'searched for about %d faces (levels %d)' % (FO.shape[0] * 4 ** levels, levels)
    log('[equal_area_refine] target area %.6g (%s): |V| %d -> %d, |F| %d -> %d'
        % (R.Astar, how, VO.shape[0], len(R.carrierType), FO.shape[0], R.F.shape[0]))
    log('[equal_area_refine] ' + stats_line('coarse faces ', tri_areas(VO, FO)))
    log('[equal_area_refine] ' + stats_line('refined faces', tri_areas(R.V, R.F)))
    return R


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bundle', required=True)
    p.add_argument('--equal_area_levels', type=int, default=2)
    p.add_argument('--equal_area_target', type=float, default=-1.0)
    p.add_argument('--out', default=None, help='write the refined coarse mesh (OBJ)')
    a = p.parse_args(argv)
    from bundle_io import load_bundle_flat
    from obj_io import write_obj
    B = load_bundle_flat(a.bundle)
    R = refine_from_args(B.coarseV, B.coarseF, a.equal_area_levels, a.equal_area_target)
    if a.out:
        write_obj(a.out, R.V, R.F)
        log('[equal_area_refine] refined coarse mesh -> %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
