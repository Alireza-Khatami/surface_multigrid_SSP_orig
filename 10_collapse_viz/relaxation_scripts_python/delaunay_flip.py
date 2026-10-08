"""Delaunay edge flips on the subdivided coarse mesh (Python-only, no C++ counterpart).

Lawson flips on the mesh S (positions S.V on the coarse geometry): an edge ab shared by
two triangles abc, bad is flipped to cd when the angles opposite it sum to more than pi,

    angle_c(a, b) + angle_d(a, b) > pi    (<=>  cot + cot < 0: a negative cotan weight),

until no flippable edge is left. Flips only change the connectivity: no vertex moves, no
sample is added or removed, the vertices' carriers (and so their c2f positions, struct
ids, roles and projection targets) stay as they are.

Never flipped:
  - seam and boundary edges: an edge whose two ends share a seam / boundary struct id
    (every seam / boundary edge is one; checked again at the end: none may disappear);
  - edges without exactly two triangles (seams have 3+, boundaries 1);
  - edges whose triangles are not consistently oriented.
Never created: an edge whose two ends share a seam / boundary id (it would act as a
curve edge in the directed relaxation graph), an edge that already exists, a degenerate
or folded triangle (new normals must agree with the old pair's).

Scope ("sheet", the default): edges inside one coarse face and coarse edges inside a
sheet (not seam / boundary). A triangle made by a flip across a coarse edge spans two
coarse faces; its coarse face (faceOrig: fold reference normal) is the one of the two
whose normal is closer to the triangle's. Scope "face": only edges inside one coarse face.
Across a bend the flip loop is not guaranteed to end; it is capped (maxFlips).

  python delaunay_flip.py --equal_area_samples 277700     # statistics only (no relaxation)
"""
import argparse
import heapq  # noqa: F401
import math
import os
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from log_util import log  # noqa: E402


def _angle(P, o, a, b):
    """Angle at o in triangle (o, a, b)."""
    ox, oy, oz = P[o]
    ux, uy, uz = P[a][0] - ox, P[a][1] - oy, P[a][2] - oz
    vx, vy, vz = P[b][0] - ox, P[b][1] - oy, P[b][2] - oz
    d = ux * vx + uy * vy + uz * vz
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return math.atan2(math.sqrt(cx * cx + cy * cy + cz * cz), d)


def _normal(P, a, b, c):
    ax, ay, az = P[a]
    ux, uy, uz = P[b][0] - ax, P[b][1] - ay, P[b][2] - az
    vx, vy, vz = P[c][0] - ax, P[c][1] - ay, P[c][2] - az
    return (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)


def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def curve_ids_per_vertex(S, B, ms, cms=None):
    """Seam / boundary struct ids of every vertex of S (from its carrier on the coarse
    mesh, the rules the relaxation uses)."""
    from struct_ids import bundle_ancestors, build_struct_sets, coarse_matstruct, split_palette
    if cms is None:
        cms = coarse_matstruct(B, ms, bundle_ancestors(B))
    pal, setId = build_struct_sets(S, B.coarseF, cms)
    sp = split_palette(pal, ms)
    perSet = [frozenset(sp.curve[k]) for k in range(pal.size())]
    return [perSet[k] for k in np.asarray(setId).tolist()]


def edge_stats(P, F):
    """(negative cotan edges, interior edges, min angle p1 / p5 / median in degrees)."""
    P = np.asarray(P, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    A = np.zeros((F.shape[0], 3))
    for c in range(3):
        o, a, b = F[:, c], F[:, (c + 1) % 3], F[:, (c + 2) % 3]
        u, v = P[a] - P[o], P[b] - P[o]
        A[:, c] = np.arctan2(np.linalg.norm(np.cross(u, v), axis=1), np.einsum('ij,ij->i', u, v))
    # angle opposite edge (F[:,c+1], F[:,c+2]) is A[:, c]
    a, b = F[:, [1, 2, 0]].reshape(-1), F[:, [2, 0, 1]].reshape(-1)
    key = (np.minimum(a, b) << 32) | np.maximum(a, b)
    opp = A.reshape(-1)
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    s = np.bincount(inv, opp, minlength=len(uk))
    two = cnt == 2
    neg = int((s[two] > math.pi + 1e-12).sum())
    mins = np.degrees(A.min(axis=1))
    q = np.percentile(mins, [1, 5, 50])
    return neg, int(two.sum()), q


def delaunay_flip(S, B, ms, scope='sheet', maxFlips=None, cms=None, logf=None):
    """Flips S.F / S.faceOrig in place. Returns a dict of statistics."""
    logf = logf or log
    t0 = time.perf_counter()
    P = [tuple(r) for r in np.asarray(S.V, dtype=np.float64)[:, :3].tolist()]
    F = [list(r) for r in np.asarray(S.F, dtype=np.int64).tolist()]
    fo = np.asarray(S.faceOrig, dtype=np.int64).tolist()
    nF = len(F)
    curves = curve_ids_per_vertex(S, B, ms, cms)
    CV, CF = np.asarray(B.coarseV, dtype=np.float64)[:, :3], np.asarray(B.coarseF, dtype=np.int64)
    cn = np.cross(CV[CF[:, 1]] - CV[CF[:, 0]], CV[CF[:, 2]] - CV[CF[:, 0]])
    cn = cn / np.maximum(np.linalg.norm(cn, axis=1, keepdims=True), 1e-300)
    cn = [tuple(r) for r in cn.tolist()]

    def key(a, b):
        return (a, b) if a < b else (b, a)

    def protected(a, b):
        return bool(curves[a] & curves[b])  # both ends on one seam / boundary

    ef = {}
    for f, r in enumerate(F):
        for c in range(3):
            ef.setdefault(key(r[c], r[(c + 1) % 3]), []).append(f)
    protEdges = [e for e in ef if protected(*e)]
    neg0, int0, q0 = edge_stats(S.V, S.F)

    st = dict(scope=scope, flips=0, flipsInFace=0, flipsAcross=0, skipProtected=0, skipNonManifold=0,
              skipOrientation=0, skipCurveChord=0, skipExists=0, skipFold=0, skipScope=0, capped=False)
    maxFlips = maxFlips or 50 * len(ef)
    queue = deque(ef.keys())
    inq = set(ef.keys())
    eps = 1e-12
    while queue:
        e = queue.popleft()
        inq.discard(e)
        fs = ef.get(e)
        if fs is None:
            continue
        a, b = e
        if len(fs) != 2:
            continue
        f1, f2 = fs
        r1, r2 = F[f1], F[f2]
        c = [v for v in r1 if v != a and v != b][0]
        d = [v for v in r2 if v != a and v != b][0]
        if _angle(P, c, a, b) + _angle(P, d, a, b) <= math.pi + eps:
            continue  # locally Delaunay
        if protected(a, b):
            st['skipProtected'] += 1
            continue
        if scope == 'face' and fo[f1] != fo[f2]:
            st['skipScope'] += 1
            continue
        # orientation: f1 runs a -> b, f2 runs b -> a (consistent)
        i1, i2 = r1.index(a), r2.index(a)
        if r1[(i1 + 1) % 3] == b:
            pass
        else:
            a, b = b, a
            i1, i2 = r1.index(a), r2.index(a)
        if r1[(i1 + 1) % 3] != b or r2[(i2 + 2) % 3] != b:
            st['skipOrientation'] += 1
            continue
        if c == d or key(c, d) in ef:
            st['skipExists'] += 1
            continue
        if curves[c] & curves[d]:
            st['skipCurveChord'] += 1
            continue
        t1, t2 = [c, a, d], [c, d, b]
        n_old = tuple(x + y for x, y in zip(_normal(P, *r1), _normal(P, *r2)))
        n1, n2 = _normal(P, *t1), _normal(P, *t2)
        a1, a2 = _dot(n1, n1), _dot(n2, n2)
        if a1 <= 1e-24 * _dot(n_old, n_old) or a2 <= 1e-24 * _dot(n_old, n_old) \
                or _dot(n1, n_old) <= 0 or _dot(n2, n_old) <= 0:
            st['skipFold'] += 1
            continue
        if st['flips'] >= maxFlips:
            st['capped'] = True
            break
        # flip
        for f in (f1, f2):
            r = F[f]
            for k in range(3):
                lst = ef[key(r[k], r[(k + 1) % 3])]
                lst.remove(f)
        del ef[key(a, b)]
        o1, o2 = fo[f1], fo[f2]
        if o1 == o2:
            st['flipsInFace'] += 1
            g1 = g2 = o1
        else:
            st['flipsAcross'] += 1
            g1 = o1 if _dot(n1, cn[o1]) >= _dot(n1, cn[o2]) else o2
            g2 = o1 if _dot(n2, cn[o1]) >= _dot(n2, cn[o2]) else o2
        F[f1], F[f2] = t1, t2
        fo[f1], fo[f2] = g1, g2
        for f in (f1, f2):
            r = F[f]
            for k in range(3):
                ef.setdefault(key(r[k], r[(k + 1) % 3]), []).append(f)
        st['flips'] += 1
        for x in (key(a, c), key(c, b), key(b, d), key(d, a)):
            if x not in inq:
                queue.append(x)
                inq.add(x)

    S.F = np.array(F, dtype=np.int64).reshape(nF, 3)
    S.faceOrig = np.array(fo, dtype=np.int64)
    # seam / boundary edges must all still be there
    lost = [e for e in protEdges if e not in ef or not ef[e]]
    if lost:
        raise RuntimeError('[delaunay_flip] %d seam / boundary edges were flipped' % len(lost))
    neg1, int1, q1 = edge_stats(S.V, S.F)
    st.update(seamBoundaryEdges=len(protEdges), seamBoundaryLost=len(lost), negCotBefore=neg0, negCotAfter=neg1,
              interiorEdges=int1, minAngleBefore=[float(x) for x in q0], minAngleAfter=[float(x) for x in q1],
              seconds=time.perf_counter() - t0)
    logf('[delaunay_flip] scope %s: %d flips (%d inside a coarse face, %d across a coarse edge)%s in %.1f s'
         % (scope, st['flips'], st['flipsInFace'], st['flipsAcross'], ' (CAPPED)' if st['capped'] else '',
            st['seconds']))
    logf('[delaunay_flip] non-Delaunay edges (negative cotan weight): %d -> %d of %d interior edges; left: '
         'seam / boundary %d, other scope %d, orientation %d, existing edge %d, curve chord %d, would fold %d '
         '(counts of the last checks)'
         % (neg0, neg1, int1, st['skipProtected'], st['skipScope'], st['skipOrientation'], st['skipExists'],
            st['skipCurveChord'], st['skipFold']))
    logf('[delaunay_flip] min angle p1 / p5 / median: %.2f / %.2f / %.2f -> %.2f / %.2f / %.2f deg; seam / '
         'boundary edges %d, flipped 0' % (q0[0], q0[1], q0[2], q1[0], q1[1], q1[2], len(protEdges)))
    return st


def flip_plys(out_dir, stem, S, Fbefore, foBefore, comment=''):
    """The mesh before and after the flips (same vertices, coarse geometry) as PLY:
    face properties coarse_face, min_angle (deg), and in the after mesh changed = 1 for the
    triangles made by flips; vertex properties carrier_type / carrier_index."""
    from ply_io import write_ply
    os.makedirs(out_dir, exist_ok=True)
    V = np.asarray(S.V, dtype=np.float64)[:, :3]

    def min_angle(F):
        A = []
        for c in range(3):
            o, a, b = F[:, c], F[:, (c + 1) % 3], F[:, (c + 2) % 3]
            u, w = V[a] - V[o], V[b] - V[o]
            A.append(np.arctan2(np.linalg.norm(np.cross(u, w), axis=1), np.einsum('ij,ij->i', u, w)))
        return np.degrees(np.min(np.stack(A, axis=1), axis=1))

    def tri_keys(F):
        s = np.sort(F, axis=1)
        return (s[:, 0] << 42) | (s[:, 1] << 21) | s[:, 2]
    Fb, Fa = np.asarray(Fbefore, dtype=np.int64), np.asarray(S.F, dtype=np.int64)
    changed = ~np.isin(tri_keys(Fa), tri_keys(Fb))
    vp = dict(carrier_type=S.carrierType.astype(np.uint8), carrier_index=S.carrierIndex.astype(np.int32))
    com = [comment] if comment else []
    pb = os.path.join(out_dir, '02a_before_flips' + stem + '.ply')
    pa = os.path.join(out_dir, '02b_after_flips' + stem + '.ply')
    write_ply(pb, V, Fb, vp, dict(coarse_face=np.asarray(foBefore).astype(np.int32),
                                  min_angle=min_angle(Fb).astype(np.float32)),
              com + ['before the Delaunay flips (coarse geometry)'])
    write_ply(pa, V, Fa, vp, dict(coarse_face=np.asarray(S.faceOrig).astype(np.int32),
                                  min_angle=min_angle(Fa).astype(np.float32), changed=changed.astype(np.uint8)),
              com + ['after the Delaunay flips (coarse geometry); changed = made by a flip'])
    log('[delaunay_flip] before / after PLYs -> %s, %s (%d of %d triangles changed)'
        % (pb, pa, int(changed.sum()), len(Fa)))
    return pb, pa


def main(argv=None):
    import run_relax
    from bundle_io import load_bundle_flat
    from matstruct import load_matstruct
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bundle', default=run_relax.default_bundle())
    p.add_argument('--matstruct_path', default=run_relax.DEFAULT_MS)
    p.add_argument('--equal_area_samples', type=int, default=277700)
    p.add_argument('--delaunay_scope', default='sheet', choices=['sheet', 'face'])
    p.add_argument('--out_dir', default=None, help='write the before / after PLYs here')
    a = p.parse_args(argv)
    B = load_bundle_flat(a.bundle)
    ms = load_matstruct(a.matstruct_path, B.fineV, B.fineF)
    ra = run_relax.parse_args(['--bundle', a.bundle, '--equal_area_samples', str(a.equal_area_samples)])
    C = run_relax.build_c2f(B, ra)
    Fb, fob = C.S.F.copy(), C.S.faceOrig.copy()
    st = delaunay_flip(C.S, B, ms, a.delaunay_scope)
    if a.out_dir:
        import datetime
        import json
        stem = '_' + run_relax.bundle_stem(a.bundle)
        flip_plys(a.out_dir, stem, C.S, Fb, fob, 'delaunay_flip.py, scope %s' % a.delaunay_scope)
        with open(os.path.join(a.out_dir, 'experiment_config.txt'), 'w', newline='\n') as f:
            f.write('experiment:   %s\n' % os.path.basename(os.path.normpath(a.out_dir)))
            f.write('description:  Delaunay edge flips on the equal-area split mesh; before / after PLYs (no relaxation)\n')
            f.write('started:      %s\n' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            f.write('implementation: python (relaxation_scripts_python/delaunay_flip.py)\n')
            f.write('bundle:       %s\n' % a.bundle)
            f.write('matstruct:    %s\n' % a.matstruct_path)
            f.write('equal-area:   largest face split until %d samples (no uniform subdivision)\n' % a.equal_area_samples)
            f.write('flips:        scope %s; seam / boundary edges never flipped\n' % a.delaunay_scope)
            f.write('command:      python delaunay_flip.py %s\n' % ' '.join(argv if argv is not None else sys.argv[1:]))
            f.write('result:       %s\n' % json.dumps(st))
    return 0


if __name__ == '__main__':
    sys.exit(main())
