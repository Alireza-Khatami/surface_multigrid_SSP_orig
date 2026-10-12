"""Is the coarse -> fine map bijective? Checks the map of the subdivided coarse mesh (after the
equal-area split and the Delaunay flips) onto the fine MAT given by the c2f query (sample i ->
P[i] on fine face fineFace[i]), and the same triangulation at the relaxed positions.

The map is piecewise linear on the subdivided triangles. It is locally bijective when
  1. vertex injectivity: no two samples land on the same fine point;
  2. orientation: no image triangle is degenerate or flipped. Flipped = its normal disagrees
     with the fine surface under it (the normal of the fine faces of its corners), after the
     orientation of its coarse face relative to the fine sheet is taken out (per coarse face,
     the majority sign of its triangles at the c2f seeds; some coarse faces are wound against
     the fine sheet under them, which is not a fold);
  3. one-ring check: at every vertex inside a sheet (every incident edge has exactly two
     faces) the image triangles around it, projected onto the fine tangent plane at the
     vertex, turn by exactly 2 pi (oriented angle sum). Less: a fold; 4 pi or more: the
     one-ring wraps twice (a branch point).
1-3 everywhere is local bijectivity. Global overlap (two far-apart parts of one sheet
covering the same fine region) is NOT tested. Also reported: the existing fold count (against
the coarse normals, the relaxation's no-new-folds reference) and the c2f walk diagnostics
(unmapped samples, clamped walk steps).

  python check_c2f_bijective.py --output_dir <dir> [run_relax.py sampling flags, e.g.
      --equal_area_samples 277700 --delaunay_flips]   (writes bijectivity.log / .json there)
run_relax.py calls bijectivity_stats() itself, on the c2f seeds and on the relaxed mesh.
"""
import json
import os
import sys

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import log_util  # noqa: E402

TWO_PI = 2.0 * np.pi


@njit(cache=True)
def _fine_normals(VO, FO):
    n = np.zeros((FO.shape[0], 3))
    for f in range(FO.shape[0]):
        a, b, c = FO[f, 0], FO[f, 1], FO[f, 2]
        ux = VO[b, 0] - VO[a, 0]; uy = VO[b, 1] - VO[a, 1]; uz = VO[b, 2] - VO[a, 2]
        vx = VO[c, 0] - VO[a, 0]; vy = VO[c, 1] - VO[a, 1]; vz = VO[c, 2] - VO[a, 2]
        x = uy * vz - uz * vy; y = uz * vx - ux * vz; z = ux * vy - uy * vx
        l = np.sqrt(x * x + y * y + z * z)
        if l > 0:
            n[f, 0] = x / l; n[f, 1] = y / l; n[f, 2] = z / l
    return n


@njit(parallel=True, cache=True)
def _tri_signs(P, F, fineFace, bary, fn, diag, sgn, area):
    """Per image triangle: sign of (image normal . summed fine normals of its corners'
    faces): +1 / -1, 0 = degenerate, 2 = a corner unmapped. Only corners strictly inside
    their fine face count (a sample on a seam / crease edge sits on the fine face of any of
    the sheets there, whose normal says nothing about this triangle's sheet); all corners
    when none is."""
    for t in prange(F.shape[0]):
        a, b, c = F[t, 0], F[t, 1], F[t, 2]
        if fineFace[a] < 0 or fineFace[b] < 0 or fineFace[c] < 0:
            sgn[t] = 2
            area[t] = 0.0
            continue
        ux = P[b, 0] - P[a, 0]; uy = P[b, 1] - P[a, 1]; uz = P[b, 2] - P[a, 2]
        vx = P[c, 0] - P[a, 0]; vy = P[c, 1] - P[a, 1]; vz = P[c, 2] - P[a, 2]
        x = uy * vz - uz * vy; y = uz * vx - ux * vz; z = ux * vy - uy * vx
        ar = 0.5 * np.sqrt(x * x + y * y + z * z)
        area[t] = ar
        if ar <= 1e-14 * diag * diag:
            sgn[t] = 0
            continue
        rx = 0.0; ry = 0.0; rz = 0.0
        nIn = 0
        for v in (a, b, c):
            if min(bary[v, 0], min(bary[v, 1], bary[v, 2])) > 1e-12:
                rx += fn[fineFace[v], 0]; ry += fn[fineFace[v], 1]; rz += fn[fineFace[v], 2]
                nIn += 1
        if nIn == 0:
            for v in (a, b, c):
                rx += fn[fineFace[v], 0]; ry += fn[fineFace[v], 1]; rz += fn[fineFace[v], 2]
        d = x * rx + y * ry + z * rz
        sgn[t] = 1 if d > 0 else -1


@njit(parallel=True, cache=True)
def _angle_sums(P, F, fineFace, fn, vf_off, vf, interior, triOrient, out):
    """Oriented angle sum of the image one-ring of every interior vertex, in the fine tangent
    plane at the vertex (normal of its fine face); each triangle's angle is signed by its
    coarse face's orientation relative to the fine sheet (triOrient)."""
    for v in prange(P.shape[0]):
        out[v] = np.nan
        if not interior[v] or fineFace[v] < 0:
            continue
        nx = fn[fineFace[v], 0]; ny = fn[fineFace[v], 1]; nz = fn[fineFace[v], 2]
        s = 0.0
        ok = True
        for q in range(vf_off[v], vf_off[v + 1]):
            t = vf[q]
            c = 0
            while F[t, c] != v:
                c += 1
            a = F[t, (c + 1) % 3]; b = F[t, (c + 2) % 3]
            if fineFace[a] < 0 or fineFace[b] < 0:
                ok = False
                break
            ax = P[a, 0] - P[v, 0]; ay = P[a, 1] - P[v, 1]; az = P[a, 2] - P[v, 2]
            bx = P[b, 0] - P[v, 0]; by = P[b, 1] - P[v, 1]; bz = P[b, 2] - P[v, 2]
            da = ax * nx + ay * ny + az * nz
            db = bx * nx + by * ny + bz * nz
            ax -= da * nx; ay -= da * ny; az -= da * nz
            bx -= db * nx; by -= db * ny; bz -= db * nz
            cx = ay * bz - az * by; cy = az * bx - ax * bz; cz = ax * by - ay * bx
            sn = cx * nx + cy * ny + cz * nz
            cs = ax * bx + ay * by + az * bz
            s += triOrient[t] * np.arctan2(sn, cs)
        if ok:
            out[v] = s


def _interior(F, Vs):
    """Vertices whose every incident edge has exactly two incident faces and whose incident
    faces form one closed fan (inside a sheet; not on a seam, boundary or non-manifold vertex)."""
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    key = e[:, 0] * Vs + e[:, 1]
    uk, cnt = np.unique(key, return_counts=True)
    bad = uk[cnt != 2]
    interior = np.ones(Vs, dtype=np.bool_)
    interior[bad // Vs] = False
    interior[bad % Vs] = False
    # one fan: #incident faces == #incident edges for an interior vertex
    nf = np.bincount(F.ravel(), minlength=Vs)
    eu = np.unique(e, axis=0)
    ne = np.bincount(eu.ravel(), minlength=Vs)
    interior &= (nf == ne) & (nf >= 3)
    return interior


class OrientRef:
    """Per coarse face: +1 / -1, how its triangles are wound relative to the fine sheet under
    them (majority at the c2f seeds). Reused for the relaxed mesh."""

    def __init__(self, sgn, faceOrig, nCoarse):
        vote = np.zeros(nCoarse, dtype=np.int64)
        ok = (sgn == 1) | (sgn == -1)
        np.add.at(vote, faceOrig[ok], sgn[ok])
        self.coarse = np.where(vote >= 0, 1, -1).astype(np.int64)


def bijectivity_stats(B, C, P, fineFace, fineBary, name, log=None, ref=None, foldRef=None):
    """Checks the map sample i -> P[i] (fine face fineFace[i], barycentrics fineBary[i]) with the
    triangulation C.S.F.
    ref: an OrientRef (None: built here, from this P; pass the seeds' one for a relaxed mesh).
    foldRef: the relaxation's fold reference (per triangle, coarse normal) to also count folds
    the usual way. Logs a summary; returns (stats dict, OrientRef)."""
    log = log or log_util.log
    VO = np.ascontiguousarray(B.fineV[:, :3], dtype=np.float64)
    FO = np.ascontiguousarray(B.fineF, dtype=np.int64)
    F = np.ascontiguousarray(C.S.F, dtype=np.int64)
    P = np.ascontiguousarray(P[:, :3], dtype=np.float64)
    ff = np.ascontiguousarray(fineFace, dtype=np.int64)
    faceOrig = np.asarray(C.S.faceOrig, dtype=np.int64)
    Vs, nT = P.shape[0], F.shape[0]
    diag = float(np.linalg.norm(VO.max(axis=0) - VO.min(axis=0)))
    fn = _fine_normals(VO, FO)
    st = dict(name=name, samples=int(Vs), triangles=int(nT))

    # 1. vertex injectivity
    mapped = ff >= 0
    st['unmapped'] = int((~mapped).sum())
    Pm = P[mapped]
    _, cnt = np.unique(Pm, axis=0, return_counts=True)
    st['coincident_exact'] = int((cnt[cnt > 1] - 1).sum())
    q = np.round(Pm / (1e-9 * diag)).astype(np.int64)
    _, cnt = np.unique(q, axis=0, return_counts=True)
    st['coincident_1e-9diag'] = int((cnt[cnt > 1] - 1).sum())

    # 2. orientation
    sgn = np.zeros(nT, np.int64)
    area = np.zeros(nT)
    fb = np.ascontiguousarray(fineBary, dtype=np.float64)
    _tri_signs(P, F, ff, fb, fn, diag, sgn, area)
    if ref is None:
        ref = OrientRef(sgn, faceOrig, len(B.coarseF))
    rel = sgn * ref.coarse[faceOrig]          # +1 agrees with its coarse face's majority
    st['tri_unmapped_corner'] = int((sgn == 2).sum())
    st['tri_degenerate'] = int((sgn == 0).sum())
    st['tri_flipped'] = int((((sgn == 1) | (sgn == -1)) & (rel == -1)).sum())
    st['coarse_faces_wound_against_fine'] = int((ref.coarse < 0).sum())
    st['coarse_faces_with_flips'] = int(len(np.unique(faceOrig[((sgn == 1) | (sgn == -1)) & (rel == -1)])))
    if foldRef is not None:
        from relax_explicit import count_folded
        st['folded_vs_coarse_normal'] = int(count_folded(P, F, np.ascontiguousarray(foldRef, dtype=np.float64)))

    # 3. one-ring angle sums at sheet-interior vertices
    interior = _interior(F, Vs)
    order = np.argsort(F.ravel(), kind='stable')
    vf = (order // 3).astype(np.int64)
    vf_off = np.zeros(Vs + 1, np.int64)
    vf_off[1:] = np.cumsum(np.bincount(F.ravel(), minlength=Vs))
    triOrient = ref.coarse[faceOrig].astype(np.float64)
    sums = np.zeros(Vs)
    _angle_sums(P, F, ff, fn, vf_off, vf, interior, triOrient, sums)
    chk = ~np.isnan(sums)
    dev = sums[chk] / TWO_PI
    st['interior_vertices'] = int(chk.sum())
    st['onering_ok'] = int((np.abs(dev - 1.0) <= 1e-6).sum())
    st['onering_fold'] = int((dev < 1.0 - 1e-6).sum())          # turns less than once
    st['onering_wrap'] = int((dev > 1.0 + 1e-6).sum())          # turns more than once
    st['onering_turns_min'] = float(dev.min()) if len(dev) else 1.0
    st['onering_turns_max'] = float(dev.max()) if len(dev) else 1.0
    st['onering_bad_ids'] = np.nonzero(chk)[0][np.abs(dev - 1.0) > 1e-6][:20].tolist()

    # 4. c2f walk diagnostics (the c2f seeds only)
    if name.startswith('c2f'):
        st['walk_clamped_queries'] = int((C.clampedSteps > 0).sum())
        st['walk_far_outside_steps'] = int(C.farSteps.sum())

    st['locally_bijective'] = bool(st['coincident_exact'] == 0 and st['tri_degenerate'] == 0
                                   and st['tri_flipped'] == 0 and st['onering_fold'] == 0
                                   and st['onering_wrap'] == 0 and st['unmapped'] == 0)
    log('[bijectivity] %s: %s | samples %d (unmapped %d), coincident %d (within 1e-9 x diag %d) | '
        'triangles %d: flipped %d (in %d coarse faces), degenerate %d, with an unmapped corner %d%s | '
        'one-ring at %d sheet-interior vertices: turns once %d, fold (< 1 turn) %d, wrap (> 1 turn) %d, '
        'turns min %.4g max %.4g%s%s | coarse faces wound against the fine sheet %d (not folds) | '
        'global overlap not tested'
        % (name, 'LOCALLY BIJECTIVE' if st['locally_bijective'] else 'NOT BIJECTIVE', Vs, st['unmapped'],
           st['coincident_exact'], st['coincident_1e-9diag'], nT, st['tri_flipped'], st['coarse_faces_with_flips'],
           st['tri_degenerate'], st['tri_unmapped_corner'],
           (', folded vs coarse normal %d' % st['folded_vs_coarse_normal']) if foldRef is not None else '',
           st['interior_vertices'], st['onering_ok'], st['onering_fold'], st['onering_wrap'],
           st['onering_turns_min'], st['onering_turns_max'],
           (' (first bad: %s)' % st['onering_bad_ids'][:8]) if st['onering_bad_ids'] else '',
           (' | c2f walk: %d queries clamped, %d steps far outside'
            % (st['walk_clamped_queries'], st['walk_far_outside_steps'])) if 'walk_clamped_queries' in st else '',
           st['coarse_faces_wound_against_fine']))
    return st, ref


def main(argv=None):
    import run_relax
    from bundle_io import load_bundle_flat
    from coarse_subdiv_relax import _fold_ref
    from matstruct import load_matstruct
    a = run_relax.parse_args(argv)
    os.makedirs(a.output_dir, exist_ok=True)
    logf = open(os.path.join(a.output_dir, 'bijectivity.log'), 'w', newline='\n')
    log_util.set_log_file(logf)
    B = load_bundle_flat(a.bundle)
    ms = load_matstruct(a.matstruct_path, B.fineV, B.fineF)
    C = run_relax.build_c2f(B, a, ms)
    foldRef = _fold_ref(np.ascontiguousarray(C.S.V), np.ascontiguousarray(C.P),
                        np.ascontiguousarray(C.S.F, dtype=np.int64),
                        np.ascontiguousarray(C.S.faceOrig, dtype=np.int64), B.coarseF.shape[0])
    st, _ = bijectivity_stats(B, C, C.P, C.fineFace, C.fineBary, 'c2f seeds (%s)' % run_relax.equal_area_text(a),
                              foldRef=foldRef)
    with open(os.path.join(a.output_dir, 'bijectivity.json'), 'w') as f:
        json.dump(st, f, indent=1)
    log_util.set_log_file(None)
    logf.close()
    return 0 if st['locally_bijective'] else 1


if __name__ == '__main__':
    sys.exit(main())
