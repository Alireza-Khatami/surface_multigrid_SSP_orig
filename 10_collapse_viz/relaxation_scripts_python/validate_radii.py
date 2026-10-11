"""Validates the radii interpolated at the relaxed samples of a coarse_subdiv_relaxed_<stem>.sdt.

The samples lie on the medial axis (the fine MAT). Every sample gets the radius interpolated
from the fine MAT's vertex radii (the "v x y z r" lines of the .ma_struct) with its stored
(fine_face, fine_bary): r = w0 r_A + w1 r_B + w2 r_C. Its sphere (centre = sample position on
the MAT, radius r) is then checked against the SURFACE mesh the MAT was computed from:

  1. touches: the distance d from the centre to the surface equals r within tol_touch
     (d < r - tol: the sphere cuts into the surface; d > r + tol: it does not reach it)
  2. two contacts: the sphere touches the surface at two distinct points (surface points
     within r + tol_touch of the centre, two of them at least sep_abs apart). Exempt: spheres
     with r <= tol_touch, the ends of the medial axis on the surface, where the two contacts
     merge into one (counted and logged, but they must still touch)
  3. inside the surface's bounding box: centre +- r within the box, up to tol_bbox (default
     = tol_touch: a sphere that cuts into the surface by e can stick out of the box by e)

Tolerances are in units of the surface's bounding-box diagonal. The MAT's own vertices
are checked the same way first (--check_mat_vertices): the defaults are set so that the
source MAT passes; an interpolated sphere can't be held to a stricter test than the
vertices it comes from.

Samples with fine_face < 0 (unmapped, see md_files/coarse_subdiv_relaxed_sdt.md) are
skipped and counted. Any failure is logged (per check: count, worst samples) and the
script raises RuntimeError (exit code 1).

  python validate_radii.py --sdt <...>.sdt --matstruct_path <...>.ma_struct --surface <surface>.obj
         [--fine_obj <MAT .obj>] [--tol_touch 1e-3] [--tol_bbox <= tol_touch>] [--sep_abs 1e-3]
"""
import argparse
import os
import sys

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'subdiv_sample_tracker', 'tools'))

from log_util import log  # noqa: E402
from obj_io import read_obj  # noqa: E402
from read_sdt import read_sdt  # noqa: E402


def read_ma_struct_vertices(path):
    """(positions n x 3, radii n) from the 'v x y z r' lines."""
    with open(path) as f:
        nv = int(f.readline().split()[0])
        P = np.empty((nv, 3))
        r = np.empty(nv)
        for i in range(nv):
            t = f.readline().split()
            if t[0] != 'v':
                raise RuntimeError('[validate_radii] %s: bad vertex line %d' % (path, i))
            P[i] = (float(t[1]), float(t[2]), float(t[3]))
            r[i] = float(t[4])
    return P, r


@njit(cache=True)
def _closest_on_tri(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
    """Closest point to p on triangle abc (Ericson, Real-Time Collision Detection 5.1.5)."""
    abx, aby, abz = bx - ax, by - ay, bz - az
    acx, acy, acz = cx - ax, cy - ay, cz - az
    apx, apy, apz = px - ax, py - ay, pz - az
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return ax, ay, az
    bpx, bpy, bpz = px - bx, py - by, pz - bz
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return bx, by, bz
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return ax + v * abx, ay + v * aby, az + v * abz
    cpx, cpy, cpz = px - cx, py - cy, pz - cz
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return cx, cy, cz
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return ax + w * acx, ay + w * acy, az + w * acz
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return bx + w * (cx - bx), by + w * (cy - by), bz + w * (cz - bz)
    den = 1.0 / (va + vb + vc)
    v = vb * den
    w = vc * den
    return ax + abx * v + acx * w, ay + aby * v + acy * w, az + abz * v + acz * w


@njit(parallel=True, cache=True)
def _sphere_contacts(C, R, SV, SF, tol, sepRel, sepAbs, d1, c1, sepMax):
    """Per sphere (C[i], R[i]): d1 = distance to the surface, c1 = closest point, sepMax =
    largest distance from c1 to another surface point within R + tol of the centre (the
    closest point of each triangle that comes that near)."""
    n, m = C.shape[0], SF.shape[0]
    for i in prange(n):
        px, py, pz = C[i, 0], C[i, 1], C[i, 2]
        best = 1e300
        bx = by = bz = 0.0
        for f in range(m):
            a, b, c = SF[f, 0], SF[f, 1], SF[f, 2]
            qx, qy, qz = _closest_on_tri(px, py, pz, SV[a, 0], SV[a, 1], SV[a, 2], SV[b, 0], SV[b, 1], SV[b, 2],
                                         SV[c, 0], SV[c, 1], SV[c, 2])
            dd = (qx - px) ** 2 + (qy - py) ** 2 + (qz - pz) ** 2
            if dd < best:
                best, bx, by, bz = dd, qx, qy, qz
        d1[i] = np.sqrt(best)
        c1[i, 0], c1[i, 1], c1[i, 2] = bx, by, bz
        reach = R[i] + tol
        s = 0.0
        for f in range(m):
            a, b, c = SF[f, 0], SF[f, 1], SF[f, 2]
            qx, qy, qz = _closest_on_tri(px, py, pz, SV[a, 0], SV[a, 1], SV[a, 2], SV[b, 0], SV[b, 1], SV[b, 2],
                                         SV[c, 0], SV[c, 1], SV[c, 2])
            if (qx - px) ** 2 + (qy - py) ** 2 + (qz - pz) ** 2 <= reach * reach:
                e = np.sqrt((qx - bx) ** 2 + (qy - by) ** 2 + (qz - bz) ** 2)
                if e > s:
                    s = e
        sepMax[i] = s


def check_spheres(name, C, R, SV, SF, tol_touch, tol_bbox, sep_rel, sep_abs, logf=log):
    """Runs the three checks on spheres (C, R). Returns (ok, dict of per-check stats)."""
    diag = float(np.linalg.norm(SV.max(0) - SV.min(0)))
    tT, tB, sA = tol_touch * diag, (tol_touch if tol_bbox is None else tol_bbox) * diag, sep_abs * diag
    n = len(R)
    d1 = np.zeros(n)
    c1 = np.zeros((n, 3))
    sepMax = np.zeros(n)
    _sphere_contacts(np.ascontiguousarray(C, dtype=np.float64), np.ascontiguousarray(R, dtype=np.float64),
                     np.ascontiguousarray(SV, dtype=np.float64), np.ascontiguousarray(SF, dtype=np.int64),
                     tT, sep_rel, sA, d1, c1, sepMax)
    e = d1 - R
    sep = np.maximum(sep_rel * R, sA)
    end = R <= tT                                       # medial-axis ends: contacts merge
    lo, hi = SV.min(0), SV.max(0)
    out = np.maximum(np.maximum(lo - (C - R[:, None]), (C + R[:, None]) - hi), 0.0).max(1)
    fails = dict(cuts_surface=e < -tT, does_not_reach=e > tT, one_contact=(sepMax < sep) & ~end,
                 leaves_bbox=out > tB)
    st = dict(n=n, diag=diag, d_minus_r=dict(min=float(e.min() / diag), max=float(e.max() / diag),
                                             p1=float(np.percentile(e, 1) / diag), p99=float(np.percentile(e, 99) / diag)),
              bbox_excess_max=float(out.max() / diag), medial_axis_ends=int(end.sum()),
              min_contact_sep=float(sepMax[~end].min() / diag) if (~end).any() else 0.0,
              fails={k: int(v.sum()) for k, v in fails.items()})
    ok = not any(v.any() for v in fails.values())
    logf('[validate_radii] %s: %d spheres | d - r (x diag): min %.3g p1 %.3g p99 %.3g max %.3g | bbox excess max '
         '%.3g x diag | smallest contact separation %.3g x diag (%d medial-axis ends, r <= tol, exempt) | '
         'tol_touch %.3g, tol_bbox %.3g, sep_abs %.3g x diag'
         % (name, n, st['d_minus_r']['min'], st['d_minus_r']['p1'], st['d_minus_r']['p99'], st['d_minus_r']['max'],
            st['bbox_excess_max'], st['min_contact_sep'], st['medial_axis_ends'], tol_touch,
            tol_touch if tol_bbox is None else tol_bbox, sep_abs))
    for k, v in fails.items():
        if v.any():
            idx = np.nonzero(v)[0]
            key = {'cuts_surface': e, 'does_not_reach': -e, 'one_contact': sepMax - sep, 'leaves_bbox': -out}[k]
            worst = idx[np.argsort(key[idx])[:5]]
            logf('[validate_radii] %s FAIL %s: %d of %d (%.3f%%); worst: %s'
                 % (name, k, len(idx), n, 100.0 * len(idx) / n,
                    '; '.join('#%d r %.4g d %.4g contacts sep %.4g bbox +%.3g' % (j, R[j], d1[j], sepMax[j], out[j])
                              for j in worst)))
    st['ok'] = ok
    return ok, st


def validate(sdt_path, matstruct_path, surface_path, fineF, tol_touch=1e-3, tol_bbox=None, sep_rel=0.0,
             sep_abs=1e-3, check_mat_vertices=False, logf=log):
    """Interpolated radii of an .sdt against the surface; raises RuntimeError on any failure.
    Returns the stats of the samples' check."""
    H, A = read_sdt(sdt_path)
    PV, rv = read_ma_struct_vertices(matstruct_path)
    FF = np.asarray(fineF, dtype=np.int64)
    if len(PV) != H['n_fine_verts'] or len(FF) != H['n_fine_faces']:
        raise RuntimeError('[validate_radii] the .ma_struct / fine faces do not match the .sdt (%d / %d vs %d / %d)'
                           % (len(PV), len(FF), H['n_fine_verts'], H['n_fine_faces']))
    SV, SF = read_obj(surface_path)
    ff, fb = A['fine_face'].astype(np.int64), A['fine_bary']
    on = ff >= 0
    logf('[validate_radii] %s: %d samples, %d unmapped (fine_face -1) skipped; surface %s: %d vertices, %d faces'
         % (sdt_path, len(ff), int((~on).sum()), surface_path, len(SV), len(SF)))
    if (fb[on] < 0).any():
        raise RuntimeError('[validate_radii] negative barycentric weights in the .sdt')
    R = (fb[on] * rv[FF[ff[on]]]).sum(1)              # interpolated radius
    C = A['sub_V'][on]
    okAll = True
    if check_mat_vertices:
        okV, _ = check_spheres('MAT vertices', PV, rv, SV, SF, tol_touch, tol_bbox, sep_rel, sep_abs, logf)
        okAll &= okV
    ok, st = check_spheres('relaxed samples', C, R, SV, SF, tol_touch, tol_bbox, sep_rel, sep_abs, logf)
    okAll &= ok
    if not okAll:
        raise RuntimeError('[validate_radii] radius validation FAILED (see the log lines above)')
    logf('[validate_radii] radius validation PASSED')
    return st


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sdt', required=True)
    p.add_argument('--matstruct_path', required=True, help='the fine MAT .ma_struct (vertex radii)')
    p.add_argument('--surface', required=True, help='the surface mesh the MAT was computed from (.obj)')
    p.add_argument('--fine_obj', default=None, help='fine MAT .obj (for the faces); default: the bundle next to the .sdt')
    p.add_argument('--bundle', default=None)
    p.add_argument('--tol_touch', type=float, default=1e-3, help='|d - r| allowed, x surface bbox diagonal')
    p.add_argument('--tol_bbox', type=float, default=None, help='sphere beyond the surface bbox allowed, x diagonal '
                   '(default: tol_touch)')
    p.add_argument('--sep_rel', type=float, default=0.0, help='second contact at least sep_rel * r from the first')
    p.add_argument('--sep_abs', type=float, default=1e-3, help='... and at least sep_abs x diagonal (distinct points)')
    p.add_argument('--check_mat_vertices', action='store_true', help='also check the MAT vertices themselves')
    a = p.parse_args(argv)

    if a.fine_obj:
        _, FF = read_obj(a.fine_obj)
    else:
        import glob
        from bundle_io import load_bundle_flat
        bpath = a.bundle or glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.sdt))),
                                                   'correspondence_*.c2f'))[0]
        FF = load_bundle_flat(bpath).fineF
    validate(a.sdt, a.matstruct_path, a.surface, FF, a.tol_touch, a.tol_bbox, a.sep_rel, a.sep_abs,
             a.check_mat_vertices)
    return 0


if __name__ == '__main__':
    sys.exit(main())
