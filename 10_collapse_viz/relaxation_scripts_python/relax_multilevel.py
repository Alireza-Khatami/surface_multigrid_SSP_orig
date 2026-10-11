"""Coarse-to-fine (multilevel) relaxation, Python-only.

The explicit / momentum relaxation is slow on large-scale imbalances (one ring per step).
This relaxes a coarser sampling first and refines the RELAXED mesh up to the final count:

  level 0       equal-area split to N / r^(L-1) samples (+ Delaunay flips), relax
  level l > 0   equal-area split to N / r^(L-1-l) samples. The longest-edge bisection splits
                the largest face first, deterministically, so level l-1's mesh is a prefix of
                level l's: the same vertices, in the same order, plus new edge midpoints.
                Old vertices start where level l-1 left them; each new vertex starts at the
                midpoint of its parent edge (the relaxed positions, in creation order) and is
                projected onto its own structure (sheet or seam / boundary curve). Then the
                Delaunay flips (on the coarse geometry, as in the single-level run) and the relaxation.
  final level   N samples (--equal_area_samples), written like run_relax.py (relaxed OBJ, .sdt,
                sliding stats, radius check, results.json, experiment_config.txt)
  every level   levels/level<l>/: the subdivision stages 00-04 as run_relax.py's relax_input/
                (coarse mesh, equal-area split, before / after the flips, subdivided mesh on the coarse
                geometry, at its c2f positions, relaxation input); 05_subdivided_relaxed (level > 0: the
                relaxed level below, subdivided, new vertices at their parents' midpoint, not yet
                projected); 06_start (the relaxation input = 04, with is_new / folded); 07_relaxed (after
                the level's iterations). 05-07 on the fine MAT, vertex role / set_id / free / fine_face /
                is_new, face coarse_face / folded

Everything per level is the existing code: run_relax.build_c2f (equal-area split, flips, c2f
walk), coarse_subdiv_relax.CoarseRelaxSession / make_relaxer (struct ids, seeds, graph, fold
reference, explicit / momentum / implicit relaxer, which takes the prolonged state as
state=...). Only the prolongation is new. Notes: md_files/relaxations_md/relaxation_multilevel_*.md.

  python relax_multilevel.py --levels 3 --level_ratio 4 --level_iters 300,100,100 \
      --output_dir <dir> <run_relax.py flags: --equal_area_samples N --delaunay_flips
      --coarse_subdiv_relax_method momentum --coarse_subdiv_relax_no_new_folds ...>

The final level's .sdt stores the c2f seeds of the final level as sub_V_seed (the positions the
samples were made at), not the prolonged start.
"""
import argparse
import copy
import dataclasses
import datetime
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import log_util  # noqa: E402
import run_relax  # noqa: E402
from bundle_io import load_bundle_flat  # noqa: E402
from coarse_subdiv_relax import CoarseRelaxSession, RelaxFailure, relaxed_obj_prefix  # noqa: E402
from matstruct import load_matstruct  # noqa: E402
from obj_io import write_obj  # noqa: E402
from ply_io import write_ply  # noqa: E402
from quality import subdiv_mesh_quality  # noqa: E402
from relax_exports import export_relax_input, relax_vertex_props  # noqa: E402
from sdt_io import export_relaxed_sdt, sliding_stats, sliding_text  # noqa: E402
from struct_ids import RELAX_CURVE, RELAX_JUNCTION, bundle_ancestors  # noqa: E402
from subdiv_mesh import SUBDIV_CARRIER_EDGE  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--levels', type=int, default=3, help='number of levels (1 = the single-level run)')
    p.add_argument('--level_ratio', type=float, default=4.0, help='sample count ratio between levels')
    p.add_argument('--level_iters', default='300,100,100',
                   help='iterations per level, coarse to fine (one value: the same on every level)')
    m, rest = p.parse_known_args(argv)
    a = run_relax.parse_args(rest)
    if a.equal_area_samples <= 0:
        raise SystemExit('relax_multilevel.py needs --equal_area_samples N (the final sample count)')
    it = [int(x) for x in m.level_iters.split(',') if x.strip()]
    if len(it) == 1:
        it = it * m.levels
    if len(it) != m.levels:
        raise SystemExit('--level_iters needs %d values' % m.levels)
    m.iters = it
    m.counts = [int(round(a.equal_area_samples / m.level_ratio ** (m.levels - 1 - l))) for l in range(m.levels)]
    return m, a, rest


def default_state(sess):
    """(X, face, edge, bary) as ExplicitRelaxer builds it from the session's seeds."""
    M = sess.M
    role = np.asarray(sess.setRole)[np.asarray(sess.setId, dtype=np.int64)]
    edge = np.full(sess.Vs, -1, dtype=np.int64)
    ce = (role == RELAX_CURVE) & (M.carrierType == SUBDIV_CARRIER_EDGE)
    edge[ce] = M.carrierIndex[ce]
    return M.V.copy(), np.asarray(M.fineFace, dtype=np.int64).copy(), edge, M.fineBary.copy(), role


def prolong(prevC, prevSess, prevRel, C, sess, nCoarseV, log):
    """Start state of a level from the relaxed previous level (see the module doc)."""
    n0 = prevC.S.V.shape[0]
    S = C.S
    if S.V.shape[0] < n0 or not np.array_equal(S.V[:n0], prevC.S.V):
        raise RelaxFailure('[relax_multilevel] level meshes are not nested (vertex positions differ)')
    if not np.array_equal(np.asarray(sess.setId)[:n0], np.asarray(prevSess.setId)):
        raise RelaxFailure('[relax_multilevel] struct sets of the shared vertices differ')
    X, face, edge, bary, role = default_state(sess)
    X[:n0], face[:n0], edge[:n0], bary[:n0] = prevRel.X, prevRel.face, prevRel.edge, prevRel.bary
    Xmid = X.copy()  # the subdivided relaxed mesh: new vertices at their parents' midpoint, not projected
    par = C.refined.splitParents
    hasTarget = np.array([len(t) > 0 for t in sess.proj.targets], dtype=bool)
    setId = np.asarray(sess.setId, dtype=np.int64)
    kept = 0
    dist = []
    for v in range(n0, S.V.shape[0]):
        a, b = par[v - nCoarseV]
        if face[v] < 0 or role[v] == RELAX_JUNCTION or not hasTarget[setId[v]]:
            kept += 1  # fixed in the relaxation anyway: keep the seed
            continue
        p = 0.5 * (X[a] + X[b])
        Xmid[v] = p
        r = sess.proj.project(setId[v], p, face[v], edge[v])
        face[v] = r[0]
        bary[v] = (r[1], r[2], r[3])
        if role[v] == RELAX_CURVE:
            edge[v] = r[4]
        X[v] = (r[5], r[6], r[7])
        dist.append(float(np.linalg.norm(X[v] - p)))
    d = np.array(dist) / sess.diag if dist else np.zeros(1)
    log('[relax_multilevel] prolongation: %d kept vertices, %d new (%d at their seed: unmapped / junction / no '
        'target); midpoint -> own structure distance mean %.3g max %.3g (x diag)'
        % (n0, S.V.shape[0] - n0, kept, d.mean(), d.max()))
    return (X, face, edge, bary), Xmid


def folded_faces(rel, X):
    """Per face: folded against the relaxer's fold reference (as relax_explicit.count_folded)."""
    if not rel.haveRef:
        return np.zeros(rel.Fi.shape[0], dtype=np.uint8)
    F = rel.Fi
    n = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
    return ((n * rel.ref).sum(axis=1) <= 0).astype(np.uint8)


def export_level(out_dir, stem, l, what, X, C, sess, rel, nOld, face, comment):
    """levels/level<l>_<what>_<stem>.ply: the level's mesh at X on the fine MAT. Vertex properties
    role, set_id, free, fine_face, is_new (made on this level; 0 on level 0); face properties
    coarse_face, folded (against the fold reference)."""
    vp = relax_vertex_props(sess, rel, face=face)
    isNew = np.zeros(sess.Vs, dtype=np.uint8)
    isNew[nOld:] = 1
    vp['is_new'] = isNew
    fo = folded_faces(rel, X)
    fp = dict(coarse_face=C.S.faceOrig.astype(np.int32), folded=fo)
    path = os.path.join(out_dir, 'levels', 'level%d' % l, '%s_%s.ply' % (what, stem))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not write_ply(path, X, sess.M.F, vp, fp, [comment]):
        raise RelaxFailure('[relax_multilevel] could not write ' + path)
    return path, int(fo.sum())


def main(argv=None):
    m, a, rest = parse_args(argv)
    cfg0 = run_relax.config_from_args(a)
    out_dir = a.output_dir
    os.makedirs(out_dir, exist_ok=True)
    stem = a.stem if a.stem is not None else run_relax.bundle_stem(a.bundle)
    out = lambda prefix, ext: os.path.join(out_dir, prefix + stem + ext)
    objPath = out(relaxed_obj_prefix(cfg0), '.obj')
    logPath = os.path.join(out_dir, 'run.log')
    logf = open(logPath, 'w', newline='\n')
    log_util.set_log_file(logf)
    log = log_util.log

    started = datetime.datetime.now()
    cmd = 'python ' + ' '.join([os.path.relpath(__file__)] + (argv if argv is not None else sys.argv[1:]))
    with open(os.path.join(out_dir, 'experiment_config.txt'), 'w', newline='\n') as f:
        f.write('experiment:   %s\n' % os.path.basename(os.path.normpath(out_dir)))
        f.write('description:  %s\n' % a.description)
        f.write('started:      %s\n' % started.strftime('%Y-%m-%d %H:%M:%S'))
        f.write('implementation: python (relaxation_scripts_python/relax_multilevel.py)\n')
        f.write('bundle:       %s\n' % a.bundle)
        f.write('matstruct:    %s\n' % a.matstruct_path)
        f.write('multilevel:   %d levels, ratio %g, samples %s, iterations %s\n'
                % (m.levels, m.level_ratio, ','.join(map(str, m.counts)), ','.join(map(str, m.iters))))
        f.write('equal-area refinement: %s (per level: split to that level\'s sample count)\n'
                % run_relax.equal_area_text(a))
        f.write('relaxation:   method %s, lambda %g, tol %g x diag, %s projection, %s graph, %s weights, '
                'no-new-folds %s, per coarse face %s\n'
                % (cfg0.method, cfg0.explicitLambda, cfg0.explicitTol, 'global' if cfg0.explicitGlobalProj else 'local',
                   'directed' if cfg0.explicitDirected and not cfg0.jointPass else 'symmetric', cfg0.weights,
                   'on' if cfg0.noNewFolds else 'off', 'on' if cfg0.perCoarseFace else 'off'))
        if cfg0.method == 'momentum':
            f.write('method params: momentum beta %g, adaptive restart %s\n' % (cfg0.momentumBeta, cfg0.momentumRestart))
        f.write('output:       %s , log %s\n' % (out_dir, logPath))
        f.write('command:      %s\n' % cmd)

    t0 = time.perf_counter()
    status = 0
    try:
        B = load_bundle_flat(a.bundle)
        ms = load_matstruct(a.matstruct_path, B.fineV, B.fineF)
        anc = bundle_ancestors(B)
        nCoarseV = B.coarseV.shape[0]
        levels = []
        prev = None
        relaxSeconds = 0.0
        for l, (n, iters) in enumerate(zip(m.counts, m.iters)):
            last = l == m.levels - 1
            log('[relax_multilevel] ===== level %d of %d: %d samples, %d iterations' % (l, m.levels, n, iters))
            tl = time.perf_counter()
            al = copy.copy(a)
            al.equal_area_samples = n
            C = run_relax.build_c2f(B, al, ms)
            sess = CoarseRelaxSession(B, C, ms, log, anc)
            state, Xmid = (None, None) if prev is None else prolong(*prev, C, sess, nCoarseV, log)
            cfg = dataclasses.replace(cfg0, explicitMaxIter=iters)
            if not last:
                cfg.snapshotIters = ()
            rel = sess.make_relaxer(cfg, objPath if last else '', state=state)
            if last and not a.no_input_ply:
                export_relax_input(os.path.join(out_dir, 'relax_input'), '_' + stem, B, C, sess, rel,
                                   'experiment %s (multilevel, final level)' % os.path.basename(os.path.normpath(out_dir)))
                # the final level's c2f seeds (the .sdt's sub_V_seed; 04_relax_input is the prolonged start)
                write_ply(os.path.join(out_dir, 'relax_input', '04s_c2f_seeds_%s.ply' % stem), sess.Vseed, sess.M.F,
                          dict(fine_face=np.asarray(sess.M.fineFace).astype(np.int32)),
                          comments=('multilevel: c2f seeds of the final level, snapped onto their own structure',))
            nOld = 0 if prev is None else prev[0].S.V.shape[0]
            lvDir = os.path.join(out_dir, 'levels', 'level%d' % l)
            # the level's subdivision stages (00-04, as run_relax.py's relax_input/): coarse mesh, equal-area
            # split, before / after the flips, subdivided mesh, at its c2f positions, relaxation input
            export_relax_input(lvDir, '_' + stem, B, C, sess, rel, 'multilevel level %d (%d samples)' % (l, sess.Vs))
            if Xmid is not None:
                export_level(out_dir, stem, l, '05_subdivided_relaxed', Xmid, C, sess, rel, nOld, rel.face.copy(),
                             'multilevel level %d: level %d relaxed, subdivided: new vertices at the midpoint of their '
                             'parents, before the projection onto their structure' % (l, l - 1))
            pStart, fStart = export_level(out_dir, stem, l, '06_start', rel.X.copy(), C, sess, rel, nOld, rel.face.copy(),
                                          'multilevel level %d (%d samples): start of the relaxation (%s)'
                                          % (l, sess.Vs, 'c2f seeds' if l == 0 else 'prolonged from level %d' % (l - 1)))
            if fStart != rel.R.foldedSeed:
                raise RelaxFailure('[relax_multilevel] folded faces in the export %d != relaxer %d' % (fStart, rel.R.foldedSeed))
            qSeed = subdiv_mesh_quality(sess.Vseed, sess.M.F)
            qStart = subdiv_mesh_quality(rel.X, sess.M.F)
            foldedSeed = rel.count_folded(sess.Vseed)
            tr = time.perf_counter()
            rel.run()
            R = rel.finish()
            dt = time.perf_counter() - tr
            relaxSeconds += dt
            M = rel.M
            qEnd = subdiv_mesh_quality(M.V, M.F, sess.Vseed)
            if R.seedOffStructure or R.fixedMoved or R.posMismatch or R.badBary or R.offStructure:
                raise RelaxFailure('[relax_multilevel] relaxation consistency checks failed on level %d' % l)
            info = dict(level=l, samples=int(sess.Vs), faces=int(M.F.shape[0]), iterations=int(R.itersSheet),
                        relaxSeconds=dt, levelSeconds=time.perf_counter() - tl, foldedSeed=int(foldedSeed),
                        foldedStart=int(R.foldedSeed), foldedEnd=int(R.foldedResult), heldBack=int(R.foldReverts),
                        quality_seed=dataclasses.asdict(qSeed), quality_start=dataclasses.asdict(qStart),
                        quality_end=dataclasses.asdict(qEnd))
            levels.append(info)
            log('[relax_multilevel] level %d: %d samples, %d iterations in %.1f s | edge CV seed %.4f, start %.4f '
                '-> %.4f | min angle p1 %.2f -> %.2f, p5 %.2f -> %.2f, median %.2f -> %.2f | folded seed %d, start '
                '%d -> %d | degenerate %d'
                % (l, sess.Vs, R.itersSheet, dt, qSeed.edgeCV, qStart.edgeCV, qEnd.edgeCV, qStart.p1, qEnd.p1,
                   qStart.p5, qEnd.p5, qStart.median, qEnd.median, foldedSeed, R.foldedSeed, R.foldedResult,
                   qEnd.degenerate))
            pEnd, fEnd = export_level(out_dir, stem, l, '07_relaxed', M.V, C, sess, rel, nOld, M.fineFace,
                                      'multilevel level %d (%d samples): after %d iterations' % (l, sess.Vs, R.itersSheet))
            if fEnd != R.foldedResult:
                raise RelaxFailure('[relax_multilevel] folded faces in the export %d != relaxer %d' % (fEnd, R.foldedResult))
            log('[relax_multilevel] level %d meshes -> %s' % (l, lvDir))
            info['plys'] = [pStart, pEnd]
            prev = (C, sess, rel)

        # ---- the final level, written as run_relax.py does
        R.foldedSeed = levels[-1]['foldedSeed']  # vs the c2f seeds of the final level (comparable with single-level runs)
        q0 = subdiv_mesh_quality(sess.Vseed, sess.M.F)
        q1 = qEnd
        if sess.Vs <= a.subdiv_obj_max_verts:
            write_obj(objPath, M.V, M.F)
            log('[relax_multilevel] relaxed mesh -> %s' % objPath)
        sdtPath = out('coarse_subdiv_relaxed_', '.sdt')
        nb = export_relaxed_sdt(sdtPath, B, C, sess, M, a.equal_area_samples)
        log('[relax_multilevel] relaxed coarse <-> fine correspondence -> %s (%.1f MB)' % (sdtPath, nb / 1048576.0))
        slide = sliding_stats(B, sess, M)
        log('[relax_multilevel] ' + sliding_text(slide))
        radii = None
        if a.surface_mesh:
            from validate_radii import validate
            radii = validate(sdtPath, a.matstruct_path, a.surface_mesh, B.fineF, a.radii_tol_touch)
        total = time.perf_counter() - t0
        result = [
            '[relax_multilevel] %d levels, samples %s, iterations %s; relaxation %.1f s, total %.1f s'
            % (m.levels, ','.join(str(x['samples']) for x in levels), ','.join(str(x['iterations']) for x in levels),
               relaxSeconds, total),
            'quality c2f seeds -> after: edge CV %.4f -> %.4f | min angle p1 %.3f -> %.3f, p5 %.3f -> %.3f, median '
            '%.3f -> %.3f deg | degenerate %d -> %d | folded %d -> %d'
            % (q0.edgeCV, q1.edgeCV, q0.p1, q1.p1, q0.p5, q1.p5, q0.median, q1.median, q0.degenerate,
               q1.degenerate, R.foldedSeed, R.foldedResult),
            sliding_text(slide)]
        res = dict(folder=os.path.basename(os.path.normpath(out_dir)), config=dataclasses.asdict(cfg0),
                   equal_area=run_relax.equal_area_text(a), n_subdiv_verts=int(sess.Vs), flips=C.flipStats,
                   multilevel=dict(levels=m.levels, ratio=m.level_ratio, counts=m.counts, iters=m.iters,
                                   relaxSeconds=relaxSeconds, perLevel=levels),
                   sliding=slide, radii=radii, n_subdiv_faces=int(M.F.shape[0]),
                   report=dataclasses.asdict(R), quality_before=dataclasses.asdict(q0),
                   quality_after=dataclasses.asdict(q1), seconds=total, relaxed_obj=objPath)
        with open(os.path.join(out_dir, 'results.json'), 'w') as f:
            json.dump(res, f, indent=1, default=float)
    except Exception as e:  # noqa: BLE001
        log('[relax_multilevel] FAILED: %s' % e)
        result = ['FAILED: %s' % e]
        status = 1
    with open(os.path.join(out_dir, 'experiment_config.txt'), 'a', newline='\n') as f:
        f.write('finished:     %s (exit code %d, %.1f s)\n'
                % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), status, time.perf_counter() - t0))
        f.write('result lines:\n')
        for s in result:
            f.write('  ' + s + '\n')
    log_util.set_log_file(None)
    logf.close()
    return status


if __name__ == '__main__':
    sys.exit(main())
