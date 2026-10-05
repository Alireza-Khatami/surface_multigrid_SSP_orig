"""Headless test of the step viewer (polyscope mock backend, no window).

1. Recording checkpoints does not change the relaxation: N viewer steps (trace on)
   give bit-identical positions / faces / barycentrics to N steps of a relaxer
   without trace, for several configurations, including a configuration switch
   in the middle of the run.
2. The checkpoint is consistent with the step: X1 == the relaxer state, held-back
   points kept X0, the others are at P, P == interpolation of (Pface, Pbary),
   Y == x + lambda (mean - x) for the free points.
3. Every viewer action runs: select, each display option, clear selection,
   concave mask, step meshes, BVH inspector (show all / hide all / tree check).
4. PLY export: each mesh button writes the relaxation's own arrays (read back and
   compared exactly), and the stages before the relaxation are written.

  python test_viewer.py [--steps 20]
"""
import argparse
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import relax_viewer  # noqa: E402
from ply_io import read_ply  # noqa: E402
from projector import interp3  # noqa: E402


def same_state(a, b):
    return (np.array_equal(a.X, b.X) and np.array_equal(a.face, b.face) and np.array_equal(a.edge, b.edge)
            and np.array_equal(a.bary, b.bary))


def check_trace(v):
    t, r = v.trace, v.rel
    bad = []
    if not np.array_equal(t.X1, r.X):
        bad.append('X1 != state')
    h = t.held.astype(bool)
    if not np.array_equal(t.X1[h], t.X0[h]):
        bad.append('held point moved')
    if not np.array_equal(t.X1[~h], t.P[~h]):
        bad.append('not-held point not at P')
    lst = r.lst
    VO, FO = r.proj.VO, r.proj.FO
    P = np.array([interp3(VO, FO, t.Pface[i], t.Pbary[i, 0], t.Pbary[i, 1], t.Pbary[i, 2]) for i in lst[:5000]])
    if not np.array_equal(P, t.P[lst[:5000]]):
        bad.append('P != interp(Pface, Pbary)')
    # Y = x + lam (mean - x), same arithmetic as the kernel
    ro, co, X0, lam = r.rowOffs, r.cols, t.X0, r.opt.lam
    for i in lst[:2000].tolist():
        m = np.zeros(3)
        ws = 0.0
        for q in range(ro[i], ro[i + 1]):
            w = 1.0 if r.uniformRow[i] else r.wts[q]
            m = m + w * X0[co[q]]
            ws += w
        m = m / ws
        if not np.array_equal(X0[i] + lam * (m - X0[i]), t.Y[i]):
            bad.append('Y mismatch at %d' % i)
            break
    return bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--export_dir', default=None, help='default: a temporary folder')
    a = p.parse_args()
    exportDir = a.export_dir or tempfile.mkdtemp(prefix='relax_viewer_export_')
    # segment 1 local projection (the default is global), segment 2 switches to global
    v = relax_viewer.main(['--mock', '--coarse_subdiv_relax_no_new_folds', '--explicit_local_proj',
                           '--export_dir', exportDir])
    sess, cfg = v.sess, v.cfg
    camDefault = v.cam['mode']
    ok = True

    def reference(cfg_, state, it0, n):
        r = sess.make_relaxer(cfg_, '', state=state, it0=it0)
        for _ in range(n):
            r.step()
        return r

    # 1 + 2: segment 1 (no new folds, local), then switch to directed + cotan + global projection
    v.select(1234)
    st0 = None
    ref = reference(cfg, st0, 0, a.steps)
    for _ in range(a.steps):
        v.step(1)
        bad = check_trace(v)
        if bad:
            print('trace check failed:', bad)
            ok = False
            break
    print('segment 1 (%s): trace on == trace off: %s' % (v.cfg_text(), same_state(v.rel, ref)))
    ok &= same_state(v.rel, ref)
    st = (v.rel.X.copy(), v.rel.face.copy(), v.rel.edge.copy(), v.rel.bary.copy())
    it0 = v.iters
    cfg.explicitDirected = True
    cfg.weights = 'cotan'
    cfg.explicitGlobalProj = True
    v.apply_config()
    ref = reference(cfg, st, it0, a.steps)
    for _ in range(a.steps):
        v.step(1)
    bad = check_trace(v)
    print('segment 2 (%s): trace on == trace off: %s; trace checks: %s'
          % (v.cfg_text(), same_state(v.rel, ref), bad or 'ok'))
    ok &= same_state(v.rel, ref) and not bad
    info = v.selection_info()
    print('selected point: winTree %d leaf %d, visited nodes %s'
          % (info['winTree'], info['winLeaf'], None if info['visited'] is None else len(info['visited'])))
    print(v.info_text(info))

    # 3: viewer actions
    for key in v.show:
        v.show[key] = not v.show[key]
        v.update_selection()
        v.show[key] = not v.show[key]
    for at in range(3):
        v.selAt = at
        v.update_selection()
    for key in v.meshOn:
        v.meshOn[key] = True
    v.refresh()
    v.colorMode = 1
    v.apply_colors()
    m = v.concave_mask()
    P = v.concParts
    print('concave mask: %d concave corners (of %d sheet border corners), %d seam / boundary edges at them, '
          '%d of %d input points marked (radius %g subdiv edges, k=%d, interior angle > 180 + %g deg)'
          % (len(P.corners), P.nBorderCorners, len(P.edges), int(m.sum()), v.Vs, v.concRadius, v.concK,
             v.concAngle))
    m2 = v.concave_mask()
    v.step(1)
    ok &= m2 is m and np.array_equal(v.concave_mask(), m)  # fixed on the input, not on the moving points
    cn = v.ps.get_curve_network('concave seam / boundary parts')
    print('curve network nodes: %d (only the vertices of the edges; fine mesh has %d)' % (cn.n_nodes(), v.B.fineV.shape[0]))
    v.clear_selection()
    for t in v.treeLabel:
        v.set_tree(t, True)
    for t in v.treeLabel:
        v.set_tree(t, False)
    errs = sum(v.tree_errors(t) for t in v.treeLabel)
    print('BVH inspector: %d trees, %d structure errors' % (len(v.treeLabel), errs))
    ok &= errs == 0

    # 4: PLY exports, read back
    want = dict(input=v.sess.Vseed, committed=v.rel.X, step=v.trace.Y, proj=v.trace.P)
    for key, _ in v.EXPORTS:
        path = v.export_mesh(key)
        V, F, vp, fp = read_ply(path)
        same = (np.array_equal(V, want[key]) and np.array_equal(F, v.F)
                and np.array_equal(fp['coarse_face'], v.C.S.faceOrig))
        if key == 'proj':
            same &= np.array_equal(vp['fine_face'], v.trace.Pface)
        if key in ('committed', 'step', 'proj'):
            same &= np.array_equal(vp['held_back'], v.trace.held.astype(np.uint8))
        print('export %-9s -> %s: %s' % (key, os.path.basename(path), 'exact' if same else 'MISMATCH'))
        ok &= bool(same)
    paths = v.export_stages()
    print('export stages: %s' % ', '.join(os.path.basename(x) for x in paths))
    V, F, vp, _ = read_ply([x for x in paths if '04_relax_input' in x][0])
    same = np.array_equal(V, v.sess.Vseed) and np.array_equal(vp['free'], v.input0.isFree)
    V3 = read_ply([x for x in paths if '03_subdiv_at_fine' in x][0])[0]
    same &= np.array_equal(V3, v.C.P)
    print('stages: relax input == seeds, at-fine == c2f positions: %s (folder %s)' % (same, exportDir))
    ok &= bool(same) and len(paths) >= 4
    # 4b: Reset, then a different configuration from iteration 0 == a fresh run of it from the seeds
    cfg.explicitDirected = False
    cfg.weights = 'meanvalue'
    cfg.explicitGlobalProj = False
    cfg.noNewFolds = False
    v.reset()
    okr = v.iters == 0 and np.array_equal(v.rel.X, v.sess.Vseed)
    ref = reference(cfg, None, 0, a.steps)
    for _ in range(a.steps):
        v.step(1)
    same = same_state(v.rel, ref)
    p2 = v.export_mesh('committed')
    okr &= same and os.path.basename(p2).startswith('run01_') and check_trace(v) == []
    print('reset -> run %d (%s): back at the seeds, %d steps == fresh run: %s; export %s'
          % (v.runId, v.cfg_text(), a.steps, same, os.path.basename(p2)))
    ok &= okr

    # 5: camera speed control (distance readings simulated; the mock backend has no view)
    L0 = relax_viewer.LEN0[0]
    rad0 = v.pc.get_radius()
    for mode in (0, 1):
        v.cam['mode'] = mode
        for d in (2.0 * L0, 0.1 * L0, 1e-6 * L0):
            v.camera_distance = lambda psim=None, d=d: d
            for _ in range(60):
                v.camera_speed()
            want = min(max(v.cam['k'] * d, v.cam['minf'] * L0), v.cam['maxf'] * L0)
            okc = abs(v.ps.get_length_scale() - want) <= 1e-6 * L0
            print('camera mode %d, d = %.3g x L0: length scale %.4g x L0 (want %.4g): %s'
                  % (mode, d / L0, v.ps.get_length_scale() / L0, want / L0, 'ok' if okc else 'WRONG'))
            ok &= okc
    del v.camera_distance
    # mode C from the real view: camera at a known position, distance to the view centre
    R = np.linalg.qr(np.random.default_rng(1).random((3, 3)))[0]
    cpos = v.B.fineV.mean(axis=0) + np.array([0.3, 0.2, 0.1]) * L0
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = -R @ cpos
    v.ps.set_camera_view_matrix(M)
    ctr = v._view()[1]  # view centre (2.5+) or scene box centre (older)
    v.cam['mode'] = 0
    d = v.camera_distance()
    okd = d is not None and abs(d - np.linalg.norm(ctr - cpos)) <= 1e-6 * L0
    print('camera mode C from the view: d = %s (want %.6g): %s' % (d, np.linalg.norm(ctr - cpos), okd))
    ok &= okd
    v.camera_speed()
    ok &= not v.cam.get('error')
    v.cam['on'] = False
    for _ in range(60):
        v.camera_speed()
    ok &= abs(v.ps.get_length_scale() - L0) <= 1e-6 * L0
    v.select(1234)  # selection markers / vectors / curves registered at a small length scale
    v.update_selection()
    okr = v.pc.get_radius() == rad0
    v.set_point_radius(0.0003)
    okr &= camDefault == 1  # default: depth under the cursor
    v.fineOn = False
    v.ps.get_surface_mesh('fine MAT').set_enabled(False)
    okr &= not v.ps.get_surface_mesh('fine MAT').is_enabled()
    v.fineOn = True
    v.ps.get_surface_mesh('fine MAT').set_enabled(True)
    okr &= abs(v.pc.get_radius() - 0.0003 * L0) <= 1e-9 * L0
    print('point radius unchanged by the length scale: %s' % okr)
    ok &= okr
    # 6: real GUI frames (mock backend): every panel's imgui / polyscope calls run
    v.cam['on'] = True
    v.colorMode = 1
    v.select(1234)
    for _ in range(3):
        v.ps.frame_tick()
    okg = not v.cam.get('error')
    print('GUI frames (all panels drawn): %s' % ('ok' if okg else 'camera error: ' + v.cam['error']))
    ok &= okg
    print('TEST VIEWER: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
