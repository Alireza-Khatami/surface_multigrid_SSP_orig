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

  python test_viewer.py [--steps 20]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import relax_viewer  # noqa: E402
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
    a = p.parse_args()
    v = relax_viewer.main(['--mock', '--coarse_subdiv_relax_no_new_folds'])
    sess, cfg = v.sess, v.cfg
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
    print('concave mask: %d concave fine edges, %d of %d points marked (k=%d, bend > %g deg)'
          % (len(v.concEdges[1]), int(m.sum()), v.Vs, v.concK, v.concAngle))
    v.clear_selection()
    for t in v.treeLabel:
        v.set_tree(t, True)
    for t in v.treeLabel:
        v.set_tree(t, False)
    errs = sum(v.tree_errors(t) for t in v.treeLabel)
    print('BVH inspector: %d trees, %d structure errors' % (len(v.treeLabel), errs))
    ok &= errs == 0
    print('TEST VIEWER: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
