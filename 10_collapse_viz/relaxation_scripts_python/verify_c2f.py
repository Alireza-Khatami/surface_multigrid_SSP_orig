"""Checks 11_correspond_viz/c2f_query.py against the C++ coarse -> fine query.

  1. load_bundle: field by field against the strict reader (bundle_io), which
     follows the C++ writer and checks that every byte is read.
  2. query_coarse_to_fine: c2f_query's walk vs this package's port of the C++
     walk (c2f_walk), on the queries coarse_subdiv_c2f_build makes for the
     subdivided coarse mesh (same initial BC / BF / FIdx). The port itself is
     checked against the C++ output coarse_subdiv_at_fine_pos_*.obj.
  3. c2f_query.sample_face_correspondence's way of starting a query (BC in
     coarseF corner order, BF = the coarse face's corners) vs the C++ start
     (BC reordered to the FUV_pre columns of the face's last collapse).

  python verify_c2f.py --cpp_dir ../output/relaxation_experiments/explicit [--every 1]
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '11_correspond_viz'))

import c2f_query  # noqa: E402
from bundle_io import compare_with_c2f_query, load_bundle_flat  # noqa: E402
from c2f_walk import coarse_subdiv_c2f_build  # noqa: E402
from obj_io import read_obj  # noqa: E402


def positions(fineV, BC, BF):
    return (BC[:, 0:1] * fineV[BF[:, 0]] + BC[:, 1:2] * fineV[BF[:, 1]]) + BC[:, 2:3] * fineV[BF[:, 2]]


def summarize(name, P_ref, P, BF_ref, BF, F_ref, F, diag, out):
    d = np.sqrt(((P - P_ref) ** 2).sum(axis=1)) / diag
    same = (d == 0) & np.all(BF == BF_ref, axis=1) & (F == F_ref)
    out.append('%s: %d queries, %d identical (same face, corners, position), %d differ; '
               'position difference max %.3g mean %.3g (x diag), %d differ by > 1e-9 x diag, %d end on another face'
               % (name, len(d), int(same.sum()), int((~same).sum()), d.max() if len(d) else 0,
                  d.mean() if len(d) else 0, int((d > 1e-9).sum()), int((F != F_ref).sum())))
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cpp_dir', required=True)
    p.add_argument('--n_coarse_subdiv_samples', type=int, default=200000)
    p.add_argument('--every', type=int, default=1, help='check every N-th query with c2f_query (pure Python, slow)')
    p.add_argument('--report', default=None)
    a = p.parse_args()
    bundle = glob.glob(os.path.join(a.cpp_dir, 'correspondence_*.c2f'))[0]
    stem = os.path.basename(bundle)[len('correspondence_'):-4]
    out = []

    B = load_bundle_flat(bundle)
    CQ = c2f_query.load_bundle(bundle)
    bad = compare_with_c2f_query(B, CQ)
    out.append('1. c2f_query.load_bundle vs strict reader (all %d bytes): %s'
               % (os.path.getsize(bundle), 'identical' if not bad else 'DIFFER in ' + ', '.join(bad)))

    C = coarse_subdiv_c2f_build(B, a.n_coarse_subdiv_samples)
    Vc, _ = read_obj(os.path.join(a.cpp_dir, 'coarse_subdiv_at_fine_pos_' + stem + '.obj'))
    lines_same = all('%.15g' % x == '%.15g' % y for x, y in zip(Vc.ravel().tolist(), C.P.ravel().tolist()))
    out.append('   C++-exact port (c2f_walk) vs C++ coarse_subdiv_at_fine_pos_*.obj: %s'
               % ('identical (all %d vertices, 15 digits)' % len(Vc) if lines_same else 'DIFFERENT'))

    d = B.fineV.max(axis=0) - B.fineV.min(axis=0)
    diag = float(np.sqrt((d ** 2).sum()))
    sel = np.arange(0, len(C.qVert), a.every)
    qv = C.qVert[sel]

    # rebuild the C++ start of the selected queries
    BC0, BF0, F0 = C.BC0[sel], C.BF0[sel], C.FIdx0[sel]
    t = time.perf_counter()
    BC1, BF1, F1 = BC0.copy(), BF0.astype(np.int32).copy(), F0.astype(np.int32).copy()
    c2f_query.query_coarse_to_fine(CQ.decInfo, CQ.decIM, CQ.faceSheetID, BC1, BF1, F1)
    out.append('2. c2f_query.query_coarse_to_fine (%.0f s) vs C++ walk, same start:' % (time.perf_counter() - t))
    summarize('   ', positions(B.fineV, C.BC[sel], C.BF[sel]), positions(B.fineV, BC1, BF1.astype(np.int64)),
              C.BF[sel], BF1, C.FIdx[sel], F1, diag, out)

    # sample_face_correspondence start: BC in coarseF order, BF = coarse face corners
    cf = C.S.fineFace[qv]
    BC2 = C.S.fineBary[qv].copy()
    BF2 = B.vtxMap[B.coarseF[cf]].astype(np.int32)
    F2 = B.faceMap[cf].astype(np.int32)
    reordered = int(np.any(BF2 != BF0, axis=1).sum())
    t = time.perf_counter()
    c2f_query.query_coarse_to_fine(CQ.decInfo, CQ.decIM, CQ.faceSheetID, BC2, BF2, F2)
    out.append('3. sample_face_correspondence start (BC in coarseF order; %d of %d queries start with other corners '
               'or corner order than the C++) (%.0f s):' % (reordered, len(qv), time.perf_counter() - t))
    summarize('   ', positions(B.fineV, C.BC[sel], C.BF[sel]), positions(B.fineV, BC2, BF2.astype(np.int64)),
              C.BF[sel], BF2, C.FIdx[sel], F2, diag, out)

    txt = '\n'.join(out)
    print(txt)
    if a.report:
        with open(a.report, 'w') as f:
            f.write('bundle: %s\n%s\n' % (bundle, txt))


if __name__ == '__main__':
    main()
