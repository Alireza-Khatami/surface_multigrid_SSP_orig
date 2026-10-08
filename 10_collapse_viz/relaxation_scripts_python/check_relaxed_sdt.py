"""Checks a coarse_subdiv_relaxed_<stem>.sdt written by run_relax.py (sdt_io.py).

Reads it with the existing reader (subdiv_sample_tracker/tools/read_sdt.py, unchanged) and
checks it against the bundle, the .ma_struct and the run's other outputs:

  1. layout: magic, version 2 / relaxed 1, header size, 15 arrays, 8-byte aligned offsets,
     the last array ends at the end of the file; same header layout as a C++ .sdt
  2. counts: fine / coarse vertex and face counts = the bundle's; array lengths
  3. fine side: (fine_face, fine_bary) interpolated on the fine mesh = sub_V; barycentrics
     >= 0 and sum 1
  4. coarse side: (coarse_face, coarse_bary) interpolated on the coarse mesh = the
     sample's position on the coarse geometry (relax_input/02_subdiv PLY)
  5. sub_V_seed = the relaxation input (relax_input/04_relax_input PLY); sub_V = the relaxed
     OBJ (15 digits); sub_F = the relaxed OBJ's faces
  6. carriers: FACE -> a fine face; EDGE -> a fine edge (orig_edges row) whose ends are fine
     vertices on one seam / boundary of the sample's struct set
  7. palette: offsets increasing, end = n_palette_ids; type mask = the .ma_struct types of
     the ids
  8. own structure: every sheet sample's fine face is in one of its sheets; every curve
     sample lies on a fine edge of one of its seams / boundaries

  python check_relaxed_sdt.py --run_dir <run_relax output dir> [--bundle ...] [--matstruct_path ...]
"""
import argparse
import glob
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'subdiv_sample_tracker', 'tools'))

import run_relax  # noqa: E402
from bundle_io import load_bundle_flat  # noqa: E402
from matstruct import load_matstruct, matstruct_edge_key  # noqa: E402
from obj_io import read_obj  # noqa: E402
from ply_io import read_ply  # noqa: E402
from read_sdt import read_sdt  # noqa: E402  (the consumer-side reader, unchanged)


def interp(V, F, face, bary):
    T = F[face]
    return (bary[:, 0:1] * V[T[:, 0], :3] + bary[:, 1:2] * V[T[:, 1], :3]) + bary[:, 2:3] * V[T[:, 2], :3]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', required=True)
    p.add_argument('--bundle', default=run_relax.default_bundle())
    p.add_argument('--matstruct_path', default=run_relax.DEFAULT_MS)
    a = p.parse_args(argv)
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print('%-4s %s%s' % ('ok' if cond else 'FAIL', name, (' | ' + detail) if detail else ''))

    path = glob.glob(os.path.join(a.run_dir, 'coarse_subdiv_relaxed_*.sdt'))[0]
    H, A = read_sdt(path)
    buf = open(path, 'rb').read()
    B = load_bundle_flat(a.bundle)
    ms = load_matstruct(a.matstruct_path, B.fineV, B.fineF)
    diag = float(np.linalg.norm(B.fineV.max(axis=0) - B.fineV.min(axis=0)))

    # 1. layout
    hb = struct.unpack_from('<I', buf, 12)[0]
    offs = struct.unpack_from('<%dQ' % H['n_arrays'], buf, 112)
    sizes = [A[k].nbytes for k in A]
    check('layout: magic, version 2, relaxed 1, 15 arrays, header %d bytes' % hb,
          buf[:8] == b'SUBDIVT\0' and H['version'] == 2 and H['relaxed'] == 1 and H['n_arrays'] == 15
          and hb == 112 + 8 * 15)
    check('layout: offsets 8-byte aligned, increasing, last array ends at the end of the file',
          all(o % 8 == 0 for o in offs) and all(offs[k] + sizes[k] <= offs[k + 1] for k in range(14))
          and offs[14] + sizes[14] == len(buf), '%d bytes' % len(buf))
    cpp = glob.glob(os.path.join(os.path.dirname(a.bundle), 'subdiv_*.sdt'))
    if cpp:
        Hc, _ = read_sdt(cpp[0])
        check('layout: the same reader reads the C++ .sdt of the source run (format reference)',
              set(Hc) == set(H), 'C++ file version %d, %d arrays' % (Hc['version'], Hc['n_arrays']))

    # 2. counts
    Vs, Fs = H['Vs'], H['Fs']
    check('counts: fine %d / %d, coarse %d / %d = the bundle' % (H['n_fine_verts'], H['n_fine_faces'],
                                                                H['n_coarse_verts'], H['n_coarse_faces']),
          (H['n_fine_verts'], H['n_fine_faces'], H['n_coarse_verts'], H['n_coarse_faces'])
          == (len(B.fineV), len(B.fineF), len(B.coarseV), len(B.coarseF)))
    check('counts: Vs %d, Fs %d, array lengths' % (Vs, Fs),
          all(len(A[k]) == Vs for k in ('sub_V', 'carrier_type', 'carrier_index', 'fine_face', 'fine_bary',
                                         'coarse_face', 'coarse_bary', 'struct_set_id', 'sub_V_seed'))
          and len(A['sub_F']) == Fs and len(A['sub_face_orig']) == Fs)

    # 3. fine side
    ff, fb = A['fine_face'].astype(np.int64), A['fine_bary']
    on = ff >= 0
    Pf = interp(B.fineV, B.fineF, ff[on], fb[on])
    err = np.abs(Pf - A['sub_V'][on]).max() / diag
    check('fine side: (fine_face, fine_bary) on the fine mesh = sub_V for %d of %d samples' % (on.sum(), Vs),
          on.all() and err == 0.0, 'max diff %.3g x diag' % err)
    check('fine side: barycentrics >= 0 and sum to 1',
          fb.min() >= 0.0 and np.abs(fb.sum(axis=1) - 1.0).max() <= 1e-12,
          'min %.3g, max |sum - 1| %.3g' % (fb.min(), np.abs(fb.sum(axis=1) - 1.0).max()))

    # 4. coarse side
    cf, cb = A['coarse_face'].astype(np.int64), A['coarse_bary']
    Pc = interp(B.coarseV, B.coarseF, cf, cb)
    sub = glob.glob(os.path.join(a.run_dir, 'relax_input', '02_subdiv_*.ply'))
    if sub:
        Vc = read_ply(sub[0])[0]
        e = np.abs(Pc - Vc).max() / diag
        check('coarse side: (coarse_face, coarse_bary) on the coarse mesh = the samples\' coarse positions',
              (cf >= 0).all() and e == 0.0, 'max diff %.3g x diag' % e)
    check('coarse side: barycentrics >= 0 and sum to 1; coarse_face / sub_face_orig in range',
          cb.min() >= 0.0 and np.abs(cb.sum(axis=1) - 1.0).max() <= 1e-12 and cf.max() < len(B.coarseF)
          and A['sub_face_orig'].min() >= 0 and A['sub_face_orig'].max() < len(B.coarseF))

    # 5. seed / relaxed / faces vs the run's files
    inp = glob.glob(os.path.join(a.run_dir, 'relax_input', '04_relax_input_*.ply'))
    if inp:
        Vi = read_ply(inp[0])[0]
        check('sub_V_seed = the relaxation input (04_relax_input PLY)', np.array_equal(Vi, A['sub_V_seed']))
    objs = [o for o in glob.glob(os.path.join(a.run_dir, 'coarse_subdiv_at_fine_pos_relaxed_*.obj'))
            if '_it' not in os.path.basename(o)]
    if objs:
        Vo, Fo = read_obj(objs[0])
        same = all('%.15g' % x == '%.15g' % y for x, y in zip(Vo.ravel().tolist(), A['sub_V'].ravel().tolist()))
        check('sub_V = the relaxed OBJ (15 digits); sub_F = its faces', same and np.array_equal(Fo, A['sub_F']))
    moved = np.linalg.norm(A['sub_V'] - A['sub_V_seed'], axis=1)
    print('     relaxation moved %d of %d samples, max %.3g x diag' % ((moved > 0).sum(), Vs, moved.max() / diag))

    # 5b. sliding: barycentrics recomputed from the relaxed xyz in the stored fine triangle
    # (independent of the relaxation code) = the stored ones; samples that slid into another
    # triangle are outside their seed triangle
    def bary_in(Pt, f):
        a_, b_, c_ = B.fineV[B.fineF[f, 0], :3], B.fineV[B.fineF[f, 1], :3], B.fineV[B.fineF[f, 2], :3]
        e1, e2, r = b_ - a_, c_ - a_, Pt - a_
        d11, d12, d22 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
        r1, r2 = (r * e1).sum(1), (r * e2).sum(1)
        den = d11 * d22 - d12 * d12
        v = (d22 * r1 - d12 * r2) / den
        w = (d11 * r2 - d12 * r1) / den
        q = a_ + v[:, None] * e1 + w[:, None] * e2
        return np.stack([1 - v - w, v, w], 1), np.linalg.norm(Pt - q, axis=1)
    bs, dist = bary_in(A['sub_V'], ff)
    check('sliding: barycentrics recomputed from the relaxed xyz in the stored triangle = the stored ones',
          np.abs(bs - fb).max() <= 1e-9 and bs.min() >= -1e-9 and dist.max() <= 1e-12 * diag,
          'max diff %.2e, min %.2e, off-plane %.2e x diag' % (np.abs(bs - fb).max(), bs.min(), dist.max() / diag))
    if inp:
        f0 = read_ply(inp[0])[2]['fine_face'].astype(np.int64)
        slid = ff != f0
        if slid.any():
            bo, do = bary_in(A['sub_V'][slid], f0[slid])
            out = int(((bo.min(1) < -1e-9) | (do > 1e-9 * diag)).sum())
            onEdge = int(slid.sum()) - out
            # a sample that stayed in its seed triangle's closure is on an edge shared with the new one
            okEdge = True
            if onEdge:
                keep = ~((bo.min(1) < -1e-9) | (do > 1e-9 * diag))
                okEdge = bool((np.abs(bo[keep]).min(1) <= 1e-9).all())
            check('sliding: %d samples moved to another fine triangle; %d of them are outside their seed triangle, '
                  '%d on an edge it shares with the new one' % (slid.sum(), out, onEdge), okEdge)

    # 6 / 7 / 8. carriers, palette, own structure
    off, ids, mask, sid = A['palette_offsets'], A['palette_ids'], A['palette_type_mask'], A['struct_set_id']
    P = H['P']
    check('palette: %d sets, offsets increasing from 0 to n_palette_ids %d' % (P, H['nPI']),
          off[0] == 0 and np.all(np.diff(off) >= 0) and off[-1] == H['nPI'] and len(off) == P + 1
          and sid.min() >= 0 and sid.max() < P)
    sets = [ids[off[k]:off[k + 1]].tolist() for k in range(P)]
    mk = []
    for st in sets:
        m = 0
        for i in st:
            m |= 1 << ms.structType[i]
        mk.append(m)
    check('palette: type mask = the .ma_struct types of the ids (bit0 sheet, 1 seam, 2 boundary, 3 junction)',
          all(int(mask[k]) == mk[k] for k in range(P)))
    curveIds = [{i for i in s if ms.structType[i] in (1, 2)} for s in sets]
    sheetIds = [{i for i in s if ms.structType[i] == 0} for s in sets]
    role = np.array([3 if m & 8 else (1 if m & 6 else 0) for m in mask.tolist()])[sid]  # 0 sheet, 1 curve, 3 junction
    E = A['orig_edges']
    ct, ci = A['carrier_type'], A['carrier_index']
    eRows = np.nonzero(ct == 1)[0]
    edgeCurve = {}
    for st in ms.structs:
        if st.type in (1, 2):
            for e in st.elements:
                edgeCurve.setdefault(matstruct_edge_key(*ms.maEdges[e]), set()).add(st.id)
    badE = sum(1 for i in eRows.tolist()
               if not (edgeCurve.get(matstruct_edge_key(E[ci[i], 0], E[ci[i], 1]), set()) & curveIds[sid[i]]))
    check('carriers: %d FACE (fine face), %d EDGE (a fine edge on one of the sample\'s seams / boundaries), '
          '%d VERTEX' % ((ct == 2).sum(), len(eRows), (ct == 0).sum()),
          set(np.unique(ct).tolist()) <= {0, 1, 2} and ci[ct == 2].max() < len(B.fineF) and badE == 0
          and (eRows.size == 0 or ci[eRows].max() < len(E)), '%d EDGE carriers off their curves' % badE)
    offSheet = sum(1 for i in np.nonzero(role == 0)[0].tolist() if not (set(ms.faceIds[ff[i]]) & sheetIds[sid[i]]))
    offCurve = 0
    for i in np.nonzero(role == 1)[0].tolist():
        nz = [B.fineF[ff[i], c] for c in range(3) if fb[i, c] != 0.0]
        if len(nz) == 1:   # on a fine vertex: some curve edge of the sample's curves at it
            v = nz[0]
            hit = any((edgeCurve.get(matstruct_edge_key(v, w), set()) & curveIds[sid[i]])
                      for f in np.nonzero((B.fineF == v).any(axis=1))[0] for w in B.fineF[f])
        else:
            hit = len(nz) == 2 and bool(edgeCurve.get(matstruct_edge_key(nz[0], nz[1]), set()) & curveIds[sid[i]])
        offCurve += not hit
    check('own structure: %d sheet samples on one of their sheets, %d curve samples on one of their curves, '
          '%d junction samples' % ((role == 0).sum(), (role == 1).sum(), (role == 3).sum()),
          offSheet == 0 and offCurve == 0, 'off: sheet %d, curve %d' % (offSheet, offCurve))
    jverts = {st.id: set(st.elements) for st in ms.structs if st.type == 3}
    offJ = 0
    for i in np.nonzero(role == 3)[0].tolist():
        c = [c for c in range(3) if fb[i, c] == 1.0]
        v = B.fineF[ff[i], c[0]] if c else -1
        offJ += not any(v in jverts.get(j, ()) for j in sets[sid[i]])
    check('junction samples on one of their junction vertices', offJ == 0, '%d off' % offJ)
    print('SDT CHECK: %s (%s)' % ('PASS' if ok else 'FAIL', path))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
