"""The SSP collapse output as an .sdt, WITHOUT relaxation, and a check that the .sdt
reconstructs the SSP output.

Builds the subdivided coarse mesh and its coarse -> fine walk exactly as run_relax.py does
(run_relax.build_c2f; same sampling flags), skips the relaxation and writes
coarse_subdiv_norelax_<stem>.sdt: the unrelaxed .sdt layout of the C++ writer (version 1,
relaxed 0, 14 arrays, no sub_V_seed), read by subdiv_sample_tracker/tools/read_sdt.py.

  default        the samples exactly where the SSP walk put them (= the C++
                 coarse_subdiv_at_fine_pos_*.obj); carriers FACE = that fine face
  --snap_seeds   the seeds snapped onto their own structure (the relaxation's iteration-0
                 state); curve samples get EDGE carriers. No longer equal to the SSP output.

Then reads the file back with the consumer reader and checks it against the SSP run
(the bundle's folder, or --ssp_dir):
  1. layout: magic, version 1, relaxed 0, 14 arrays, aligned offsets, file size; the same
     header fields as the C++ subdiv_*.sdt of that run
  2. counts = the bundle; the coarse mesh = the SSP simplified_*.obj
  3. fine side: (fine_face, fine_bary) on the fine mesh = sub_V; barycentrics >= 0, sum 1
  4. sub_V = the SSP coarse_subdiv_at_fine_pos_*.obj (15 digits; unsnapped, default sampling)
     and sub_F = its faces
  5. coarse side: (coarse_face, coarse_bary) on the SSP coarse mesh = the SSP
     coarse_subdiv_*.obj; the first |VC| samples are the coarse vertices
  6. carriers, palette (type mask = the .ma_struct types)

  python export_sdt_norelax.py --output_dir ../output/relaxation_experiments/py_norelax_sdt
  python export_sdt_norelax.py --output_dir ... --equal_area_samples 277700 --delaunay_flips
  (+ any run_relax.py sampling flag; with a non-default sampling there is no SSP OBJ to
  compare against, and checks 4 / 5 compare against this run's own build instead)
"""
import argparse
import datetime
import glob
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'subdiv_sample_tracker', 'tools'))

import log_util  # noqa: E402
import run_relax  # noqa: E402
from bundle_io import load_bundle_flat  # noqa: E402
from coarse_subdiv_relax import CoarseRelaxSession  # noqa: E402
from matstruct import load_matstruct  # noqa: E402
from obj_io import read_obj  # noqa: E402
from read_sdt import read_sdt  # noqa: E402  (the consumer-side reader, unchanged)
from sdt_io import export_unrelaxed_sdt  # noqa: E402


def interp(V, F, face, bary):
    T = F[face]
    return (bary[:, 0:1] * V[T[:, 0], :3] + bary[:, 1:2] * V[T[:, 1], :3]) + bary[:, 2:3] * V[T[:, 2], :3]


def same15(X, Y):
    return X.shape == Y.shape and all('%.15g' % x == '%.15g' % y for x, y in zip(X.ravel().tolist(),
                                                                                  Y.ravel().tolist()))


def check_sdt(path, B, C, ms, ssp_dir, stem, snapped, default_sampling, check):
    """Reads path back and checks it against the SSP run in ssp_dir (see the module doc)."""
    H, A = read_sdt(path)
    buf = open(path, 'rb').read()
    diag = float(np.linalg.norm(B.fineV.max(axis=0) - B.fineV.min(axis=0)))

    # 1. layout
    n = H['n_arrays']
    hb = struct.unpack_from('<I', buf, 12)[0]
    offs = struct.unpack_from('<%dQ' % n, buf, 112)
    sizes = [A[k].nbytes for k in A]
    check('layout: magic, version 1, relaxed 0, 14 arrays (no sub_V_seed), header %d bytes' % hb,
          buf[:8] == b'SUBDIVT\0' and H['version'] == 1 and H['relaxed'] == 0 and n == 14
          and hb == 112 + 8 * 14 and 'sub_V_seed' not in A)
    check('layout: offsets 8-byte aligned, increasing, last array ends at the end of the file',
          all(o % 8 == 0 for o in offs) and all(offs[k] + sizes[k] <= offs[k + 1] for k in range(n - 1))
          and offs[-1] + sizes[-1] == len(buf), '%d bytes' % len(buf))
    cpp = glob.glob(os.path.join(ssp_dir, 'subdiv_*.sdt'))
    if cpp:
        Hc, Ac = read_sdt(cpp[0])
        check('layout: same header fields, version, array names as the C++ subdiv_*.sdt of the SSP run',
              set(Hc) == set(H) and Hc['version'] == H['version'] and Hc['relaxed'] == H['relaxed']
              and list(Ac) == list(A), 'C++ file: version %d, %d arrays' % (Hc['version'], Hc['n_arrays']))

    # 2. counts, coarse mesh
    Vs = H['Vs']
    check('counts: fine %d / %d, coarse %d / %d = the bundle' % (H['n_fine_verts'], H['n_fine_faces'],
                                                                H['n_coarse_verts'], H['n_coarse_faces']),
          (H['n_fine_verts'], H['n_fine_faces'], H['n_coarse_verts'], H['n_coarse_faces'])
          == (len(B.fineV), len(B.fineF), len(B.coarseV), len(B.coarseF)))
    check('counts: Vs %d, Fs %d, array lengths' % (Vs, H['Fs']),
          all(len(A[k]) == Vs for k in ('sub_V', 'carrier_type', 'carrier_index', 'fine_face', 'fine_bary',
                                         'coarse_face', 'coarse_bary', 'struct_set_id'))
          and len(A['sub_F']) == H['Fs'] and len(A['sub_face_orig']) == H['Fs'])
    simp = glob.glob(os.path.join(ssp_dir, 'simplified_' + stem + '.obj'))
    if simp:
        Vsimp, Fsimp = read_obj(simp[0])
        check('coarse mesh: the bundle\'s coarse mesh = the SSP simplified_*.obj (15 digits, same faces)',
              same15(Vsimp, B.coarseV[:, :3]) and np.array_equal(Fsimp, B.coarseF))

    # 3. fine side
    ff, fb = A['fine_face'].astype(np.int64), A['fine_bary']
    cf, cb = A['coarse_face'].astype(np.int64), A['coarse_bary']
    on = ff >= 0
    Pf = interp(B.fineV, B.fineF, ff[on], fb[on])
    err = np.abs(Pf - A['sub_V'][on]).max() / diag
    check('fine side: (fine_face, fine_bary) on the fine mesh = sub_V, all %d mapped samples (%d unmapped)'
          % (on.sum(), (~on).sum()), err == 0.0, 'max diff %.3g x diag' % err)
    check('fine side: barycentrics >= 0 and sum to 1',
          fb[on].min() >= 0.0 and np.abs(fb[on].sum(axis=1) - 1.0).max() <= 1e-12,
          'min %.3g, max |sum - 1| %.3g' % (fb[on].min(), np.abs(fb[on].sum(axis=1) - 1.0).max()))

    # 4. sub_V / sub_F = the SSP coarse -> fine output
    ref = os.path.join(ssp_dir, 'coarse_subdiv_at_fine_pos_' + stem + '.obj')
    if default_sampling and os.path.exists(ref):
        Vr, Fr = read_obj(ref)
        refName = 'the SSP coarse_subdiv_at_fine_pos_*.obj'
    else:
        Vr, Fr = C.P, C.S.F
        refName = 'this run\'s coarse -> fine walk (no SSP OBJ for this sampling)'
    d = np.linalg.norm(A['sub_V'] - Vr, axis=1) / diag if Vr.shape == A['sub_V'].shape else None
    check('faces: sub_F = %s faces (%d)' % (refName, len(Fr)), np.array_equal(Fr, A['sub_F']))
    if not snapped:
        check('positions: sub_V = %s, all %d samples (15 digits)' % (refName, len(Vr)),
              same15(Vr, A['sub_V']), 'max diff %.3g x diag' % (d.max() if d is not None else np.nan))
        # the same positions rebuilt from the .sdt alone (fine mesh + face + barycentrics)
        Pall = A['sub_V'].copy()
        Pall[on] = Pf
        check('reconstruction: interpolated from (fine_face, fine_bary) = %s (15 digits)' % refName,
              same15(Vr, Pall))
    else:
        moved = d > 0
        print('     snapped seeds vs %s: %d of %d samples moved by the snap, max %.3g mean %.3g x diag'
              % (refName, moved.sum(), len(d), d.max(), d[moved].mean() if moved.any() else 0.0))

    # 5. coarse side
    Pc = interp(B.coarseV, B.coarseF, cf[on], cb[on])
    refc = os.path.join(ssp_dir, 'coarse_subdiv_' + stem + '.obj')
    if default_sampling and os.path.exists(refc):
        Vc, Fc = read_obj(refc)
        refcName = 'the SSP coarse_subdiv_*.obj'
    else:
        Vc, Fc = C.S.V, C.S.F
        refcName = 'this run\'s subdivided coarse mesh'
    # not bit-exact: the subdivision makes the coarse positions as edge midpoints, the .sdt
    # rebuilds them from barycentrics, and the OBJ has 15 digits; both round at ~1e-15 x diag
    e = np.abs(Pc - Vc[on]).max() / diag
    check('coarse side: (coarse_face, coarse_bary) on the coarse mesh = %s (positions to 1e-12 x diag, '
          'faces)' % refcName, e <= 1e-12 and np.array_equal(Fc, A['sub_F']), 'max diff %.3g x diag' % e)
    nCV = len(B.coarseV)
    iv = np.nonzero(on[:nCV])[0]
    check('coarse side: the first %d samples are the coarse vertices (exact; %d of them on no coarse face)'
          % (nCV, nCV - len(iv)), np.array_equal(interp(B.coarseV, B.coarseF, cf[iv], cb[iv]), B.coarseV[iv, :3]))
    check('coarse side: barycentrics >= 0 and sum 1; coarse_face / sub_face_orig in range',
          cb[on].min() >= 0.0 and np.abs(cb[on].sum(axis=1) - 1.0).max() <= 1e-12 and cf.max() < len(B.coarseF)
          and A['sub_face_orig'].min() >= 0 and A['sub_face_orig'].max() < len(B.coarseF)
          and np.array_equal(cf < 0, ~on))

    # 6. carriers, palette
    ct, ci = A['carrier_type'], A['carrier_index']
    if not snapped:
        check('carriers: all FACE, carrier_index = fine_face', np.all(ct == 2) and np.array_equal(ci, A['fine_face']))
    else:
        check('carriers: FACE %d (index = fine_face), EDGE %d (index into orig_edges)' % ((ct == 2).sum(), (ct == 1).sum()),
              set(np.unique(ct).tolist()) <= {1, 2} and np.array_equal(ci[ct == 2], A['fine_face'][ct == 2])
              and ((ct == 1).sum() == 0 or ci[ct == 1].max() < len(A['orig_edges'])))
    off, ids, mask, sid = A['palette_offsets'], A['palette_ids'], A['palette_type_mask'], A['struct_set_id']
    P = H['P']
    mk = []
    for k in range(P):
        m = 0
        for i in ids[off[k]:off[k + 1]].tolist():
            m |= 1 << ms.structType[i]
        mk.append(m)
    check('palette: %d sets, offsets 0 .. n_palette_ids, type mask = the .ma_struct types of the ids' % P,
          off[0] == 0 and np.all(np.diff(off) >= 0) and off[-1] == H['nPI'] and sid.min() >= 0 and sid.max() < P
          and all(int(mask[k]) == mk[k] for k in range(P)))


def main(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--snap_seeds', action='store_true')
    p.add_argument('--ssp_dir', default=None, help='the SSP (collapse_viz_bin) run folder; default: the bundle\'s')
    x, rest = p.parse_known_args(argv)
    a = run_relax.parse_args(rest)
    out_dir = a.output_dir
    os.makedirs(out_dir, exist_ok=True)
    stem = a.stem if a.stem is not None else run_relax.bundle_stem(a.bundle)
    ssp_dir = x.ssp_dir or os.path.dirname(os.path.abspath(a.bundle))
    default_sampling = not (a.equal_area_samples > 0 or a.equal_area_levels > 0
                            or run_relax.target_on(a.equal_area_target) or a.delaunay_flips)
    logf = open(os.path.join(out_dir, 'run.log'), 'w', newline='\n')
    log_util.set_log_file(logf)
    log = log_util.log
    started = datetime.datetime.now()
    cmd = 'python ' + ' '.join([os.path.relpath(__file__)] + (argv if argv is not None else sys.argv[1:]))
    with open(os.path.join(out_dir, 'experiment_config.txt'), 'w', newline='\n') as f:
        f.write('experiment:   %s\n' % os.path.basename(os.path.normpath(out_dir)))
        f.write('description:  %s\n' % (a.description or 'SSP output as .sdt, no relaxation'))
        f.write('started:      %s\n' % started.strftime('%Y-%m-%d %H:%M:%S'))
        f.write('implementation: python (relaxation_scripts_python/export_sdt_norelax.py)\n')
        f.write('bundle:       %s\n' % a.bundle)
        f.write('matstruct:    %s\n' % a.matstruct_path)
        f.write('SSP run:      %s\n' % ssp_dir)
        f.write('n_coarse_subdiv_samples: %d\n' % a.n_coarse_subdiv_samples)
        f.write('equal-area refinement: %s\n' % run_relax.equal_area_text(a))
        f.write('relaxation:   none; samples %s\n' % ('snapped onto their own structure (iteration-0 state)'
                                                     if x.snap_seeds else 'as the SSP coarse -> fine walk put them'))
        f.write('command:      %s\n' % cmd)

    t0 = time.perf_counter()
    B = load_bundle_flat(a.bundle)
    ms = load_matstruct(a.matstruct_path, B.fineV, B.fineF)
    C = run_relax.build_c2f(B, a, ms)
    sess = CoarseRelaxSession(B, C, ms, log)   # struct ids, palette, fine edges (+ the snapped seeds)
    sdtPath = os.path.join(out_dir, 'coarse_subdiv_norelax%s_%s.sdt' % ('_snapped' if x.snap_seeds else '', stem))
    nReq = a.equal_area_samples if a.equal_area_samples > 0 else a.n_coarse_subdiv_samples
    nb = export_unrelaxed_sdt(sdtPath, B, C, sess, nReq, snapped=x.snap_seeds)
    log('[export_sdt_norelax] %d samples, %d faces, no relaxation -> %s (%.1f MB)'
        % (C.S.V.shape[0], C.S.F.shape[0], sdtPath, nb / 1048576.0))

    lines, ok = [], [True]

    def check(name, cond, detail=''):
        ok[0] &= bool(cond)
        s = '%-4s %s%s' % ('ok' if cond else 'FAIL', name, (' | ' + detail) if detail else '')
        lines.append(s)
        print(s)
    check_sdt(sdtPath, B, C, ms, ssp_dir, stem, x.snap_seeds, default_sampling, check)
    verdict = 'SDT vs SSP CHECK: %s (%d checks, %s)' % ('PASS' if ok[0] else 'FAIL', len(lines), sdtPath)
    log(verdict)
    status = 0 if ok[0] else 1
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(dict(folder=os.path.basename(os.path.normpath(out_dir)), sdt=sdtPath, snapped=x.snap_seeds,
                       n_subdiv_verts=int(C.S.V.shape[0]), n_subdiv_faces=int(C.S.F.shape[0]),
                       checks=lines, passed=bool(ok[0]), seconds=time.perf_counter() - t0), f, indent=1)
    with open(os.path.join(out_dir, 'experiment_config.txt'), 'a', newline='\n') as f:
        f.write('finished:     %s (exit code %d, %.1f s)\n'
                % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), status, time.perf_counter() - t0))
        f.write('result lines:\n')
        for s in lines + [verdict]:
            f.write('  ' + s + '\n')
    log_util.set_log_file(None)
    logf.close()
    return status


if __name__ == '__main__':
    sys.exit(main())
