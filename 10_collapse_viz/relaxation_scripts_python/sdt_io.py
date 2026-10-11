"""The .sdt format of collapse_viz_bin (subdiv_sample_tracker/subdiv_tracker.cpp,
subdiv_tracker_save), written for the relaxed subdivided COARSE mesh.

Same layout, byte for byte, as the C++ writer (little-endian, header 112 + 8 * n_arrays
bytes, each array 8-byte aligned, version 2 / relaxed = 1, 15 arrays). The samples are the
vertices of the subdivided coarse mesh after run_relax.py's relaxation:

  sub_V          relaxed positions on the fine mesh
  sub_F          the faces of the relaxed mesh
  sub_face_orig  coarse face of each sub face (the C++ file stores the fine face here)
  orig_edges     fine-mesh edges (min, max), indexed by EDGE carriers
  carrier_type / carrier_index   carrier of the seed (before relaxation) on the fine mesh
  fine_face / fine_bary          the relaxed position on the fine mesh (gFO order)
  coarse_face / coarse_bary      where the sample was made on the coarse mesh (unchanged)
  struct_set_id + palette        the samples' struct ids (as the relaxation uses them)
  sub_V_seed     positions before the relaxation (the relaxation input)

Spec for consumers: md_files/coarse_subdiv_relaxed_sdt.md. Reader:
subdiv_sample_tracker/tools/read_sdt.py (unchanged).
"""
import os
import struct

import numpy as np

from obj_io import long_path

MAGIC = b'SUBDIVT\0'


def write_sdt(path, header, arrays):
    """header: n_levels, relaxed, n_samples_requested, Vs, Fs, n_fine_verts, n_fine_faces, nE,
    n_coarse_verts, n_coarse_faces, P, nPI. arrays: the 14 / 15 arrays in file order, each a
    numpy array of its final dtype. Returns the number of bytes written."""
    n = len(arrays)
    hdr = 112 + 8 * n
    offs, at = [], hdr
    for a in arrays:
        at = (at + 7) // 8 * 8
        offs.append(at)
        at += a.nbytes
    out = bytearray()
    out += MAGIC
    out += struct.pack('<4I', 2 if header['relaxed'] else 1, hdr, header['n_levels'], 1 if header['relaxed'] else 0)
    out += struct.pack('<11Q', header['n_samples_requested'], header['Vs'], header['Fs'], header['n_fine_verts'],
                       header['n_fine_faces'], header['nE'], header['n_coarse_verts'], header['n_coarse_faces'],
                       header['P'], header['nPI'], n)
    out += struct.pack('<%dQ' % n, *offs)
    assert len(out) == hdr
    for a, o in zip(arrays, offs):
        out += b'\0' * (o - len(out))
        out += np.ascontiguousarray(a).tobytes()
    d = os.path.dirname(path)
    if d:
        os.makedirs(long_path(d), exist_ok=True)
    with open(long_path(path), 'wb') as f:
        f.write(out)
    return len(out)


def sliding_stats(B, sess, M):
    """How the relaxation moved the samples across fine triangles: the relaxed location
    (M.fineFace / M.fineBary) against the seed's fine face (sess.M.fineFace). The barycentrics
    of the relaxed xyz in the seed triangle are recomputed from the fine mesh: a sample that
    changed triangle is either outside its seed triangle or on an edge it shares with the new
    one. Also the unmapped samples and the stored barycentrics' range."""
    V, F = np.asarray(B.fineV, dtype=np.float64)[:, :3], np.asarray(B.fineF, dtype=np.int64)
    ff, fb = np.asarray(M.fineFace, dtype=np.int64), np.asarray(M.fineBary, dtype=np.float64)
    f0 = np.asarray(sess.M.fineFace, dtype=np.int64)
    X = np.asarray(M.V, dtype=np.float64)
    on = ff >= 0
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    changed = on & (f0 >= 0) & (ff != f0)
    idx = np.nonzero(changed)[0]
    outside = 0
    if len(idx):
        a, b, c = V[F[f0[idx], 0]], V[F[f0[idx], 1]], V[F[f0[idx], 2]]
        e1, e2, r = b - a, c - a, X[idx] - a
        d11, d12, d22 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
        r1, r2 = (r * e1).sum(1), (r * e2).sum(1)
        den = d11 * d22 - d12 * d12
        v = (d22 * r1 - d12 * r2) / den
        w = (d11 * r2 - d12 * r1) / den
        q = a + v[:, None] * e1 + w[:, None] * e2
        bo = np.stack([1 - v - w, v, w], 1)
        outside = int(((bo.min(1) < -1e-9) | (np.linalg.norm(X[idx] - q, axis=1) > 1e-9 * diag)).sum())
    n = int(len(ff))
    return dict(samples=n, unmapped=int((~on).sum()), unmappedIds=np.nonzero(~on)[0].tolist()[:20],
                changedTriangle=int(len(idx)), changedPercent=100.0 * len(idx) / max(1, int(on.sum())),
                outsideSeedTriangle=outside, onSharedEdge=int(len(idx)) - outside,
                minBary=float(fb[on].min()) if on.any() else 0.0,
                maxBarySumErr=float(np.abs(fb[on].sum(1) - 1.0).max()) if on.any() else 0.0)


def sliding_text(st):
    return ('sliding: %d of %d mapped samples (%.1f%%) ended on another fine triangle than their seed '
            '(%d outside their seed triangle, %d on an edge it shares with the new one); the .sdt stores '
            'the new triangle | stored barycentrics min %.3g, max |sum - 1| %.3g | unmapped samples (face -1, '
            'on no coarse face) %d%s'
            % (st['changedTriangle'], st['samples'] - st['unmapped'], st['changedPercent'],
               st['outsideSeedTriangle'], st['onSharedEdge'], st['minBary'], st['maxBarySumErr'], st['unmapped'],
               (' %s' % st['unmappedIds']) if st['unmapped'] else ''))


def export_relaxed_sdt(path, B, C, sess, M, nRequested):
    """The relaxed subdivided coarse mesh as .sdt. M: the relaxed mesh (rel.M after
    finish(): V, F, fineFace, fineBary on the fine mesh); sess: its CoarseRelaxSession
    (seeds, seed carriers, struct palette); C: the c2f build (coarse face / barycentric of
    every sample)."""
    from subdiv_mesh import SUBDIV_CARRIER_FACE
    S, M0 = C.S, sess.M
    Vs = M.V.shape[0]
    ct = np.asarray(M0.carrierType, dtype=np.uint8)
    # carrier of the seed: FACE -> the seed's fine face, EDGE -> the seed's fine edge
    ci = np.where(ct == SUBDIV_CARRIER_FACE, np.asarray(M0.fineFace), np.asarray(M0.carrierIndex))
    pal = sess.pal
    arrays = [
        np.asarray(M.V, dtype='<f8').reshape(Vs, 3),                     # 0  sub_V (relaxed)
        np.asarray(M.F, dtype='<i4').reshape(-1, 3),                     # 1  sub_F
        np.asarray(S.faceOrig, dtype='<i4'),                             # 2  sub_face_orig (coarse face)
        np.asarray(M.origEdges, dtype='<i4').reshape(-1, 2),             # 3  orig_edges (fine mesh)
        ct.astype('u1'),                                                 # 4  carrier_type (seed)
        ci.astype('<i4'),                                                # 5  carrier_index (seed)
        np.asarray(M.fineFace, dtype='<i4'),                             # 6  fine_face (relaxed)
        np.asarray(M.fineBary, dtype='<f8').reshape(Vs, 3),              # 7  fine_bary (relaxed)
        np.asarray(C.S.fineFace, dtype='<i4'),                           # 8  coarse_face
        np.asarray(C.S.fineBary, dtype='<f8').reshape(Vs, 3),            # 9  coarse_bary
        np.asarray(sess.setId, dtype='<i4'),                             # 10 struct_set_id
        np.asarray(pal.offsets, dtype='<i4'),                            # 11 palette_offsets
        np.asarray(pal.ids, dtype='<i4'),                                # 12 palette_ids
        np.asarray(pal.typeMask, dtype='u1'),                            # 13 palette_type_mask
        np.asarray(sess.Vseed, dtype='<f8').reshape(Vs, 3),              # 14 sub_V_seed
    ]
    header = dict(n_levels=int(S.nLevels), relaxed=True, n_samples_requested=int(nRequested), Vs=Vs,
                  Fs=int(arrays[1].shape[0]), n_fine_verts=int(B.fineV.shape[0]), n_fine_faces=int(B.fineF.shape[0]),
                  nE=int(arrays[3].shape[0]), n_coarse_verts=int(B.coarseV.shape[0]),
                  n_coarse_faces=int(B.coarseF.shape[0]), P=int(len(pal.typeMask)), nPI=int(len(pal.ids)))
    return write_sdt(path, header, arrays)
