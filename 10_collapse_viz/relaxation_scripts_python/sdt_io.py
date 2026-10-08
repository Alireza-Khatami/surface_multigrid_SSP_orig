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
