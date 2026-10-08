"""PLY exports of the meshes that lead to the relaxation, and of the relaxation's meshes.

Stages written by export_relax_input (run_relax.py, every experiment; folder relax_input/):
  00_coarse_<stem>.ply              the bundle's coarse mesh
  01_coarse_equal_area_<stem>.ply   the equal-area refined coarse mesh (only when that mode is on)
  02a_before_flips / 02b_after_flips_<stem>.ply   the mesh before / after the Delaunay flips (only
                                    with --delaunay_flips; face properties min_angle, changed)
  02_subdiv_<stem>.ply              the subdivided coarse mesh, on the coarse geometry
  03_subdiv_at_fine_<stem>.ply      the same vertices at their c2f positions on the fine mesh
  04_relax_input_<stem>.ply         what the relaxation starts from: the relaxer's positions at its
                                    initialization (seeds snapped onto their own structure)

Vertex properties of the subdivided meshes: carrier_type / carrier_index (on the coarse mesh:
0 vertex, 1 edge, 2 face), role (0 sheet, 1 curve, 2 junction), set_id (struct-id set),
free (moved by the relaxation), fine_face, snap (|seed - c2f position|, input only).
Face property: coarse_face.
"""
import os

import numpy as np

from log_util import log
from ply_io import write_ply


def _carrier_props(S):
    return dict(carrier_type=S.carrierType.astype(np.uint8), carrier_index=S.carrierIndex.astype(np.int32))


def relax_vertex_props(sess, rel, face=None, held=None, colors=None):
    """Per-vertex properties from the relaxation (role, set, free, fine face, held, colour)."""
    p = dict(role=np.asarray(rel.G.role).astype(np.uint8), set_id=np.asarray(rel.setId).astype(np.int32),
             free=np.asarray(rel.isFree).astype(np.uint8),
             fine_face=np.asarray(rel.face if face is None else face).astype(np.int32))
    if held is not None:
        p['held_back'] = np.asarray(held).astype(np.uint8)
    if colors is not None:
        c = np.clip(np.round(np.asarray(colors) * 255.0), 0, 255).astype(np.uint8)
        p['red'], p['green'], p['blue'] = c[:, 0], c[:, 1], c[:, 2]
    return p


def export_relax_input(out_dir, stem, B, C, sess, rel, comment=''):
    """Writes the stages (see the module doc) to out_dir; call right after the relaxer is
    built, before its first step. Returns the written paths."""
    os.makedirs(out_dir, exist_ok=True)
    S = C.S
    cf = dict(coarse_face=S.faceOrig.astype(np.int32))
    com = [comment] if comment else []
    out = []

    def w(name, V, F, vp=None, fp=None, extra=()):
        p = os.path.join(out_dir, name + stem + '.ply')  # stem: '_<bundle stem>' or ''
        if write_ply(p, V, F, vp, fp, list(com) + list(extra)):
            out.append(p)
        else:
            log('[relax_exports] could not write %s' % p)

    w('00_coarse', B.coarseV, B.coarseF, fp=dict(coarse_face=np.arange(B.coarseF.shape[0], dtype=np.int32)),
      extra=['the bundle coarse mesh'])
    R = getattr(C, 'refined', None)
    if R is not None:
        w('01_coarse_equal_area', R.V, R.F,
          vp=dict(carrier_type=np.asarray(R.carrierType, dtype=np.uint8),
                  carrier_index=np.asarray(R.carrierIndex, dtype=np.int32)),
          fp=dict(coarse_face=R.faceOrig.astype(np.int32)),
          extra=['equal-area refined coarse mesh, target area %.6g' % R.Astar])
    if getattr(C, 'flipStats', None) is not None:
        from delaunay_flip import flip_plys
        out += list(flip_plys(out_dir, stem, S, C.Fbefore, C.faceOrigBefore, comment))
    w('02_subdiv', S.V, S.F, _carrier_props(S), cf, ['subdivided coarse mesh, coarse geometry'])
    vp = _carrier_props(S)
    vp['fine_face'] = C.fineFace.astype(np.int32)
    w('03_subdiv_at_fine', C.P, S.F, vp, cf, ['subdivided coarse mesh at its c2f positions on the fine mesh'])
    vp = relax_vertex_props(sess, rel)
    d = rel.X - C.P
    vp['snap'] = np.sqrt(((d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2]))
    w('04_relax_input', rel.X, S.F, vp, cf, ['relaxation input: positions at the relaxer initialization'])
    log('[relax_exports] relaxation input stages -> %s (%d files)' % (out_dir, len(out)))
    return out
