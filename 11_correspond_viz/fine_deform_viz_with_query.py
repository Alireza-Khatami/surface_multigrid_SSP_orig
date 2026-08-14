#!/usr/bin/env python3
"""
fine_deform_viz.py  —  Show fine mesh vs deformed-fine mesh with correspondence arrows.

The "deformed fine mesh" replaces each tracked fine vertex by its coarse
correspondence position (loaded from samples_vertices_*.txt).  The deformed
mesh is lifted by a Z offset so both meshes are visible simultaneously.
Ambient arrows go from each tracked fine vertex up to its deformed position.

Click a face on the deformed mesh to:
  - Show only the 3 correspondence arrows for that face's corner vertices
  - Highlight every face on the original fine mesh that shares one of those
    3 original fine vertex indices (i.e. the "source" neighbourhood)

Usage:
    python fine_deform_viz.py  <bundle.c2f>

Dependencies:
    pip install polyscope numpy
    (c2f_query.py must be in the same directory or on PYTHONPATH)
"""

import sys
import os
import math
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from scipy.spatial import cKDTree

from c2f_query import load_bundle, query_vertex_f2c_intermediates, compute_f2c_correspondences

# ---- structure names ----
FINE_MESH     = "fine_mesh"
DEFORM_MESH   = "deformed_fine_mesh"
DEFORM_PC     = "deformed_fine_verts"
ARROWS_PC     = "fine_to_deformed"
SEL_ARROWS_PC = "sel_fine_to_deformed"
FLIPPED_PC    = "flipped_face_verts"
FLIPPED_MESH  = "flipped_faces_mesh"
TRACKED_VTX_PC = "tracked_vtx"
TRAJ_CURVE     = "vtx_trajectory"
UV_SHEET_MESH  = "uv_sheet"
UV_TRACKED_PC  = "uv_tracked_vtx"
FINE_VTX_PC    = "fine_mesh_verts"   # selectable point cloud over fine mesh
STEP_POS_PC    = "step_tracked_pos"  # main view: single dot at tracked vertex (current step)
STEP_RING_PC   = "step_ring_pts"     # main view: one-ring vertices for current step
F2C_DEFORM_MESH = "f2c_deformed_fine_mesh"   # F2C-query-based deformed mesh (green)
F2C_DEFORM_PC   = "f2c_deformed_fine_verts"  # F2C tracked vertex positions
FLIPPED_F2C_PC  = "flipped_face_f2c_query"  # on-the-fly F2C final positions for clicked flipped face
CV_RING_MESH   = "cv_ring"           # canonical: 3D one-ring (blue)
CV_UV_PRE      = "cv_uv_pre"         # canonical: UV_pre panel (blue)
CV_UV_POST     = "cv_uv_post"        # canonical: UV_post panel (orange)
CV_TRACKED     = "cv_tracked_vtx"    # canonical: tracked vertex dot + arrow
CV_TRACKED_POST = "cv_tracked_post"  # canonical: tracked vertex dot at UV_post
_CV_NAMES      = [CV_RING_MESH, CV_UV_PRE, CV_UV_POST, CV_TRACKED]

# ---- mutable state ----
_bundle        = None
_bundle_dir    = ""

_fine_ids      = None   # (N,)   tracked fine vertex indices
_fine_pts      = None   # (N,3)  original fine positions for tracked verts
_deform_pts    = None   # (N,3)  coarse-correspondence positions (no Z offset)
_deform_mesh_v = None   # (nFineV,3)  full deformed fine mesh vertices (no offset)

# index into _fine_ids / _fine_pts for fast lookup by fine vertex id
_fine_id_to_row = None  # dict: fine_vertex_id -> row index in _fine_ids

_z_offset               = 1.0
_sample_step            = 10
_show_arrows            = True
_selected_deform_face   = -1
_face_highlight_active  = False
_show_flipped             = True
_show_flipped_verts       = True
_show_flipped_arrows      = True
_flipped_highlight_active = False
_selected_flipped_face    = -1       # local index into FLIPPED_MESH / _flipped_faces_glob
_flipped_faces_glob       = None     # (K,3) global fine vertex IDs per flipped face
_flipped_vtx_colors       = None     # cached (NV,3) color array for the quantity
_flipped_vtx_arrows       = None     # cached (NV,3) arrow vectors for the quantity

# F2C batch deformed mesh
_f2c_deform_mesh_v  = None   # (NF, 3) fineV with F2C-tracked verts moved
_f2c_tracked_mask   = None   # (NF,) bool
_use_f2c_deform     = False  # True = F2C mode, False = bundle mode
_use_incident_steps = True   # True = incident faces only (direct, fewer steps);
                              # False = full one-ring via batch map (more steps)

# collapse-trace state
_selected_vtx         = -1
_vtx_collapse_steps   = []   # list of (collapse_idx, SheetData, local_vtx_idx)
_vtx_query_positions  = []   # list of np.ndarray(3,) — actual C2F query intermediates
_current_step_idx     = 0
_uv_plane_z         = -1.5  # Z position of UV flat-mesh in 3D viewport
_uv_scale           = 3.0   # scale applied to normalised UV coords
_canonical_view     = False
_uv_post_offset     = 1.5   # normalised separation: UV_post sits this far above UV_pre
_canonical_domain   = 'uv'  # 'uv' = show UV_pre/post panels  |  'ring' = show 3D one-ring


# ---------------------------------------------------------------------------

def _compute_vertex_correspondences():
    """
    For each compact coarse vertex i, its fine-mesh correspondent is at
    coarseV[i] + corrVec[i].  A KD-tree search on fineV snaps that position
    to the nearest fine vertex, giving us the full fine→coarse mapping without
    any external txt files.
    """
    global _fine_ids, _fine_pts, _deform_pts, _deform_mesh_v, _fine_id_to_row

    if not _bundle.has_ssp_data or _bundle.vtxMap is None:
        print("[vtx_corr] Bundle has no SSP data — cannot compute correspondences.")
        return False

    NC = _bundle.coarseV.shape[0]

    # fine-mesh position of each compact coarse vertex's correspondent
    fine_pos_of_coarse = _bundle.coarseV + _bundle.corrVec   # (NC, 3)

    # snap to nearest fine vertex
    tree = cKDTree(_bundle.fineV)
    _, fine_indices = tree.query(fine_pos_of_coarse, k=1)    # (NC,)
    fine_indices = fine_indices.astype(np.int32)

    _fine_ids       = fine_indices
    _fine_pts       = _bundle.fineV[fine_indices]            # (NC, 3) original fine positions
    _deform_pts     = _bundle.coarseV.copy()                 # (NC, 3) coarse positions (no Z offset)
    _fine_id_to_row = {int(fine_indices[i]): i for i in range(NC)}

    deform_v = _bundle.fineV.copy()
    deform_v[fine_indices] = _bundle.coarseV
    _deform_mesh_v = deform_v

    n_unique = len(np.unique(fine_indices))
    print(f"[vtx_corr] {NC} coarse verts → {n_unique} unique fine verts mapped")
    return True


def _compute_f2c_deformed_mesh():
    """
    Run F2C query for every fine vertex (batch), store the result as a
    deformed vertex array — fine vertices replaced by their F2C final position.
    """
    global _f2c_deform_mesh_v, _f2c_tracked_mask
    print("[f2c] computing F2C correspondences for all fine vertices...")
    _f2c_deform_mesh_v, _f2c_tracked_mask = compute_f2c_correspondences(_bundle)


# ---------------------------------------------------------------------------
# Rebuild helpers
# ---------------------------------------------------------------------------

def _active_deform_mesh_v():
    """Return the deformed vertex array for whichever mode is active."""
    if _use_f2c_deform and _f2c_deform_mesh_v is not None:
        return _f2c_deform_mesh_v
    return _deform_mesh_v


def _rebuild_meshes():
    fm = ps.register_surface_mesh(FINE_MESH, _bundle.fineV, _bundle.fineF)
    fm.set_color((0.55, 0.55, 0.55))
    fm.set_edge_width(0.3)
    fm.set_smooth_shade(False)
    fm.set_transparency(0.4)

    active_v = _active_deform_mesh_v()
    if active_v is not None:
        dv = active_v.copy()
        dv[:, 2] += _z_offset
        dm = ps.register_surface_mesh(DEFORM_MESH, dv, _bundle.fineF)
        dm.set_color((0.95, 0.50, 0.10))
        dm.set_edge_width(0.7)
        dm.set_smooth_shade(False)
        dm.set_transparency(0.7)


def _rebuild_deform_pc():
    if ps.has_point_cloud(DEFORM_PC):
        ps.remove_point_cloud(DEFORM_PC)

    if _deform_pts is None:
        return

    step = max(1, _sample_step)
    pts  = _deform_pts[::step].copy()
    pts[:, 2] += _z_offset

    pc = ps.register_point_cloud(DEFORM_PC, pts)
    pc.set_color((0.95, 0.50, 0.10))
    pc.set_radius(0.004, relative=True)


def _rebuild_arrows():
    """Global subsampled arrows — hidden when a face is selected."""
    if ps.has_point_cloud(ARROWS_PC):
        ps.remove_point_cloud(ARROWS_PC)

    if _fine_pts is None or _deform_pts is None:
        return
    if _selected_deform_face >= 0:
        return  # selection arrows take over

    step = max(1, _sample_step)
    src  = _fine_pts[::step]
    dst  = _deform_pts[::step].copy()
    dst[:, 2] += _z_offset
    vecs = dst - src

    pc = ps.register_point_cloud(ARROWS_PC, src)
    pc.set_color((1.0, 0.85, 0.0))
    pc.set_radius(0.003, relative=True)
    pc.add_vector_quantity("to_deformed", vecs,
                           vectortype="ambient",
                           enabled=_show_arrows,
                           color=(1.0, 0.55, 0.05))


def _rebuild_selection():
    """
    When a deformed face is selected:
      1. Show arrows only for the (up to 3) tracked corner vertices.
      2. Highlight all fine mesh faces that share any of those vertex indices.
    """
    global _face_highlight_active

    # --- clear selection arrows ---
    if ps.has_point_cloud(SEL_ARROWS_PC):
        ps.remove_point_cloud(SEL_ARROWS_PC)

    # --- clear fine mesh highlight ---
    if ps.has_surface_mesh(FINE_MESH):
        fine_mesh = ps.get_surface_mesh(FINE_MESH)
        if _face_highlight_active:
            fine_mesh.remove_quantity("face_highlight")
            _face_highlight_active = False

    if _selected_deform_face < 0 or _fine_id_to_row is None:
        return

    # 3 corner fine vertex indices of the selected deformed face
    corner_vids = _bundle.fineF[_selected_deform_face]  # shape (3,)

    # ---- selection arrows: only the tracked corners ----
    rows = [_fine_id_to_row[int(vid)]
            for vid in corner_vids if int(vid) in _fine_id_to_row]

    if rows:
        src  = _fine_pts[rows]
        dst  = _deform_pts[rows].copy()
        dst[:, 2] += _z_offset
        vecs = dst - src

        pc = ps.register_point_cloud(SEL_ARROWS_PC, src)
        pc.set_color((1.0, 0.10, 0.10))
        pc.set_radius(0.008, relative=True)
        pc.add_vector_quantity("to_deformed", vecs,
                               vectortype="ambient",
                               enabled=True,
                               color=(1.0, 0.20, 0.20))
        print(f"[sel] deformed face {_selected_deform_face}: "
              f"{len(rows)}/3 corners tracked, arrows shown")
    else:
        print(f"[sel] deformed face {_selected_deform_face}: no tracked corners")

    # ---- highlight fine faces sharing any corner vertex ----
    if ps.has_surface_mesh(FINE_MESH):
        fine_mesh = ps.get_surface_mesh(FINE_MESH)
        corner_set = set(int(v) for v in corner_vids)
        fineF = _bundle.fineF
        nF    = fineF.shape[0]

        # face colour: yellow-highlight if it touches a corner vertex, else default grey
        fc = np.tile([0.55, 0.55, 0.55], (nF, 1))
        for fi in range(nF):
            if corner_set & {int(fineF[fi, 0]), int(fineF[fi, 1]), int(fineF[fi, 2])}:
                fc[fi] = [1.0, 0.85, 0.0]

        fine_mesh.add_color_quantity("face_highlight", fc, defined_on='faces')
        _face_highlight_active = True


def _rebuild_flipped_viz():
    """
    Flipped faces (dot(orig_normal, deformed_normal) < 0):
      - DEFORM_MESH: transparency 0.4 (non-flipped faces are dimmed, keep orange).
      - FLIPPED_MESH: re-indexed sub-mesh of flipped faces only, red, fully opaque.
    On disable: remove FLIPPED_MESH, restore DEFORM_MESH transparency to 0.7.
    """
    global _flipped_highlight_active, _flipped_faces_glob, _selected_flipped_face

    if ps.has_surface_mesh(FLIPPED_MESH):
        ps.remove_surface_mesh(FLIPPED_MESH)
    if ps.has_surface_mesh(DEFORM_MESH):
        if _flipped_highlight_active:
            _flipped_highlight_active = False
        if not _show_flipped:
            ps.get_surface_mesh(DEFORM_MESH).set_transparency(0.7)

    active_v = _active_deform_mesh_v()
    if not _show_flipped or active_v is None:
        _rebuild_flipped_vtx_colors()
        return

    fineF = _bundle.fineF
    fineV = _bundle.fineV
    nF    = fineF.shape[0]

    v0o = fineV[fineF[:, 0]];  v1o = fineV[fineF[:, 1]];  v2o = fineV[fineF[:, 2]]
    orig_n = np.cross(v1o - v0o, v2o - v0o)
    v0d = active_v[fineF[:, 0]];  v1d = active_v[fineF[:, 1]];  v2d = active_v[fineF[:, 2]]
    deform_n = np.cross(v1d - v0d, v2d - v0d)

    flipped   = np.sum(orig_n * deform_n, axis=1) < 0
    n_flipped = int(flipped.sum())
    print(f"[flip] {n_flipped} / {nF} faces flipped")

    if n_flipped == 0:
        return

    # dim the deformed mesh so non-flipped faces are at transparency 0.4
    if ps.has_surface_mesh(DEFORM_MESH):
        ps.get_surface_mesh(DEFORM_MESH).set_transparency(0.4)

    # build flipped sub-mesh at the deformed mesh Z level
    dv = active_v.copy()
    dv[:, 2] += _z_offset

    flip_face_idx   = np.where(flipped)[0]
    flip_faces_glob = fineF[flip_face_idx]                          # (K, 3) global ids
    unique_verts    = np.unique(flip_faces_glob)
    old_to_new      = np.full(len(fineV), -1, dtype=np.int32)
    old_to_new[unique_verts] = np.arange(len(unique_verts), dtype=np.int32)
    new_verts = dv[unique_verts]                                    # positions at deformed Z
    new_faces = old_to_new[flip_faces_glob]                         # re-indexed

    # push vertices along averaged face normals to avoid z-fighting with DEFORM_MESH
    fn  = np.cross(new_verts[new_faces[:, 1]] - new_verts[new_faces[:, 0]],
                   new_verts[new_faces[:, 2]] - new_verts[new_faces[:, 0]])
    nl  = np.linalg.norm(fn, axis=1, keepdims=True)
    fn  = fn / np.where(nl > 1e-10, nl, 1.0)
    vn  = np.zeros_like(new_verts)
    for i in range(3):
        np.add.at(vn, new_faces[:, i], fn)
    vnl = np.linalg.norm(vn, axis=1, keepdims=True)
    vn  = vn / np.where(vnl > 1e-10, vnl, 1.0)
    span    = float((new_verts.max(axis=0) - new_verts.min(axis=0)).max())
    new_verts = new_verts + vn * span * 0.002

    fm = ps.register_surface_mesh(FLIPPED_MESH, new_verts, new_faces)
    fm.set_color((1.0, 0.10, 0.10))
    fm.set_edge_width(0.7)
    fm.set_smooth_shade(False)
    fm.set_transparency(1.0)
    _flipped_highlight_active = True

    _flipped_faces_glob    = flip_faces_glob   # store for per-face pick lookup
    _selected_flipped_face = -1                # reset selection on full rebuild
    _rebuild_flipped_vtx_colors()
    _rebuild_flipped_arrows()


def _rebuild_flipped_vtx_colors():
    """
    Paint a color quantity on the FINE_VTX_PC point cloud:
      - Flipped-face vertices → red
      - All others            → default grey
    When a single flipped face is selected, only its 3 corners go red.
    Colors are cached in _flipped_vtx_colors so the checkbox can re-upload
    with enabled=True/False without recomputing.
    """
    global _flipped_vtx_colors
    if not ps.has_point_cloud(FINE_VTX_PC):
        return
    pc = ps.get_point_cloud(FINE_VTX_PC)

    if not _show_flipped or _flipped_faces_glob is None:
        try:
            pc.remove_quantity("flipped_verts")
        except Exception:
            pass
        _flipped_vtx_colors = None
        return

    NV     = _bundle.fineV.shape[0]
    colors = np.tile(np.array([0.75, 0.75, 0.75], dtype=np.float32), (NV, 1))

    if _selected_flipped_face >= 0 and _selected_flipped_face < len(_flipped_faces_glob):
        vids = _flipped_faces_glob[_selected_flipped_face]
        colors[vids] = [1.0, 0.1, 0.1]
    else:
        for face_vids in _flipped_faces_glob:
            colors[face_vids] = [1.0, 0.1, 0.1]

    _flipped_vtx_colors = colors
    pc.add_color_quantity("flipped_verts", colors, enabled=_show_flipped_verts)
    print ("dfaaf")


def _rebuild_flipped_arrows():
    """Vector quantity on FINE_VTX_PC: fine → deformed position for flipped vertices."""
    global _flipped_vtx_arrows
    if not ps.has_point_cloud(FINE_VTX_PC):
        return
    pc = ps.get_point_cloud(FINE_VTX_PC)

    active_v = _active_deform_mesh_v()
    if not _show_flipped or _flipped_faces_glob is None or active_v is None:
        try:
            pc.remove_quantity("flipped_arrows")
        except Exception:
            pass
        _flipped_vtx_arrows = None
        return

    NV   = _bundle.fineV.shape[0]
    vecs = np.zeros((NV, 3), dtype=np.float32)
    dv   = active_v.copy()
    dv[:, 2] += _z_offset

    if _selected_flipped_face >= 0 and _selected_flipped_face < len(_flipped_faces_glob):
        vids = _flipped_faces_glob[_selected_flipped_face]
    else:
        vids = np.unique(_flipped_faces_glob)

    vecs[vids] = (dv[vids] - _bundle.fineV[vids]).astype(np.float32)
    _flipped_vtx_arrows = vecs
    pc.add_vector_quantity("flipped_arrows", vecs, vectortype="ambient",
                           enabled=_show_flipped_arrows, color=(1.0, 0.30, 0.30))


def _rebuild_fine_vtx_pc():
    """Selectable point cloud over all fine mesh vertices."""
    if ps.has_point_cloud(FINE_VTX_PC):
        ps.remove_point_cloud(FINE_VTX_PC)
    pc = ps.register_point_cloud(FINE_VTX_PC, _bundle.fineV)
    pc.set_color((0.75, 0.75, 0.75))
    pc.set_radius(0.003, relative=True)
    _rebuild_flipped_vtx_colors()
    _rebuild_flipped_arrows()


def _log_step_info():
    n_steps = len(_vtx_collapse_steps)
    if _selected_vtx < 0 or n_steps == 0:
        return
    step_idx = max(0, min(_current_step_idx, n_steps - 1))

    # intermediate position from F2C query
    if _vtx_query_positions:
        qi  = min(step_idx + 1, len(_vtx_query_positions) - 1)
        pos = _vtx_query_positions[qi]
    else:
        pos = _bundle.fineV[_selected_vtx]

    # final deformed position (no Z offset)
    if _f2c_deform_mesh_v is not None:
        final_pos = _f2c_deform_mesh_v[_selected_vtx]
    elif _fine_id_to_row is not None and _selected_vtx in _fine_id_to_row:
        final_pos = _deform_pts[_fine_id_to_row[_selected_vtx]]
    else:
        final_pos = None

    print(f"[step] {step_idx + 1}/{n_steps}  "
          f"pos=({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})", end="")
    if final_pos is not None:
        print(f"  |  final=({final_pos[0]:.6f}, {final_pos[1]:.6f}, {final_pos[2]:.6f})", end="")
    print()


def _rebuild_step_viz():
    """
    Main-view (non-canonical) overlay for the current collapse step:
      STEP_POS_PC  — single dot that interpolates between the fine-mesh position
                     (step 0) and the coarse correspondence position (last step),
                     so it slides along the F2C mapping as you press Prev/Next.
      STEP_RING_PC — one-ring vertices for the current step's sheet (cyan).
    Both are cleared when no vertex is selected.
    """
    for name in (STEP_POS_PC, STEP_RING_PC):
        if ps.has_point_cloud(name):
            ps.remove_point_cloud(name)

    if _selected_vtx < 0 or not _vtx_collapse_steps:
        return

    step_idx = max(0, min(_current_step_idx, len(_vtx_collapse_steps) - 1))
    _ci, sheet, _local_v = _vtx_collapse_steps[step_idx]

    # Use the actual C2F query intermediate position for this step.
    if _vtx_query_positions:
        qi = min(step_idx + 1, len(_vtx_query_positions) - 1)
        current_pos = _vtx_query_positions[qi]
    else:
        current_pos = _bundle.fineV[_selected_vtx]

    pc = ps.register_point_cloud(STEP_POS_PC, current_pos[np.newaxis])
    pc.set_color((1.0, 0.35, 0.0))
    pc.set_radius(0.009, relative=True)

    # Ring point cloud: all subsetVIdx vertices for this step
    # subV     = np.clip(sheet.subsetVIdx, 0, _bundle.fineV.shape[0] - 1)
    # ring_pts = _bundle.fineV[subV]
    # pc_ring  = ps.register_point_cloud(STEP_RING_PC, ring_pts)
    # pc_ring.set_color((0.3, 0.85, 1.0))
    # pc_ring.set_radius(0.006, relative=True)
    # pc_ring.set_radius(0.006, relative=True)


def _vtx_world_pos_after_collapse(sheet, local_v: int) -> np.ndarray:
    """
    Returns the 3-D world-space position of the tracked vertex after the
    collapse described by `sheet`, by projecting UV_post[local_v] through
    the ring's local tangent frame.  This is the actual intermediate result
    of the F2C mapping at this collapse step.
    """
    subV    = np.clip(sheet.subsetVIdx, 0, len(_bundle.fineV) - 1)
    verts   = _bundle.fineV[subV]
    uv_pre  = sheet.UV_pre
    F       = sheet.FUV_pre

    centroid = verts.mean(axis=0)
    span     = (verts.max(axis=0) - verts.min(axis=0)).max()
    if span < 1e-10:
        span = 1.0

    v0 = verts[F[:, 0]]; v1 = verts[F[:, 1]]; v2 = verts[F[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0).sum(axis=0)
    n   = np.linalg.norm(nrm)
    nrm = nrm / n if n > 1e-10 else np.array([0., 0., 1.])

    arb = np.array([1., 0., 0.]) if abs(nrm[0]) < 0.9 else np.array([0., 1., 0.])
    t1  = arb - nrm * nrm.dot(arb); t1 /= np.linalg.norm(t1)
    t2  = np.cross(nrm, t1);        t2 /= np.linalg.norm(t2)

    uc     = (uv_pre[:, 0].max() + uv_pre[:, 0].min()) * 0.5
    vc     = (uv_pre[:, 1].max() + uv_pre[:, 1].min()) * 0.5
    pv     = verts - centroid
    xu     = pv @ t1;  yu  = pv @ t2
    du     = uv_pre[:, 0] - uc;  dv = uv_pre[:, 1] - vc
    A_sum  = float((xu * du + yu * dv).sum())
    B_sum  = float((yu * du - xu * dv).sum())
    beta   = math.atan2(B_sum, A_sum)
    cb, sb = math.cos(beta), math.sin(beta)
    t1_old = t1.copy()
    t1 =  cb * t1_old + sb * t2
    t2 = -sb * t1_old + cb * t2

    uv_span  = max(uv_pre[:, 0].max() - uv_pre[:, 0].min(),
                   uv_pre[:, 1].max() - uv_pre[:, 1].min())
    uv_scale = (1.0 / uv_span) if uv_span > 1e-10 else 1.0

    uv = sheet.UV_post[local_v]
    dU = (uv[0] - uc) * uv_scale * span
    dV = (uv[1] - vc) * uv_scale * span
    return centroid + dU * t1 + dV * t2


def _compute_canonical(sheet, local_v: int) -> dict:
    """
    Port of compute_ring_geometry() + show_canonical_view() from visualizer.cpp.

    3D ring  : fineV[subsetVIdx]  (original fine-mesh positions; approximation
               of the true intermediate one-ring, which is not stored in the bundle)
    UV panels: UV_pre / UV_post from SheetData  (exact)

    Returns a dict with everything needed to register the canonical structures.
    """
    subV      = np.clip(sheet.subsetVIdx, 0, len(_bundle.fineV) - 1)
    verts     = _bundle.fineV[subV]   # (N, 3)
    uv_pre    = sheet.UV_pre          # (N, 2)
    uv_post   = sheet.UV_post         # (N, 2)
    F         = sheet.FUV_pre         # (M, 3)

    # ---- centroid and span ----
    centroid = verts.mean(axis=0)
    span     = (verts.max(axis=0) - verts.min(axis=0)).max()
    if span < 1e-10:
        span = 1.0

    # ---- average face normal (vectorised) ----
    v0 = verts[F[:, 0]]; v1 = verts[F[:, 1]]; v2 = verts[F[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0).sum(axis=0)
    n   = np.linalg.norm(nrm)
    nrm = nrm / n if n > 1e-10 else np.array([0.0, 0.0, 1.0])

    # ---- initial tangent frame (t1, t2) ----
    arb = np.array([1.0, 0.0, 0.0]) if abs(nrm[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1  = arb - nrm * nrm.dot(arb);  t1 /= np.linalg.norm(t1)
    t2  = np.cross(nrm, t1);          t2 /= np.linalg.norm(t2)

    # ---- 2-D Procrustes: rotate (t1,t2) to align UV panel with ring projection ----
    # (β = atan2(Σ(y·u − x·v),  Σ(x·u + y·v))  as in visualizer.cpp §"Rotate (t1,t2) by β")
    uc    = (uv_pre[:, 0].max() + uv_pre[:, 0].min()) * 0.5
    vc    = (uv_pre[:, 1].max() + uv_pre[:, 1].min()) * 0.5
    pv    = verts - centroid
    xu    = pv @ t1;   yu    = pv @ t2
    du    = uv_pre[:, 0] - uc;  dv = uv_pre[:, 1] - vc
    A_sum = float((xu * du + yu * dv).sum())
    B_sum = float((yu * du - xu * dv).sum())
    beta  = math.atan2(B_sum, A_sum)
    cb, sb = math.cos(beta), math.sin(beta)
    t1_old = t1.copy()
    t1 =  cb * t1_old + sb * t2
    t2 = -sb * t1_old + cb * t2

    # ---- UV scale: map UV extent → 1 unit (in normalised space) ----
    uv_span = max(uv_pre[:, 0].max() - uv_pre[:, 0].min(),
                  uv_pre[:, 1].max() - uv_pre[:, 1].min())
    uv_scale = (1.0 / uv_span) if uv_span > 1e-10 else 1.0

    # UV_pre at ring level (centroid); UV_post offset above by _uv_post_offset normalised units.
    # Both panels use the same tangent frame so they're visually aligned.
    def to3d(UV, panel_origin):
        dU = (UV[:, 0:1] - uc) * uv_scale * span
        dV = (UV[:, 1:2] - vc) * uv_scale * span
        return panel_origin + dU * t1 + dV * t2

    uv_pre_3d  = to3d(uv_pre,  centroid)
    uv_post_3d = to3d(uv_post, centroid + nrm * _uv_post_offset * span)

    # ---- canonical rotation: nrm → world Y-up ----
    world_up = np.array([0.0, 1.0, 0.0])
    ax       = np.cross(nrm, world_up)
    sin_a    = np.linalg.norm(ax)
    cos_a    = float(nrm.dot(world_up))
    if sin_a < 1e-6:
        R = np.eye(3) if cos_a > 0 else np.diag([-1.0, -1.0, 1.0])
    else:
        ax    /= sin_a
        theta  = math.atan2(sin_a, cos_a)
        K      = np.array([[   0.0, -ax[2],  ax[1]],
                            [ ax[2],    0.0, -ax[0]],
                            [-ax[1],  ax[0],    0.0]])
        R = np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)

    # Normalise by span so every collapse step renders at unit scale — prevents
    # UV panels jumping to wildly different heights between steps.
    def rot_norm(P):
        return (R @ ((P - centroid) / span).T).T

    vr   = rot_norm(verts)
    upr  = rot_norm(uv_pre_3d)
    upor = rot_norm(uv_post_3d)

    return {
        'V_ring':       vr,
        'uv_pre':       upr,
        'uv_post':      upor,
        'F':            F,
        'tracked_ring': vr[local_v],
        'tracked_pre':  upr[local_v],
        'tracked_post': upor[local_v],
        'span':         span,
    }


def _clear_canonical():
    for name in _CV_NAMES:
        if ps.has_surface_mesh(name):
            ps.remove_surface_mesh(name)
        if ps.has_point_cloud(name):
            ps.remove_point_cloud(name)


def _rebuild_canonical_view():
    _clear_canonical()
    if not _canonical_view or _selected_vtx < 0 or not _vtx_collapse_steps:
        return

    step_idx = max(0, min(_current_step_idx, len(_vtx_collapse_steps) - 1))
    _ci, sheet, local_v = _vtx_collapse_steps[step_idx]

    geo = _compute_canonical(sheet, local_v)

    if _canonical_domain == 'ring':
        # Spatial domain — show the 3D one-ring only
        rm = ps.register_surface_mesh(CV_RING_MESH, geo['V_ring'], geo['F'])
        rm.set_color((0.3, 0.55, 1.0))
        rm.set_edge_width(1.5)
        rm.set_smooth_shade(False)
        rm.set_transparency(0.5)

        pt3d = geo['tracked_ring'][np.newaxis]
        pc   = ps.register_point_cloud(CV_TRACKED, pt3d)
        pc.set_color((1.0, 0.3, 0.0))
        pc.set_radius(0.025, relative=True)

    else:
        # UV domain — show UV_pre (blue) and UV_post (orange) panels
        pm = ps.register_surface_mesh(CV_UV_PRE, geo['uv_pre'], geo['F'])
        pm.set_color((0.3, 0.55, 1.0))
        pm.set_edge_width(1.5)
        pm.set_smooth_shade(False)
        pm.set_transparency(1.0)

        qm = ps.register_surface_mesh(CV_UV_POST, geo['uv_post'], geo['F'])
        qm.set_color((1.0, 0.5, 0.15))
        qm.set_edge_width(1.5)
        qm.set_smooth_shade(False)
        qm.set_transparency(1.0)

        # Tracked vertex: orange dot at UV_pre position + yellow arrow → UV_post
        pt3d  = geo['tracked_pre'][np.newaxis]
        arrow = (geo['tracked_post'] - geo['tracked_pre'])[np.newaxis]
        pc    = ps.register_point_cloud(CV_TRACKED, pt3d)
        pc.set_color((1.0, 0.3, 0.0))
        pc.set_radius(0.015, relative=True)
        pc.add_vector_quantity("uv_disp", arrow, vectortype="ambient",
                               enabled=True, color=(1.0, 0.85, 0.0))

        # Tracked vertex: red  dot at UV_post  position  
        pt3d  = geo['tracked_post'][np.newaxis]
        pc    = ps.register_point_cloud(CV_TRACKED_POST, pt3d)
        pc.set_color((1.0, 0.0, 0.0))
        pc.set_radius(0.015, relative=True)


def _find_vtx_collapse_steps(v: int):
    """
    Walk decIM forward and collect every collapse step that touches vertex v
    via its incident faces only (direct method — fewer steps than full one-ring).
    Returns list stored in _vtx_collapse_steps:
        (collapse_idx, SheetData, local_idx_of_v_in_sheet)
    """
    global _vtx_collapse_steps
    fineF = _bundle.fineF

    # all fine faces that contain v
    faces_of_v = np.where(np.any(fineF == v, axis=1))[0]

    # union of collapse indices from decIM for those faces
    seen: set = set()
    for f in faces_of_v:
        fi = int(f)
        if fi < len(_bundle.decIM):
            for ci in _bundle.decIM[fi]:
                seen.add(ci)

    steps = []
    for ci in sorted(seen):
        cd = _bundle.decInfo[ci]
        # Collect all sheets where v has a valid UV slot (local_v < uvRows).
        valid_sheets = []
        for sheet in cd.sheets:
            uvRows = len(sheet.UV_post)
            hits = np.where(sheet.subsetVIdx == v)[0]
            if len(hits) and int(hits[0]) < uvRows:
                valid_sheets.append((sheet, int(hits[0])))
        if len(valid_sheets) > 1:
            print(f"[trace][multi-sheet] ci={ci} vertex={v}: "
                  f"valid UV slot in {len(valid_sheets)} sheets — using first")
        if valid_sheets:
            steps.append((ci, valid_sheets[0][0], valid_sheets[0][1]))
        else:
            # fallback: record even if no valid UV slot (will be skipped in query)
            for sheet in cd.sheets:
                hits = np.where(sheet.subsetVIdx == v)[0]
                if len(hits):
                    steps.append((ci, sheet, int(hits[0])))
                    break
    _vtx_collapse_steps = steps
    print(f"[trace] vertex {v}: {len(steps)} collapse steps "
          f"(incident faces only, from {len(faces_of_v)} fine faces)")


def _load_vtx_collapse_steps(v: int):
    """
    Populate _vtx_collapse_steps for vertex v.
    _use_incident_steps=True  → incident faces only (direct, fewer steps).
    _use_incident_steps=False → full one-ring via batch map (more steps).
    """
    global _vtx_collapse_steps
    if _use_incident_steps:
        _find_vtx_collapse_steps(v)
        return
    batch_map = getattr(_bundle, '_f2c_vtx_steps', None)
    if batch_map is not None and v in batch_map:
        _vtx_collapse_steps = batch_map[v]
        print(f"[trace] vertex {v}: {len(_vtx_collapse_steps)} collapse steps (full one-ring / batch map)")
    else:
        _find_vtx_collapse_steps(v)


def _run_query_intermediates(v: int):
    """
    Track fine vertex v forward through F2C collapse steps (fine→coarse direction).
    _vtx_collapse_steps must already be populated by _load_vtx_collapse_steps(v).
    """
    global _vtx_query_positions
    _vtx_query_positions = query_vertex_f2c_intermediates(
        _vtx_collapse_steps,
        _bundle.fineV,
        _bundle.fineV[v],
    )
    print(f"[f2c_intermediates] vertex {v}: {len(_vtx_query_positions)} positions "
          f"({len(_vtx_collapse_steps)} steps)")


def _rebuild_vtx_trajectory():
    """
    3D view: green dot at clicked fine vertex + green curve to its coarse
    correspondence (if it's a tracked vertex in samples_vertices_*.txt).
    """
    try:
        if ps.has_curve_network(TRAJ_CURVE):
            ps.remove_curve_network(TRAJ_CURVE)
    except Exception:
        pass
    if ps.has_point_cloud(TRACKED_VTX_PC):
        ps.remove_point_cloud(TRACKED_VTX_PC)

    if _selected_vtx < 0:
        return

    fine_pos = _bundle.fineV[_selected_vtx]
    pc = ps.register_point_cloud(TRACKED_VTX_PC, fine_pos[np.newaxis])
    pc.set_color((0.1, 1.0, 0.3))
    pc.set_radius(0.0035, relative=True)

    # if _fine_id_to_row is not None and _selected_vtx in _fine_id_to_row:
    #     row  = _fine_id_to_row[_selected_vtx]
    #     corr = _deform_pts[row].copy()
    #     corr[2] += _z_offset
    #     nodes = np.array([fine_pos, corr])
    #     edges = np.array([[0, 1]], dtype=np.int32)
    #     try:
    #         cn = ps.register_curve_network(TRAJ_CURVE, nodes, edges)
    #         cn.set_color((0.1, 1.0, 0.3))
    #         cn.set_radius(0.003, relative=True)
    #     except Exception:
    #         pass


def _rebuild_uv_sheet_viz():
    """
    UV canonical view: render the sheet's UV_pre triangulation as a flat mesh
    at Z = _uv_plane_z.  Orange dot marks the tracked vertex's UV_pre position;
    a yellow arrow points to its UV_post position (the UV displacement caused by
    this collapse step).
    """
    if ps.has_surface_mesh(UV_SHEET_MESH):
        ps.remove_surface_mesh(UV_SHEET_MESH)
    if ps.has_point_cloud(UV_TRACKED_PC):
        ps.remove_point_cloud(UV_TRACKED_PC)

    if _selected_vtx < 0 or not _vtx_collapse_steps:
        return

    step_idx = max(0, min(_current_step_idx, len(_vtx_collapse_steps) - 1))
    ci, sheet, local_v = _vtx_collapse_steps[step_idx]

    uv     = sheet.UV_pre          # (uvRows, 2)
    center = uv.mean(axis=0)
    span   = (uv.max(axis=0) - uv.min(axis=0)).max()
    if span < 1e-10:
        span = 1.0
    norm   = (uv - center) / span  # ~ [-0.5, 0.5]

    verts3d = np.column_stack([
        norm[:, 0] * _uv_scale,
        norm[:, 1] * _uv_scale,
        np.full(len(uv), _uv_plane_z),
    ])
    sm = ps.register_surface_mesh(UV_SHEET_MESH, verts3d, sheet.FUV_pre)
    sm.set_color((0.4, 0.6, 1.0))
    sm.set_edge_width(1.0)
    sm.set_smooth_shade(False)
    sm.set_transparency(0.5)

    # tracked vertex: orange dot + yellow arrow (UV_pre → UV_post)
    uv_pre_n  = (uv[local_v]             - center) / span * _uv_scale
    uv_post_n = (sheet.UV_post[local_v]  - center) / span * _uv_scale
    pt3d  = np.array([[uv_pre_n[0], uv_pre_n[1], _uv_plane_z]])
    arrow = np.array([[uv_post_n[0] - uv_pre_n[0],
                       uv_post_n[1] - uv_pre_n[1],
                       0.0]])
    pc = ps.register_point_cloud(UV_TRACKED_PC, pt3d)
    pc.set_color((1.0, 0.3, 0.0))
    pc.set_radius(0.010, relative=True)
    pc.add_vector_quantity("uv_disp", arrow, vectortype="ambient",
                           enabled=True, color=(1.0, 0.85, 0.0))


def _rebuild_all():
    _rebuild_meshes()
    _rebuild_deform_pc()
    _rebuild_arrows()
    _rebuild_selection()
    _rebuild_flipped_viz()
    _rebuild_fine_vtx_pc()
    _rebuild_vtx_trajectory()
    _rebuild_step_viz()
    # _rebuild_uv_sheet_viz()


# ---------------------------------------------------------------------------
# UI callback
# ---------------------------------------------------------------------------

def ui_callback():
    global _z_offset, _sample_step, _show_arrows, _selected_deform_face, _show_flipped
    global _show_flipped_verts, _show_flipped_arrows, _selected_flipped_face, _flipped_vtx_colors, _flipped_vtx_arrows
    global _selected_vtx, _current_step_idx, _uv_scale, _uv_plane_z, _canonical_view, _uv_post_offset, _canonical_domain, _vtx_query_positions
    global _use_f2c_deform, _use_incident_steps

    changed = False

    try:
        psim.SetNextWindowSize((330, 280), psim.ImGuiCond_FirstUseEver)
    except Exception:
        pass
    psim.Begin("Fine -> Deformed", True)

    c, v = psim.Checkbox("Use F2C query", _use_f2c_deform)
    if c:
        _use_f2c_deform = v
        _rebuild_all()
    psim.SameLine()
    label = "[ F2C query ]" if _use_f2c_deform else "[ Bundle ]"
    psim.TextUnformatted(label)

    psim.Separator()

    c, v = psim.SliderFloat("Z offset (deformed)", _z_offset, -2.0, 2.0)
    if c:
        _z_offset = v
        changed = True

    c, v = psim.SliderInt("Arrow every N verts", _sample_step, 1, 500)
    if c:
        _sample_step = v
        changed = True

    c, v = psim.Checkbox("Show arrows", _show_arrows)
    if c:
        _show_arrows = v
        changed = True

    if _fine_pts is not None:
        step  = max(1, _sample_step)
        shown = (len(_fine_pts) + step - 1) // step
        psim.TextUnformatted(f"Arrows: {shown} / {len(_fine_pts)} tracked verts")

    psim.Separator()

    if _selected_deform_face >= 0:
        corner_vids = _bundle.fineF[_selected_deform_face]
        psim.TextUnformatted(
            f"Deformed face {_selected_deform_face}  "
            f"(fine verts {corner_vids[0]}, {corner_vids[1]}, {corner_vids[2]})")
        if psim.Button("Clear selection"):
            _selected_deform_face = -1
            _rebuild_arrows()
            _rebuild_selection()
    else:
        psim.TextUnformatted("Click a deformed mesh face to inspect")

    psim.Separator()
    psim.TextUnformatted("Flip detection")

    c, v = psim.Checkbox("Show flipped faces", _show_flipped)
    if c:
        _show_flipped = v
        _rebuild_flipped_viz()

    if _show_flipped:
        psim.Indent()
        c, v = psim.Checkbox("Highlight flipped vertices on fine mesh", _show_flipped_verts)
        if c:
            _show_flipped_verts = v
            if _flipped_vtx_colors is not None and ps.has_point_cloud(FINE_VTX_PC):
                ps.get_point_cloud(FINE_VTX_PC).add_color_quantity(
                    "flipped_verts", _flipped_vtx_colors, enabled=v)
        c, v = psim.Checkbox("Show flipped arrows", _show_flipped_arrows)
        if c:
            _show_flipped_arrows = v
            if _flipped_vtx_arrows is not None and ps.has_point_cloud(FINE_VTX_PC):
                ps.get_point_cloud(FINE_VTX_PC).add_vector_quantity(
                    "flipped_arrows", _flipped_vtx_arrows, vectortype="ambient",
                    enabled=v, color=(1.0, 0.30, 0.30))
        psim.Unindent()

    if _show_flipped and _active_deform_mesh_v() is not None:
        fineF    = _bundle.fineF
        fineV    = _bundle.fineV
        v0o = fineV[fineF[:, 0]];  v1o = fineV[fineF[:, 1]];  v2o = fineV[fineF[:, 2]]
        _av = _active_deform_mesh_v()
        v0d = _av[fineF[:, 0]];  v1d = _av[fineF[:, 1]];  v2d = _av[fineF[:, 2]]
        orig_n   = np.cross(v1o - v0o, v2o - v0o)
        deform_n = np.cross(v1d - v0d, v2d - v0d)
        n_flipped = int((np.sum(orig_n * deform_n, axis=1) < 0).sum())
        psim.TextUnformatted(f"Flipped: {n_flipped} / {fineF.shape[0]} faces")
        if _selected_flipped_face >= 0 and _flipped_faces_glob is not None:
            vids = _flipped_faces_glob[_selected_flipped_face]
            psim.TextUnformatted(
                f"Selected flipped face {_selected_flipped_face}  "
                f"(verts {vids[0]}, {vids[1]}, {vids[2]})")
            if psim.Button("Clear flipped face selection"):
                _selected_flipped_face = -1
                _rebuild_flipped_vtx_colors()
                _rebuild_flipped_arrows()
        else:
            psim.TextDisabled("Click a flipped face to isolate its vertices")

    psim.Separator()
    psim.TextUnformatted("Collapse Trace  (click fine_mesh_verts point cloud)")
    c, v = psim.Checkbox("Incident faces only (direct, fewer steps)", _use_incident_steps)
    if c:
        _use_incident_steps = v
        if _selected_vtx >= 0:
            _current_step_idx = 0
            _load_vtx_collapse_steps(_selected_vtx)
            _run_query_intermediates(_selected_vtx)
            _rebuild_step_viz()
            if _canonical_view:
                _rebuild_canonical_view()
    psim.SameLine()
    psim.TextDisabled("(unchecked = full one-ring)")

    if _selected_vtx >= 0:
        n_steps = len(_vtx_collapse_steps)
        psim.TextUnformatted(f"Vertex {_selected_vtx}  |  {n_steps} collapse steps")

        if n_steps > 0:
            ci_cur = _vtx_collapse_steps[_current_step_idx][0]
            psim.TextUnformatted(
                f"Step {_current_step_idx + 1} / {n_steps}  "
                f"(collapse #{ci_cur})")

            if psim.Button("< Prev"):
                if _current_step_idx > 0:
                    _current_step_idx -= 1
                    if _canonical_view:
                        _rebuild_canonical_view()
                    else:
                        _rebuild_step_viz()
                    _log_step_info()
            psim.SameLine()
            if psim.Button("Next >"):
                if _current_step_idx < n_steps - 1:
                    _current_step_idx += 1
                    if _canonical_view:
                        _rebuild_canonical_view()
                    else:
                        _rebuild_step_viz()
                    _log_step_info()
        else:
            psim.TextUnformatted("(vertex not involved in any collapse)")

        # ---- canonical / main view switch (mirrors visualizer.cpp) ----
        psim.Separator()
        if not _canonical_view:
            if psim.Button("Canonical View"):
                _canonical_view = True
                try:
                    ps.remove_all_structures()
                except Exception:
                    pass
                _rebuild_canonical_view()
                try:
                    ps.reset_camera_to_home_view()
                except Exception:
                    pass
            psim.SameLine()
            psim.TextUnformatted("(ring + UV panels only)")

            # flat UV sheet controls shown only in main view
            c, v = psim.SliderFloat("UV scale", _uv_scale, 0.5, 10.0)
            if c:
                _uv_scale = v
                _rebuild_uv_sheet_viz()
            c, v = psim.SliderFloat("UV plane Z", _uv_plane_z, -5.0, 5.0)
            if c:
                _uv_plane_z = v
                _rebuild_uv_sheet_viz()
        else:
            if psim.Button("Back to Main View"):
                _canonical_view = False
                try:
                    ps.remove_all_structures()
                except Exception:
                    pass
                _rebuild_all()
            # Domain toggle — only one active at a time
            c, chk = psim.Checkbox("UV domain (pre/post panels)", _canonical_domain == 'uv')
            if c and chk:
                _canonical_domain = 'uv'
                _rebuild_canonical_view()
            psim.SameLine()
            c, chk = psim.Checkbox("Spatial domain (ring)", _canonical_domain == 'ring')
            if c and chk:
                _canonical_domain = 'ring'
                _rebuild_canonical_view()

            if _canonical_domain == 'uv':
                psim.TextDisabled("Blue = UV_pre (down)   Orange = UV_post (offset above)")
                c, v = psim.SliderFloat("Pre/Post offset", _uv_post_offset, 0.0, 5.0)
                if c:
                    _uv_post_offset = v
                    _rebuild_canonical_view()
            else:
                psim.TextDisabled("Blue = 3-D one-ring")

        psim.Separator()
        if psim.Button("Clear vertex"):
            was_canonical = _canonical_view
            _selected_vtx = -1
            _vtx_collapse_steps.clear()
            _vtx_query_positions.clear()
            _current_step_idx = 0
            _canonical_view   = False
            if was_canonical:
                try:
                    ps.remove_all_structures()
                except Exception:
                    pass
                _rebuild_all()
            else:
                _rebuild_vtx_trajectory()
                _rebuild_step_viz()
                _clear_canonical()
    else:
        psim.TextUnformatted("No vertex selected — click fine_mesh_verts")

    psim.End()

    if changed and not _canonical_view:
        _rebuild_all()

    # ---- pick detection ----
    io = psim.GetIO()
    if io.MouseClicked[0]:
        try:
            pr = ps.pick(screen_coords=io.MousePos)
        except Exception:
            pr = None
        if pr is not None and getattr(pr, 'is_hit', False):
            sname = getattr(pr, 'structure_name', None)
            sdata = getattr(pr, 'structure_data', None) or {}

            if sname == FINE_VTX_PC:
                vidx = int(sdata.get('index', -1))
                nV   = _bundle.fineV.shape[0]
                if 0 <= vidx < nV:
                    _selected_vtx     = vidx
                    _current_step_idx = 0
                    _load_vtx_collapse_steps(vidx)
                    _run_query_intermediates(vidx)
                    _rebuild_vtx_trajectory()
                    _rebuild_step_viz()
                    _rebuild_canonical_view()
                    # Re-register to clear Polyscope's built-in selection indicator
                    # (picked point is rendered at larger radius by default)
                    _rebuild_fine_vtx_pc()

            elif sname == DEFORM_MESH:
                etype = sdata.get('element_type', None)
                fidx  = int(sdata.get('index', -1))
                nF    = _bundle.fineF.shape[0]
                if etype == 'face' and 0 <= fidx < nF and fidx != _selected_deform_face:
                    _selected_deform_face = fidx
                    _rebuild_arrows()
                    _rebuild_selection()

            elif sname == FLIPPED_MESH:
                etype = sdata.get('element_type', None)
                fidx  = int(sdata.get('index', -1))
                if (etype == 'face' and _flipped_faces_glob is not None
                        and 0 <= fidx < len(_flipped_faces_glob)):
                    _selected_flipped_face = -1 if fidx == _selected_flipped_face else fidx
                    _rebuild_flipped_vtx_colors()
                    _rebuild_flipped_arrows()
                    # clear previous on-the-fly query points
                    if ps.has_point_cloud(FLIPPED_F2C_PC):
                        ps.remove_point_cloud(FLIPPED_F2C_PC)
                    if _selected_flipped_face >= 0:
                        vids     = _flipped_faces_glob[_selected_flipped_face]
                        active_v = _active_deform_mesh_v()
                        batch_map = getattr(_bundle, '_f2c_vtx_steps', None)
                        mode_str  = "F2C" if _use_f2c_deform else "Bundle"
                        print(f"\n[flipped face {_selected_flipped_face}]  mode={mode_str}")
                        query_pts = []
                        for k, vid in enumerate(vids):
                            fine_pos   = _bundle.fineV[vid]
                            deform_pos = active_v[vid] if active_v is not None else fine_pos
                            # run f2c query fresh for this vertex
                            if batch_map is not None and vid in batch_map:
                                steps = batch_map[vid]
                                positions = query_vertex_f2c_intermediates(
                                    steps, _bundle.fineV, fine_pos)
                                query_final = positions[-1]
                                n_steps = len(steps)
                            else:
                                query_final = fine_pos
                                n_steps = 0
                            query_pts.append(query_final)
                            print(f"  v{k}: fine_id={vid}  n_steps={n_steps}")
                            print(f"        fine       =({fine_pos[0]:.6f}, {fine_pos[1]:.6f}, {fine_pos[2]:.6f})")
                            print(f"        deformed   =({deform_pos[0]:.6f}, {deform_pos[1]:.6f}, {deform_pos[2]:.6f})")
                            print(f"        query_final=({query_final[0]:.6f}, {query_final[1]:.6f}, {query_final[2]:.6f})")
                            print(f"        deformed==query: {np.allclose(deform_pos, query_final, atol=1e-6)}")
                        # visualize the 3 on-the-fly query positions at deformed Z
                        pts = np.array(query_pts)
                        pts[:, 2] += _z_offset
                        pc = ps.register_point_cloud(FLIPPED_F2C_PC, pts)
                        pc.set_color((1.0, 1.0, 0.0))   # yellow — distinct from everything else
                        pc.set_radius(0.012, relative=True)

            elif sname == FINE_MESH:
                etype = sdata.get('element_type', None)
                vidx  = int(sdata.get('index', -1))
                nV    = _bundle.fineV.shape[0]
                if etype == 'vertex' and 0 <= vidx < nV:
                    _selected_vtx     = vidx
                    _current_step_idx = 0
                    _load_vtx_collapse_steps(vidx)
                    _rebuild_vtx_trajectory()
                    _rebuild_uv_sheet_viz()
                    _rebuild_canonical_view()



# ---------------------------------------------------------------------------
# Diagnostic: decIM / subsetVIdx coverage
# ---------------------------------------------------------------------------

def _log_decim_stats():
    if not _bundle.has_ssp_data:
        return

    fineF = _bundle.fineF
    NV    = _bundle.fineV.shape[0]
    NF    = fineF.shape[0]
    nFO   = len(_bundle.decIM)

    # ---- face-level: how many original faces have any collapse entry ----
    nonempty_faces = sum(1 for d in _bundle.decIM if len(d) > 0)
    print(f"\n[decIM] ---- decIM coverage ----")
    print(f"[decIM] decIM entries  : {nFO}  (fine faces: {NF})")
    print(f"[decIM] non-empty rows : {nonempty_faces} / {nFO}  "
          f"({100*nonempty_faces/nFO:.1f}%)" if nFO > 0 else "")

    # ---- vertex-level: which fine verts have at least one incident face with decIM ----
    vtx_face_covered = np.zeros(NV, dtype=bool)
    for fi in range(min(NF, nFO)):
        if len(_bundle.decIM[fi]) > 0:
            vtx_face_covered[fineF[fi, 0]] = True
            vtx_face_covered[fineF[fi, 1]] = True
            vtx_face_covered[fineF[fi, 2]] = True
    n_face_covered = int(vtx_face_covered.sum())
    print(f"[decIM] fine verts with ≥1 incident non-empty decIM face : "
          f"{n_face_covered} / {NV}  ({100*n_face_covered/NV:.1f}%)")
    print(f"[decIM] fine verts with NO  incident non-empty decIM face : "
          f"{NV - n_face_covered} / {NV}  ({100*(NV-n_face_covered)/NV:.1f}%)")

    # ---- vertex-level: which fine verts appear in any sheet's subsetVIdx ----
    vtx_in_sheet = np.zeros(NV, dtype=bool)
    for cd in _bundle.decInfo:
        for sheet in cd.sheets:
            for v in sheet.subsetVIdx:
                if 0 <= v < NV:
                    vtx_in_sheet[v] = True
    n_in_sheet = int(vtx_in_sheet.sum())
    print(f"[decIM] fine verts appearing in ≥1 sheet subsetVIdx       : "
          f"{n_in_sheet} / {NV}  ({100*n_in_sheet/NV:.1f}%)")
    print(f"[decIM] fine verts never in any sheet subsetVIdx           : "
          f"{NV - n_in_sheet} / {NV}  ({100*(NV-n_in_sheet)/NV:.1f}%)")

    # ---- total collapse-step count across all vertices ----
    total_entries = sum(len(d) for d in _bundle.decIM)
    print(f"[decIM] total (face, collapse) pairs in decIM              : {total_entries}")
    print(f"[decIM] SSP collapses recorded                             : {len(_bundle.decInfo)}")

    # ---- simulate _find_vtx_collapse_steps for every fine vertex ----
    # A face in decIM[fi] for collapse ci doesn't mean v is in ci's one-ring;
    # this sweep finds how many vertices actually get ≥1 step.
    print(f"[decIM] sweeping all {NV} fine vertices for reachable steps …")
    step_counts = np.zeros(NV, dtype=np.int32)
    for v in range(NV):
        faces_of_v = np.where(np.any(fineF == v, axis=1))[0]
        seen: set = set()
        for f in faces_of_v:
            fi = int(f)
            if fi < nFO:
                for ci in _bundle.decIM[fi]:
                    seen.add(ci)
        count = 0
        for ci in seen:
            cd = _bundle.decInfo[ci]
            for sheet in cd.sheets:
                if np.any(sheet.subsetVIdx == v):
                    count += 1
                    break
        step_counts[v] = count

    zero_steps = int((step_counts == 0).sum())
    print(f"[decIM] fine verts with 0 reachable collapse steps         : {zero_steps} / {NV}  ({100*zero_steps/NV:.1f}%)")
    print(f"[decIM] fine verts with ≥1 reachable collapse steps        : {NV-zero_steps} / {NV}  ({100*(NV-zero_steps)/NV:.1f}%)")
    if NV > zero_steps:
        nz = step_counts[step_counts > 0]
        print(f"[decIM] steps per vertex (non-zero): min={nz.min()}  median={int(np.median(nz))}  max={nz.max()}  mean={nz.mean():.1f}")
    print(f"[decIM] --------------------------------\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global _bundle, _bundle_dir

    c2f_path    = rf"C:\\Users\\alirz\\Projects\\Graphics\\Neural QMAT\\external\\surf_subgrid_SSP_orig\\10_collapse_viz\\correspondence_01_00040057_f8f78dbd17414efda75bc437_trimesh_000_mat_initial.c2f"
    _bundle_dir = os.path.dirname(os.path.abspath(c2f_path))
    _bundle     = load_bundle(c2f_path)

    print(f"Fine:   {_bundle.fineV.shape[0]} verts  {_bundle.fineF.shape[0]} faces")
    if _bundle.has_ssp_data:
        print(f"SSP:    {len(_bundle.decInfo)} collapses")

    ok = _compute_vertex_correspondences()
    if not ok:
        print("ERROR: could not compute vertex correspondences — exiting.")
        sys.exit(1)

    _compute_f2c_deformed_mesh()

    _log_decim_stats()

    ps.init()
    ps.set_program_name("Fine -> Deformed Fine")
    ps.set_up_dir("y_up")

    _rebuild_all()

    ps.set_user_callback(ui_callback)
    ps.show()


if __name__ == "__main__":
    main()
