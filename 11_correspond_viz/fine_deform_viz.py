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
import glob
import os
import math
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from c2f_query import load_bundle

# ---- structure names ----
FINE_MESH     = "fine_mesh"
DEFORM_MESH   = "deformed_fine_mesh"
DEFORM_PC     = "deformed_fine_verts"
ARROWS_PC     = "fine_to_deformed"
SEL_ARROWS_PC = "sel_fine_to_deformed"
FLIPPED_PC    = "flipped_face_verts"
TRACKED_VTX_PC = "tracked_vtx"
TRAJ_CURVE     = "vtx_trajectory"
UV_SHEET_MESH  = "uv_sheet"
UV_TRACKED_PC  = "uv_tracked_vtx"
FINE_VTX_PC    = "fine_mesh_verts"   # selectable point cloud over fine mesh
CV_RING_MESH   = "cv_ring"           # canonical: 3D one-ring (blue)
CV_UV_PRE      = "cv_uv_pre"         # canonical: UV_pre panel (blue)
CV_UV_POST     = "cv_uv_post"        # canonical: UV_post panel (orange)
CV_TRACKED     = "cv_tracked_vtx"    # canonical: tracked vertex dot + arrow
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
_show_flipped           = True
_flipped_highlight_active = False

# collapse-trace state
_selected_vtx       = -1
_vtx_collapse_steps = []   # list of (collapse_idx, SheetData, local_vtx_idx)
_current_step_idx   = 0
_uv_plane_z         = -1.5  # Z position of UV flat-mesh in 3D viewport
_uv_scale           = 3.0   # scale applied to normalised UV coords
_canonical_view     = False
_uv_elev            = 1.5   # UV panel elevation above ring (× ring_span), like gUVOffset


# ---------------------------------------------------------------------------

def _load_vertex_tracker(bundle_path: str):
    global _fine_ids, _fine_pts, _deform_pts, _deform_mesh_v, _fine_id_to_row

    bundle_dir = os.path.dirname(os.path.abspath(bundle_path))
    vtx_files  = sorted(glob.glob(os.path.join(bundle_dir, "samples_vertices_*.txt")))

    if not vtx_files:
        print("[vtx] No samples_vertices_*.txt found in", bundle_dir)
        return False

    print(f"[vtx] Loading {os.path.basename(vtx_files[0])}")

    rows = []
    with open(vtx_files[0]) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            rows.append([float(x) for x in parts[:8]])

    if not rows:
        print("[vtx] Empty file — aborting.")
        return False

    data = np.array(rows, dtype=np.float64)
    N    = len(data)
    print(f"[vtx] {N} tracked fine vertices")

    fine_ids = data[:, 0].astype(np.int32)

    if not _bundle.has_ssp_data or _bundle.vtxMap is None:
        print("[vtx] Bundle has no vtxMap — cannot compute coarse positions.")
        return False

    gv_to_cv = {int(_bundle.vtxMap[i]): i for i in range(len(_bundle.vtxMap))}
    coarseV  = _bundle.coarseV
    bc       = data[:, 2:5]
    bv0      = data[:, 5].astype(np.int32)
    bv1      = data[:, 6].astype(np.int32)
    bv2      = data[:, 7].astype(np.int32)
    cv0 = np.array([gv_to_cv.get(int(v), 0) for v in bv0], dtype=np.int32)
    cv1 = np.array([gv_to_cv.get(int(v), 0) for v in bv1], dtype=np.int32)
    cv2 = np.array([gv_to_cv.get(int(v), 0) for v in bv2], dtype=np.int32)
    coarse_corr = (bc[:, 0:1] * coarseV[cv0]
                 + bc[:, 1:2] * coarseV[cv1]
                 + bc[:, 2:3] * coarseV[cv2])

    _fine_ids       = fine_ids
    _fine_pts       = _bundle.fineV[fine_ids]
    _deform_pts     = coarse_corr
    _fine_id_to_row = {int(vid): row for row, vid in enumerate(fine_ids)}

    deform_v = _bundle.fineV.copy()
    deform_v[fine_ids] = coarse_corr
    _deform_mesh_v = deform_v

    return True


# ---------------------------------------------------------------------------
# Rebuild helpers
# ---------------------------------------------------------------------------

def _rebuild_meshes():
    fm = ps.register_surface_mesh(FINE_MESH, _bundle.fineV, _bundle.fineF)
    fm.set_color((0.55, 0.55, 0.55))
    fm.set_edge_width(0.3)
    fm.set_smooth_shade(False)
    fm.set_transparency(0.4)

    if _deform_mesh_v is not None:
        dv = _deform_mesh_v.copy()
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
    For every face where dot(orig_normal, deformed_normal) < 0 (winding reversed):
      - Color it red on the deformed mesh via a face color quantity.
      - Draw fine->deformed arrows for the unique tracked corner vertices.
    A uniform Z offset doesn't change face normals, so we compare without it.
    """
    global _flipped_highlight_active

    if ps.has_point_cloud(FLIPPED_PC):
        ps.remove_point_cloud(FLIPPED_PC)
    if ps.has_surface_mesh(DEFORM_MESH):
        dm = ps.get_surface_mesh(DEFORM_MESH)
        if _flipped_highlight_active:
            dm.remove_quantity("flip_highlight")
            _flipped_highlight_active = False

    if not _show_flipped or _deform_mesh_v is None:
        return

    fineF = _bundle.fineF
    fineV = _bundle.fineV
    nF    = fineF.shape[0]

    # original face normals
    v0o = fineV[fineF[:, 0]];  v1o = fineV[fineF[:, 1]];  v2o = fineV[fineF[:, 2]]
    orig_n = np.cross(v1o - v0o, v2o - v0o)

    # deformed face normals (no Z offset needed — uniform shift preserves normals)
    v0d = _deform_mesh_v[fineF[:, 0]];  v1d = _deform_mesh_v[fineF[:, 1]];  v2d = _deform_mesh_v[fineF[:, 2]]
    deform_n = np.cross(v1d - v0d, v2d - v0d)

    flipped = np.sum(orig_n * deform_n, axis=1) < 0  # (nF,) bool
    n_flipped = int(flipped.sum())
    print(f"[flip] {n_flipped} / {nF} faces flipped")

    if n_flipped == 0:
        return

    # color quantity on deformed mesh: red for flipped, default orange elsewhere
    if ps.has_surface_mesh(DEFORM_MESH):
        dm = ps.get_surface_mesh(DEFORM_MESH)
        fc = np.tile([0.95, 0.50, 0.10], (nF, 1))
        fc[flipped] = [1.0, 0.10, 0.10]
        dm.add_color_quantity("flip_highlight", fc, defined_on='faces')
        _flipped_highlight_active = True

    # arrows for unique tracked corner vertices of all flipped faces
    if _fine_id_to_row is None:
        return

    flipped_corners = np.unique(fineF[flipped])  # unique fine vertex ids
    rows = [_fine_id_to_row[int(vid)]
            for vid in flipped_corners if int(vid) in _fine_id_to_row]
    if not rows:
        return

    src  = _fine_pts[rows]
    dst  = _deform_pts[rows].copy()
    dst[:, 2] += _z_offset
    vecs = dst - src

    pc = ps.register_point_cloud(FLIPPED_PC, src)
    pc.set_color((1.0, 0.10, 0.10))
    pc.set_radius(0.005, relative=True)
    pc.add_vector_quantity("to_deformed", vecs,
                           vectortype="ambient", enabled=True,
                           color=(1.0, 0.30, 0.30))


def _rebuild_fine_vtx_pc():
    """Selectable point cloud over all fine mesh vertices."""
    if ps.has_point_cloud(FINE_VTX_PC):
        ps.remove_point_cloud(FINE_VTX_PC)
    pc = ps.register_point_cloud(FINE_VTX_PC, _bundle.fineV)
    pc.set_color((0.75, 0.75, 0.75))
    pc.set_radius(0.003, relative=True)


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

    # ---- UV scale: map UV extent → ring_span ----
    uv_span = max(uv_pre[:, 0].max() - uv_pre[:, 0].min(),
                  uv_pre[:, 1].max() - uv_pre[:, 1].min())
    uv_scale = span / uv_span if uv_span > 1e-10 else 1.0

    # ---- UV panels → 3D, floating above ring along nrm by _uv_elev * span ----
    panel_c = centroid + nrm * _uv_elev * span

    def to3d(UV):
        dU = (UV[:, 0:1] - uc) * uv_scale
        dV = (UV[:, 1:2] - vc) * uv_scale
        return panel_c + dU * t1 + dV * t2

    uv_pre_3d  = to3d(uv_pre)
    uv_post_3d = to3d(uv_post)

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

    def rot(P):
        return (R @ (P - centroid).T).T

    vr   = rot(verts)
    upr  = rot(uv_pre_3d)
    upor = rot(uv_post_3d)

    return {
        'V_ring':       vr,
        'uv_pre':       upr,
        'uv_post':      upor,
        'F':            F,
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
    """
    Canonical view matching visualizer.cpp show_canonical_view():
      Blue  = 3D one-ring (approx) + UV_pre panel
      Orange = UV_post panel
      Orange dot + yellow arrow = tracked vertex UV_pre → UV_post displacement
    """
    _clear_canonical()
    if not _canonical_view or _selected_vtx < 0 or not _vtx_collapse_steps:
        return

    step_idx = max(0, min(_current_step_idx, len(_vtx_collapse_steps) - 1))
    _ci, sheet, local_v = _vtx_collapse_steps[step_idx]

    geo = _compute_canonical(sheet, local_v)

    # Blue — 3D one-ring (approximate from original fine mesh)
    rm = ps.register_surface_mesh(CV_RING_MESH, geo['V_ring'], geo['F'])
    rm.set_color((0.3, 0.55, 1.0))
    rm.set_edge_width(1.5)
    rm.set_smooth_shade(False)
    rm.set_transparency(0.5)

    # Blue — UV_pre flat panel
    pm = ps.register_surface_mesh(CV_UV_PRE, geo['uv_pre'], geo['F'])
    pm.set_color((0.3, 0.55, 1.0))
    pm.set_edge_width(1.5)
    pm.set_smooth_shade(False)

    # Orange — UV_post flat panel
    qm = ps.register_surface_mesh(CV_UV_POST, geo['uv_post'], geo['F'])
    qm.set_color((1.0, 0.5, 0.15))
    qm.set_edge_width(1.5)
    qm.set_smooth_shade(False)
    qm.set_transparency(0.35)

    # Tracked vertex: orange dot at UV_pre position + yellow arrow → UV_post
    pt3d  = geo['tracked_pre'][np.newaxis]
    arrow = (geo['tracked_post'] - geo['tracked_pre'])[np.newaxis]
    pc = ps.register_point_cloud(CV_TRACKED, pt3d)
    pc.set_color((1.0, 0.3, 0.0))
    pc.set_radius(0.015, relative=True)
    pc.add_vector_quantity("uv_disp", arrow, vectortype="ambient",
                           enabled=True, color=(1.0, 0.85, 0.0))


def _find_vtx_collapse_steps(v: int):
    """
    Walk decIM forward and collect every collapse step that touches vertex v.
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
        for sheet in cd.sheets:
            hits = np.where(sheet.subsetVIdx == v)[0]
            if len(hits):
                steps.append((ci, sheet, int(hits[0])))
                break   # v appears in at most one sheet per collapse
    _vtx_collapse_steps = steps
    print(f"[trace] vertex {v}: {len(steps)} collapse steps "
          f"(from {len(faces_of_v)} incident fine faces)")


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
    pc.set_radius(0.012, relative=True)

    if _fine_id_to_row is not None and _selected_vtx in _fine_id_to_row:
        row  = _fine_id_to_row[_selected_vtx]
        corr = _deform_pts[row].copy()
        corr[2] += _z_offset
        nodes = np.array([fine_pos, corr])
        edges = np.array([[0, 1]], dtype=np.int32)
        try:
            cn = ps.register_curve_network(TRAJ_CURVE, nodes, edges)
            cn.set_color((0.1, 1.0, 0.3))
            cn.set_radius(0.003, relative=True)
        except Exception:
            pass


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
    _rebuild_uv_sheet_viz()


# ---------------------------------------------------------------------------
# UI callback
# ---------------------------------------------------------------------------

def ui_callback():
    global _z_offset, _sample_step, _show_arrows, _selected_deform_face, _show_flipped
    global _selected_vtx, _current_step_idx, _uv_scale, _uv_plane_z, _canonical_view, _uv_elev

    changed = False

    try:
        psim.SetNextWindowSize((330, 280), psim.ImGuiCond_FirstUseEver)
    except Exception:
        pass
    psim.Begin("Fine -> Deformed", True)

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

    if _show_flipped and _deform_mesh_v is not None:
        fineF    = _bundle.fineF
        fineV    = _bundle.fineV
        v0o = fineV[fineF[:, 0]];  v1o = fineV[fineF[:, 1]];  v2o = fineV[fineF[:, 2]]
        v0d = _deform_mesh_v[fineF[:, 0]];  v1d = _deform_mesh_v[fineF[:, 1]];  v2d = _deform_mesh_v[fineF[:, 2]]
        orig_n   = np.cross(v1o - v0o, v2o - v0o)
        deform_n = np.cross(v1d - v0d, v2d - v0d)
        n_flipped = int((np.sum(orig_n * deform_n, axis=1) < 0).sum())
        psim.TextUnformatted(f"Flipped: {n_flipped} / {fineF.shape[0]} faces")

    psim.Separator()
    psim.TextUnformatted("Collapse Trace  (click fine_mesh_verts point cloud)")

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
                        _rebuild_uv_sheet_viz()
            psim.SameLine()
            if psim.Button("Next >"):
                if _current_step_idx < n_steps - 1:
                    _current_step_idx += 1
                    if _canonical_view:
                        _rebuild_canonical_view()
                    else:
                        _rebuild_uv_sheet_viz()
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
            psim.TextDisabled("Blue = ring + UV_pre   Orange = UV_post")
            c, v = psim.SliderFloat("UV panel elev", _uv_elev, 0.0, 5.0)
            if c:
                _uv_elev = v
                _rebuild_canonical_view()

        psim.Separator()
        if psim.Button("Clear vertex"):
            was_canonical = _canonical_view
            _selected_vtx = -1
            _vtx_collapse_steps.clear()
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
                _rebuild_uv_sheet_viz()
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

            if sname == DEFORM_MESH:
                etype = sdata.get('element_type', None)
                fidx  = int(sdata.get('index', -1))
                nF    = _bundle.fineF.shape[0]
                if etype == 'face' and 0 <= fidx < nF and fidx != _selected_deform_face:
                    _selected_deform_face = fidx
                    _rebuild_arrows()
                    _rebuild_selection()

            elif sname == FINE_MESH:
                etype = sdata.get('element_type', None)
                vidx  = int(sdata.get('index', -1))
                nV    = _bundle.fineV.shape[0]
                if etype == 'vertex' and 0 <= vidx < nV:
                    _selected_vtx     = vidx
                    _current_step_idx = 0
                    _find_vtx_collapse_steps(vidx)
                    _rebuild_vtx_trajectory()
                    _rebuild_uv_sheet_viz()
                    _rebuild_canonical_view()

            elif sname == FINE_VTX_PC:
                etype = sdata.get('element_type', None)
                vidx  = int(sdata.get('index', -1))
                nV    = _bundle.fineV.shape[0]
                if etype == 'point' and 0 <= vidx < nV:
                    _selected_vtx     = vidx
                    _current_step_idx = 0
                    _find_vtx_collapse_steps(vidx)
                    _rebuild_vtx_trajectory()
                    _rebuild_uv_sheet_viz()
                    _rebuild_canonical_view()


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

    ok = _load_vertex_tracker(c2f_path)
    if not ok:
        print("ERROR: could not load vertex tracker — exiting.")
        sys.exit(1)

    ps.init()
    ps.set_program_name("Fine -> Deformed Fine")
    ps.set_up_dir("y_up")

    _rebuild_all()

    ps.set_user_callback(ui_callback)
    ps.show()


if __name__ == "__main__":
    main()
