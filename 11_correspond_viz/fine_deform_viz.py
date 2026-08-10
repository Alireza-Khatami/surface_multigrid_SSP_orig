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


def _rebuild_all():
    _rebuild_meshes()
    _rebuild_deform_pc()
    _rebuild_arrows()
    _rebuild_selection()
    _rebuild_flipped_viz()


# ---------------------------------------------------------------------------
# UI callback
# ---------------------------------------------------------------------------

def ui_callback():
    global _z_offset, _sample_step, _show_arrows, _selected_deform_face, _show_flipped

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

    psim.End()

    if changed:
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
