#!/usr/bin/env python3
"""
c2f_viz.py  —  Python Polyscope visualizer for .c2f coarse-to-fine bundles.

Mirrors 11_correspond_viz/ (C++ with Polyscope):
  - Fine mesh: semi-transparent grey surface
  - Coarse mesh: blue, floating above with Z offset
  - Arrows: every N-th coarse vertex → its fine correspondent
  - Click any sample point  → highlight that correspondence
  - Click any coarse face   → fold-sample N barycentric points via SSP query
  - Convex hull of the fine sample points drawn as a teal patch

Usage:
    python c2f_viz.py  <bundle.c2f>

Dependencies:
    pip install polyscope numpy scipy
    (c2f_query.py must be in the same directory or on PYTHONPATH)
"""

import sys
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from scipy.spatial import ConvexHull, QhullError

from c2f_query import load_bundle, query_coarse_to_fine

# ---- structure name constants ----
FINE_MESH          = "fine_mesh"
COARSE_MESH        = "coarse_mesh"
SAMPLE_COARSE      = "sample_coarse"
SAMPLE_FINE        = "sample_fine"
SEL_COARSE         = "sel_coarse"
SEL_FINE           = "sel_fine"
FACE_COARSE        = "face_sample_coarse"
FACE_FINE          = "face_sample_fine"
FACE_HULL          = "face_sample_hull"

# ---- mutable global state (mirrors C++ globals) ----
_bundle            = None   # Bundle loaded from .c2f

_z_offset          = 1.0
_sample_step       = 1
_show_arrows       = True
_selected_sample   = -1   # index into the subsampled cloud (not raw coarse index)
_selected_face     = -1

_face_n            = 20
_show_face_coarse  = True
_show_face_fine    = True

# polyscope object references kept for visibility toggles
_coarse_pc         = None
_fine_pc           = None
_face_coarse_pc    = None
_face_fine_pc      = None

_face_sample_reg   = False
_face_hull_reg     = False
_face_highlight_active = False


# ---------------------------------------------------------------------------
# Convex hull helpers
# ---------------------------------------------------------------------------

def _convex_hull_faces(pts: np.ndarray):
    """
    PCA-project pts to best-fit plane, compute 2D convex hull, fan-triangulate.
    Returns (verts, faces) where faces indexes into pts, or (None, None) on failure.
    """
    n = len(pts)
    if n < 3:
        return None, None

    center = pts.mean(axis=0)
    C = pts - center
    try:
        _, _, Vt = np.linalg.svd(C, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None

    ax0 = Vt[0]   # dominant axis
    ax1 = Vt[1]   # second axis

    p2 = np.column_stack([C @ ax0, C @ ax1])

    try:
        hull2d = ConvexHull(p2)
    except (QhullError, ValueError):
        return None, None

    # hull2d.vertices is CCW ordered for 2-D input
    verts = hull2d.vertices
    h = len(verts)
    if h < 3:
        return None, None

    # Fan triangulation from verts[0]
    faces = np.zeros((h - 2, 3), dtype=np.int32)
    for i in range(h - 2):
        faces[i] = [verts[0], verts[i + 1], verts[i + 2]]
    return pts, faces


# ---------------------------------------------------------------------------
# Coarse mesh (Z-offset copy of coarseV)
# ---------------------------------------------------------------------------

def rebuild_coarse_viz():
    V = _bundle.coarseV.copy()
    V[:, 2] += _z_offset
    m = ps.register_surface_mesh(COARSE_MESH, V, _bundle.coarseF)
    m.set_color((0.25, 0.52, 0.95))
    m.set_edge_width(0.5)
    m.set_smooth_shade(False)


# ---------------------------------------------------------------------------
# Sample-level correspondence arrows (every _sample_step coarse vertices)
# ---------------------------------------------------------------------------

def rebuild_sample_viz():
    global _coarse_pc, _fine_pc

    NC   = _bundle.coarseV.shape[0]
    step = max(1, _sample_step)
    idx  = list(range(0, NC, step))
    NS   = len(idx)

    coarse_pts  = np.empty((NS, 3))
    fine_pts    = np.empty((NS, 3))
    arrow_vecs  = np.empty((NS, 3))

    for j, i in enumerate(idx):
        cp                = _bundle.coarseV[i].copy()
        cp[2]            += _z_offset
        coarse_pts[j]     = cp
        fine_pts[j]       = _bundle.coarseV[i] + _bundle.corrVec[i]
        arr               = _bundle.corrVec[i].copy()
        arr[2]           -= _z_offset       # compensate for coarse lift
        arrow_vecs[j]     = arr

    _coarse_pc = ps.register_point_cloud(SAMPLE_COARSE, coarse_pts)
    _coarse_pc.set_color((1.0, 0.85, 0.0))
    _coarse_pc.set_radius(0.006, relative=True)

    _coarse_pc.add_vector_quantity("to_fine", arrow_vecs,
        vectortype="ambient", enabled=_show_arrows, color=(1.0, 0.35, 0.05))

    _fine_pc = ps.register_point_cloud(SAMPLE_FINE, fine_pts)
    _fine_pc.set_color((0.15, 0.85, 0.40))
    _fine_pc.set_radius(0.006, relative=True)


# ---------------------------------------------------------------------------
# Single-vertex selection highlight
# ---------------------------------------------------------------------------

def rebuild_selection_viz():
    for name in (SEL_COARSE, SEL_FINE):
        if ps.has_point_cloud(name):
            ps.remove_point_cloud(name)

    if _selected_sample < 0:
        return

    step      = max(1, _sample_step)
    ci        = _selected_sample * step
    NC        = _bundle.coarseV.shape[0]
    if ci >= NC:
        return

    cp        = _bundle.coarseV[ci].copy()
    cp[2]    += _z_offset
    fp        = _bundle.coarseV[ci] + _bundle.corrVec[ci]
    arrow     = _bundle.corrVec[ci].copy()
    arrow[2] -= _z_offset

    sc = ps.register_point_cloud(SEL_COARSE, cp[np.newaxis])
    sc.set_color((1.0, 0.15, 0.15))
    sc.set_radius(0.012, relative=True)
    sc.add_vector_quantity("to_fine", arrow[np.newaxis],
        vectortype="ambient", enabled=True, color=(1.0, 0.0, 0.0))

    sf = ps.register_point_cloud(SEL_FINE, fp[np.newaxis])
    sf.set_color((1.0, 0.15, 0.15))
    sf.set_radius(0.012, relative=True)


# ---------------------------------------------------------------------------
# Face-level sampling  (mirrors face_sample_viz.cpp)
# ---------------------------------------------------------------------------

def _fold_sample_bc(n: int, seed: int = 42) -> np.ndarray:
    """Uniform fold-sampled barycentric coordinates, shape (n, 3)."""
    rng = np.random.default_rng(seed)
    r   = rng.random((n, 2))
    mask = (r[:, 0] + r[:, 1]) > 1.0
    r[mask] = 1.0 - r[mask]
    bc = np.empty((n, 3))
    bc[:, 0] = r[:, 0]
    bc[:, 1] = r[:, 1]
    bc[:, 2] = 1.0 - r[:, 0] - r[:, 1]
    return bc


def rebuild_face_sample_viz():
    global _face_coarse_pc, _face_fine_pc, _face_sample_reg, _face_hull_reg
    global _face_highlight_active

    # Remove old structures
    for name in (FACE_COARSE, FACE_FINE):
        if ps.has_point_cloud(name):
            ps.remove_point_cloud(name)
    if ps.has_surface_mesh(FACE_HULL):
        ps.remove_surface_mesh(FACE_HULL)
    _face_sample_reg = False
    _face_hull_reg   = False
    _face_coarse_pc  = None
    _face_fine_pc    = None

    # Face highlight on the coarse mesh
    if ps.has_surface_mesh(COARSE_MESH):
        mesh = ps.get_surface_mesh(COARSE_MESH)
        FC   = _bundle.coarseF.shape[0]
        if _selected_face < 0:
            if _face_highlight_active:
                mesh.remove_quantity("face_highlight")
                _face_highlight_active = False
        else:
            fc = np.tile([0.25, 0.52, 0.95], (FC, 1))
            fc[_selected_face] = [1.0, 0.65, 0.0]
            q = mesh.add_face_color_quantity("face_highlight", fc)
            q.set_enabled(True)
            _face_highlight_active = True

    if _selected_face < 0:
        return

    n   = max(1, _face_n)
    cf  = _selected_face
    ci0, ci1, ci2 = _bundle.coarseF[cf]
    cv0 = _bundle.coarseV[ci0]
    cv1 = _bundle.coarseV[ci1]
    cv2 = _bundle.coarseV[ci2]

    BC = _fold_sample_bc(n)

    # Coarse 3D positions (with Z offset for display)
    coarse_pts      = (BC[:, 0:1] * cv0 + BC[:, 1:2] * cv1 + BC[:, 2:3] * cv2)
    coarse_pts_disp = coarse_pts.copy()
    coarse_pts_disp[:, 2] += _z_offset

    if _bundle.has_ssp_data:
        gvi0 = int(_bundle.vtxMap[ci0])
        gvi1 = int(_bundle.vtxMap[ci1])
        gvi2 = int(_bundle.vtxMap[ci2])
        gfi  = int(_bundle.faceMap[cf])

        BF   = np.tile([gvi0, gvi1, gvi2], (n, 1)).astype(np.int32)
        FIdx = np.full(n, gfi, dtype=np.int32)

        query_coarse_to_fine(
            _bundle.decInfo,
            _bundle.decIM,
            _bundle.faceSheetID,
            BC, BF, FIdx,
        )

        fine_pts = (BC[:, 0:1] * _bundle.fineV[BF[:, 0]]
                  + BC[:, 1:2] * _bundle.fineV[BF[:, 1]]
                  + BC[:, 2:3] * _bundle.fineV[BF[:, 2]])

        # Convex hull of fine sample points
        hull_pts, hull_F = _convex_hull_faces(fine_pts)
        if hull_pts is not None:
            hm = ps.register_surface_mesh(FACE_HULL, hull_pts, hull_F)
            hm.set_color((0.10, 0.90, 0.75))
            hm.set_transparency(0.0)
            hm.set_edge_width(0.0)
            hm.set_smooth_shade(False)
            _face_hull_reg = True
    else:
        fine_pts = coarse_pts_disp.copy()
        print('[face_sample] No SSP data — need a v2 bundle for correspondence')

    arrow_vecs = fine_pts - coarse_pts_disp

    _face_coarse_pc = ps.register_point_cloud(FACE_COARSE, coarse_pts_disp)
    _face_coarse_pc.set_color((0.90, 0.40, 1.00))
    _face_coarse_pc.set_radius(0.0048, relative=True)
    _face_coarse_pc.set_enabled(_show_face_coarse)
    _face_coarse_pc.add_vector_quantity("to_fine", arrow_vecs,
        vectortype="ambient", enabled=True, color=(0.70, 0.00, 1.00))

    _face_fine_pc = ps.register_point_cloud(FACE_FINE, fine_pts)
    _face_fine_pc.set_color((0.20, 1.00, 0.85))
    _face_fine_pc.set_radius(0.0048, relative=True)
    _face_fine_pc.set_enabled(_show_face_fine)

    _face_sample_reg = True


def update_face_sample_visibility():
    if _face_coarse_pc is not None:
        _face_coarse_pc.set_enabled(_show_face_coarse)
    if _face_fine_pc is not None:
        _face_fine_pc.set_enabled(_show_face_fine)


def rebuild_all():
    rebuild_coarse_viz()
    rebuild_sample_viz()
    rebuild_selection_viz()
    rebuild_face_sample_viz()


# ---------------------------------------------------------------------------
# Polyscope UI callback
# ---------------------------------------------------------------------------

def ui_callback():
    global _z_offset, _sample_step, _show_arrows
    global _selected_sample, _selected_face
    global _face_n, _show_face_coarse, _show_face_fine

    NC = _bundle.coarseV.shape[0]

    # ---- pick detection ----
    new_sel  = _selected_sample
    new_face = _selected_face

    try:
        if ps.have_selection():
            struct, idx = ps.get_selection()
            name = getattr(struct, 'name', None)

            if name == SAMPLE_COARSE:
                new_sel = int(idx)

            elif name == COARSE_MESH:
                nV = _bundle.coarseV.shape[0]
                # Polyscope: indices 0..nV-1 = vertices, nV..nV+nF-1 = faces
                if idx >= nV:
                    new_face = int(idx - nV)
    except Exception:
        pass

    changed = False

    if new_sel != _selected_sample:
        _selected_sample = new_sel
        rebuild_selection_viz()

    if new_face != _selected_face:
        _selected_face = new_face
        rebuild_face_sample_viz()

    # ---- ImGui panel ----
    try:
        psim.SetNextWindowSize((350, 260), psim.ImGuiCond_FirstUseEver)
    except Exception:
        pass
    psim.Begin("Correspondence", True)

    c, v = psim.SliderFloat("Z offset (coarse)", _z_offset, -2.0, 2.0)
    if c:
        _z_offset = v
        changed = True

    c, v = psim.SliderInt("Sample every X pts", _sample_step, 1, 40)
    if c:
        _sample_step = v
        _selected_sample = -1
        changed = True

    c, v = psim.Checkbox("Show arrows", _show_arrows)
    if c:
        _show_arrows = v
        changed = True

    step  = max(1, _sample_step)
    shown = (NC + step - 1) // step
    psim.TextUnformatted(f"Showing {shown} / {NC} coarse vertices")

    if _selected_sample >= 0:
        ci = _selected_sample * step
        psim.TextUnformatted(f"Selected: sample[{_selected_sample}] → coarse[{ci}]")
        if psim.Button("Clear selection"):
            _selected_sample = -1
            rebuild_selection_viz()

    if changed:
        rebuild_all()

    psim.Separator()
    psim.TextUnformatted("Face sampling  (click a coarse face)")

    c, v = psim.InputInt("n samples", _face_n)
    if c:
        _face_n = max(1, v)
        if _selected_face >= 0:
            rebuild_face_sample_viz()

    vis_changed = False
    c, v = psim.Checkbox("coarse samples", _show_face_coarse)
    if c:
        _show_face_coarse = v
        vis_changed = True
    psim.SameLine()
    c, v = psim.Checkbox("fine samples", _show_face_fine)
    if c:
        _show_face_fine = v
        vis_changed = True

    if vis_changed:
        update_face_sample_visibility()

    if _selected_face >= 0:
        psim.TextUnformatted(f"Face {_selected_face}  |  {_face_n} samples")
        if psim.Button("Clear face"):
            _selected_face = -1
            rebuild_face_sample_viz()

    psim.End()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global _bundle

    # if len(sys.argv) < 2:
    #     print("Usage: python c2f_viz.py  <bundle.c2f>", file=sys.stderr)
    #     sys.exit(1)
    c2f_path = rf"C:\\Users\\alirz\\Projects\\Graphics\\Neural QMAT\\external\\surf_subgrid_SSP_orig\\10_collapse_viz\\c2f_bundle.c2f"
    _bundle = load_bundle(c2f_path)
    print(f"Coarse: {_bundle.coarseV.shape[0]} verts  {_bundle.coarseF.shape[0]} faces")
    print(f"Fine:   {_bundle.fineV.shape[0]} verts  {_bundle.fineF.shape[0]} faces")
    if _bundle.has_ssp_data:
        print(f"SSP:    {len(_bundle.decInfo)} collapses  "
              f"{len(_bundle.decIM)} tracked faces")

    ps.init()
    ps.set_program_name("Coarse-to-Fine Correspondence")
    ps.set_up_dir("z_up")

    # Fine mesh — always shown at ground level, semi-transparent
    fm = ps.register_surface_mesh(FINE_MESH, _bundle.fineV, _bundle.fineF)
    fm.set_color((0.55, 0.55, 0.55))
    fm.set_edge_width(0.3)
    fm.set_smooth_shade(False)
    fm.set_transparency(0.6)

    rebuild_all()

    ps.set_user_callback(ui_callback)
    ps.show()


if __name__ == "__main__":
    main()
