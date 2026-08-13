#!/usr/bin/env python3
"""
c2f_viz.py  —  Python Polyscope visualizer for .c2f coarse-to-fine bundles.

Mirrors 11_correspond_viz/ (C++ with Polyscope):
  - Fine mesh: semi-transparent grey surface
  - Coarse mesh: blue, floating above with Z offset
  - Arrows: every N-th coarse vertex -> its fine correspondent
  - Click any sample point  -> highlight that correspondence
  - Click any coarse face   -> fold-sample N barycentric points via SSP query
  - Convex hull of the fine sample points drawn as a teal patch
  - Sample-tracker overlay: loads samples_fine_*.txt / samples_coarse_*.txt
    produced by 10_collapse_viz, shows forward-tracked coarse->fine correspondence
  - Vertex-tracker overlay: loads samples_vertices_*.txt, shows original fine
    vertices tracked to their coarse correspondences

Usage:
    python c2f_viz.py  <bundle.c2f>

Dependencies:
    pip install polyscope numpy scipy
    (c2f_query.py must be in the same directory or on PYTHONPATH)
"""

import sys
import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from scipy.spatial import ConvexHull, QhullError, cKDTree
import trimesh

from c2f_query import load_bundle, query_coarse_to_fine

# ---- structure name constants ----
FINE_MESH           = "fine_mesh"
COARSE_MESH         = "coarse_mesh"
SAMPLE_COARSE       = "sample_coarse"
SAMPLE_FINE         = "sample_fine"
SEL_COARSE          = "sel_coarse"
SEL_FINE            = "sel_fine"
FACE_COARSE         = "face_sample_coarse"
FACE_FINE           = "face_sample_fine"
FACE_HULL           = "face_sample_hull"
FACE_CENTS          = "coarse_face_cents"
TRACKER_FINE        = "tracker_fine"
TRACKER_COARSE      = "tracker_coarse"
FACE_TRACKER_FINE   = "face_tracker_fine"
FACE_TRACKER_COARSE = "face_tracker_coarse"
TRACKER_VTX_FINE    = "vtx_tracker_fine"
TRACKER_VTX_COARSE  = "vtx_tracker_coarse"
VTX_FACE_FINE       = "vtx_face_tracker_fine"
VTX_FACE_COARSE     = "vtx_face_tracker_coarse"

# ---- mutable global state ----
_bundle             = None
_bundle_dir         = ""   # directory containing the loaded .c2f file

_z_offset           = 1.0
_sample_step        = 500
_show_arrows        = True
_selected_sample    = -1
_selected_face      = -1

_face_n             = 20
_show_face_coarse   = False
_show_face_fine     = False

_coarse_pc          = None
_fine_pc            = None
_face_coarse_pc     = None
_face_fine_pc       = None

_face_sample_reg        = False
_face_hull_reg          = False
_face_highlight_active  = False

# Sample-tracker data (from samples_fine_*.txt / samples_coarse_*.txt)
_tracker_fine_pts    = None   # (N,3)  fine-mesh 3D positions
_tracker_coarse_pts  = None   # (N,3)  coarse-mesh 3D positions (no Z offset)
_tracker_coarse_fids = None   # (N,)   global gF face index per sample
_show_tracker        = True

# Vertex-tracker data (from samples_vertices_*.txt)
_tracker_vtx_fine_pts    = None   # (Nv,3)  original fine vertex positions
_tracker_vtx_coarse_pts  = None   # (Nv,3)  coarse correspondence positions (no Z offset)
_tracker_vtx_coarse_fids = None   # (Nv,)   global gF face index per vertex track
_show_vtx_tracker         = True
_deformed_fine_pts        = None   # (nFineV,3)  fineV with positions replaced by coarse corr.


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

    ax0 = Vt[0]
    ax1 = Vt[1]
    p2 = np.column_stack([C @ ax0, C @ ax1])

    try:
        hull2d = ConvexHull(p2)
    except (QhullError, ValueError):
        return None, None

    verts = hull2d.vertices
    h = len(verts)
    if h < 3:
        return None, None

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
        cp             = _bundle.coarseV[i].copy()
        cp[2]         += _z_offset
        coarse_pts[j]  = cp
        fine_pts[j]    = _bundle.coarseV[i] + _bundle.corrVec[i]
        arr            = _bundle.corrVec[i].copy()
        arr[2]        -= _z_offset
        arrow_vecs[j]  = arr

    # _coarse_pc = ps.register_point_cloud(SAMPLE_COARSE, coarse_pts)
    # _coarse_pc.set_color((1.0, 0.85, 0.0))
    # _coarse_pc.set_radius(0.006, relative=True)
    # _coarse_pc.add_vector_quantity("to_fine", arrow_vecs,
    #     vectortype="ambient", enabled=_show_arrows, color=(1.0, 0.35, 0.05))

    # _fine_pc = ps.register_point_cloud(SAMPLE_FINE, fine_pts)
    # _fine_pc.set_color((0.15, 0.85, 0.40))
    # _fine_pc.set_radius(0.006, relative=True)


# ---------------------------------------------------------------------------
# Single-vertex selection highlight
# ---------------------------------------------------------------------------

def rebuild_selection_viz():
    for name in (SEL_COARSE, SEL_FINE):
        if ps.has_point_cloud(name):
            ps.remove_point_cloud(name)

    if _selected_sample < 0:
        return

    step = max(1, _sample_step)
    ci   = _selected_sample * step
    NC   = _bundle.coarseV.shape[0]
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
    rng  = np.random.default_rng(seed)
    r    = rng.random((n, 2))
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

    for name in (FACE_COARSE, FACE_FINE):
        if ps.has_point_cloud(name):
            ps.remove_point_cloud(name)
    if ps.has_surface_mesh(FACE_HULL):
        ps.remove_surface_mesh(FACE_HULL)
    _face_sample_reg = False
    _face_hull_reg   = False
    _face_coarse_pc  = None
    _face_fine_pc    = None

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
            mesh.add_color_quantity("face_highlight", fc, defined_on='faces')
            mesh.set_enabled(True)
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


# ---------------------------------------------------------------------------
# SSP-based correspondence (replaces file-loading tracker functions)
# ---------------------------------------------------------------------------

def compute_sample_correspondences():
    """
    Use query_coarse_to_fine to sample every coarse face with a few barycentric
    points and compute where each lands on the fine mesh.  Populates the same
    globals that the old tracker file-loaders filled so all downstream viz
    functions work without change.
    """
    global _tracker_fine_pts, _tracker_coarse_pts, _tracker_coarse_fids

    if not _bundle.has_ssp_data:
        print("[ssp_corr] Bundle has no SSP data — skipping sample correspondences.")
        return

    FC          = _bundle.coarseF.shape[0]
    n_per_face  = 3

    all_coarse = []
    all_fine   = []
    all_fids   = []

    print(f"[ssp_corr] Querying {FC} coarse faces × {n_per_face} samples …")
    for cf in range(FC):
        ci0, ci1, ci2 = _bundle.coarseF[cf]
        cv0 = _bundle.coarseV[ci0]
        cv1 = _bundle.coarseV[ci1]
        cv2 = _bundle.coarseV[ci2]

        BC = _fold_sample_bc(n_per_face, seed=cf)
        # compute coarse 3-D positions BEFORE BC is overwritten by the query
        coarse_pts = BC[:, 0:1] * cv0 + BC[:, 1:2] * cv1 + BC[:, 2:3] * cv2

        gvi0 = int(_bundle.vtxMap[ci0])
        gvi1 = int(_bundle.vtxMap[ci1])
        gvi2 = int(_bundle.vtxMap[ci2])
        gfi  = int(_bundle.faceMap[cf])

        BF   = np.tile([gvi0, gvi1, gvi2], (n_per_face, 1)).astype(np.int32)
        FIdx = np.full(n_per_face, gfi, dtype=np.int32)

        query_coarse_to_fine(
            _bundle.decInfo, _bundle.decIM, _bundle.faceSheetID,
            BC, BF, FIdx,
        )

        fine_pts = (BC[:, 0:1] * _bundle.fineV[BF[:, 0]]
                  + BC[:, 1:2] * _bundle.fineV[BF[:, 1]]
                  + BC[:, 2:3] * _bundle.fineV[BF[:, 2]])

        all_coarse.append(coarse_pts)
        all_fine.append(fine_pts)
        all_fids.extend([gfi] * n_per_face)

    _tracker_coarse_pts  = np.vstack(all_coarse)
    _tracker_fine_pts    = np.vstack(all_fine)
    _tracker_coarse_fids = np.array(all_fids, dtype=np.int32)
    print(f"[ssp_corr] {len(_tracker_fine_pts)} coarse→fine samples ready")


def compute_vertex_correspondences():
    """
    For each compact coarse vertex i, its fine-mesh correspondent is at
    coarseV[i] + corrVec[i].  A KD-tree search on fineV finds the nearest
    fine vertex, giving a clean fine→coarse mapping without any txt files.
    Populates _tracker_vtx_* and _deformed_fine_pts.
    """
    global _tracker_vtx_fine_pts, _tracker_vtx_coarse_pts
    global _tracker_vtx_coarse_fids, _deformed_fine_pts

    if not _bundle.has_ssp_data or _bundle.vtxMap is None:
        print("[vtx_corr] Bundle has no SSP data — skipping vertex correspondences.")
        return

    NC = _bundle.coarseV.shape[0]

    # Fine-mesh positions of each compact coarse vertex's correspondent
    fine_pos_of_coarse = _bundle.coarseV + _bundle.corrVec  # (NC, 3)

    # Nearest fine vertex for each correspondent position
    tree = cKDTree(_bundle.fineV)
    _, fine_indices = tree.query(fine_pos_of_coarse, k=1)   # (NC,)

    _tracker_vtx_fine_pts   = _bundle.fineV[fine_indices]   # (NC, 3)
    _tracker_vtx_coarse_pts = _bundle.coarseV.copy()        # (NC, 3)

    # Assign each coarse vertex to its first adjacent coarse face (for face filtering)
    coarse_vtx_to_face = np.full(NC, -1, dtype=np.int32)
    for fi in range(_bundle.coarseF.shape[0]):
        for v in _bundle.coarseF[fi]:
            if coarse_vtx_to_face[v] < 0:
                coarse_vtx_to_face[v] = fi
    _tracker_vtx_coarse_fids = np.array(
        [int(_bundle.faceMap[coarse_vtx_to_face[i]]) if coarse_vtx_to_face[i] >= 0 else -1
         for i in range(NC)], dtype=np.int32)

    # Build deformed fine mesh: replace each matched fine vertex with its coarse position
    _deformed_fine_pts = _bundle.fineV.copy()
    _deformed_fine_pts[fine_indices] = _bundle.coarseV

    n_unique = len(np.unique(fine_indices))
    print(f"[vtx_corr] {NC} coarse verts → {n_unique} unique fine verts mapped")


def export_deformed_fine_mesh(out_path: str):
    if _deformed_fine_pts is None:
        print("[export] No vertex tracker data loaded — cannot export.")
        return
    mesh = trimesh.Trimesh(vertices=_deformed_fine_pts, faces=_bundle.fineF, process=False)
    mesh.export(out_path)
    print(f"[export] Deformed fine mesh -> {out_path}  "
          f"({len(_deformed_fine_pts)} verts, {len(_bundle.fineF)} faces)")


def rebuild_tracker_viz():
    """Global tracker overlay, subsampled by _sample_step."""
    for name in (TRACKER_FINE, TRACKER_COARSE):
        if ps.has_point_cloud(name):
            ps.remove_point_cloud(name)

    if _tracker_fine_pts is None or _tracker_coarse_pts is None:
        return

    step       = max(1, _sample_step)
    fine_pts   = _tracker_fine_pts[::step]
    coarse_pts = _tracker_coarse_pts[::step].copy()
    coarse_pts[:, 2] += _z_offset
    arrows = fine_pts - coarse_pts

    pc_c = ps.register_point_cloud(TRACKER_COARSE, coarse_pts)
    pc_c.set_color((1.0, 0.55, 0.05))
    pc_c.set_radius(0.0035, relative=True)
    pc_c.set_enabled(_show_tracker)
    pc_c.add_vector_quantity("to_fine", arrows,
                             vectortype="ambient",
                             enabled=_show_tracker,
                             color=(1.0, 0.85, 0.1))

    pc_f = ps.register_point_cloud(TRACKER_FINE, fine_pts)
    pc_f.set_color((0.25, 0.90, 0.45))
    pc_f.set_radius(0.0035, relative=True)
    pc_f.set_enabled(_show_tracker)


def rebuild_vtx_tracker_viz():
    """Vertex-tracker overlay: original fine vertices -> coarse correspondences."""
    for name in (TRACKER_VTX_FINE, TRACKER_VTX_COARSE):
        if ps.has_point_cloud(name):
            ps.remove_point_cloud(name)

    if _tracker_vtx_fine_pts is None or _tracker_vtx_coarse_pts is None:
        return

    coarse_pts = _tracker_vtx_coarse_pts.copy()
    coarse_pts[:, 2] += _z_offset
    arrows = _tracker_vtx_fine_pts - coarse_pts

    pc_c = ps.register_point_cloud(TRACKER_VTX_COARSE, coarse_pts)
    pc_c.set_color((0.20, 0.70, 1.00))
    pc_c.set_radius(0.004, relative=True)
    pc_c.set_enabled(_show_vtx_tracker)
    pc_c.add_vector_quantity("to_fine", arrows,
                             vectortype="ambient",
                             enabled=_show_vtx_tracker,
                             color=(1.0, 0.80, 0.10))

    pc_f = ps.register_point_cloud(TRACKER_VTX_FINE, _tracker_vtx_fine_pts)
    pc_f.set_color((1.00, 0.35, 0.10))
    pc_f.set_radius(0.004, relative=True)
    pc_f.set_enabled(_show_vtx_tracker)


def rebuild_face_tracker_viz():
    """
    For the currently selected coarse face, show only the tracker samples whose
    cur_FIdx matches that face. faceMap[compact_face] gives the global gF face
    index stored in cur_FIdx, so we filter _tracker_coarse_fids on that value.
    """
    for name in (FACE_TRACKER_FINE, FACE_TRACKER_COARSE):
        if ps.has_point_cloud(name):
            ps.remove_point_cloud(name)

    if (_selected_face < 0
            or _tracker_coarse_fids is None
            or not _bundle.has_ssp_data
            or _bundle.faceMap is None):
        return

    global_fid = int(_bundle.faceMap[_selected_face])
    mask = _tracker_coarse_fids == global_fid
    if not mask.any():
        return

    fine_pts   = _tracker_fine_pts[mask]
    coarse_pts = _tracker_coarse_pts[mask].copy()
    coarse_pts[:, 2] += _z_offset
    arrows = fine_pts - coarse_pts

    pc_c = ps.register_point_cloud(FACE_TRACKER_COARSE, coarse_pts)
    pc_c.set_color((1.0, 0.20, 0.85))
    pc_c.set_radius(0.007, relative=True)
    pc_c.add_vector_quantity("to_fine", arrows,
                             vectortype="ambient", enabled=True,
                             color=(1.0, 0.55, 1.0))

    pc_f = ps.register_point_cloud(FACE_TRACKER_FINE, fine_pts)
    pc_f.set_color((0.45, 1.0, 0.20))
    pc_f.set_radius(0.007, relative=True)


def rebuild_vtx_face_tracker_viz():
    """
    For the selected coarse face, show all fine vertices whose coarse
    correspondence landed on that face, with ambient arrows pointing
    from the fine position toward the (Z-offset) coarse correspondence.
    """
    for name in (VTX_FACE_FINE, VTX_FACE_COARSE):
        if ps.has_point_cloud(name):
            ps.remove_point_cloud(name)

    if (_selected_face < 0
            or _tracker_vtx_coarse_fids is None
            or _tracker_vtx_fine_pts is None
            or not _bundle.has_ssp_data
            or _bundle.faceMap is None):
        return

    global_fid = int(_bundle.faceMap[_selected_face])
    mask = _tracker_vtx_coarse_fids == global_fid
    if not mask.any():
        print(f"[vtx_face] no fine vertices map to global face {global_fid}")
        return

    fine_pts   = _tracker_vtx_fine_pts[mask]
    coarse_pts = _tracker_vtx_coarse_pts[mask].copy()
    coarse_pts[:, 2] += _z_offset
    arrows = coarse_pts - fine_pts  # fine -> coarse direction

    pc_f = ps.register_point_cloud(VTX_FACE_FINE, fine_pts)
    pc_f.set_color((1.00, 0.35, 0.10))
    pc_f.set_radius(0.007, relative=True)
    pc_f.add_vector_quantity("to_coarse", arrows,
                             vectortype="ambient", enabled=True,
                             color=(1.0, 0.75, 0.15))

    pc_c = ps.register_point_cloud(VTX_FACE_COARSE, coarse_pts)
    pc_c.set_color((0.20, 0.80, 1.00))
    pc_c.set_radius(0.007, relative=True)

    print(f"[vtx_face] {mask.sum()} fine verts -> coarse face {_selected_face} (global {global_fid})")


def rebuild_all():
    rebuild_coarse_viz()
    rebuild_sample_viz()
    rebuild_selection_viz()
    # rebuild_face_sample_viz()
    rebuild_tracker_viz()
    rebuild_vtx_tracker_viz()
    rebuild_face_tracker_viz()
    rebuild_vtx_face_tracker_viz()


# ---------------------------------------------------------------------------
# Polyscope UI callback
# ---------------------------------------------------------------------------

def ui_callback():
    global _z_offset, _sample_step, _show_arrows
    global _selected_sample, _selected_face
    global _face_n, _show_face_coarse, _show_face_fine
    global _show_tracker, _show_vtx_tracker

    NC      = _bundle.coarseV.shape[0]
    changed = False

    try:
        psim.SetNextWindowSize((350, 420), psim.ImGuiCond_FirstUseEver)
    except Exception:
        pass
    psim.Begin("Correspondence", True)

    c, v = psim.SliderFloat("Z offset (coarse)", _z_offset, -2.0, 2.0)
    if c:
        _z_offset = v
        changed = True

    c, v = psim.SliderInt("Sample every X pts", _sample_step, 1, 1000)
    if c:
        _sample_step     = v
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
        psim.TextUnformatted(f"Selected: sample[{_selected_sample}] -> coarse[{ci}]")
        if psim.Button("Clear selection"):
            _selected_sample = -1
            rebuild_selection_viz()

    if changed:
        rebuild_all()

    psim.Separator()
    psim.TextUnformatted("Face sampling  (click a coarse face)")

    c, v = psim.SliderInt("n samples", _face_n, 1, 200)
    # if c:
    #     _face_n = max(1, v)
    #     if _selected_face >= 0:
    #         rebuild_face_sample_viz()

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
            # rebuild_face_sample_viz()
            rebuild_face_tracker_viz()
            rebuild_vtx_face_tracker_viz()

    psim.Separator()
    psim.TextUnformatted("Sample tracker  (samples_fine / samples_coarse)")
    if _tracker_fine_pts is not None:
        t_shown = (len(_tracker_fine_pts) + step - 1) // step
        psim.TextUnformatted(f"{t_shown} / {len(_tracker_fine_pts)} samples shown")
        c, v = psim.Checkbox("Show tracker", _show_tracker)
        if c:
            _show_tracker = v
            rebuild_tracker_viz()
    else:
        psim.TextUnformatted("(no sample files found)")

    psim.Separator()
    psim.TextUnformatted("Vertex tracker  (samples_vertices_*.txt)")
    if _tracker_vtx_fine_pts is not None:
        psim.TextUnformatted(f"{len(_tracker_vtx_fine_pts)} fine vertices tracked")
        c, v = psim.Checkbox("Show vtx tracker", _show_vtx_tracker)
        if c:
            _show_vtx_tracker = v
            rebuild_vtx_tracker_viz()
        if psim.Button("Export deformed fine mesh"):
            out_path = os.path.join(_bundle_dir, "deformed_fine_mesh.obj")
            export_deformed_fine_mesh(out_path)
    else:
        psim.TextUnformatted("(no samples_vertices_*.txt found)")

    psim.End()

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

            if sname == SAMPLE_COARSE:
                pt_idx = int(sdata.get('index', -1))
                if pt_idx >= 0 and pt_idx != _selected_sample:
                    _selected_sample = pt_idx
                    rebuild_selection_viz()

            elif sname == COARSE_MESH:
                etype = sdata.get('element_type', None)
                fidx  = int(sdata.get('index', -1))
                nF    = _bundle.coarseF.shape[0]
                if etype == 'face' and 0 <= fidx < nF and fidx != _selected_face:
                    _selected_face = fidx
                    # rebuild_face_sample_viz()
                    rebuild_face_tracker_viz()
                    rebuild_vtx_face_tracker_viz()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global _bundle, _bundle_dir

    c2f_path   = rf"C:\\Users\\alirz\\Projects\\Graphics\\Neural QMAT\\external\\surf_subgrid_SSP_orig\\10_collapse_viz\\correspondence_01_00040057_f8f78dbd17414efda75bc437_trimesh_000_mat_initial.c2f"
    _bundle_dir = os.path.dirname(os.path.abspath(c2f_path))
    _bundle    = load_bundle(c2f_path)
    print(f"Coarse: {_bundle.coarseV.shape[0]} verts  {_bundle.coarseF.shape[0]} faces")
    print(f"Fine:   {_bundle.fineV.shape[0]} verts  {_bundle.fineF.shape[0]} faces")
    if _bundle.has_ssp_data:
        print(f"SSP:    {len(_bundle.decInfo)} collapses  "
              f"{len(_bundle.decIM)} tracked faces")

    compute_sample_correspondences()
    compute_vertex_correspondences()

    ps.init()
    ps.set_program_name("Coarse-to-Fine Correspondence")
    ps.set_up_dir("y_up")

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
