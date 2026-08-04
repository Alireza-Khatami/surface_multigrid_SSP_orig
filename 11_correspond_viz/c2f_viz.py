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
import glob
import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from scipy.spatial import ConvexHull, QhullError
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

# ---- mutable global state ----
_bundle             = None
_bundle_dir         = ""   # directory containing the loaded .c2f file

_z_offset           = 1.0
_sample_step        = 1
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
_tracker_vtx_fine_pts   = None   # (Nv,3)  original fine vertex positions
_tracker_vtx_coarse_pts = None   # (Nv,3)  coarse correspondence positions (no Z offset)
_show_vtx_tracker        = True
_deformed_fine_pts       = None   # (nFineV,3)  fineV with positions replaced by coarse corr.


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
# Sample-tracker correspondence  (samples_fine_*.txt / samples_coarse_*.txt)
# ---------------------------------------------------------------------------

def _load_fine_file(path: str):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 1:
                continue
            rows.append([float(x) for x in parts[:6]])
    return np.array(rows, dtype=np.float64) if rows else None


def _load_coarse_file(path: str):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 1:
                continue
            rows.append([float(x) for x in parts[:9]])
    return np.array(rows, dtype=np.float64) if rows else None


def load_tracker_samples(bundle_path: str):
    """
    Auto-discover samples_fine_*.txt / samples_coarse_*.txt next to the bundle,
    compute 3D positions, and store in globals for rebuild_tracker_viz().
    """
    global _tracker_fine_pts, _tracker_coarse_pts, _tracker_coarse_fids

    bundle_dir   = os.path.dirname(os.path.abspath(bundle_path))
    fine_files   = sorted(glob.glob(os.path.join(bundle_dir, "samples_fine_*.txt")))
    coarse_files = sorted(glob.glob(os.path.join(bundle_dir, "samples_coarse_*.txt")))

    if not fine_files or not coarse_files:
        print("[tracker] No sample files found in", bundle_dir)
        return

    print(f"[tracker] Loading {os.path.basename(fine_files[0])}")
    print(f"[tracker]          {os.path.basename(coarse_files[0])}")

    fine_data   = _load_fine_file(fine_files[0])
    coarse_data = _load_coarse_file(coarse_files[0])

    if fine_data is None or coarse_data is None:
        print("[tracker] Empty sample files — skipping.")
        return

    N = min(len(fine_data), len(coarse_data))
    print(f"[tracker] {N} samples")

    # Fine 3D positions: fine_face_id indexes directly into fineF (= gFO)
    fineV    = _bundle.fineV
    fineF    = _bundle.fineF
    face_ids = fine_data[:N, 2].astype(np.int32)
    bc       = fine_data[:N, 3:6]
    _tracker_fine_pts = (bc[:, 0:1] * fineV[fineF[face_ids, 0]]
                       + bc[:, 1:2] * fineV[fineF[face_ids, 1]]
                       + bc[:, 2:3] * fineV[fineF[face_ids, 2]])

    # Coarse 3D positions: bv0/bv1/bv2 are global gV indices.
    # vtxMap[compact_idx] = global_gV_idx  ->  build inverse map.
    if not _bundle.has_ssp_data or _bundle.vtxMap is None:
        print("[tracker] Bundle has no vtxMap — cannot map coarse samples.")
        return

    gv_to_cv = {int(_bundle.vtxMap[i]): i for i in range(len(_bundle.vtxMap))}
    coarseV  = _bundle.coarseV
    bc_c     = coarse_data[:N, 3:6]
    bv0      = coarse_data[:N, 6].astype(np.int32)
    bv1      = coarse_data[:N, 7].astype(np.int32)
    bv2      = coarse_data[:N, 8].astype(np.int32)
    cv0 = np.array([gv_to_cv.get(int(v), 0) for v in bv0], dtype=np.int32)
    cv1 = np.array([gv_to_cv.get(int(v), 0) for v in bv1], dtype=np.int32)
    cv2 = np.array([gv_to_cv.get(int(v), 0) for v in bv2], dtype=np.int32)
    _tracker_coarse_pts  = (bc_c[:, 0:1] * coarseV[cv0]
                           + bc_c[:, 1:2] * coarseV[cv1]
                           + bc_c[:, 2:3] * coarseV[cv2])

    # Global gF face index per sample — used by rebuild_face_tracker_viz()
    _tracker_coarse_fids = coarse_data[:N, 2].astype(np.int32)


def load_vertex_tracker(bundle_path: str):
    """
    Auto-discover samples_vertices_*.txt next to the bundle.
    Format per row: fine_vertex_id  coarse_face_id  b0 b1 b2  bv0 bv1 bv2
    Fine positions come directly from fineV[fine_vertex_id].
    Coarse positions are interpolated using the inverse vtxMap.
    """
    global _tracker_vtx_fine_pts, _tracker_vtx_coarse_pts

    bundle_dir = os.path.dirname(os.path.abspath(bundle_path))
    vtx_files  = sorted(glob.glob(os.path.join(bundle_dir, "samples_vertices_*.txt")))

    if not vtx_files:
        print("[vtx_tracker] No samples_vertices_*.txt found in", bundle_dir)
        return

    print(f"[vtx_tracker] Loading {os.path.basename(vtx_files[0])}")

    rows = []
    with open(vtx_files[0]) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 1:
                continue
            rows.append([float(x) for x in parts[:8]])

    if not rows:
        print("[vtx_tracker] Empty vertex file — skipping.")
        return

    data = np.array(rows, dtype=np.float64)
    N    = len(data)
    print(f"[vtx_tracker] {N} vertex tracks")

    # Fine positions: fine_vertex_id indexes directly into fineV
    fine_ids = data[:, 0].astype(np.int32)
    _tracker_vtx_fine_pts = _bundle.fineV[fine_ids]

    # Coarse positions: bv0/bv1/bv2 are global gV indices
    if not _bundle.has_ssp_data or _bundle.vtxMap is None:
        print("[vtx_tracker] Bundle has no vtxMap — cannot map coarse positions.")
        return

    gv_to_cv = {int(_bundle.vtxMap[i]): i for i in range(len(_bundle.vtxMap))}
    coarseV  = _bundle.coarseV
    bc       = data[:, 2:5]
    bv0      = data[:, 5].astype(np.int32)
    bv1      = data[:, 6].astype(np.int32)
    bv2      = data[:, 7].astype(np.int32)
    cv0 = np.array([gv_to_cv.get(int(v), 0) for v in bv0], dtype=np.int32)
    cv1 = np.array([gv_to_cv.get(int(v), 0) for v in bv1], dtype=np.int32)
    cv2 = np.array([gv_to_cv.get(int(v), 0) for v in bv2], dtype=np.int32)
    _tracker_vtx_coarse_pts = (bc[:, 0:1] * coarseV[cv0]
                              + bc[:, 1:2] * coarseV[cv1]
                              + bc[:, 2:3] * coarseV[cv2])

    # Build deformed fine mesh: fineV-indexed array where each vertex is replaced
    # by its coarse correspondence.  Unreferenced vertices fall back to fineV.
    global _deformed_fine_pts
    _deformed_fine_pts = _bundle.fineV.copy()
    _deformed_fine_pts[fine_ids] = _tracker_vtx_coarse_pts


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


def rebuild_all():
    rebuild_coarse_viz()
    rebuild_sample_viz()
    rebuild_selection_viz()
    # rebuild_face_sample_viz()
    rebuild_tracker_viz()
    rebuild_vtx_tracker_viz()
    rebuild_face_tracker_viz()


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

    c, v = psim.SliderInt("Sample every X pts", _sample_step, 1, 40)
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global _bundle, _bundle_dir

    c2f_path   = rf"C:\\Users\\alirz\\Projects\\Graphics\\Neural QMAT\\external\\surf_subgrid_SSP_orig\\10_collapse_viz\\c2f_bundle.c2f"
    _bundle_dir = os.path.dirname(os.path.abspath(c2f_path))
    _bundle    = load_bundle(c2f_path)
    print(f"Coarse: {_bundle.coarseV.shape[0]} verts  {_bundle.coarseF.shape[0]} faces")
    print(f"Fine:   {_bundle.fineV.shape[0]} verts  {_bundle.fineF.shape[0]} faces")
    if _bundle.has_ssp_data:
        print(f"SSP:    {len(_bundle.decInfo)} collapses  "
              f"{len(_bundle.decIM)} tracked faces")

    load_tracker_samples(c2f_path)
    load_vertex_tracker(c2f_path)

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
