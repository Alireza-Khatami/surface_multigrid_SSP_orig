#!/usr/bin/env python3
"""
fine_2_coarse_samples_viz_with_query copy.py
  Two modes for visualising fine → coarse correspondence:

  Mode 0 — Calculated (F2C query)
    Sample random interior points on fine mesh faces and map each to the
    coarse mesh via the SSP F2C correspondence (c2f_query.py).
    Per-vertex F2C map is computed once; each sample is interpolated from
    the surrounding fine-face vertices.

  Mode 1 — SSP Sample Tracker
    Read pre-saved samples_fine_*.txt / samples_coarse_*.txt produced by
    sample_tracker_save() in face_sample_tracker.cpp.
    Fine position  = BC on fine face (from the fine file).
    Coarse position = BC with global coarse vertex indices (from the coarse file).

Two sampling sub-modes for Mode 0 (UI checkbox):
  "Per face"     — exactly N samples on every fine face
  "Total (area)" — N total samples distributed by face area

Visualization:
  fine_mesh           grey at Z=z_offset      (source / lifted)
  coarse_mesh_src     green at Z=0            (target background)
  fine_samples        blue dots at Z=z_offset (sampled fine positions)
  coarse_landings     orange dots at Z=0      (F2C landed positions)
  sample_arrows       yellow ambient arrows   (fine sample → coarse landing)

Click a blue sample dot to inspect its details.
Click a fine mesh face in Pick-face mode to filter samples (Mode 0 only).

Usage:
    python "fine_2_coarse_samples_viz_with_query copy.py"

Dependencies:
    pip install polyscope numpy
    (c2f_query.py must be in the same directory or on PYTHONPATH)
"""

import sys
import os
import glob
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from c2f_query import load_bundle, compute_f2c_correspondences

# ---------------------------------------------------------------------------
# Structure names
# ---------------------------------------------------------------------------

FINE_MESH         = "fine_mesh"
COARSE_MESH_VIZ   = "coarse_mesh_src"
FINE_SAMPLE_PC    = "fine_samples"
COARSE_LANDING_PC = "coarse_landings"
ARROWS_PC         = "sample_arrows"
SEL_FINE_PC       = "sel_fine_dot"
SEL_COARSE_PC     = "sel_coarse_dot"

# ---------------------------------------------------------------------------
# Mutable state — shared
# ---------------------------------------------------------------------------

_bundle     = None
_bundle_dir = ""
_z_offset   = 1.0
_mesh_span  = 1.0

# viz mode: 0 = calculated F2C query, 1 = SSP sample tracker files
_viz_mode = 0

# ---------------------------------------------------------------------------
# Mutable state — Mode 0 (Calculated)
# ---------------------------------------------------------------------------

_f2c_v        = None   # (NF, 3)  each fine vertex → coarse-side landing position
_tracked_mask = None   # (NF,)  bool

_n_samples      = 1000
_seed           = 42
_per_face_mode  = False

# ---------------------------------------------------------------------------
# Mutable state — Mode 1 (Tracker files)
# ---------------------------------------------------------------------------

# Paths to the tracker files (auto-detected from bundle directory)
_tracker_fine_path   = ""
_tracker_coarse_path = ""

# tracker-mode sample metadata (parallel with _fine_sample_pts/_coarse_land_pts)
_tracker_sample_ids   = None   # (N,) int
_tracker_is_vertex    = None   # (N,) bool
_tracker_fine_face_id = None   # (N,) int
_tracker_fine_bc      = None   # (N, 3)
_tracker_coarse_face_id = None  # (N,) int
_tracker_coarse_bc    = None   # (N, 3)
_tracker_coarse_bv    = None   # (N, 3) int — global coarse vertex indices

# ---------------------------------------------------------------------------
# Mutable state — per-sample data (both modes)
# ---------------------------------------------------------------------------

_sample_face_ids  = None   # (N,)   fine face index
_sample_bcs       = None   # (N, 3) barycentric coords on fine face
_fine_sample_pts  = None   # (N, 3) 3-D position on fine mesh at Z=0
_coarse_land_pts  = None   # (N, 3) F2C landed position at Z=0
_sample_tracked   = None   # (N,)   bool

# display
_show_arrows      = True
_show_fine_dots   = True
_show_coarse_dots = True
_selected_sample  = -1

# face filter (Mode 0 only)
_pick_face_mode  = False
_selected_face   = -1


# ---------------------------------------------------------------------------
# Sampling helpers (Mode 0)
# ---------------------------------------------------------------------------

def _uniform_random_bc(n: int, rng: np.random.Generator) -> np.ndarray:
    r1 = rng.random(n)
    r2 = rng.random(n)
    sr = np.sqrt(r1)
    return np.stack([1.0 - sr, sr * (1.0 - r2), sr * r2], axis=1).astype(np.float64)


def _face_areas(V, F):
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _area_weighted_face_ids(areas, N, rng):
    total = areas.sum()
    if total < 1e-30:
        areas = np.ones(len(areas))
        total = float(len(areas))
    exact     = areas * (N / total)
    counts    = np.floor(exact).astype(np.int64)
    remainder = exact - counts
    leftover  = N - int(counts.sum())
    counts[np.argsort(-remainder)[:leftover]] += 1
    return np.repeat(np.arange(len(areas), dtype=np.int32), counts)


# ---------------------------------------------------------------------------
# Compute samples — Mode 0
# ---------------------------------------------------------------------------

def _compute_f2c_samples():
    global _sample_face_ids, _sample_bcs
    global _fine_sample_pts, _coarse_land_pts, _sample_tracked

    b = _bundle
    if b is None or _f2c_v is None:
        return False

    FF  = b.fineF.shape[0]
    rng = np.random.default_rng(_seed)

    if _per_face_mode:
        N        = _n_samples * FF
        face_ids = np.repeat(np.arange(FF, dtype=np.int32), _n_samples)
        print(f"[f2c_samples] {_n_samples} samples/face × {FF} faces = {N} total  seed={_seed}")
    else:
        N        = _n_samples
        areas    = _face_areas(b.fineV, b.fineF)
        face_ids = _area_weighted_face_ids(areas, N, rng)
        print(f"[f2c_samples] {N} total samples, area-weighted across {FF} faces  seed={_seed}")

    bc = _uniform_random_bc(N, rng)

    f0 = b.fineV[b.fineF[face_ids, 0]]
    f1 = b.fineV[b.fineF[face_ids, 1]]
    f2 = b.fineV[b.fineF[face_ids, 2]]
    fine_pts = bc[:, 0:1] * f0 + bc[:, 1:2] * f1 + bc[:, 2:3] * f2

    c0 = _f2c_v[b.fineF[face_ids, 0]]
    c1 = _f2c_v[b.fineF[face_ids, 1]]
    c2 = _f2c_v[b.fineF[face_ids, 2]]
    coarse_pts = bc[:, 0:1] * c0 + bc[:, 1:2] * c1 + bc[:, 2:3] * c2

    t0 = _tracked_mask[b.fineF[face_ids, 0]]
    t1 = _tracked_mask[b.fineF[face_ids, 1]]
    t2 = _tracked_mask[b.fineF[face_ids, 2]]
    tracked = t0 & t1 & t2

    _sample_face_ids  = face_ids
    _sample_bcs       = bc
    _fine_sample_pts  = fine_pts
    _coarse_land_pts  = coarse_pts
    _sample_tracked   = tracked

    n_tracked = int(tracked.sum())
    print(f"[f2c_samples] {n_tracked}/{N} samples on fully-tracked faces  "
          f"({100*n_tracked/N:.1f}%)")
    return True


# ---------------------------------------------------------------------------
# Load tracker files — Mode 1
# ---------------------------------------------------------------------------

def _load_tracker_samples(fine_path: str, coarse_path: str) -> bool:
    """
    Read samples_fine_*.txt and samples_coarse_*.txt written by
    sample_tracker_save() and populate the shared sample arrays.

    Fine file columns:  sample_id  is_vertex  fine_face_id  b0  b1  b2
    Coarse file columns: sample_id  is_vertex  coarse_face_id  b0  b1  b2  bv0  bv1  bv2

    Coarse landing position = b0*coarseV[bv0] + b1*coarseV[bv1] + b2*coarseV[bv2]
    """
    global _sample_face_ids, _sample_bcs
    global _fine_sample_pts, _coarse_land_pts, _sample_tracked
    global _tracker_sample_ids, _tracker_is_vertex, _tracker_fine_face_id, _tracker_fine_bc
    global _tracker_coarse_face_id, _tracker_coarse_bc, _tracker_coarse_bv

    b = _bundle
    if b is None:
        print("[tracker] ERROR: no bundle loaded")
        return False
    if not os.path.isfile(fine_path):
        print(f"[tracker] ERROR: fine file not found: {fine_path}")
        return False
    if not os.path.isfile(coarse_path):
        print(f"[tracker] ERROR: coarse file not found: {coarse_path}")
        return False

    def _read_txt(path, skip_header=True):
        with open(path, "r") as fh:
            lines = fh.readlines()
        # strip comment lines starting with '#'
        data_lines = [l for l in lines if not l.strip().startswith("#")]
        # first data line is count
        count = int(data_lines[0].strip())
        rows  = [l.split() for l in data_lines[1:count + 1]]
        return rows

    fine_rows   = _read_txt(fine_path)
    coarse_rows = _read_txt(coarse_path)

    N = len(fine_rows)
    if len(coarse_rows) != N:
        print(f"[tracker] WARNING: fine has {N} samples but coarse has {len(coarse_rows)} — using min")
        N = min(N, len(coarse_rows))

    print(f"[tracker] Loading {N} samples from tracker files")

    sample_ids    = np.zeros(N, dtype=np.int32)
    is_vertex     = np.zeros(N, dtype=bool)
    fine_face_ids = np.zeros(N, dtype=np.int32)
    fine_bc       = np.zeros((N, 3), dtype=np.float64)

    coarse_face_ids = np.zeros(N, dtype=np.int32)
    coarse_bc       = np.zeros((N, 3), dtype=np.float64)
    coarse_bv       = np.zeros((N, 3), dtype=np.int32)

    for i in range(N):
        fr = fine_rows[i]
        cr = coarse_rows[i]
        sample_ids[i]    = int(fr[0])
        is_vertex[i]     = bool(int(fr[1]))
        fine_face_ids[i] = int(fr[2])
        fine_bc[i]       = [float(fr[3]), float(fr[4]), float(fr[5])]

        coarse_face_ids[i] = int(cr[2])
        coarse_bc[i]       = [float(cr[3]), float(cr[4]), float(cr[5])]
        coarse_bv[i]       = [int(cr[6]),   int(cr[7]),   int(cr[8])]

    # Fine positions — BC on original fine mesh faces (gFO in C++)
    ff = fine_face_ids
    fv = b.fineV
    fi = b.fineF
    fine_pts = (fine_bc[:, 0:1] * fv[fi[ff, 0]]
              + fine_bc[:, 1:2] * fv[fi[ff, 1]]
              + fine_bc[:, 2:3] * fv[fi[ff, 2]])

    # Coarse positions — BC with global coarse vertex indices
    cv = b.coarseV
    bv0, bv1, bv2 = coarse_bv[:, 0], coarse_bv[:, 1], coarse_bv[:, 2]
    # clamp to valid range
    nCV = cv.shape[0]
    bv0 = np.clip(bv0, 0, nCV - 1)
    bv1 = np.clip(bv1, 0, nCV - 1)
    bv2 = np.clip(bv2, 0, nCV - 1)
    coarse_pts = (coarse_bc[:, 0:1] * cv[bv0]
                + coarse_bc[:, 1:2] * cv[bv1]
                + coarse_bc[:, 2:3] * cv[bv2])

    _tracker_sample_ids    = sample_ids
    _tracker_is_vertex     = is_vertex
    _tracker_fine_face_id  = fine_face_ids
    _tracker_fine_bc       = fine_bc
    _tracker_coarse_face_id = coarse_face_ids
    _tracker_coarse_bc     = coarse_bc
    _tracker_coarse_bv     = coarse_bv

    _sample_face_ids  = fine_face_ids
    _sample_bcs       = fine_bc
    _fine_sample_pts  = fine_pts
    _coarse_land_pts  = coarse_pts
    _sample_tracked   = is_vertex   # vertex samples are "tracked" in tracker terminology

    n_verts = int(is_vertex.sum())
    print(f"[tracker] {N} samples  ({n_verts} vertex, {N - n_verts} interior)")
    return True


# ---------------------------------------------------------------------------
# Auto-detect tracker files in a directory
# ---------------------------------------------------------------------------

def _find_tracker_files(directory: str):
    """Return (fine_path, coarse_path) found in directory, or ('', '')."""
    fine_candidates   = sorted(glob.glob(os.path.join(directory, "samples_fine_*.txt")))
    coarse_candidates = sorted(glob.glob(os.path.join(directory, "samples_coarse_*.txt")))
    if fine_candidates and coarse_candidates:
        return fine_candidates[0], coarse_candidates[0]
    return "", ""


# ---------------------------------------------------------------------------
# Active mask (face filter — Mode 0 only)
# ---------------------------------------------------------------------------

def _active_mask():
    if _sample_face_ids is None:
        return None
    if _viz_mode == 1 or _selected_face < 0:
        return np.ones(len(_sample_face_ids), dtype=bool)
    return _sample_face_ids == _selected_face


# ---------------------------------------------------------------------------
# Rebuild helpers
# ---------------------------------------------------------------------------

def _rebuild_meshes():
    fv = _bundle.fineV.copy()
    fv[:, 2] += _z_offset
    fm = ps.register_surface_mesh(FINE_MESH, fv, _bundle.fineF)
    fm.set_color((0.55, 0.55, 0.55))
    fm.set_edge_width(0.3)
    fm.set_smooth_shade(False)
    fm.set_transparency(0.4)

    cm = ps.register_surface_mesh(COARSE_MESH_VIZ, _bundle.coarseV, _bundle.coarseF)
    cm.set_color((0.15, 0.75, 0.35))
    cm.set_edge_width(1.0)
    cm.set_smooth_shade(False)
    cm.set_transparency(0.55)


def _rebuild_fine_sample_pc():
    if ps.has_point_cloud(FINE_SAMPLE_PC):
        ps.remove_point_cloud(FINE_SAMPLE_PC)
    if not _show_fine_dots or _fine_sample_pts is None:
        return
    mask = _active_mask()
    pts = _fine_sample_pts[mask].copy()
    pts[:, 2] += _z_offset
    pc = ps.register_point_cloud(FINE_SAMPLE_PC, pts)
    pc.set_color((0.25, 0.55, 1.0))
    pc.set_radius(0.005, relative=True)


def _rebuild_coarse_landing_pc():
    if ps.has_point_cloud(COARSE_LANDING_PC):
        ps.remove_point_cloud(COARSE_LANDING_PC)
    if not _show_coarse_dots or _coarse_land_pts is None:
        return
    mask = _active_mask()
    pc = ps.register_point_cloud(COARSE_LANDING_PC, _coarse_land_pts[mask])
    pc.set_color((0.95, 0.45, 0.10))
    pc.set_radius(0.005, relative=True)


def _rebuild_arrows():
    if ps.has_point_cloud(ARROWS_PC):
        ps.remove_point_cloud(ARROWS_PC)
    if _fine_sample_pts is None or _coarse_land_pts is None:
        return
    mask = _active_mask()
    src = _fine_sample_pts[mask].copy()
    src[:, 2] += _z_offset
    vecs = _coarse_land_pts[mask] - src
    pc = ps.register_point_cloud(ARROWS_PC, src)
    pc.set_color((1.0, 0.85, 0.0))
    pc.set_radius(0.002, relative=True)
    pc.add_vector_quantity("to_coarse", vecs, vectortype="ambient",
                           enabled=_show_arrows, color=(1.0, 0.55, 0.05))


def _rebuild_selection():
    for name in (SEL_FINE_PC, SEL_COARSE_PC):
        if ps.has_point_cloud(name):
            ps.remove_point_cloud(name)
    if _selected_sample < 0 or _fine_sample_pts is None:
        return

    fpos = _fine_sample_pts[_selected_sample].copy()
    fpos[2] += _z_offset
    pc_f = ps.register_point_cloud(SEL_FINE_PC, fpos[np.newaxis])
    pc_f.set_color((1.0, 1.0, 0.0))
    pc_f.set_radius(0.012, relative=True)

    cpos = _coarse_land_pts[_selected_sample].copy()
    pc_c = ps.register_point_cloud(SEL_COARSE_PC, cpos[np.newaxis])
    pc_c.set_color((1.0, 0.0, 0.3))
    pc_c.set_radius(0.012, relative=True)


def _rebuild_all():
    _rebuild_meshes()
    _rebuild_fine_sample_pc()
    _rebuild_coarse_landing_pc()
    _rebuild_arrows()
    _rebuild_selection()


# ---------------------------------------------------------------------------
# UI callback
# ---------------------------------------------------------------------------

def ui_callback():
    global _z_offset, _mesh_span, _n_samples, _seed, _per_face_mode
    global _show_arrows, _show_fine_dots, _show_coarse_dots, _selected_sample
    global _pick_face_mode, _selected_face
    global _viz_mode
    global _tracker_fine_path, _tracker_coarse_path

    changed = False

    try:
        psim.SetNextWindowSize((400, 520), psim.ImGuiCond_FirstUseEver)
    except Exception:
        pass
    psim.Begin("Fine -> Coarse (Face Samples)", True)

    # ---- Mode selector ----
    psim.TextUnformatted("Correspondence Mode")
    clicked0 = psim.RadioButton("Mode 0: Calculated F2C (c2f_query)", _viz_mode == 0)
    if clicked0 and _viz_mode != 0:
        _viz_mode = 0
        _selected_face   = -1
        _selected_sample = -1
        if _bundle is not None and _f2c_v is not None:
            _compute_f2c_samples()
            _rebuild_all()

    clicked1 = psim.RadioButton("Mode 1: SSP Sample Tracker files", _viz_mode == 1)
    if clicked1 and _viz_mode != 1:
        _viz_mode = 1
        _selected_face   = -1
        _selected_sample = -1
        if _tracker_fine_path and _tracker_coarse_path:
            _load_tracker_samples(_tracker_fine_path, _tracker_coarse_path)
            _rebuild_all()

    psim.Separator()

    # ---- Z offset ----
    _slider_range = max(2.0, _mesh_span * 5)
    c, v = psim.SliderFloat("Z offset", _z_offset, -_slider_range, _slider_range)
    if c:
        _z_offset = v
        changed = True

    psim.Separator()

    # ---- Mode-specific controls ----
    if _viz_mode == 0:
        psim.TextUnformatted("[ Mode 0: Sampling on fine faces via F2C query ]")

        c, v = psim.Checkbox("Per face (N samples per fine face)", _per_face_mode)
        if c:
            _per_face_mode = v

        if _per_face_mode:
            c, v = psim.InputInt("Samples per face", _n_samples)
            if c:
                _n_samples = max(1, min(v, 1000))
            if _bundle is not None:
                FF = _bundle.fineF.shape[0]
                psim.TextDisabled(f"  = {_n_samples} x {FF} = {_n_samples * FF} total")
        else:
            c, v = psim.InputInt("Total samples", _n_samples)
            if c:
                _n_samples = max(1, min(v, 500000))
            psim.TextDisabled("  distributed by face area")

        c, v = psim.InputInt("Seed", _seed)
        if c:
            _seed = max(0, v)

        if psim.Button("Resample"):
            ok = _compute_f2c_samples()
            if ok:
                _selected_sample = -1
                _selected_face   = -1
                _rebuild_all()

        if _fine_sample_pts is not None:
            N    = len(_fine_sample_pts)
            n_tr = int(_sample_tracked.sum()) if _sample_tracked is not None else 0
            psim.TextUnformatted(f"{N} samples  |  {n_tr} on fully-tracked faces")

        psim.Separator()
        psim.TextUnformatted("Face Filter")

        c, v = psim.Checkbox("Pick face mode (click fine mesh)", _pick_face_mode)
        if c:
            _pick_face_mode = v
            if not v:
                _selected_face   = -1
                _selected_sample = -1
                _rebuild_fine_sample_pc()
                _rebuild_coarse_landing_pc()
                _rebuild_arrows()
                _rebuild_selection()

        if _selected_face >= 0:
            n_on = int((_sample_face_ids == _selected_face).sum()) if _sample_face_ids is not None else 0
            psim.TextUnformatted(f"  Face {_selected_face}  ({n_on} samples shown)")
            if psim.Button("Clear face filter"):
                _selected_face   = -1
                _selected_sample = -1
                _rebuild_fine_sample_pc()
                _rebuild_coarse_landing_pc()
                _rebuild_arrows()
                _rebuild_selection()
        elif _pick_face_mode:
            psim.TextDisabled("  Click a fine mesh face to filter")

    else:
        # Mode 1 — Tracker files
        psim.TextUnformatted("[ Mode 1: SSP Sample Tracker output files ]")

        psim.TextDisabled("Fine file:")
        psim.TextUnformatted(f"  {os.path.basename(_tracker_fine_path) if _tracker_fine_path else '(none)'}")
        psim.TextDisabled("Coarse file:")
        psim.TextUnformatted(f"  {os.path.basename(_tracker_coarse_path) if _tracker_coarse_path else '(none)'}")

        if psim.Button("Reload tracker files"):
            if _tracker_fine_path and _tracker_coarse_path:
                ok = _load_tracker_samples(_tracker_fine_path, _tracker_coarse_path)
                if ok:
                    _selected_sample = -1
                    _rebuild_all()
            else:
                print("[tracker] No tracker file paths set — check _tracker_fine_path")

        if _fine_sample_pts is not None:
            N     = len(_fine_sample_pts)
            n_vtx = int(_tracker_is_vertex.sum()) if _tracker_is_vertex is not None else 0
            psim.TextUnformatted(f"{N} samples  |  {n_vtx} vertex  {N - n_vtx} interior")

    # ---- Display ----
    psim.Separator()
    psim.TextUnformatted("Display")

    c, v = psim.Checkbox("Show fine sample dots", _show_fine_dots)
    if c:
        _show_fine_dots = v
        _rebuild_fine_sample_pc()

    c, v = psim.Checkbox("Show coarse landing dots", _show_coarse_dots)
    if c:
        _show_coarse_dots = v
        _rebuild_coarse_landing_pc()

    c, v = psim.Checkbox("Show arrows", _show_arrows)
    if c:
        _show_arrows = v
        _rebuild_arrows()

    # ---- Sample inspector ----
    psim.Separator()
    if _selected_sample >= 0 and _fine_sample_pts is not None:
        si  = _selected_sample
        fi  = int(_sample_face_ids[si])
        bc  = _sample_bcs[si]
        fp  = _fine_sample_pts[si]
        cp  = _coarse_land_pts[si]
        dist = float(np.linalg.norm(cp - fp))

        if _viz_mode == 0:
            trk = bool(_sample_tracked[si]) if _sample_tracked is not None else False
            psim.TextUnformatted(f"Sample {si}  {'[tracked]' if trk else '[untracked]'}")
            psim.TextUnformatted(f"  Fine face   : {fi}  BC=({bc[0]:.3f},{bc[1]:.3f},{bc[2]:.3f})")
        else:
            is_vtx = bool(_tracker_is_vertex[si]) if _tracker_is_vertex is not None else False
            sid    = int(_tracker_sample_ids[si]) if _tracker_sample_ids is not None else si
            cfi    = int(_tracker_coarse_face_id[si]) if _tracker_coarse_face_id is not None else -1
            bv     = _tracker_coarse_bv[si] if _tracker_coarse_bv is not None else [0, 0, 0]
            psim.TextUnformatted(f"Sample {sid}  {'[vertex]' if is_vtx else '[interior]'}")
            psim.TextUnformatted(f"  Fine face   : {fi}  BC=({bc[0]:.3f},{bc[1]:.3f},{bc[2]:.3f})")
            psim.TextUnformatted(f"  Coarse face : {cfi}  verts=({bv[0]},{bv[1]},{bv[2]})")

        psim.TextUnformatted(f"  Fine pos    : ({fp[0]:.5f},{fp[1]:.5f},{fp[2]:.5f})")
        psim.TextUnformatted(f"  Coarse land : ({cp[0]:.5f},{cp[1]:.5f},{cp[2]:.5f})")
        psim.TextUnformatted(f"  Dist F->C   : {dist:.6f}")
        if psim.Button("Clear selection"):
            _selected_sample = -1
            _rebuild_selection()
    else:
        psim.TextDisabled("Click a blue fine sample dot to inspect")

    psim.End()

    # ---- pick handler ----
    io = psim.GetIO()
    if io.MouseClicked[0]:
        try:
            pr = ps.pick(screen_coords=io.MousePos)
        except Exception:
            pr = None
        if pr is not None and getattr(pr, 'is_hit', False):
            sname = getattr(pr, 'structure_name', None)
            sdata = getattr(pr, 'structure_data', None) or {}

            if sname == FINE_MESH and _pick_face_mode and _viz_mode == 0:
                etype = sdata.get('element_type', None)
                fidx  = int(sdata.get('index', -1))
                nF    = _bundle.fineF.shape[0]
                if etype == 'face' and 0 <= fidx < nF:
                    _selected_face   = -1 if fidx == _selected_face else fidx
                    _selected_sample = -1
                    _rebuild_fine_sample_pc()
                    _rebuild_coarse_landing_pc()
                    _rebuild_arrows()
                    _rebuild_selection()
                    if _selected_face >= 0:
                        n_on = int((_sample_face_ids == _selected_face).sum()) \
                               if _sample_face_ids is not None else 0
                        print(f"\n[face filter] fine face {_selected_face}  ({n_on} samples)")

            elif sname == FINE_SAMPLE_PC:
                idx = int(sdata.get('index', -1))
                if _fine_sample_pts is not None and idx >= 0:
                    mask    = _active_mask()
                    indices = np.where(mask)[0]
                    global_idx = int(indices[idx]) if idx < len(indices) else -1
                    if global_idx >= 0:
                        _selected_sample = -1 if global_idx == _selected_sample else global_idx
                        _rebuild_selection()
                        if _selected_sample >= 0:
                            si  = _selected_sample
                            fi  = int(_sample_face_ids[si])
                            bc  = _sample_bcs[si]
                            fp  = _fine_sample_pts[si]
                            cp  = _coarse_land_pts[si]
                            print(f"\n[sample {si}]  fine face={fi}  "
                                  f"BC=({bc[0]:.4f},{bc[1]:.4f},{bc[2]:.4f})")
                            print(f"  fine pos    : ({fp[0]:.6f},{fp[1]:.6f},{fp[2]:.6f})")
                            print(f"  coarse land : ({cp[0]:.6f},{cp[1]:.6f},{cp[2]:.6f})")
                            print(f"  dist F->C   : {np.linalg.norm(cp-fp):.6f}")
                            if _viz_mode == 1 and _tracker_is_vertex is not None:
                                is_vtx = bool(_tracker_is_vertex[si])
                                print(f"  type        : {'vertex' if is_vtx else 'interior'}")

    if changed and _fine_sample_pts is not None:
        _rebuild_all()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global _bundle, _bundle_dir, _z_offset, _mesh_span, _f2c_v, _tracked_mask
    global _tracker_fine_path, _tracker_coarse_path

    c2f_path = rf"C:\\Users\\alirz\\Projects\\Graphics\\Neural QMAT\\external\\surf_subgrid_SSP_orig\\10_collapse_viz\\output\\01_00040057_f8f78dbd17414efda75bc437_trimesh_000\\correspondence_01_00040057_f8f78dbd17414efda75bc437_trimesh_000_mat_initial.c2f"
    # c2f_path = rf"C:\\Users\\alirz\\Projects\\Graphics\\Neural QMAT\\external\\surf_subgrid_SSP_orig\\10_collapse_viz\\output\\0002000_partstudio_14_model_ste_00_1024\\correspondence_0002000_partstudio_14_model_ste_00_1024.c2f"
    # c2f_path = rf"C:\\Users\\alirz\\Projects\\Graphics\\Neural QMAT\\external\\surf_subgrid_SSP_orig\\10_collapse_viz\\output\\hand\\correspondence_hand.c2f"
    # c2f_path = rf"C:\\Users\\alirz\\Projects\\Graphics\\Neural QMAT\\external\\surf_subgrid_SSP_orig\\10_collapse_viz\\output\\subDiv_cube\\correspondence_subDiv_cube.c2f"

    _bundle_dir = os.path.dirname(os.path.abspath(c2f_path))
    _bundle     = load_bundle(c2f_path)

    fv = _bundle.fineV
    mesh_span = float((fv.max(axis=0) - fv.min(axis=0)).max())
    if mesh_span > 1e-10:
        _z_offset  = mesh_span
        _mesh_span = mesh_span

    print(f"Fine:   {_bundle.fineV.shape[0]} verts  {_bundle.fineF.shape[0]} faces")
    print(f"Coarse: {_bundle.coarseV.shape[0]} verts  {_bundle.coarseF.shape[0]} faces")
    if _bundle.has_ssp_data:
        print(f"SSP:    {len(_bundle.decInfo)} collapses")

    # Auto-detect tracker files in the same output directory as the c2f file
    fine_found, coarse_found = _find_tracker_files(_bundle_dir)
    if fine_found and coarse_found:
        _tracker_fine_path   = fine_found
        _tracker_coarse_path = coarse_found
        print(f"[tracker] Found fine   : {os.path.basename(fine_found)}")
        print(f"[tracker] Found coarse : {os.path.basename(coarse_found)}")
    else:
        print(f"[tracker] No tracker files found in: {_bundle_dir}")

    print("\n[main] Computing per-vertex F2C map ...")
    _f2c_v, _tracked_mask = compute_f2c_correspondences(_bundle)
    n_tr = int(_tracked_mask.sum())
    print(f"[main] {n_tr}/{_bundle.fineV.shape[0]} fine vertices tracked by F2C")

    # Start in Mode 0 (calculated)
    ok = _compute_f2c_samples()
    if not ok:
        print("ERROR: could not compute F2C samples — exiting.")
        sys.exit(1)

    ps.init()
    ps.set_program_name("Fine -> Coarse (Face Samples)")
    ps.set_up_dir("y_up")

    _rebuild_all()

    ps.set_user_callback(ui_callback)
    ps.show()


if __name__ == "__main__":
    main()
