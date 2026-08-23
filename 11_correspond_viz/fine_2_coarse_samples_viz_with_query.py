#!/usr/bin/env python3
"""
fine_2_coarse_samples_viz_with_query.py
  — Sample random interior points on fine mesh faces and map each to the
    coarse mesh via the SSP F2C correspondence.

Strategy: compute the per-vertex F2C map once with compute_f2c_correspondences,
then for each sample at barycentric (BC, fine-face) interpolate:
    coarse_pos = BC[0]*f2c_v[v0] + BC[1]*f2c_v[v1] + BC[2]*f2c_v[v2]
This is the correct piecewise-linear approximation of the F2C map and gives
full coverage of the fine mesh.

Two sampling modes (UI checkbox):
  "Per face"     — exactly N samples on every fine face  (total = N × FF)
  "Total (area)" — N total samples distributed by face area (largest-remainder)

Visualization:
  fine_mesh           grey at Z=z_offset      (source / lifted)
  coarse_mesh_src     green at Z=0            (target background)
  fine_samples        blue dots at Z=z_offset (sampled fine positions)
  coarse_landings     orange dots at Z=0      (F2C landed positions)
  sample_arrows       yellow ambient arrows   (fine sample → coarse landing)

Click a blue sample dot to inspect its details.
Click a fine mesh face in Pick-face mode to filter samples.

Usage:
    python fine_2_coarse_samples_viz_with_query.py  <bundle.c2f>

Dependencies:
    pip install polyscope numpy
    (c2f_query.py must be in the same directory or on PYTHONPATH)
"""

import sys
import os
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
# Mutable state
# ---------------------------------------------------------------------------

_bundle     = None
_bundle_dir = ""
_z_offset   = 1.0
_mesh_span  = 1.0

# pre-computed per-vertex F2C map
_f2c_v        = None   # (NF, 3)  each fine vertex → coarse-side landing position
_tracked_mask = None   # (NF,)  bool — True if vertex was actually moved by F2C

# sampling config
_n_samples      = 1000
_seed           = 42
_per_face_mode  = False   # False = total+area  |  True = per-face

# per-sample data
_sample_face_ids  = None   # (N,)   fine face index for each sample
_sample_bcs       = None   # (N, 3) barycentric coords on fine face
_fine_sample_pts  = None   # (N, 3) 3-D position on fine mesh at Z=0
_coarse_land_pts  = None   # (N, 3) F2C landed position at Z=0
_sample_tracked   = None   # (N,)   bool — all 3 face verts have F2C tracking

# display
_show_arrows      = True
_show_fine_dots   = True
_show_coarse_dots = True
_selected_sample  = -1

# face filter
_pick_face_mode  = False
_selected_face   = -1


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _uniform_random_bc(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random BCs via sqrt-Osada transform (uniform over triangle area)."""
    r1 = rng.random(n)
    r2 = rng.random(n)
    sr = np.sqrt(r1)
    return np.stack([1.0 - sr, sr * (1.0 - r2), sr * r2], axis=1).astype(np.float64)


def _face_areas(V, F):
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _area_weighted_face_ids(areas, N, rng):
    """
    Distribute N samples across faces proportionally to area.
    Uses largest-remainder method so total == N exactly.
    """
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


def _compute_f2c_samples():
    """
    Build sample positions on the fine mesh and compute their F2C landings
    by interpolating the pre-computed per-vertex F2C map.
    """
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

    # Fine 3-D positions
    f0 = b.fineV[b.fineF[face_ids, 0]]
    f1 = b.fineV[b.fineF[face_ids, 1]]
    f2 = b.fineV[b.fineF[face_ids, 2]]
    fine_pts = bc[:, 0:1] * f0 + bc[:, 1:2] * f1 + bc[:, 2:3] * f2

    # F2C landed positions via interpolated per-vertex map
    c0 = _f2c_v[b.fineF[face_ids, 0]]
    c1 = _f2c_v[b.fineF[face_ids, 1]]
    c2 = _f2c_v[b.fineF[face_ids, 2]]
    coarse_pts = bc[:, 0:1] * c0 + bc[:, 1:2] * c1 + bc[:, 2:3] * c2

    # Track flag: True if all 3 vertices of the face have F2C tracking
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
# Active mask (face filter)
# ---------------------------------------------------------------------------

def _active_mask():
    if _sample_face_ids is None:
        return None
    if _selected_face < 0:
        return np.ones(len(_sample_face_ids), dtype=bool)
    return _sample_face_ids == _selected_face


# ---------------------------------------------------------------------------
# Rebuild helpers
# ---------------------------------------------------------------------------

def _rebuild_meshes():
    # Fine mesh lifted to z_offset (source)
    fv = _bundle.fineV.copy()
    fv[:, 2] += _z_offset
    fm = ps.register_surface_mesh(FINE_MESH, fv, _bundle.fineF)
    fm.set_color((0.55, 0.55, 0.55))
    fm.set_edge_width(0.3)
    fm.set_smooth_shade(False)
    fm.set_transparency(0.4)

    # Coarse mesh at Z=0 (target)
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

    changed = False

    try:
        psim.SetNextWindowSize((360, 420), psim.ImGuiCond_FirstUseEver)
    except Exception:
        pass
    psim.Begin("Fine -> Coarse (Face Samples)", True)

    psim.TextUnformatted("[ Samples on fine faces  →  coarse mesh F2C query ]")
    psim.Separator()

    _slider_range = max(2.0, _mesh_span * 5)
    c, v = psim.SliderFloat("Z offset", _z_offset, -_slider_range, _slider_range)
    if c:
        _z_offset = v
        changed = True

    psim.Separator()
    psim.TextUnformatted("Sampling mode")

    c, v = psim.Checkbox("Per face (N samples per fine face)", _per_face_mode)
    if c:
        _per_face_mode = v

    if _per_face_mode:
        c, v = psim.InputInt("Samples per face", _n_samples)
        if c:
            _n_samples = max(1, min(v, 1000))
        if _bundle is not None:
            FF = _bundle.fineF.shape[0]
            psim.TextDisabled(f"  = {_n_samples} × {FF} = {_n_samples * FF} total")
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
        N = len(_fine_sample_pts)
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

    psim.Separator()
    if _selected_sample >= 0 and _fine_sample_pts is not None:
        si  = _selected_sample
        fi  = int(_sample_face_ids[si])
        bc  = _sample_bcs[si]
        fp  = _fine_sample_pts[si]
        cp  = _coarse_land_pts[si]
        trk = bool(_sample_tracked[si]) if _sample_tracked is not None else False
        dist = float(np.linalg.norm(cp - fp))
        psim.TextUnformatted(f"Sample {si}  {'[tracked]' if trk else '[untracked]'}")
        psim.TextUnformatted(f"  Fine face   : {fi}  BC=({bc[0]:.3f},{bc[1]:.3f},{bc[2]:.3f})")
        psim.TextUnformatted(f"  Fine pos    : ({fp[0]:.5f},{fp[1]:.5f},{fp[2]:.5f})")
        psim.TextUnformatted(f"  Coarse land : ({cp[0]:.5f},{cp[1]:.5f},{cp[2]:.5f})")
        psim.TextUnformatted(f"  Dist F→C    : {dist:.6f}")
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

            if sname == FINE_MESH and _pick_face_mode:
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
                            trk = bool(_sample_tracked[si]) if _sample_tracked is not None else False
                            print(f"\n[sample {si}]  fine face={fi}  "
                                  f"BC=({bc[0]:.4f},{bc[1]:.4f},{bc[2]:.4f})  "
                                  f"{'tracked' if trk else 'UNTRACKED'}")
                            print(f"  fine pos    : ({fp[0]:.6f},{fp[1]:.6f},{fp[2]:.6f})")
                            print(f"  coarse land : ({cp[0]:.6f},{cp[1]:.6f},{cp[2]:.6f})")
                            print(f"  dist F→C    : {np.linalg.norm(cp-fp):.6f}")

    if changed and _fine_sample_pts is not None:
        _rebuild_all()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global _bundle, _bundle_dir, _z_offset, _mesh_span, _f2c_v, _tracked_mask

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

    print("\n[main] Computing per-vertex F2C map …")
    _f2c_v, _tracked_mask = compute_f2c_correspondences(_bundle)
    n_tr = int(_tracked_mask.sum())
    print(f"[main] {n_tr}/{_bundle.fineV.shape[0]} fine vertices tracked by F2C")

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
