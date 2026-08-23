#!/usr/bin/env python3
"""
coarse_2_fine_samples_viz_with_query.py
  — Sample N random interior points PER coarse face and map each to the
    fine mesh via the SSP C2F query.  Total samples = N × num_coarse_faces.

Visualization:
  fine_mesh          grey transparent at Z=0      (target / background)
  coarse_mesh_src    green at Z=z_offset           (source mesh)
  coarse_samples     blue dots at Z=z_offset       (sampled coarse positions)
  fine_landings      orange dots at Z=0            (C2F landed positions on fine)
  sample_arrows      yellow ambient arrows          (coarse sample → fine landing)

Click a sample point to print its BC, face, and landing info.

Usage:
    python coarse_2_fine_samples_viz_with_query.py  <bundle.c2f>

Dependencies:
    pip install polyscope numpy
    (c2f_query.py must be in the same directory or on PYTHONPATH)
"""

import sys
import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from c2f_query import load_bundle, query_coarse_to_fine

# ---------------------------------------------------------------------------
# Structure names
# ---------------------------------------------------------------------------

FINE_MESH        = "fine_mesh"
COARSE_MESH_VIZ  = "coarse_mesh_src"
COARSE_SAMPLE_PC = "coarse_samples"
FINE_LANDING_PC  = "fine_landings"
ARROWS_PC        = "sample_arrows"
SEL_COARSE_PC    = "sel_coarse_dot"
SEL_FINE_PC      = "sel_fine_dot"

# ---------------------------------------------------------------------------
# Mutable state
# ---------------------------------------------------------------------------

_bundle      = None
_bundle_dir  = ""
_z_offset    = 1.0
_mesh_span   = 1.0

_n_samples   = 100
_seed        = 42

# per-sample data (set by _compute_c2f_samples)
_sample_face_ids   = None   # (N,) compact coarse face index for each sample
_sample_bcs        = None   # (N, 3) barycentric coords on coarse face (source)
_coarse_sample_pts = None   # (N, 3) 3-D position on coarse mesh at Z=0
_fine_landing_pts  = None   # (N, 3) fine-mesh landing position at Z=0
_fine_landing_bfs  = None   # (N, 3) global fine vertex indices of each landing tri

_show_arrows       = True
_show_coarse_dots  = True
_show_fine_dots    = True
_selected_sample   = -1


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _uniform_random_bc(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw n uniform random barycentric coordinates inside a triangle.
    Uses the sqrt-based Osada transform: (sqrt(r1), sqrt(r1)*(1-r2), sqrt(r1)*r2)
    so that the distribution is uniform over area.
    """
    r1 = rng.random(n)
    r2 = rng.random(n)
    sr = np.sqrt(r1)
    bc = np.stack([1.0 - sr,
                   sr * (1.0 - r2),
                   sr * r2], axis=1)
    return bc.astype(np.float64)


def _compute_c2f_samples():
    """
    Sample _n_samples random points PER coarse face, run query_coarse_to_fine,
    store results in _coarse_sample_pts / _fine_landing_pts.
    Total samples = _n_samples * FC.
    """
    global _sample_face_ids, _sample_bcs
    global _coarse_sample_pts, _fine_landing_pts, _fine_landing_bfs

    b = _bundle
    if b is None or not b.has_ssp_data:
        print("[samples] Bundle has no SSP data — cannot compute C2F samples.")
        return False

    FC  = b.coarseF.shape[0]
    N   = _n_samples * FC   # total samples
    rng = np.random.default_rng(_seed)

    # Each coarse face gets exactly _n_samples samples
    face_ids = np.repeat(np.arange(FC, dtype=np.int32), _n_samples)
    bc = _uniform_random_bc(N, rng)

    # Build BF (global SSP vertex indices) and FIdx (global SSP face indices)
    BC   = bc.copy()
    BF   = np.zeros((N, 3), dtype=np.int32)
    FIdx = np.zeros(N, dtype=np.int32)

    for i in range(N):
        cf = face_ids[i]
        ci0, ci1, ci2 = int(b.coarseF[cf, 0]), int(b.coarseF[cf, 1]), int(b.coarseF[cf, 2])
        BF[i]   = [int(b.vtxMap[ci0]), int(b.vtxMap[ci1]), int(b.vtxMap[ci2])]
        FIdx[i] = int(b.faceMap[cf])

    # 3-D coarse positions (at Z=0; z_offset is added in rebuild for viz)
    coarse_pts = (bc[:, 0:1] * b.coarseV[b.coarseF[face_ids, 0]] +
                  bc[:, 1:2] * b.coarseV[b.coarseF[face_ids, 1]] +
                  bc[:, 2:3] * b.coarseV[b.coarseF[face_ids, 2]])

    # Run the C2F query — modifies BC and BF in-place
    query_coarse_to_fine(b.decInfo, b.decIM, b.faceSheetID, BC, BF, FIdx)

    # Fine landing positions (BC/BF now refer to fine mesh)
    fine_pts = (BC[:, 0:1] * b.fineV[BF[:, 0]] +
                BC[:, 1:2] * b.fineV[BF[:, 1]] +
                BC[:, 2:3] * b.fineV[BF[:, 2]])

    _sample_face_ids   = face_ids
    _sample_bcs        = bc
    _coarse_sample_pts = coarse_pts
    _fine_landing_pts  = fine_pts
    _fine_landing_bfs  = BF.copy()

    print(f"[samples] {_n_samples} samples/face × {FC} faces = {N} total  (seed={_seed})")
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

    cv = _bundle.coarseV.copy()
    cv[:, 2] += _z_offset
    cm = ps.register_surface_mesh(COARSE_MESH_VIZ, cv, _bundle.coarseF)
    cm.set_color((0.15, 0.75, 0.35))
    cm.set_edge_width(1.0)
    cm.set_smooth_shade(False)
    cm.set_transparency(0.55)


def _rebuild_coarse_sample_pc():
    if ps.has_point_cloud(COARSE_SAMPLE_PC):
        ps.remove_point_cloud(COARSE_SAMPLE_PC)
    if not _show_coarse_dots or _coarse_sample_pts is None:
        return
    pts = _coarse_sample_pts.copy()
    pts[:, 2] += _z_offset
    pc = ps.register_point_cloud(COARSE_SAMPLE_PC, pts)
    pc.set_color((0.25, 0.55, 1.0))   # blue
    pc.set_radius(0.005, relative=True)


def _rebuild_fine_landing_pc():
    if ps.has_point_cloud(FINE_LANDING_PC):
        ps.remove_point_cloud(FINE_LANDING_PC)
    if not _show_fine_dots or _fine_landing_pts is None:
        return
    pc = ps.register_point_cloud(FINE_LANDING_PC, _fine_landing_pts)
    pc.set_color((0.95, 0.45, 0.10))  # orange
    pc.set_radius(0.005, relative=True)


def _rebuild_arrows():
    if ps.has_point_cloud(ARROWS_PC):
        ps.remove_point_cloud(ARROWS_PC)
    if _coarse_sample_pts is None or _fine_landing_pts is None:
        return
    src = _coarse_sample_pts.copy()
    src[:, 2] += _z_offset
    vecs = _fine_landing_pts - src
    pc = ps.register_point_cloud(ARROWS_PC, src)
    pc.set_color((1.0, 0.85, 0.0))
    pc.set_radius(0.002, relative=True)
    pc.add_vector_quantity("to_fine", vecs, vectortype="ambient",
                           enabled=_show_arrows, color=(1.0, 0.55, 0.05))


def _rebuild_selection():
    for name in (SEL_COARSE_PC, SEL_FINE_PC):
        if ps.has_point_cloud(name):
            ps.remove_point_cloud(name)

    if _selected_sample < 0 or _coarse_sample_pts is None:
        return

    cpos = _coarse_sample_pts[_selected_sample].copy()
    cpos[2] += _z_offset
    pc_c = ps.register_point_cloud(SEL_COARSE_PC, cpos[np.newaxis])
    pc_c.set_color((1.0, 1.0, 0.0))  # yellow
    pc_c.set_radius(0.012, relative=True)

    fpos = _fine_landing_pts[_selected_sample].copy()
    pc_f = ps.register_point_cloud(SEL_FINE_PC, fpos[np.newaxis])
    pc_f.set_color((1.0, 0.0, 0.3))  # red-pink
    pc_f.set_radius(0.012, relative=True)


def _rebuild_all():
    _rebuild_meshes()
    _rebuild_coarse_sample_pc()
    _rebuild_fine_landing_pc()
    _rebuild_arrows()
    _rebuild_selection()


# ---------------------------------------------------------------------------
# UI callback
# ---------------------------------------------------------------------------

def ui_callback():
    global _z_offset, _mesh_span, _n_samples, _seed
    global _show_arrows, _show_coarse_dots, _show_fine_dots, _selected_sample

    changed = False

    try:
        psim.SetNextWindowSize((340, 320), psim.ImGuiCond_FirstUseEver)
    except Exception:
        pass
    psim.Begin("Coarse -> Fine (Face Samples)", True)

    psim.TextUnformatted("[ Samples on coarse faces  →  fine mesh C2F query ]")
    psim.Separator()

    _slider_range = max(2.0, _mesh_span * 5)
    c, v = psim.SliderFloat("Z offset", _z_offset, -_slider_range, _slider_range)
    if c:
        _z_offset = v
        changed = True

    psim.Separator()
    psim.TextUnformatted("Sampling")

    c, v = psim.InputInt("Samples per face", _n_samples)
    if c:
        _n_samples = max(1, min(v, 1000))

    c, v = psim.InputInt("Seed", _seed)
    if c:
        _seed = max(0, v)

    if psim.Button("Resample & Requery"):
        ok = _compute_c2f_samples()
        if ok:
            _selected_sample = -1
            _rebuild_all()

    if _coarse_sample_pts is not None:
        FC = _bundle.coarseF.shape[0]
        N  = len(_coarse_sample_pts)
        psim.TextUnformatted(f"{_n_samples}/face × {FC} faces = {N} total")

    psim.Separator()
    psim.TextUnformatted("Display")

    c, v = psim.Checkbox("Show coarse sample dots", _show_coarse_dots)
    if c:
        _show_coarse_dots = v
        _rebuild_coarse_sample_pc()

    c, v = psim.Checkbox("Show fine landing dots", _show_fine_dots)
    if c:
        _show_fine_dots = v
        _rebuild_fine_landing_pc()

    c, v = psim.Checkbox("Show arrows", _show_arrows)
    if c:
        _show_arrows = v
        _rebuild_arrows()

    psim.Separator()
    if _selected_sample >= 0 and _coarse_sample_pts is not None:
        si = _selected_sample
        fi = int(_sample_face_ids[si])
        bc = _sample_bcs[si]
        cp = _coarse_sample_pts[si]
        fp = _fine_landing_pts[si]
        bf = _fine_landing_bfs[si]
        dist = float(np.linalg.norm(fp - cp))
        psim.TextUnformatted(f"Sample {si}")
        psim.TextUnformatted(f"  Coarse face : {fi}  BC=({bc[0]:.3f},{bc[1]:.3f},{bc[2]:.3f})")
        psim.TextUnformatted(f"  Coarse pos  : ({cp[0]:.5f},{cp[1]:.5f},{cp[2]:.5f})")
        psim.TextUnformatted(f"  Fine landing: ({fp[0]:.5f},{fp[1]:.5f},{fp[2]:.5f})")
        psim.TextUnformatted(f"  Landing tri : {int(bf[0])},{int(bf[1])},{int(bf[2])}")
        psim.TextUnformatted(f"  Dist C→F    : {dist:.6f}")
        if psim.Button("Clear selection"):
            _selected_sample = -1
            _rebuild_selection()
    else:
        psim.TextDisabled("Click a coarse sample dot to inspect")

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
            if sname == COARSE_SAMPLE_PC:
                idx = int(sdata.get('index', -1))
                if 0 <= idx < len(_coarse_sample_pts):
                    _selected_sample = -1 if idx == _selected_sample else idx
                    _rebuild_selection()
                    if _selected_sample >= 0:
                        fi = int(_sample_face_ids[_selected_sample])
                        bc = _sample_bcs[_selected_sample]
                        cp = _coarse_sample_pts[_selected_sample]
                        fp = _fine_landing_pts[_selected_sample]
                        bf = _fine_landing_bfs[_selected_sample]
                        print(f"\n[sample {_selected_sample}]  coarse face={fi}  "
                              f"BC=({bc[0]:.4f},{bc[1]:.4f},{bc[2]:.4f})")
                        print(f"  coarse pos   : ({cp[0]:.6f},{cp[1]:.6f},{cp[2]:.6f})")
                        print(f"  fine landing : ({fp[0]:.6f},{fp[1]:.6f},{fp[2]:.6f})")
                        print(f"  landing tri  : gv={int(bf[0])},{int(bf[1])},{int(bf[2])}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global _bundle, _bundle_dir, _z_offset, _mesh_span

    c2f_path    = rf"C:\\Users\\alirz\\Projects\\Graphics\\Neural QMAT\\external\\surf_subgrid_SSP_orig\\10_collapse_viz\\output\\01_00040057_f8f78dbd17414efda75bc437_trimesh_000\\correspondence_01_00040057_f8f78dbd17414efda75bc437_trimesh_000_mat_initial.c2f"
    # c2f_path  = rf"C:\\Users\\alirz\\Projects\\Graphics\\Neural QMAT\\external\\surf_subgrid_SSP_orig\\10_collapse_viz\\output\\0002000_partstudio_14_model_ste_00_1024\\correspondence_0002000_partstudio_14_model_ste_00_1024.c2f"
    # c2f_path  = rf"C:\\Users\\alirz\\Projects\\Graphics\\Neural QMAT\\external\\surf_subgrid_SSP_orig\\10_collapse_viz\\output\\hand\\correspondence_hand.c2f"
    # c2f_path  = rf"C:\\Users\\alirz\\Projects\\Graphics\\Neural QMAT\\external\\surf_subgrid_SSP_orig\\10_collapse_viz\\output\\subDiv_cube\\correspondence_subDiv_cube.c2f"

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

    ok = _compute_c2f_samples()
    if not ok:
        print("ERROR: could not compute C2F samples — exiting.")
        sys.exit(1)

    ps.init()
    ps.set_program_name("Coarse -> Fine (Face Samples)")
    ps.set_up_dir("y_up")

    _rebuild_all()

    ps.set_user_callback(ui_callback)
    ps.show()


if __name__ == "__main__":
    main()
