#!/usr/bin/env python3
"""
run_viz_for_mesh.py  —  Launch the fine-deform visualizer with a GUI mesh switcher.

Scans Experiments/ for all meshes that have a c2f_n1000000/ folder containing
a .c2f bundle plus samples_coarse_*.txt and samples_fine_*.txt, then exposes a
"Mesh Switcher" dropdown inside the Polyscope window to hot-swap between them.

Usage:
    python run_viz_for_mesh.py                          # opens first found mesh
    python run_viz_for_mesh.py <mesh_name>              # opens the named mesh
"""

import sys
import os
import glob
import argparse

EXPERIMENTS_ROOT = r"C:\Users\alirz\Projects\Graphics\Neural QMAT\Experiments"
C2F_DIR_NAME     = "c2f_n1000000"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import fine_deform_viz_with_query as viz
import polyscope as ps
import polyscope.imgui as psim

# ---------------------------------------------------------------------------
# Mesh discovery
# ---------------------------------------------------------------------------

# Each entry: (display_name, c2f_path, bundle_dir)
_available_meshes: list = []
_current_mesh_idx: int  = 0
_loading: bool          = False   # guard against re-entrant loads


def _scan_meshes() -> list:
    """Return (mesh_name, c2f_path, bundle_dir) for every valid mesh."""
    found = []
    if not os.path.isdir(EXPERIMENTS_ROOT):
        print(f"[scan] EXPERIMENTS_ROOT not found: {EXPERIMENTS_ROOT}")
        return found

    for mesh_name in sorted(os.listdir(EXPERIMENTS_ROOT)):
        mesh_dir = os.path.join(EXPERIMENTS_ROOT, mesh_name)
        if not os.path.isdir(mesh_dir):
            continue
        for dirpath, _, _ in os.walk(mesh_dir):
            if os.path.basename(dirpath) != C2F_DIR_NAME:
                continue
            c2f_files = glob.glob(os.path.join(dirpath, "*.c2f"))
            coarse    = glob.glob(os.path.join(dirpath, "samples_coarse_*.txt"))
            fine      = glob.glob(os.path.join(dirpath, "samples_fine_*.txt"))
            if c2f_files and coarse and fine:
                found.append((mesh_name, c2f_files[0], dirpath))
                break   # one c2f_n1000000 per mesh is enough
    return found


# ---------------------------------------------------------------------------
# State reset + mesh load
# ---------------------------------------------------------------------------

def _reset_data_state():
    """Reset all data-dependent viz globals so a new mesh can be loaded cleanly."""
    viz._bundle          = None
    viz._bundle_dir      = ""
    viz._fine_ids        = None
    viz._fine_pts        = None
    viz._deform_pts      = None
    viz._deform_mesh_v   = None
    viz._fine_id_to_row  = None
    viz._z_offset        = 1.0
    viz._mesh_span       = 1.0

    viz._f2c_deform_mesh_v   = None
    viz._f2c_tracked_mask    = None
    viz._f2c_onering_mesh_v  = None
    viz._f2c_incident_mesh_v = None

    viz._flipped_highlight_active = False
    viz._selected_flipped_face    = -1
    viz._flipped_faces_glob       = None
    viz._flipped_vtx_colors       = None
    viz._flipped_vtx_arrows       = None

    viz._selected_deform_face = -1
    viz._selected_vtx         = -1
    viz._vtx_collapse_steps.clear()
    viz._vtx_query_positions.clear()
    viz._current_step_idx = 0
    viz._canonical_view   = False

    viz._c2f_mode            = False
    viz._selected_coarse_vtx = -1
    viz._coarse_vtx_input_id = 0
    viz._c2f_result_pos      = None
    viz._c2f_result_bf       = None
    viz._c2f_result_bc       = None


def _load_mesh(mesh_name: str, c2f_path: str, bundle_dir: str) -> bool:
    """Reload viz globals with the given bundle and rebuild all Polyscope structures."""
    _reset_data_state()

    print(f"\n[switch] Loading: {mesh_name}")
    print(f"         Bundle : {c2f_path}")

    viz._bundle_dir      = bundle_dir
    viz._bundle          = viz.load_bundle(c2f_path)
    viz._simplified_ps_name = f"simplified_{os.path.basename(bundle_dir)}"

    fv        = viz._bundle.fineV
    mesh_span = float((fv.max(axis=0) - fv.min(axis=0)).max())
    if mesh_span > 1e-10:
        viz._z_offset  = mesh_span
        viz._mesh_span = mesh_span

    print(f"[switch] Fine  : {viz._bundle.fineV.shape[0]} verts  "
          f"{viz._bundle.fineF.shape[0]} faces")
    if viz._bundle.has_ssp_data:
        print(f"[switch] SSP   : {len(viz._bundle.decInfo)} collapses")

    ok = viz._compute_vertex_correspondences()
    if not ok:
        print("[switch] ERROR: could not compute vertex correspondences")
        return False

    viz._compute_f2c_deformed_mesh()
    viz._log_decim_stats()

    ps.remove_all_structures()
    viz._rebuild_all()
    ps.set_program_name(f"Fine -> Deformed  |  {mesh_name}")
    try:
        ps.reset_camera_to_home_view()
    except Exception:
        pass

    return True


# ---------------------------------------------------------------------------
# Combined UI callback
# ---------------------------------------------------------------------------

def _mesh_switcher_panel():
    """Draw the Mesh Switcher window above the main viz panel."""
    global _current_mesh_idx, _loading

    psim.SetNextWindowSize((460, 70), psim.ImGuiCond_FirstUseEver)
    psim.Begin("Mesh Switcher")

    mesh_names = [m[0] for m in _available_meshes]
    changed, new_idx = psim.Combo("##mesh", _current_mesh_idx, mesh_names)

    if changed and new_idx != _current_mesh_idx and not _loading:
        _loading = True
        _current_mesh_idx = new_idx
        name, c2f_path, bundle_dir = _available_meshes[new_idx]
        _load_mesh(name, c2f_path, bundle_dir)
        _loading = False

    psim.SameLine()
    psim.TextUnformatted(f"  {_current_mesh_idx + 1} / {len(_available_meshes)}")

    psim.End()


def combined_callback():
    _mesh_switcher_panel()
    viz.ui_callback()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global _available_meshes, _current_mesh_idx

    parser = argparse.ArgumentParser(
        description="Fine-deform visualizer with GUI mesh switcher."
    )
    parser.add_argument(
        "mesh_name", nargs="?", default=None,
        help="Optional starting mesh name (folder inside Experiments/). "
             "Defaults to the first mesh found.",
    )
    args = parser.parse_args()

    print(f"[scan] Scanning {EXPERIMENTS_ROOT} …")
    _available_meshes = _scan_meshes()
    if not _available_meshes:
        print(f"ERROR: no valid meshes found in {EXPERIMENTS_ROOT}")
        sys.exit(1)
    print(f"[scan] Found {len(_available_meshes)} mesh(es)")

    # Resolve starting mesh
    start_idx = 0
    if args.mesh_name:
        for i, (name, _, _) in enumerate(_available_meshes):
            if name == args.mesh_name:
                start_idx = i
                break
        else:
            print(f"[warn] mesh '{args.mesh_name}' not found — using first available")

    _current_mesh_idx = start_idx
    name, c2f_path, bundle_dir = _available_meshes[start_idx]

    # --- initial load (before ps.init so heavy work happens outside the render loop) ---
    viz._bundle_dir      = bundle_dir
    viz._bundle          = viz.load_bundle(c2f_path)
    viz._simplified_ps_name = f"simplified_{os.path.basename(bundle_dir)}"

    fv        = viz._bundle.fineV
    mesh_span = float((fv.max(axis=0) - fv.min(axis=0)).max())
    if mesh_span > 1e-10:
        viz._z_offset  = mesh_span
        viz._mesh_span = mesh_span

    print(f"Fine  : {viz._bundle.fineV.shape[0]} verts  {viz._bundle.fineF.shape[0]} faces")
    if viz._bundle.has_ssp_data:
        print(f"SSP   : {len(viz._bundle.decInfo)} collapses")

    ok = viz._compute_vertex_correspondences()
    if not ok:
        print("ERROR: could not compute vertex correspondences — exiting.")
        sys.exit(1)

    viz._compute_f2c_deformed_mesh()
    viz._log_decim_stats()

    ps.init()
    ps.set_program_name(f"Fine -> Deformed  |  {name}")
    ps.set_up_dir("y_up")

    viz._rebuild_all()
    ps.set_user_callback(combined_callback)
    ps.show()


if __name__ == "__main__":
    main()
