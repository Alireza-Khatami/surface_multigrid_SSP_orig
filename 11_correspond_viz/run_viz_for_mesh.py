#!/usr/bin/env python3
"""
run_viz_for_mesh.py  —  Launch the fine-deform visualizer for any experiment mesh.

Searches Experiments/<mesh_name>/**/c2f_n1000000/ for a .c2f bundle, then runs
fine_deform_viz_with_query with that bundle (which auto-discovers samples_vertices_*.txt
from the same directory).

Usage:
    python run_viz_for_mesh.py <mesh_name>
    python run_viz_for_mesh.py 01_00040049_5c84f0ac4aea4ad28f79872b_trimesh_000
"""

import sys
import os
import glob
import argparse

EXPERIMENTS_ROOT = r"C:\Users\alirz\Projects\Graphics\Neural QMAT\Experiments"

# Ensure this script's directory is on the path so sibling modules are importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def find_c2f_bundle(mesh_name: str):
    """
    Walk <EXPERIMENTS_ROOT>/<mesh_name>/ looking for a c2f_n1000000 directory
    that contains a .c2f file.  Returns (c2f_path, bundle_dir) or (None, None).
    """
    mesh_dir = os.path.join(EXPERIMENTS_ROOT, mesh_name)
    if not os.path.isdir(mesh_dir):
        print(f"ERROR: mesh directory not found: {mesh_dir}")
        return None, None

    for dirpath, dirnames, filenames in os.walk(mesh_dir):
        if os.path.basename(dirpath) == "c2f_n1000000":
            c2f_files = glob.glob(os.path.join(dirpath, "*.c2f"))
            if c2f_files:
                return c2f_files[0], dirpath

    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Run fine_deform_viz_with_query for a named experiment mesh."
    )
    parser.add_argument(
        "mesh_name",
        help="Folder name inside Experiments/ (e.g. 01_00040049_5c84f0ac4aea4ad28f79872b_trimesh_000)",
    )
    args = parser.parse_args()

    c2f_path, bundle_dir = find_c2f_bundle(args.mesh_name)
    if c2f_path is None:
        print(f"ERROR: no .c2f file found under c2f_n1000000/ in Experiments/{args.mesh_name}")
        sys.exit(1)

    samples_coarse = glob.glob(os.path.join(bundle_dir, "samples_coarse_*.txt"))
    samples_fine   = glob.glob(os.path.join(bundle_dir, "samples_fine_*.txt"))
    print(f"Bundle       : {c2f_path}")
    print(f"Bundle dir   : {bundle_dir}")
    print(f"samples_coarse: {samples_coarse[0] if samples_coarse else 'NOT FOUND'}")
    print(f"samples_fine  : {samples_fine[0]   if samples_fine   else 'NOT FOUND'}")

    # Import the visualizer module and drive it with the found bundle path,
    # replicating the logic of fine_deform_viz_with_query.main() but dynamically.
    import fine_deform_viz_with_query as viz
    import polyscope as ps

    viz._bundle_dir     = bundle_dir
    viz._bundle         = viz.load_bundle(c2f_path)

    mesh_dir_name           = os.path.basename(bundle_dir)
    viz._simplified_ps_name = f"simplified_{mesh_dir_name}"

    fv        = viz._bundle.fineV
    mesh_span = float((fv.max(axis=0) - fv.min(axis=0)).max())
    if mesh_span > 1e-10:
        viz._z_offset  = mesh_span
        viz._mesh_span = mesh_span

    print(f"Fine mesh : {viz._bundle.fineV.shape[0]} verts  {viz._bundle.fineF.shape[0]} faces")
    if viz._bundle.has_ssp_data:
        print(f"SSP       : {len(viz._bundle.decInfo)} collapses")

    ok = viz._compute_vertex_correspondences()
    if not ok:
        print("ERROR: could not compute vertex correspondences — exiting.")
        sys.exit(1)

    viz._compute_f2c_deformed_mesh()
    viz._log_decim_stats()

    ps.init()
    ps.set_program_name(f"Fine -> Deformed — {args.mesh_name}")
    ps.set_up_dir("y_up")

    viz._rebuild_all()
    ps.set_user_callback(viz.ui_callback)
    ps.show()


if __name__ == "__main__":
    main()
