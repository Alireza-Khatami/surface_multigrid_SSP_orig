#!/usr/bin/env bash
# One coarse-subdivision relaxation run on the test mesh (ABC 00040057), Release
# build, output in output/relaxation_experiments/<name>/ and <name>.log.
# Keep <name> short: long folder names hit the Windows 260-character path limit
# (the end-of-run [SANITY] check then fails to read the JSON).
#
# usage (from 10_collapse_viz):  bash test_scripts/run_coarse_relax.sh <name> [extra flags]
# extra flags (all off by default):
#   --coarse_subdiv_relax_method newton|solve_project|explicit   (default newton)
#       explicit = small Laplacian steps on everything at once, no linear solve
#       (coarse_subdiv_relax_explicit.cpp); also writes *_it<N>.obj snapshots
#   --explicit_lambda L (0.5)  --explicit_max_iter N (20000)  --explicit_tol T (1e-7 x diag)
#   --explicit_global_proj     global closest point instead of the local one
#   --coarse_subdiv_relax_per_face     relax only each coarse face's interior
#   --coarse_subdiv_relax_no_new_folds no step may fold an unfolded triangle (newton)
#   --coarse_subdiv_relax_local_proj   closest point reachable from the current location (newton)
#   --coarse_subdiv_relax_joint        curves + sheets in one pass, symmetric graph (newton)
#   --coarse_subdiv_relax_joint_solve  solve_project: one LU solve of curves + sheets (directed graph)
#   --coarse_subdiv_relax_max_iter N
set -e
name=$1; shift
D="D:/datasets/abc_full_10k/out_ABC_v6_knn_poission40_20_15_10/01_00040057_f8f78dbd17414efda75bc437_trimesh_000/mat/mat_01_00040057_f8f78dbd17414efda75bc437_trimesh_000.obj__2025-05-06_02_38_00"
R=output/relaxation_experiments
rm -rf "$R/$name"; mkdir -p "$R/$name"
build/release/Release/collapse_viz_bin.exe --mesh_path "$D.obj" --matstruct_path "$D.ma_struct" \
    --target_faces 200 --mode qslim --n_subdiv_samples 200000 --mat_struct_check \
    --output_dir "./$R/$name/" "$@" > "$R/$name.log" 2>&1
grep -hE "^\[relax_explicit\] (converged|NOT)|^\[subdiv_relax\] (newton:|joint:|curve:|sheet:)|projection:|no-new-folds|local projection:|relaxation \(|checks:" "$R/$name.log"
