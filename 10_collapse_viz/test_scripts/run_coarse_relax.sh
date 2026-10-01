#!/usr/bin/env bash
# One coarse-subdivision relaxation run on the test mesh (ABC 00040057), Release
# build, output in output/relaxation_experiments/<name>/ and <name>.log.
# Writes <name>/experiment_config.txt with the parameters of the run (and the
# result lines once it finishes). Keep <name> short: long folder names hit the
# Windows 260-character path limit (the end-of-run [SANITY] check then fails to
# read the JSON).
#
# usage (from 10_collapse_viz):
#   EXP_DESC="what this run tests" bash test_scripts/run_coarse_relax.sh <name> [extra flags]
# extra flags (all off by default):
#   --coarse_subdiv_relax_method newton|solve_project|explicit   (default newton)
#       explicit = small Laplacian steps on everything at once, no linear solve
#       (coarse_subdiv_relax_explicit.cpp); also writes *_it<N>.obj snapshots
#   --explicit_lambda L (0.5)  --explicit_max_iter N (20000)  --explicit_tol T (1e-7 x diag)
#   --explicit_global_proj     global closest point instead of the local one
#   --explicit_directed_graph  explicit: curves ignore sheets (directed graph); default symmetric
#   --coarse_subdiv_relax_weights uniform|cotan|meanvalue   explicit: neighbour weights (default uniform)
#   --coarse_subdiv_relax_per_face     relax only each coarse face's interior
#   --coarse_subdiv_relax_no_new_folds no step may fold an unfolded triangle (newton, explicit)
#   --coarse_subdiv_relax_local_proj   closest point reachable from the current location (newton)
#   --coarse_subdiv_relax_joint        curves + sheets in one pass, symmetric graph (newton)
#   --coarse_subdiv_relax_joint_solve  solve_project: one LU solve of curves + sheets (directed graph)
#   --coarse_subdiv_relax_max_iter N
set -e
name=$1; shift
D="D:/datasets/abc_full_10k/out_ABC_v6_knn_poission40_20_15_10/01_00040057_f8f78dbd17414efda75bc437_trimesh_000/mat/mat_01_00040057_f8f78dbd17414efda75bc437_trimesh_000.obj__2025-05-06_02_38_00"
R=output/relaxation_experiments
EXE=build/release/Release/collapse_viz_bin.exe
FIXED=(--target_faces 200 --mode qslim --n_subdiv_samples 200000 --mat_struct_check)
rm -rf "$R/$name"; mkdir -p "$R/$name"

cfg="$R/$name/experiment_config.txt"
{
    echo "experiment:   $name"
    echo "description:  ${EXP_DESC:-(none given; set EXP_DESC)}"
    echo "started:      $(date '+%Y-%m-%d %H:%M:%S')"
    echo "git commit:   $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet HEAD -- . 2>/dev/null || echo ' (uncommitted changes)')"
    echo "binary:       $EXE"
    echo "mesh:         $D.obj"
    echo "matstruct:    $D.ma_struct"
    echo "fixed params: ${FIXED[*]}"
    echo "extra flags:  ${*:-(none)}"
    echo "defaults not overridden above: method newton; explicit lambda 0.5, max_iter 20000, tol 1e-7 x diag,"
    echo "              local projection, symmetric graph, uniform weights; no-new-folds off; global (no per-face)"
    echo "output:       $R/$name/ , log $R/$name.log"
    echo "command:      $EXE --mesh_path $D.obj --matstruct_path $D.ma_struct ${FIXED[*]} --output_dir ./$R/$name/ $*"
} > "$cfg"

"$EXE" --mesh_path "$D.obj" --matstruct_path "$D.ma_struct" "${FIXED[@]}" \
    --output_dir "./$R/$name/" "$@" > "$R/$name.log" 2>&1 || status=$?

summary=$(grep -hE "^\[relax_explicit\] (converged|NOT)|^\[subdiv_relax\] (newton:|joint:|curve:|sheet:)|projection:|no-new-folds|local projection:|relaxation \(|checks:" "$R/$name.log" || true)
{
    echo "finished:     $(date '+%Y-%m-%d %H:%M:%S') (exit code ${status:-0})"
    echo "result lines:"
    echo "$summary" | sed 's/^/  /'
} >> "$cfg"
echo "$summary"
