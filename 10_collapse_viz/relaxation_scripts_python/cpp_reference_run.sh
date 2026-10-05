#!/usr/bin/env bash
# Headless Release run of the C++ pipeline on ABC 00040057 into
# output/relaxation_experiments/<name>/ (+ <name>.log, experiment_config.txt),
# to get C++ references for flags no earlier run used.
# Usage (from 10_collapse_viz): bash relaxation_scripts_python/cpp_reference_run.sh <name> "<description>" <extra flags...>
set -u
NAME=$1; DESC=$2; shift 2
BIN=build/release/Release/collapse_viz_bin.exe
D=D:/datasets/abc_full_10k/out_ABC_v6_knn_poission40_20_15_10/01_00040057_f8f78dbd17414efda75bc437_trimesh_000/mat
MESH=$D/mat_01_00040057_f8f78dbd17414efda75bc437_trimesh_000.obj__2025-05-06_02_38_00.obj
MS=$D/mat_01_00040057_f8f78dbd17414efda75bc437_trimesh_000.obj__2025-05-06_02_38_00.ma_struct
FIXED="--target_faces 200 --mode qslim --n_subdiv_samples 200000 --validity-checks --mat_struct_check"
OUT=./output/relaxation_experiments/$NAME/
mkdir -p "$OUT"
CMD="$BIN --mesh_path $MESH --matstruct_path $MS $FIXED --output_dir $OUT $*"
{
echo "experiment:   $NAME"
echo "description:  $DESC"
echo "started:      $(date '+%Y-%m-%d %H:%M:%S')"
echo "git commit:   $(git rev-parse --short HEAD)$(git diff --quiet -- . ../src || echo " (uncommitted C++ changes)")"
echo "binary:       $BIN"
echo "mesh:         $MESH"
echo "matstruct:    $MS"
echo "fixed params: $FIXED"
echo "extra flags:  $*"
echo "output:       $OUT , log ./output/relaxation_experiments/$NAME.log"
echo "command:      $CMD"
} > "$OUT/experiment_config.txt"
$CMD > "./output/relaxation_experiments/$NAME.log" 2>&1
rc=$?
{
echo "finished:     $(date '+%Y-%m-%d %H:%M:%S') (exit code $rc)"
echo "result lines:"
grep -E "^Validity checks|^\[relax_explicit\] (NOT CONVERGED|converged)|^\[relax_explicit\]   (projection:|checks)|relaxation \(explicit" "./output/relaxation_experiments/$NAME.log" | sed 's/^/  /'
} >> "$OUT/experiment_config.txt"
echo "$NAME exit $rc"
