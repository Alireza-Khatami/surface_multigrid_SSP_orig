#!/usr/bin/env bash
# The equal-area relaxation experiments on ABC 00040057, from scratch:
#
#   1. C++ source run (Release, headless): qslim decimation to 200 faces WITH
#      --validity-checks and --mat_struct_check -> the bundle (.c2f) every experiment
#      starts from (output/relaxation_experiments/src_qslim200_valid/). Skipped when
#      its bundle already exists (FORCE_CPP=1 reruns it).
#   2. The Python relaxation experiments (run_relax.py), one folder each:
#      output/relaxation_experiments/<name>_valid/ with experiment_config.txt, run.log,
#      results.json, the relaxed OBJ (+ _it1/_it10/_it100/_it1000 snapshots) and
#      relax_input/*.ply (coarse, equal-area refined, subdivided, c2f positions,
#      relaxation input).
#   3. The results table: output/relaxation_experiments/py_experiments_results_equal_area_valid.{csv,md}
#
# Usage (from anywhere):
#   bash relaxation_scripts_python/run_equal_area_experiments.sh            # all
#   bash relaxation_scripts_python/run_equal_area_experiments.sh eqareamin   # only names containing "eqareamin"
# Environment: PY (python to use, default: python), FORCE_CPP=1 (rerun the C++ source run).
#
# Every experiment: explicit relaxation, uniform weights unless said, no new folds,
# lambda 0.5 (default), 2000 iterations, 200,000 subdivision samples (default target).
set -u
HERE=$(cd "$(dirname "$0")" && (pwd -W 2>/dev/null || pwd))   # Windows form (C:/...) for python
ROOT=$(cd "$HERE/.." && (pwd -W 2>/dev/null || pwd))   # 10_collapse_viz
EXP=$ROOT/output/relaxation_experiments
PY=${PY:-python}
FILTER=${1:-}
SRC=src_qslim200_valid
MS=D:/datasets/abc_full_10k/out_ABC_v6_knn_poission40_20_15_10/01_00040057_f8f78dbd17414efda75bc437_trimesh_000/mat/mat_01_00040057_f8f78dbd17414efda75bc437_trimesh_000.obj__2025-05-06_02_38_00.ma_struct

# ---- 1. C++ source run --------------------------------------------------------------
BUNDLE=$(ls "$EXP/$SRC"/correspondence_*.c2f 2>/dev/null | head -1)
if [ -z "$BUNDLE" ] || [ "${FORCE_CPP:-0}" = 1 ]; then
    echo "=== C++ source run $SRC"
    (cd "$ROOT" && bash relaxation_scripts_python/cpp_reference_run.sh "$SRC" \
        "source run for the Python relaxation experiments: qslim to 200 faces WITH --validity-checks and --mat_struct_check; C++ explicit + no new folds, 1000 iterations as reference" \
        --coarse_subdiv_relax_method explicit --coarse_subdiv_relax_no_new_folds --explicit_max_iter 1000) || exit 1
    BUNDLE=$(ls "$EXP/$SRC"/correspondence_*.c2f | head -1)
fi
grep -q "Validity checks: ENABLED" "$EXP/$SRC/experiment_config.txt" \
    || { echo "source run $SRC does not have validity checks ENABLED"; exit 1; }
echo "bundle: $BUNDLE"

# ---- 2. experiments -----------------------------------------------------------------
BASE="--coarse_subdiv_relax_method explicit --coarse_subdiv_relax_no_new_folds --explicit_max_iter 2000"
run() {  # name, description, flags...
    local name=$1 desc=$2; shift 2
    case "$name" in *"$FILTER"*) ;; *) return ;; esac
    echo "=== ${name}_valid: $desc"
    "$PY" "$HERE/run_relax.py" --bundle "$BUNDLE" --matstruct_path "$MS" \
        --output_dir "$EXP/${name}_valid" --no_subdiv_objs \
        --description "$desc [bundle of C++ run $SRC]" $BASE "$@" \
        || echo "!!! ${name}_valid failed (see $EXP/${name}_valid/run.log)"
}

run py_nofold_2k \
    "baseline: uniform weights, symmetric graph, no new folds, lambda 0.5, local projection" \
    --explicit_local_proj
run py_eqarea1_nofold_2k \
    "as baseline, coarse faces refined to equal area first (levels 1: ~4|F| faces)" \
    --explicit_local_proj --equal_area_levels 1
run py_eqarea2_nofold_2k \
    "as baseline, coarse faces refined to equal area first (levels 2: ~16|F| faces)" \
    --explicit_local_proj --equal_area_levels 2
run py_eqarea2_nofold_global_2k \
    "uniform weights, symmetric (two-sided) graph, no new folds, global projection, equal-area levels 2" \
    --explicit_global_proj --equal_area_levels 2
run py_eqareamin_nofold_global_2k \
    "uniform weights, symmetric (two-sided) graph, no new folds, global projection, equal-area target = the smallest coarse face area, then the uniform subdivision" \
    --explicit_global_proj --equal_area_target min
run py_cotan_nofold_dir_2k \
    "cotangent weights, directed graph, no new folds, local projection" \
    --explicit_local_proj --coarse_subdiv_relax_weights cotan --explicit_directed_graph
run py_eqarea2_cotan_nofold_dir_2k \
    "cotan, directed, no new folds, local projection, equal-area levels 2" \
    --explicit_local_proj --coarse_subdiv_relax_weights cotan --explicit_directed_graph --equal_area_levels 2

# ---- 3. table -----------------------------------------------------------------------
"$PY" "$HERE/run_experiments.py" --set equal_area --source_run "$SRC" --suffix _valid --table_only > /dev/null
echo "table: $EXP/py_experiments_results_equal_area_valid.md"
