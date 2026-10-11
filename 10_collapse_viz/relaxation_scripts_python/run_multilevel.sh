#!/usr/bin/env bash
# Coarse-to-fine (multilevel) relaxation run with setup B (relax_multilevel.py):
# equal-area split to 277,700 samples + Delaunay flips, uniform weights, two-sided graph,
# global projection, no new folds, momentum beta 0.9; levels N/16 -> N/4 -> N (ratio 4).
# Input: the default bundle (output/relaxation_experiments/src_qslim200_valid, validity checks on)
# and the ABC 00040057 .ma_struct. Notes: md_files/relaxations_md/relaxation_multilevel_*.md
#
#   bash run_multilevel.sh [NAME] [LEVEL_ITERS] [LEVELS] [extra relax_multilevel.py / run_relax.py flags...]
#
#   NAME         output folder in output/relaxation_experiments (default py_ml3_momentum_300_100_100_valid)
#   LEVEL_ITERS  iterations per level, coarse to fine (default 300,100,100)
#   LEVELS       number of levels (default 3)
#
# Examples:
#   bash run_multilevel.sh
#   bash run_multilevel.sh py_ml3_momentum_3000x3_valid 3000,3000,3000
#   bash run_multilevel.sh py_ml3_explicit_valid 1000,300,300 3 --coarse_subdiv_relax_method explicit
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${1:-py_ml3_momentum_300_100_100_valid}"
ITERS="${2:-300,100,100}"
LEVELS="${3:-3}"
shift $(( $# < 3 ? $# : 3 ))
PYTHON="${PYTHON:-python}"
OUT="$HERE/../output/relaxation_experiments/$NAME"

status=0
echo "=== $NAME: $LEVELS levels, iterations $ITERS -> $OUT"
"$PYTHON" "$HERE/relax_multilevel.py" \
    --levels "$LEVELS" --level_ratio 4 --level_iters "$ITERS" \
    --output_dir "$OUT" --no_subdiv_objs \
    --description "multilevel $LEVELS levels (ratio 4), B momentum beta 0.9, iterations $ITERS (run_multilevel.sh)" \
    --coarse_subdiv_relax_method momentum --momentum_beta 0.9 \
    --coarse_subdiv_relax_no_new_folds \
    --explicit_global_proj --equal_area_samples 277700 --delaunay_flips \
    "$@" || status=$?
tail -n 4 "$OUT/experiment_config.txt"
"$PYTHON" "$HERE/check_relaxed_sdt.py" --run_dir "$OUT" | tail -n 1
exit $status
