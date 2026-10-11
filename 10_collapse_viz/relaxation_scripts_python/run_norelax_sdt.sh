#!/usr/bin/env bash
# The SSP collapse output as an .sdt WITHOUT relaxation (export_sdt_norelax.py), then the check that
# the .sdt reconstructs the SSP output (coarse_subdiv_at_fine_pos_*.obj, coarse_subdiv_*.obj,
# simplified_*.obj, the C++ subdiv_*.sdt layout) of the SSP run folder.
# Input: the default bundle (output/relaxation_experiments/src_qslim200_valid, validity checks on)
# and the ABC 00040057 .ma_struct. Notes: md_files/relaxations_md/coarse_subdiv_relaxed_sdt.md
#
#   bash run_norelax_sdt.sh [NAME] [extra export_sdt_norelax.py / run_relax.py flags...]
#
#   NAME   output folder in output/relaxation_experiments (default py_norelax_sdt_valid)
#
# Examples:
#   bash run_norelax_sdt.sh                                         # SSP sampling (uniform, 200,000 target)
#   bash run_norelax_sdt.sh py_norelax_snapped_sdt_valid --snap_seeds
#   bash run_norelax_sdt.sh py_norelax_eqs278k_flip_sdt_valid --equal_area_samples 277700 --delaunay_flips
#   bash run_norelax_sdt.sh my_run --bundle <ssp run>/correspondence_<stem>.c2f --matstruct_path <.ma_struct>
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${1:-py_norelax_sdt_valid}"
shift $(( $# < 1 ? $# : 1 ))
PYTHON="${PYTHON:-python}"
OUT="$HERE/../output/relaxation_experiments/$NAME"

status=0
echo "=== $NAME: SSP output as .sdt, no relaxation -> $OUT"
"$PYTHON" "$HERE/export_sdt_norelax.py" --output_dir "$OUT" "$@" || status=$?
tail -n 2 "$OUT/experiment_config.txt"
exit $status
