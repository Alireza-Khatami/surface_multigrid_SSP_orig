#!/usr/bin/env bash
# Verifies the Python relaxation against the C++ runs in output/relaxation_experiments.
# Usage: bash verify_all.sh <out_root> [prefix_iters]
# Each run: verify_relax.py with the C++ run's flags; runs longer than prefix_iters
# (default 1000) are compared on their first prefix_iters iterations.
set -u
OUT=${1:?out root}
N=${2:-1000}
EXP=../output/relaxation_experiments
E="--coarse_subdiv_relax_method explicit"
run() {  # name, C++ iterations, flags...
    local name=$1 iters=$2; shift 2
    local it=$(( iters < N ? iters : N ))
    python verify_relax.py --cpp_dir $EXP/$name --out_dir "$OUT/$name" -- $E "$@" --explicit_max_iter $it \
        > "$OUT/$name.verify.txt" 2> "$OUT/$name.stderr.txt"
    echo "$name ($it of $iters iterations): $(tail -1 "$OUT/$name.verify.txt")"
}
mkdir -p "$OUT"
run lam1          1000  --coarse_subdiv_relax_no_new_folds --explicit_lambda 1
run explicit      20000
run explicit_dir  20000 --explicit_directed_graph
run nofold_dir    5000  --coarse_subdiv_relax_no_new_folds --explicit_directed_graph
run cotan_nofold  5000  --coarse_subdiv_relax_no_new_folds --coarse_subdiv_relax_weights cotan
run mv_nofold     5000  --coarse_subdiv_relax_no_new_folds --coarse_subdiv_relax_weights meanvalue
run lam01         10000 --coarse_subdiv_relax_no_new_folds --explicit_lambda 0.1
run lam025        4000  --coarse_subdiv_relax_no_new_folds --explicit_lambda 0.25
