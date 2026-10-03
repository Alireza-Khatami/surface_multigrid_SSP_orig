# Python port of the explicit relaxation: verification and experiments

Date: 2026-10-03. Mesh: ABC 00040057 (MAT, 1506 vertices / 2715 faces; simplified to 261 / 400,
subdivided to 207,000 vertices / 409,600 faces).

Code: `10_collapse_viz/relaxation_scripts_python/` (see its `README.md`). It ports
`coarse_subdiv_relax_explicit.cpp` and all of its dependencies to Python + numba. It starts
from a C++ run's bundle (`correspondence_*.c2f`) and the fine `.ma_struct`, with the same
flags as `collapse_viz_bin`.

## Verification against the C++

Each check runs the port on the C++ run's bundle with that run's flags. It compares the
OBJs byte for byte (subdivided mesh, c2f seeds, `_it<N>` snapshots, final result when run
to the same length) and every logged value. Reports: `py_verify/`.

| C++ run | flags | iterations compared | result |
|---|---|---|---|
| `clamp_check` | no new folds | all 1000 | identical |
| `lam1` | no new folds, lambda 1 | all 1000 | identical |
| `explicit` | uniform weights | first 1000 of 20000 | identical |
| `explicit_dir` | directed graph | first 1000 of 20000 | identical |
| `nofold_dir` | directed graph, no new folds | first 1000 of 5000 | identical |
| `cotan_nofold` | cotan weights, no new folds | first 1000 of 5000 | identical |
| `mv_nofold` | mean-value weights, no new folds | first 1000 of 5000 | identical |
| `lam01` | no new folds, lambda 0.1 | first 1000 of 10000 | identical |
| `lam025` | no new folds, lambda 0.25 | first 1000 of 4000 | identical |
| `cpp_ref_global_nofold` (new C++ run) | global projection, no new folds | all 300 | identical |
| `cpp_ref_perface_dir_cotan` (new C++ run) | per face, directed graph, cotan, lambda 0.3 | all 300 | identical |

For runs compared on a prefix, the snapshots and log lines inside the compared iterations
were checked, not the final OBJ.

Other checks:

- **Projector:** 62 trees, 0 structure errors, 0 of 18,600 BVH queries differ from brute
  force (`test_projector.py`).
- **`11_correspond_viz/c2f_query.py`:** `load_bundle` returns the same data as a strict
  reader that follows the C++ writer and consumes every byte. Its walk gives the same face,
  corners and position as the C++ walk on all 207,000 queries, with both ways of starting a
  query (`py_verify/c2f_query_check.txt`). Its code still differs from the C++ in three places
  that never trigger on this mesh:
  1. It picks the collapse sheet by searching all sheets for the face; the C++ matches
     `faceSheetID`.
  2. When the point is far outside every triangle, it takes the least-outside triangle; the
     C++ takes the first one.
  3. It skips normalising the barycentrics when their sum is <= 1e-12; the C++ always divides.

  The port follows the C++.

## Experiments

All runs use the no-new-folds rule, 2000 iterations and lambda 0.5 unless stated, and start
from the `clamp_check` bundle. Seed values for all runs: 5168 folded triangles, edge length
CV 1.0839, min-angle p1 / p5 / median 0.645 / 1.397 / 14.660 deg.

| folder | scheme | converged | folded after | moves held back | edge CV after | min angle p1 / p5 / median (deg) | degenerate | flipped vs seed | move max / mean (x diag) |
|---|---|---|---|---|---|---|---|---|---|
| `py_nofold_2k` | uniform, symmetric graph, local projection | no | 2371 | 26,085,155 | 1.0214 | 0.090 / 1.511 / 17.714 | 30 | 2456 | 0.138 / 0.0111 |
| `py_nofold_global_2k` | as above, global projection | no | 2371 | 26,085,155 | 1.0214 | 0.090 / 1.511 / 17.714 | 30 | 2456 | 0.138 / 0.0111 |
| `py_nofold_perface_2k` | uniform, coarse vertices / edges held | yes (iter 1384) | 3504 | 15,274,258 | 1.0834 | 0.348 / 1.133 / 15.162 | 40 | 1330 | 0.0195 / 0.00077 |
| `py_cotan_nofold_dir_2k` | cotan weights, directed graph | no | 2780 | 10,541,967 | 1.0178 | 0.483 / 1.457 / 13.862 | 85 | 1930 | 0.0989 / 0.0104 |
| `py_mv_nofold_dir_2k` | mean-value weights, directed graph | no | 2666 | 42,830,803 | 1.7237 | 0.000 / 0.066 / 9.822 | 2553 | 2083 | 0.349 / 0.0152 |
| `py_mv_nofold_lam1_2k` | mean-value weights, symmetric graph, lambda 1 | not run (stopped, low memory) | | | | | | | |

Same table as CSV: `py_experiments_results.csv`.

### Observations

- **Local vs global projection:** identical results, down to the number of held-back moves.
  Local projection never needed more than the first ring of triangles, so both pick the same
  point.
- **Per face:** the only run that converged. It moves vertices least, but leaves the most
  folded triangles (3504) and barely improves edge spacing.
- **Mean-value weights on the directed graph:** clearly the worst. Edge CV goes 1.08 -> 1.72,
  2553 triangles become degenerate, and one vertex moves 0.35 x diagonal. The C++ `mv_nofold`
  run used the symmetric graph and did not show this, so the bad combination looks like
  mean-value weights with the directed graph.
- **Cotan on the directed graph:** best p1 minimum angle of these runs (0.483 deg), but a lower
  median angle and more folded triangles than the uniform baseline.

To finish the last run: `python run_experiments.py --only py_mv_nofold_lam1_2k`
(about 5 minutes; it also rewrites the CSV / table).
