# Test scripts: coarse-subdivision relaxation

Analysis scripts for the relaxation of the subdivided coarse mesh on the fine
MAT (`coarse_subdiv_relax.cpp`, results in `md_files/coarse_subdiv_relax_results.md`).
Run them from this folder; a `<run_dir>` is one run's `--output_dir`, e.g.
`../output/relaxation_experiments/base_newton`.

| script | what it reports |
|---|---|
| `run_coarse_relax.sh` | runs one experiment on the test mesh (from `10_collapse_viz`): `bash test_scripts/run_coarse_relax.sh <name> [flags]` |
| `fold_table.py` | seed vs relaxed: folded triangles, seed folds removed / kept, new folds, degenerate, edge CV (whole mesh, and median within each coarse face) |
| `seam_distance_folds.py` | folded / degenerate rates by ring distance from the nearest seam/boundary vertex |
| `orientation_vs_coarse.py` | why "flipped vs seed" and "flipped vs coarse normal" mislead (whole coarse faces oriented against the fine sheet) |
| `snapshot_table.py` | explicit runs: folded / removed / new / degenerate / edge CV at every `_it<N>` snapshot |
| `step_size_table.py` | explicit runs with different lambda: folded / degenerate / energy at equal lambda x iterations, from the logs |
| `c2f_clamp_vs_folds.py` | how often the coarse -> fine walk clamped (snapped a point onto a triangle border) and whether the seed folds where it did (needs `coarse_subdiv_c2f_clamp_*.csv`) |
| `check_graph_ply.py` | validates the `laplacian_graph/*.ply` files of a run |
| `relax_metrics_common.py` | shared loaders and metrics |

**Folded** (all scripts): a triangle whose orientation disagrees with the
majority of the triangles of its coarse face. The coarse face of each subdivided
face is cached in `<run_dir>/_coarse_face_of.npy`.

Examples:

```
python fold_table.py ../output/relaxation_experiments/base_newton ../output/relaxation_experiments/local_newton
python seam_distance_folds.py ../output/relaxation_experiments/base_newton
python check_graph_ply.py ../output/relaxation_experiments/base_newton
```
