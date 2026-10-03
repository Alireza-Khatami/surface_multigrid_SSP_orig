# relaxation_scripts_python

Python port of the explicit relaxation of the subdivided coarse mesh on the fine MAT
(`coarse_subdiv_relax_explicit.cpp`) and of everything it depends on, starting from a
C++ run's bundle (`correspondence_<stem>.c2f`) and the fine `.ma_struct`.

The arithmetic is the C++ one, operation by operation (same evaluation order, no
FMA, sequential sums where the C++ sums sequentially). On ABC 00040057, the outputs
are byte-identical to the C++ ones (see "Verification").

## Modules (C++ source each one ports)

| module | ports |
|---|---|
| `bundle_io.py` | bundle reader (layout of `coarse_fine_save_bundle`, `coarse_fine_viz.cpp`); checks every byte is read |
| `matstruct.py` | `load_matstruct.h/.cpp` |
| `subdiv_mesh.py` | `subdiv_sample_tracker/subdiv_mesh.h/.cpp` (midpoint subdivision, carriers) |
| `c2f_walk.py` | `src/query_coarse_to_fine.cpp`, `src/compute_barycentric.cpp`, `coarse_subdiv_c2f.cpp` |
| `struct_ids.py` | `coarse_matstruct` (`coarse_subdiv_relax.cpp`), `subdiv_struct_ids.h/.cpp`, palette helpers of `subdiv_relax_projector.h` |
| `relax_graph.py` | `build_relax_graph` (`subdiv_sample_tracker/subdiv_relax.cpp`) |
| `projector.py` | `subdiv_relax_projector.h` (closest points, BVH, `project`, `project_local`) |
| `weights.py` | `cotan_weights`, `meanvalue_weights` (`coarse_subdiv_relax.cpp`) |
| `relax_explicit.py` | `coarse_subdiv_relax_explicit.h/.cpp` |
| `quality.py` | `subdiv_mesh_quality` (`subdiv_relax.cpp`) |
| `coarse_subdiv_relax.py` | `coarse_subdiv_relax_export`, explicit method (`coarse_subdiv_relax.h/.cpp`) |
| `obj_io.py`, `log_util.py` | OBJ in `igl::writeOBJ`'s format; one log sink |

The per-vertex kernels are compiled with numba (`pip install numba`); about 2-3x slower
than the C++ build (about 0.13 s per iteration on 207k vertices).

Two things come from the bundle instead of the tracker's memory: the coarse vertices'
ancestors (each collapse's survivor / absorbed pair, `b`, folded in collapse order, as
`simp_viz_tracker_on_collapse` does) and their struct IDs (the union over ancestors;
checked against the fine vertex's IDs, as the C++ does).

Not ported: the Newton and solve_project methods, the Laplacian-graph PLY export and
the decimation itself (the bundle is its output).

## Running

```
python run_relax.py --bundle ../output/relaxation_experiments/clamp_check/correspondence_<stem>.c2f \
    --matstruct_path <...>.ma_struct --output_dir ../output/relaxation_experiments/<name> \
    --coarse_subdiv_relax_method explicit --coarse_subdiv_relax_no_new_folds --explicit_max_iter 2000
```

Flags are those of `collapse_viz_bin` (`main.cpp`): `--coarse_subdiv_relax_method explicit`,
`--coarse_subdiv_relax_per_face`, `--coarse_subdiv_relax_no_new_folds`, `--explicit_lambda`,
`--explicit_max_iter`, `--explicit_tol`, `--explicit_global_proj`, `--explicit_directed_graph`,
`--coarse_subdiv_relax_weights uniform|cotan|meanvalue`, `--n_coarse_subdiv_samples`,
`--subdiv_obj_max_verts` (and the Newton-only flags, which only change the file name or
graph as in C++). Python-only: `--log_every`, `--snapshot_iters`, `--simplified_obj`,
`--no_subdiv_objs`, `--description`, `--stem`.

Outputs, named as in C++: `coarse_subdiv_<stem>.obj`, `coarse_subdiv_at_fine_pos_<stem>.obj`,
`coarse_subdiv_c2f_clamp_<stem>.csv`, `coarse_subdiv_at_fine_pos_relaxed_<flags>_<stem>.obj`
and its `_it<N>.obj` snapshots, plus `run.log`, `experiment_config.txt`, `results.json`.

`run_experiments.py` runs a list of schemes and writes
`output/relaxation_experiments/py_experiments_results.{csv,md}`.

## Verification

- `verify_relax.py --cpp_dir <C++ run> --out_dir <dir> -- <flags>`: runs the port on that
  run's bundle and compares the OBJs byte for byte and the log lines value for value.
  `verify_all.sh <out_root> [N]` does it for the earlier C++ runs (first N iterations).
- `verify_c2f.py --cpp_dir <C++ run>`: `11_correspond_viz/c2f_query.py` (loader and walk)
  against the C++-exact walk.
- `test_projector.py`: BVH structure and BVH vs brute-force closest points.
- `cpp_reference_run.sh`: a headless C++ Release run, for flags no earlier run used.
