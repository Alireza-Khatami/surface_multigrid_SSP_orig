# Subdivided-sample relaxation: results (2026-09-27)

Plan and implementation notes: `md_files/subdiv_relax_plan.md`.

Relaxation is implemented and on by default. At every level tested (0–5), the
result passes the definition: one more relaxation step
`x <- Pi(x + 0.5 L x)` moves no vertex by more than about 3e-16 x the bbox
diagonal. The tracker then follows the relaxed samples through the whole
decimation with every error counter at 0.

## Results on ABC 00040057 (Release, qslim to 200 faces)

| | level 4 (350k samples) | level 5 (1.4M samples) |
|---|---|---|
| relaxation time | 68 s | 561 s |
| largest move from one more step | 2.9e-16 x diag | 3.5e-16 x diag |
| vertex off its own sheet or seam, or junction moved | 0 | 0 |
| tracker counters over 1168 collapses | all 0 | all 0 |
| `verify_sdt.py` / `verify_slg.py` (independent) | pass | pass |

- **Regression.** With `--no_subdiv_relax`, the `.sdt` is byte-identical to the
  pre-relaxation code. With relaxation on, `subdiv_fine_seed_<stem>.obj` is
  byte-identical to the old `subdiv_fine_<stem>.obj`.
- **Graph.** It matches an independent rebuild from the `.ma_struct` exactly:
  2,019,517 entries at level 4 and 8,209,197 at level 5. Only the five allowed
  kinds occur: S<->S, C<->C, S<-C, S<-J, C<-J. Junction rows are empty.

## Things to know

- **The end state is not unique.** Seam polylines bend slightly at every
  original vertex, and sheets have creases, so relaxation can stop in several
  places.
  - Plain relaxation with step 0.5 and with step 0.25 already ends 0.008 x diag
    apart.
  - The solver used here reaches a third stopping point, with lower Dirichlet
    energy than either.
  - All three are true fixed points. So the tests check the stopping
    definition, not agreement between methods.
- **One vertex is pinned per closed seam loop.** 26 of the 38 seam pieces on
  this mesh are closed loops with no junction; there are only 6 junction
  vertices. Such a loop can slide around itself, so its lowest-index vertex
  (an original MAT vertex) is held fixed.
- **Quality improves only in the middle.** Level 5:
  - median per-triangle min angle 40.7 -> 41.9 deg;
  - worst 1% 16.1 -> 11.6 deg;
  - 13 degenerate and 30 flipped triangles, where sheet vertices stop on a
    concave rim next to a seam vertex;
  - edge-length spread barely changes (CV 0.368 -> 0.366), because samples
    cannot move between sheets and each original face keeps its sample count.
- **Levels 6 and above are too slow for now.** Cost grows about 8x per level,
  so 5.6M / 22M samples would need a faster sheet solve.
- **The Debug launch entry now relaxes in Debug.** "Debug – collapse_viz
  subdiv tracker (4 levels)" will be much slower. Add `--no_subdiv_relax` to it
  if you only want the viewer.

## Solver

Plain repeated relaxation took 37,596 steps at level 1, so it cannot reach
level 4. The solver used instead:

- **Newton step.** Each step solves the tangent-plane system of the Dirichlet
  energy directly (sparse LDLT), curves first, then sheets.
- **Energy line search.** Every accepted step must lower the energy, so the
  iteration cannot cycle.
- **Active-set frames.** A vertex sitting exactly on a polyline bend or a
  crease moves along the edge or face its pull actually points into.
- **Stopping test.** A pass ends only when one plain relaxation step moves
  nothing.

## Outputs

- **Flag.** `--no_subdiv_relax` turns relaxation off. There is no step-size or
  iteration-count flag.
- **Files.** A relaxed run writes:
  - `subdiv_fine_<stem>.obj` (after relaxation);
  - `subdiv_fine_seed_<stem>.obj` (before relaxation);
  - `subdiv_graph_<stem>.slg`.
- **`.sdt`.** Version 2, with a 15th array `sub_V_seed` (positions before
  relaxation). The header's `relaxed` field replaces the old `reserved` field.
- **`subdiv_graph_viewer`.** A standalone program that only parses the `.slg`.
  - Points are colored by role.
  - Each edge kind is its own toggleable set.
  - One-way edges are drawn as half segments from the pulled vertex toward the
    vertex that pulls it.
  - A vertex query shows who pulls a vertex and whom it pulls.
  - Launch entry: "Release – subdiv graph viewer".

## Files

- `subdiv_sample_tracker/subdiv_relax.{h,cpp}`: graph, per-structure
  projection (one BVH per structure id, exact closest point), solver, checks.
- `subdiv_sample_tracker/subdiv_graph_viewer.cpp`: viewer.
- `subdiv_sample_tracker/tests/subdiv_relax_test.cpp`: standalone checks
  (CMake option `SUBDIV_TESTS`).
- Python verifiers (session scratchpad, not in the repo): `verify_sdt.py`
  (handles v2) and `verify_slg.py`.
