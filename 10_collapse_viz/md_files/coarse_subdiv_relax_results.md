# Relaxing the subdivided coarse mesh on the fine MAT

Status (2026-09-28): implemented and verified; **the relaxed mesh folds**.

**Update 2026-09-30:** see `coarse_subdiv_relax_experiments.md`. The "flipped vs seed" column below is misleading (a seed fold the relaxation fixes counts as a flip); the experiments doc measures absolute folds against each coarse face's majority orientation, and adds the explicit (small-step, all-at-once) relaxation, which beats Newton and improves on the seed in its first ~10-100 steps.
Code: `coarse_subdiv_relax.{h,cpp}` (commit `83b95db`). Next step undecided,
see "Options".

## Goal

After decimation the simplified mesh is subdivided (`build_subdiv_mesh`, the
same midpoint 1->4 subdivision as the fine tracker) and every subdivided vertex
is mapped to the fine MAT with `query_coarse_to_fine`
(`coarse_subdiv_at_fine_pos_*.obj`). Those vertices lie exactly on the fine
MAT but are unevenly spaced, because the coarse -> fine map stretches some
regions and compresses others.

The goal is to even out the spacing with the existing structure-aware
relaxation, with every vertex projected back onto its own structure **of the
fine mesh**: sheet vertices onto their sheet's faces, seam/boundary vertices
onto their curve's edges, junctions fixed. That needs structure IDs for every
subdivided vertex.

## Pipeline

1. **Coarse structure IDs** (`coarse_matstruct`), from the simp tracker
   (`simp_viz_tracker`, the source of `*_simp_visualize_info.json`):
   - vertex: its tracked struct IDs. With the struct-ID gate
     (`--mat_struct_check`) a collapse only merges vertices with identical ID
     sets, so these equal the fine vertex's own IDs.
   - face: `ms.faceIds[gF row]`; a coarse face keeps the row, and so the
     sheet, of the fine face it started as.
   - edge: a coarse edge `(a, b)` gets seam/boundary id `k` when a fine curve
     edge of `k` joins an **ancestor** of `a` to an ancestor of `b` (ancestors =
     fine vertices merged into the coarse vertex, tracked by the simp tracker).
     Collapsing a seam splits its fine vertices among the coarse vertices along
     it, so exactly the coarse seam edges get the ID.
2. `build_struct_sets` (unchanged) spreads them to the subdivided vertices by
   the usual carrier rules (vertex / edge / face of the coarse mesh).
3. A copy of the subdivided mesh located on the fine mesh: positions and
   `fineFace`/`fineBary` from the query, `origEdges` = edges of `gFO`.
4. **Seed snap**: each vertex projected once onto its own fine structure with
   the shared projector (`subdiv_relax_projector.h`); junctions snapped to the
   nearest fine vertex of their junction. The relaxation keeps any seed that is
   not on its structure fixed, so this is required.
5. Existing graph (`build_relax_graph`) and solver (`subdiv_relax`, Newton, or
   `subdiv_relax_solve_project`).
6. Export `coarse_subdiv_at_fine_pos_relaxed_<method>_<stem>.obj` (same
   vertices and faces as `coarse_subdiv_*.obj`).

Flags: on by default; `--no_coarse_subdiv_relax`,
`--coarse_subdiv_relax_method newton|solve_project` (default `newton`),
`--coarse_subdiv_relax_max_iter N`. Needs `--matstruct_path` and
`--mat_struct_check`, otherwise skipped with a message.

## Refactor

`Projector`, `PrimBVH` and the palette / closest-point helpers moved from
`subdiv_relax_solve_project.cpp` to the header-only
`subdiv_sample_tracker/subdiv_relax_projector.h` (namespace `subdiv_proj`,
everything inline), included by both users. The fine-subdivision relaxation
outputs (`.sdt`, `.slg`, all `subdiv_*` OBJ/PLY) are byte-identical before and
after the move (`--n_subdiv_samples 50000 --subdiv_relax`).

## Verification (ABC 00040057, target 200, qslim, `--n_subdiv_samples 200000`)

Coarse mesh 261 vertices / 400 faces, subdivided 5 levels: 207,000 vertices,
409,600 faces.

| check | result |
|---|---|
| coarse vertex IDs vs the fine vertex's own IDs | 261 of 261 equal |
| coarse curve edges | 254, from 1261 fine curve edges (1007 collapsed inside one coarse vertex) |
| fine curve edges without an owner / not on a coarse edge | 0 / 0 |
| owner conflicts; edge IDs not on both endpoints | 0; 0 |
| roles | 198,878 sheet, 8,116 curve, 6 junction |
| seed snap, max move (x diag) | sheet 2.6e-16, curve 2.4e-14, junction 0 |
| relaxation checks (seed off structure, fixed moved, pos != interp, bad bary, off own structure) | all 0 |
| Newton convergence | one more step moves 2.4e-14 x diag; residual ~1e-11; 156 s |

## Result: the relaxed mesh folds

| method | edge CV | median angle | degenerate | flipped vs seed | max move (x diag) |
|---|---|---|---|---|---|
| seed (no relaxation) | 1.084 | 14.7 | 0 | 0 | - |
| `solve_project` | 1.020 | 14.7 | 71,431 | 46,732 | 0.122 |
| `newton`, converged | 0.837 | 10.8 | 10,599 | 15,586 | 0.155 |

For comparison, the same code on the fine subdivision (well-shaped seed, close
to rest) gives 2 degenerate / 5 flipped (`solve_project`) and 1 / 0 (`newton`).

Stopping Newton early does not help; the first iteration is a full linear
solve and folds the most:

| Newton iterations per class | edge CV | median angle | degenerate | flipped |
|---|---|---|---|---|
| 1 | 0.944 | 14.7 | 75,624 | 48,949 |
| 2 | 0.901 | 14.1 | 50,487 | 41,895 |
| 5 | 0.858 | 12.6 | 20,609 | 20,776 |
| 20 | 0.839 | 11.0 | 9,777 | 14,716 |
| until converged | 0.837 | 10.8 | 10,599 | 15,586 |

## Why

The relaxation weights every neighbour equally (1/degree), so its resting
state evens out vertex **counts**. The subdivided coarse mesh gives every
coarse face exactly 1024 triangles, while coarse face areas differ by about
2000x (24.6 to 48,210). So vertices move from small coarse faces into large
ones' territory:

| coarse face area / mean of its neighbourhood | faces | flips per coarse face | median area growth of its region |
|---|---|---|---|
| < 0.5 | 186 | 34 | x1.22 (0-0.5) |
| 0.5 - 1 | 135 | 37 | x1.18 |
| 1 - 2 | 57 | 57 | x0.74 |
| > 2 | 22 | 51 | x0.56 |

Correlation of log(area ratio) with log(region growth): -0.43. Large faces are
compressed and fold the most. Flips are spread over 283 of 400 coarse faces
and grow with the valence of the coarse corners (up to 20): 41 flips per face
when a corner has valence >= 10, 26 for valence 7-9. They do not concentrate on
skinny coarse faces.

The fine subdivision does not show this: its connectivity comes from the
well-shaped fine mesh and its seed is already close to the resting state.

## Options

1. **Per-coarse-face relaxation**: fix the vertices on coarse vertices and
   coarse edges, relax only each coarse triangle's interior. No vertex count
   can move between coarse faces, so the compression of large faces goes away;
   uses the existing code with more fixed vertices. Spacing is evened out only
   inside each coarse face. (Recommended first.)
2. **Geometric weights** (e.g. cotangent weights from the seed) instead of
   uniform ones. Targets the cause, but changes the shared relaxation code (make
   it optional to keep the fine relaxation as is).
3. **Flip-preventing line search**: reject steps that flip a triangle. No folds
   by construction, but where the resting state itself is folded it stops early.

## Outputs of these runs

`output/relaxation_experiments/`:
`coarse_subdiv_relax/` (solve_project), `coarse_subdiv_relax_newton/`,
`coarse_subdiv_relax_newton_it{1,2,5,20}/`, and
`projector_extract_{before,after}/` for the refactor check.
