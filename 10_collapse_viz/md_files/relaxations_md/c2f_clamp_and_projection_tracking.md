# Clamp tracking in the coarse -> fine query and in the relaxation projection

2026-10-01. Test mesh ABC 00040057, target 200, qslim, `--n_subdiv_samples 200000`
(207,000 subdivided vertices, 409,600 triangles). Run folder:
`output/relaxation_experiments/clamp_check/` (explicit + no new folds, lambda 0.5,
1,000 iterations; see its `experiment_config.txt`).

## Question

Both steps can snap a point:

- **Coarse -> fine query** (`src/query_coarse_to_fine.cpp`): at every undone
  collapse the query point is mapped into the pre-collapse UV chart. If it lies
  outside every pre-collapse triangle, the least-outside triangle is taken, its
  negative barycentric coordinates are set to 0 and the rest renormalized
  ("clamp and renormalize"). This moves the point onto the triangle's border,
  along the line from the clamped corner (not to the nearest border point).
- **Relaxation projection** (`Projector`, `subdiv_relax_projector.h`): every step
  is projected back onto the vertex's sheet or curve; the closest point can be on
  a triangle's edge or vertex.

Do these snaps happen often, and do they cause the seed's folds?

## What is tracked

| where | what | output |
|---|---|---|
| `query_coarse_to_fine` (optional `C2FQueryStats*`, default off, so other projects are unchanged) | per query: walk steps, steps that had to clamp (most negative coordinate < -1e-12), steps outside every triangle by >= 1, largest / summed negative coordinate clamped away, largest snap distance / longest edge of the chosen UV triangle | log line `[coarse_subdiv] c2f clamp: ...`; per subdivided vertex `coarse_subdiv_c2f_clamp_<stem>.csv` (vid, mapped, steps, clamped_steps, far_steps, max_neg_bary, sum_neg_bary, max_snap_rel) |
| `Projector` (all solvers) | projections, results on an edge / vertex of their triangle (or an end of their edge), distance from the stepped point to the projected one (mean, max), barycentric fix-ups beyond rounding | explicit: every 100 iterations and for the whole run; Newton: for the whole run |
| `test_scripts/c2f_clamp_vs_folds.py <run>` | fold rate of the seed's triangles that touch a clamped vertex vs not, and by clamp strength | console |

## Results

### Coarse -> fine query: the clamp practically never happens

| quantity | value |
|---|---|
| queries (subdivided vertices) | 207,000 |
| walk steps (collapses undone) | 1,539,929 (7.4 per query, max 34) |
| steps that had to clamp | **1** (0.00%) |
| its largest negative coordinate / snap | 1.1e-12 / 1.7e-13 of a triangle edge (rounding) |
| steps far outside (>= 1) | 0 |

At every collapse the query point lands inside, or exactly on the border of, a
pre-collapse triangle; the clamp only removes rounding.

| seed triangles | count | folded |
|---|---|---|
| all | 409,600 | 1.26% (5,168) |
| never touching a clamped vertex | 409,580 | 1.26% |
| touching the one clamped vertex | 20 | 15.0% (3) |

**The clamp does not cause the seed's folds.** The noise of the coarse -> fine
mapping comes from the charts themselves: the pre- and post-collapse UV layouts
of a collapse do not match exactly, so their composition distorts (and in
places folds) the map, while every point still lands inside a triangle.

### Relaxation projection: gentle

Explicit relaxation + no new folds, 1,000 iterations:

| iterations | on an edge / vertex | off-surface distance mean | max (cumulative) |
|---|---|---|---|
| 1-100 | 0.12% | 1.1e-5 x diag | 0.0021 x diag |
| 401-500 | 0.22% | 1.1e-5 x diag | 0.0034 x diag |
| 901-1000 | 0.25% | 1.0e-5 x diag | 0.0036 x diag |
| all (207M projections) | 0.21% | 1.1e-5 x diag | 0.0036 x diag |

Barycentric fix-ups beyond rounding: 0; all consistency checks 0. The stepped
points stay very close to the surface and almost always project into the
interior of a triangle (or edge).

## Conclusion

Neither snap is a meaningful source of distortion. The seed's folds come from
the composed collapse charts, and the relaxation's folds (without the
no-new-folds rule) from the uniform Laplacian's count-evening resting state
(`coarse_subdiv_relax_experiments.md`).
