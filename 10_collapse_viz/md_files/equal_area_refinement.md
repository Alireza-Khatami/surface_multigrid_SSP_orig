# Equal-area refinement of the coarse mesh before subdivision

Code: `relaxation_scripts_python/equal_area_refine.py` (Python only, no C++ counterpart).
Flags: `--equal_area_levels K` or `--equal_area_target A` on `run_relax.py` / `relax_viewer.py`.

## Why

Uniform subdivision puts the same number of samples in every coarse face. On ABC 00040057
the coarse face areas differ by a factor of about 2000 (median 799, p90/p10 = 21, CV 2.06),
so the sample density differs by the same factor. The relaxation has to move samples across
coarse edges to even this out. That is a smooth, large-scale correction, which explicit
(Jacobi) steps make very slowly: after 1000 iterations of the baseline the per-coarse-face
area spread is still max/min = 961.

## What it does

- Before the uniform subdivision, the largest coarse face is split first, until every face is
  below a target area A*.
- The split is longest-edge bisection (Rivara): a face is split at the midpoint of its longest
  edge, and every face on that edge is split at the same point. If that edge is not the
  longest edge of a neighbour, the neighbour's own longest edge is split first. The mesh stays
  conforming, also on non-manifold edges, and the angles do not get worse.
- With `--equal_area_levels K`, A* is searched so that the refined mesh has about |F| * 4^K
  faces. The uniform subdivision then needs K levels fewer, so the sample count stays the same
  as without refinement (205k vs 205k vertices).
- Every refined vertex keeps an exact carrier on the original coarse mesh, so the rest of the
  pipeline is unchanged: c2f walk, struct ids, roles, projection, per-face holds and fold
  reference.
- Faces are only split, never merged, so faces already smaller than A* keep their size.

Refined coarse faces (areas on the coarse geometry):

| levels | faces | max/min | p90/p10 | CV |
|---|---|---|---|---|
| off | 400 | 1962 | 21.1 | 2.06 |
| 1 | 1602 | 41 | 5.6 | 0.52 |
| 2 | 6401 | 60 | 3.9 | 0.41 |

## Checks

- Refinement off: `verify_relax.py` on clamp_check (1000 iterations) still gives byte-identical
  OBJs and matching logs against the C++.
- Refinement on, checked in the code itself:
  - the refined faces cover each coarse face's area;
  - each refined edge has the right number of faces (no T-junctions);
  - the faces around an edge agree on its midpoint carrier.
- c2f walk on the refined samples: all 205,688 mapped; 2 clamped steps of about 1e-12.
- `test_viewer.py` passes.

## Results (2000 iterations, no new folds, lambda 0.5, local projection)

Table: `output/relaxation_experiments/py_experiments_results_equal_area.md`.
The edge CV and angle columns show the value at the start -> the value after relaxation.

| run | edge CV | min angle p1 | p5 | median | folded (seed -> result) | held back | flipped vs seed | degenerate |
|---|---|---|---|---|---|---|---|---|
| uniform, off | 1.084 -> 1.021 | 0.65 -> 0.09 | 1.40 -> 1.51 | 14.7 -> 17.7 | 5168 -> 2371 | 26.1M | 2456 | 30 |
| uniform, levels 1 | 0.615 -> 0.587 | 1.01 -> 0.53 | 2.57 -> 4.09 | 16.2 -> 23.8 | 4418 -> 2446 | 12.2M | 1628 | 28 |
| uniform, levels 2 | 0.565 -> 0.530 | 1.35 -> 1.34 | 3.40 -> 8.18 | 21.5 -> 32.9 | 3514 -> 1986 | 7.0M | 1107 | 28 |
| cotan + directed, off | 1.084 -> 1.018 | 0.65 -> 0.48 | 1.40 -> 1.46 | 14.7 -> 13.9 | 5168 -> 2780 | 10.5M | 1930 | 85 |
| cotan + directed, levels 2 | 0.565 -> 0.633 | 1.35 -> 0.62 | 3.40 -> 2.66 | 21.5 -> 22.9 | 3514 -> 2152 | 7.5M | 929 | 1213 |

- With uniform weights the refinement helps on every measure. At levels 2:
  - the start is already more even than the baseline's end (edge CV 0.57 vs 1.02);
  - after relaxation, the median min angle is 33 vs 18 degrees, p5 is 8.2 vs 1.5, and p1 is
    1.34 vs 0.09;
  - fewer triangles are folded (1986 vs 2371);
  - 3.7x fewer moves are held back;
  - fewer triangles are flipped compared to the seed (1107 vs 2456).
- Levels 2 is better than levels 1.
- Cotan + directed gets worse with the refinement: 1213 degenerate triangles, and the edge CV
  goes up. One possible cause, not tested: the bisected faces have large angles, whose
  cotangents are small or negative. Use uniform weights with this mode.
- No run converges in 2000 iterations.
