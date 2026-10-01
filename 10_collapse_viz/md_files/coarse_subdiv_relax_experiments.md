# Coarse-subdivision relaxation: experiments of 2026-09-30

Follow-up to `coarse_subdiv_relax_results.md` (the relaxed subdivided coarse mesh
folds). Test mesh ABC 00040057, target 200, qslim, `--n_subdiv_samples 200000`:
coarse mesh 261 vertices / 400 faces, subdivided to 207,000 vertices / 409,600
faces (1024 per coarse face). Runs in `output/relaxation_experiments/<name>/`,
analysis scripts in `test_scripts/` (see its README).

## Summary: best method so far

**Best: the explicit relaxation (small Laplacian steps on everything at once)
with the no-new-folds rule, step lambda = 0.5**
(`--coarse_subdiv_relax_method explicit --coarse_subdiv_relax_no_new_folds`).

- **Folds:** 2,383, 54% fewer than the seed's 5,168. It removes 2,799 seed folds
  and creates only 14 new ones.
- **Degenerate triangles:** 33 (plain explicit 4,764, Newton 10,606).
- **Smallest angles:** median 14.7 -> 17.5 deg.
- **Spacing:** only slightly more even than the seed (edge CV 1.084 -> 1.014;
  Newton reaches 0.837). Blocking folds also blocks most of the squeezing that
  evens out the spacing.
- **Step size:** lambda = 0.5, 0.25 and 0.1 give the same result at equal
  lambda x iterations (the path does not depend on the step); smaller steps only
  cost time. lambda = 1 is worse (it overshoots).

It is the first method that is better than the seed at convergence; what is
still missing is the even spacing.

| method | folded | degenerate | edge CV |
|---|---|---|---|
| seed (no relaxation) | 5,168 | 0 | 1.084 |
| **explicit + no new folds, lambda 0.5 (5,000 it.)** | **2,383** | **33** | 1.014 |
| explicit + no new folds, lambda 0.25 (4,000 it.) | 2,386 | 33 | 1.018 |
| explicit + no new folds, lambda 0.1 (10,000 it.) | 2,379 | 32 | 1.017 |
| explicit + no new folds, lambda 1.0 (1,000 it.) | 2,568 | 37 | 1.027 |
| explicit + no new folds, cotan weights (5,000 it.) | 2,568 | 34 | 1.135 |
| explicit, iteration 10 | 3,722 | 47 | 1.062 |
| explicit, iteration 100 | 4,357 | 655 | 1.020 |
| explicit, final (20,000) | 11,347 | 4,764 | 0.867 |
| Newton, 20 iterations per class | 14,663 | 9,782 | 0.839 |
| Newton | 15,381 | 10,606 | **0.837** |
| Newton, local projection | 15,353 | 10,636 | 0.837 |
| Newton, per coarse face | 18,512 | 1,105 | 1.080 |
| solve_project, per coarse face | 22,104 | 6,030 | 1.085 |
| solve_project, two passes | 55,738 | 71,431 | 1.020 |
| solve_project, joint solve | 64,985 | 85,090 | 1.021 |

### What we learned

1. **"Flipped vs seed" was the wrong measure.** The seed already has 5,168
   folds from the noisy coarse -> fine mapping, and removing them is the goal.
   Every method removes 55-80% of them, but without the no-new-folds rule they
   create many more new ones.
2. **Folds come from the uniform Laplacian's resting state, not the
   projection.** At rest, the Laplacian evens out vertex *counts*. Every coarse
   face has 1,024 triangles while their areas differ ~2,000x, so large faces get
   squeezed. In the explicit run, the early steps remove noise and the late
   steps create folds.
3. **The no-new-folds rule keeps the good part and blocks the bad part:** the
   unfolding continues (2,799 seed folds removed), the squeezing that folds is
   stopped (14 new). The price is that the spacing barely evens out.
4. **The projection is correct.** Its BVHs (over the fine MAT) match brute force
   exactly; the local projection gave identical quality; projecting onto a
   level-4 subdivided fine mesh would give the same points.
5. **Damage is densest next to seams.** Relaxing everything at once in small
   steps reduces it everywhere; with the no-new-folds rule the seams fold half as
   often as in the seed. Solving everything at once in one linear system makes it
   worse at the seams, because the sheets then lean on the seam positions before
   projection.
6. **One-shot linear solves don't suit this case.** solve_project places sheet
   vertices off the curved MAT and then piles them up when projecting back.
7. **Per-coarse-face relaxation** cuts degenerate triangles but doesn't reduce
   folds, and it barely evens out the spacing.
8. **Step size:** below 0.5 nothing changes but the run time; 1.0 overshoots.
9. **No relaxation makes the spacing inside each coarse face more even.** The
   median edge CV within a coarse face rises from 0.384 (seed) to ~0.5 for every
   method; the whole-mesh edge CV improves only because spacing *between* coarse
   faces is equalized (vertices migrate from small into large coarse faces).
10. **Cotangent weights (from the coarse positions) did not help:** 24% of them
    are negative (obtuse coarse triangles) and are clamped to 0, which loses the
    linear reproduction they were chosen for.

### Recommended next step

Decide which spacing is wanted: even across the whole mesh (what uniform
weights optimize, at the cost of the regular layout inside each coarse face), or
even inside each coarse face and proportional to its size (what geometric
weights aim for). For the second, try **mean-value weights** (always positive, no
clamping, reproduce the coarse layout on flat regions) as a third value of
`--coarse_subdiv_relax_weights`, keeping the no-new-folds rule.

## 1. How folds are measured now

The goal of the relaxation is to smooth the noise of the coarse -> fine mapping.
The seed (every vertex at its coarse -> fine correspondence) already has folds,
so removing them is success. Two earlier measures are wrong for this:

- **flipped vs seed** counts a seed fold that the relaxation fixes as a "flip".
- **flipped vs the coarse normal**: 131 coarse faces have triangles oriented
  against their coarse face in the seed (2 entirely, many nearly: 1018, 1014, ...
  of 1024), because the coarse face is oriented against the fine sheet under it.
  That is not folding.

**Folded** (used below): a triangle whose orientation disagrees with the majority
of the triangles of its coarse face, in that mesh. The seed has **5,168** folded
triangles in 129 coarse faces, spread evenly (about 1.3% at every distance from
the seams). For each run: seed folds **removed**, **kept**, **new**, and the total
after (`test_scripts/fold_table.py`).

## 2. Results

| run | folded | removed | kept | new | degenerate | edge CV |
|---|---|---|---|---|---|---|
| seed (no relaxation) | 5,168 | - | - | - | 0 | 1.084 |
| `solve_project` | 55,738 | 2,807 | 2,361 | 53,377 | 71,431 | 1.020 |
| Newton (curves, then sheets) | 15,381 | 3,874 | 1,294 | 14,087 | 10,606 | 0.837 |
| Newton, local projection | 15,353 | 3,874 | 1,294 | 14,059 | 10,636 | 0.837 |
| Newton, per coarse face | 18,512 | 2,278 | 2,890 | 15,622 | 1,105 | 1.080 |
| `solve_project`, per coarse face | 22,104 | 2,282 | 2,886 | 19,218 | 6,030 | 1.085 |
| **explicit, everything at once** | **11,347** | 4,151 | 1,017 | 10,330 | **4,764** | 0.867 |

Every method removes most seed folds (Newton 75%, explicit 80%), but creates
3-10x more new ones, so all end with more folds than the seed.

### Explicit relaxation over time

Snapshots of the explicit run (`test_scripts/snapshot_table.py`):

| state | folded | removed | new | degenerate | edge CV |
|---|---|---|---|---|---|
| seed | 5,168 | 0 | 0 | 0 | 1.084 |
| iteration 1 | 4,419 | 1,139 | 390 | 12 | 1.075 |
| **iteration 10** | **3,722** | 2,176 | 730 | 47 | 1.062 |
| iteration 100 | 4,357 | 2,982 | 2,171 | 655 | 1.020 |
| iteration 1000 | 10,448 | 3,711 | 8,991 | 3,600 | 0.937 |
| iteration 10000 | 11,251 | 4,109 | 10,192 | 4,668 | 0.874 |
| final (20,000, not converged: last move 2e-5 x diag) | 11,347 | 4,151 | 10,330 | 4,764 | 0.867 |

The first 10-100 small steps do what the relaxation is for: they smooth out the
mapping noise (28% fewer folds than the seed at iteration 10, almost no
degenerate triangles). Further steps approach the resting state of the uniform
Laplacian, which evens out vertex **counts**; that squeezes the large coarse
faces (areas differ ~2000x, all have 1024 triangles) and folds them. The early
improvement is not kept at convergence.

### Explicit with the no-new-folds rule

Run `nofold` (lambda 0.5, 5,000 iterations; folds flat at 2,369 since iteration
3,000, last move 6e-6 x diag). The solver's own count (2,369) fixes each coarse
face's majority orientation from the seed; the scripts recompute it per mesh from
the OBJ (2,383).

| state | folded | removed | new | degenerate | edge CV |
|---|---|---|---|---|---|
| seed | 5,168 | 0 | 0 | 0 | 1.084 |
| iteration 1 | 4,092 | 1,077 | 1 | 1 | 1.076 |
| iteration 10 | 3,175 | 1,998 | 5 | 12 | 1.065 |
| iteration 100 | 2,606 | 2,571 | 9 | 21 | 1.041 |
| iteration 1000 | 2,402 | 2,780 | 14 | 29 | 1.028 |
| **final (5,000)** | **2,383** | 2,799 | 14 | **33** | **1.014** |

Median smallest angle 14.7 -> 17.5 deg, 1st percentile 0.65 -> 0.08 deg. All
consistency checks 0. 75.5M vertex moves were held back over the run.

### Step size (explicit + no new folds)

Runs `lam1` (lambda 1.0, 1,000 iterations), `nofold` (0.5), `lam025` (0.25,
4,000), `lam01` (0.1, 10,000), compared at equal lambda x iterations from the
logs (`test_scripts/step_size_table.py`); energy = 1/2 sum of squared edge
lengths, lower = more even:

| lambda x iterations | lambda 1.0 | lambda 0.5 | lambda 0.25 | lambda 0.1 |
|---|---|---|---|---|
| 100 (folded / degenerate / energy) | 2,779 / 31 / 4.702e6 | 2,529 / 23 / 4.662e6 | 2,506 / 29 / 4.668e6 | 2,500 / 25 / 4.669e6 |
| 500 | 2,570 / 37 / 4.072e6 | 2,388 / 29 / 4.034e6 | 2,383 / 33 / 4.043e6 | 2,376 / 30 / 4.039e6 |
| 1000 | 2,551 / 37 / 3.981e6 | 2,371 / 30 / 3.947e6 | 2,370 / 33 / 3.959e6 | 2,362 / 32 / 3.954e6 |
| time to 1000 | 104 s | 160 s | 338 s | 730 s |

Final meshes (`fold_table.py`): lambda 1.0: 2,568 folded / 37 degenerate / edge
CV 1.027; 0.25: 2,386 / 33 / 1.018; 0.1: 2,379 / 32 / 1.017; 0.5 (to 5,000): 2,383
/ 33 / 1.014.

- lambda 0.5, 0.25 and 0.1 agree within 0.4% (folds) and 0.3% (energy) at every
  budget: the result depends on the total step, not on the step size. Smaller
  steps only cost time (2x, 4.5x).
- lambda 1.0 moves each vertex all the way to its neighbours' mean and
  overshoots: 8% more folds, higher energy, still moving at the end (3e-4 x diag).
- **Keep lambda = 0.5.**

### Weights: uniform vs cotangent (explicit + no new folds)

`--coarse_subdiv_relax_weights uniform|cotan` (explicit only, default uniform).
Cotangent weights w_ij = 1/2 (cot a + cot b) come from the subdivided coarse
mesh at its coarse positions, computed once; 294,080 of 1,233,216 (24%) are
negative (obtuse coarse triangles; midpoint subdivision keeps the angles) and
are clamped to 0. The update uses the weighted neighbour mean; the logged energy
is 1/2 sum w_ij |x_i - x_j|^2. Run `cotan_nofold` vs `nofold` (lambda 0.5, 5,000
iterations). "CV/face" = median over coarse faces of the edge-length CV inside
that face (`fold_table.py`):

| run | folded | removed | new | degenerate | edge CV | CV/face | median smallest angle |
|---|---|---|---|---|---|---|---|
| seed | 5,168 | - | - | 0 | 1.084 | **0.384** | 14.7 |
| uniform + no new folds | **2,383** | 2,799 | 14 | 33 | **1.014** | 0.494 | **17.5** |
| cotan + no new folds | 2,568 | 2,616 | 16 | 34 | 1.135 | 0.505 | 16.6 |
| explicit, uniform, no rule | 11,347 | 4,151 | 10,330 | 4,764 | 0.867 | 0.506 | |
| Newton | 15,381 | 3,874 | 14,087 | 10,606 | 0.837 | 0.506 | |

- Cotan is slightly worse in folds and no better inside each coarse face.
- Every relaxation raises the within-face CV from 0.384 to ~0.5: they all trade
  the regular layout inside coarse faces for evener spacing between them.

### Newton stopped early

The Newton runs capped at N iterations per class
(`--coarse_subdiv_relax_max_iter N`, from `coarse_subdiv_relax_results.md`),
re-measured with the fold measure of section 1:

| Newton iterations per class | folded | removed | kept | new | degenerate | edge CV |
|---|---|---|---|---|---|---|
| 1 | 59,518 | 2,604 | 2,564 | 56,954 | 75,624 | 0.944 |
| 2 | 48,009 | 2,943 | 2,225 | 45,784 | 50,490 | 0.901 |
| 5 | 21,629 | 3,438 | 1,730 | 19,899 | 20,610 | 0.858 |
| **20** | **14,663** | 3,845 | 1,323 | 13,340 | **9,782** | 0.839 |
| until converged | 15,381 | 3,874 | 1,294 | 14,087 | 10,606 | 0.837 |

Unlike the explicit method, Newton gets better with more iterations: its first
iteration is a full linear solve with a large step and folds the most. 20
iterations are slightly better than convergence, but never close to the seed.

### Distance from the seams

Folded / degenerate rate by ring distance of a triangle from the nearest
seam/boundary/junction vertex (`test_scripts/seam_distance_folds.py`):

| ring | 0 | 1-2 | 3-5 | 6-10 | 11+ |
|---|---|---|---|---|---|
| triangles | 23,688 | 47,424 | 71,256 | 119,080 | 148,152 |
| seed | 1.29% / 0% | 1.32% / 0% | 1.28% / 0% | 1.29% / 0% | 1.21% / 0% |
| Newton | 4.66% / 4.81% | 4.82% / 3.70% | 4.44% / 3.48% | 4.10% / 2.71% | 2.66% / 1.35% |
| Newton, local projection | 4.66% / 4.81% | 4.79% / 3.71% | 4.44% / 3.50% | 4.11% / 2.71% | 2.64% / 1.36% |
| explicit | 4.41% / 3.59% | 3.88% / 2.06% | 3.55% / 1.82% | 2.90% / 0.98% | 1.67% / 0.32% |
| **explicit + no new folds** | 0.69% / 0.13% | 0.75% / 0.00% | 0.67% / 0.00% | 0.63% / 0.00% | 0.43% / 0.00% |
| Newton, per coarse face | 3.12% / 1.49% | 3.00% / 0.56% | 3.39% / 0.33% | 4.59% / 0.13% | 5.71% / 0.07% |
| solve_project, per coarse face | 5.02% / 4.18% | 4.90% / 3.28% | 4.78% / 2.42% | 5.31% / 1.11% | 5.98% / 0.29% |

The relaxation's damage is densest at the seams (Newton: 1.7x the folds and
3.6x the degenerate triangles of the far interior), but not confined to them.
Relaxing everything at once lowers it in every band, least at the seam itself.
Per coarse face reverses the pattern: the seams (on coarse edges, held fixed)
fold least, and the folds move into the interiors of the coarse faces, which are
still squeezed toward even vertex counts within each face.

### Reproducibility

The relaxed OBJs of `coarse_subdiv_relax_newton` (2026-09-28),
`coarse_subdiv_relax_newton_graphcheck` and `base_newton` (2026-09-30) are
byte-identical, so the graph PLY export, the BVH check, `holdFixed` and the
fold-rule code (switched off) do not change the default Newton result.
`base_newton` was run before the projector consolidation, so it does not cover
that change (section 5).

### solve_project: one joint solve instead of two passes

`solve_project` solves L x = 0 in 3D once per pass (curves with junctions and
Douglas-Peucker anchors fixed, project; then sheets against the projected
curves, project). `--coarse_subdiv_relax_joint_solve` (restored from commit
84b44fc) solves all free curve and sheet vertices in one system instead (the
directed graph makes it non-symmetric; sparse LU, relative residual 1.2e-14).
Runs `sp` and `sp_joint`:

| run | folded | removed | new | degenerate | edge CV |
|---|---|---|---|---|---|
| `solve_project`, two passes | 55,738 | 2,807 | 53,377 | 71,431 | 1.020 |
| `solve_project`, joint solve | 64,985 | 2,787 | 62,604 | 85,090 | 1.021 |

| ring from seam | 0 | 1-2 | 3-5 | 6-10 | 11+ |
|---|---|---|---|---|---|
| two passes (folded / degenerate) | 16.8% / 22.3% | 16.2% / 21.2% | 15.4% / 19.9% | 14.0% / 17.9% | 11.1% / 13.9% |
| joint | 25.6% / 33.4% | 21.5% / 28.5% | 18.0% / 24.2% | 15.3% / 20.1% | 12.0% / 15.2% |

- The curves come out identical (curve rows never see sheets); the sheets differ
  only in seeing the curves **before** projection, on the chords between the
  anchors. The extra damage sits at the seams and fades with distance. Same
  conclusion as on the fine subdivision
  (`subdiv_relax_solve_project_experiments.md`): keep the two passes.
- Both variants are far worse than the iterative methods on the coarse
  subdivision: one 3D solve places sheet vertices up to 0.027 x diag off the
  curved MAT, and projecting them back piles them up (17-33% degenerate).

## 3. Projection

`Projector::project` (`subdiv_sample_tracker/subdiv_relax_projector.h`) returns
the closest point to the stepped position over **all** faces of the vertex's
sheet (interior, edge or corner of a triangle) or all edges of its seam/boundary
(interior or end of an edge). The search uses one BVH per structure, built over
the **fine MAT mesh** (1,506 vertices, 2,715 faces): 24 sheet trees (triangles)
and 38 curve trees (edges).

- **BVH check** (runs every time, `verify_projector_bvh`): boxes nest, every leaf
  box contains its primitives, every primitive appears once, and closest-point
  queries on / near / far from each structure match brute force exactly. 62
  trees, 0 errors, 0 of 1,860 queries differ. Brute force would give identical
  results.
- **Projecting onto the level-4 subdivided fine mesh** would not change anything:
  the subdivision is linear (1 -> 4 at edge midpoints), so it covers exactly the
  same surface; only primitive ids and cost change.
- **Jumps**: a committed move whose new fine face shares no vertex with the old
  one. Newton: 74,308 of 15.1M moves, the largest 0.151 x diag.
- **Local projection** (`Projector::project_local`, `--coarse_subdiv_relax_local_proj`):
  the closest point over the current face (edge) and its neighbours, grown by one
  ring until the point is a local minimum on the structure, so a vertex only
  reaches points connected to where it is. Result: the same 73k "jumps" grew
  through connected rings and never needed the global fallback; quality is
  unchanged (table above). The jumps were long slides in the large early Newton
  steps, not teleports. **The projection does not cause the folds.**
- The Newton solver had its own copy of the projector (identical apart from the
  error function); it now uses the shared header, the copy is retired to
  `legacy_relax/subdiv_relax_private_projector.inc`.

## 4. What was added

| flag | what |
|---|---|
| `--coarse_subdiv_relax_method explicit` | `coarse_subdiv_relax_explicit.cpp`: x <- Pi(x + lambda (mean - x)) for all free vertices at once, symmetric mesh graph (curves are also pulled by their sheet neighbours, slide along their curve; junctions fixed), local projection, no linear solve. Logs energy / folded / degenerate every 100 iterations, writes `*_it<N>.obj` snapshots at 1, 10, 100, 1000, 10000. `--explicit_lambda` (0.5), `--explicit_max_iter` (20000), `--explicit_tol` (1e-7 x diag), `--explicit_global_proj` |
| `--coarse_subdiv_relax_per_face` | hold the vertices on coarse vertices / edges at their seeds; only each coarse face's interior relaxes (`RelaxOptions::holdFixed`) |
| `--coarse_subdiv_relax_no_new_folds` | Newton and explicit: a step may not fold an unfolded triangle (its moved vertices are held back for that step); folded ones may unfold |
| `--coarse_subdiv_relax_local_proj` | Newton: local projection (section 3) |
| `--coarse_subdiv_relax_joint` | Newton: one pass over curves and sheets on the symmetric graph |
| `--coarse_subdiv_relax_weights uniform\|cotan` | explicit: neighbour weights; cotan from the coarse positions, negatives clamped to 0 |
| `--coarse_subdiv_relax_joint_solve` | solve_project: one LU solve of curves and sheets on the directed graph |
| (always) | relaxation graph at the seed as `laplacian_graph/{sheets,curves,junctions}_coarse_subdiv_<stem>.ply`, coloured by structure id; projector BVH check; projection jump counter |

The options are collected in `CoarseSubdivRelaxConfig` (`coarse_subdiv_relax.h`).

## 5. Not done yet

- **No-new-folds with Newton**: implemented, no run finished (with the explicit
  method it is done, see section 2).
- **Joint Newton pass**: implemented, the run was stopped before it finished.
- **Regression check** of the default Newton path after the projector
  consolidation (expected byte-identical to `base_newton`): not run yet. Newton
  with local projection on reproduces the same quality, which suggests the shared
  projector behaves the same.
- Long run-folder names exceed the Windows 260-character path limit and make the
  end-of-run `[SANITY]` check fail to read the JSON (it does not use the
  long-path form); keep run names short.

## 6. Run index

Coarse-subdivision runs in `output/relaxation_experiments/` (each folder has a
`<name>.log` next to it):

| folder | what | status |
|---|---|---|
| `coarse_subdiv_relax` | solve_project, two passes (2026-09-28) | done |
| `coarse_subdiv_relax_newton` | Newton (2026-09-28) | done |
| `coarse_subdiv_relax_newton_it{1,2,5,20}` | Newton capped at N iterations per class | done |
| `coarse_subdiv_relax_newton_graphcheck` | Newton rerun with graph PLY export + BVH check | done, identical to `coarse_subdiv_relax_newton` |
| `coarse_subdiv_relax_newton_perface` | Newton, per coarse face | done (end-of-run `[SANITY]` failed: long path) |
| `coarse_subdiv_relax_sp_perface` | solve_project, per coarse face | done |
| `base_newton` | Newton, with the projection jump counter | done, identical to `coarse_subdiv_relax_newton` |
| `local_newton` | Newton, local projection | done |
| `explicit` | explicit, 20,000 iterations, with `_it<N>` snapshots | done (not converged) |
| `sp` | solve_project, two passes (rerun) | done |
| `sp_joint` | solve_project, joint solve | done |
| `nofold` | explicit + no new folds, lambda 0.5, 5,000 iterations | done (best) |
| `lam1`, `lam025`, `lam01` | explicit + no new folds, lambda 1.0 / 0.25 / 0.1 | done |
| `cotan_nofold` | explicit + no new folds, cotangent weights | done |
| `nf_newton`, `nf_newton_pf` | Newton with no-new-folds (+ per face) | **stopped, incomplete: not results** |
| `joint_newton` | Newton, joint pass | **stopped, incomplete: not results** |
| `projector_extract_{before,after}` | projector refactor check (2026-09-28) | done |

The other folders (`l3_*`, `l4_*`, `m563_*`, `relax_*`, `c2f_*`, ...) belong to
the fine-subdivision relaxation and the coarse -> fine query work, documented in
their own md files (e.g. `subdiv_relax_solve_project_experiments.md`).
