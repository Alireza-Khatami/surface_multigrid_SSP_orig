# Coarse-subdivision relaxation: experiments of 2026-09-30

Follow-up to `coarse_subdiv_relax_results.md` (the relaxed subdivided coarse mesh
folds). Test mesh ABC 00040057, target 200, qslim, `--n_subdiv_samples 200000`:
coarse mesh 261 vertices / 400 faces, subdivided to 207,000 vertices / 409,600
faces (1024 per coarse face). Runs in `output/relaxation_experiments/<name>/`,
analysis scripts in `test_scripts/` (see its README).

## Summary: best method so far

**The explicit relaxation (small Laplacian steps on everything at once) is the
best, but no method fully works yet.**

- **Stopped early (~10 iterations),** it is the only setting better than the
  seed: 28% fewer folds (3,722 vs 5,168) and almost no degenerate triangles
  (47). It barely evens out the spacing, though (edge CV 1.084 -> 1.062).
- **Run to the end,** it beats every other method (11,347 folds, 4,764
  degenerate), but is still worse than the seed.
- **Newton evens out the spacing most** (edge CV 0.837), but leaves 15,381 folds
  and 10,606 degenerate triangles.

No method yet both evens out the spacing and keeps folds below the seed.

| method | folded | degenerate | edge CV |
|---|---|---|---|
| seed (no relaxation) | 5,168 | 0 | 1.084 |
| **explicit, iteration 10** | **3,722** | **47** | 1.062 |
| explicit, iteration 100 | 4,357 | 655 | 1.020 |
| explicit, final (20,000) | 11,347 | 4,764 | 0.867 |
| Newton | 15,381 | 10,606 | **0.837** |
| Newton, local projection | 15,353 | 10,636 | 0.837 |
| Newton, per coarse face | 18,512 | 1,105 | 1.080 |
| solve_project, per coarse face | 22,104 | 6,030 | 1.085 |
| solve_project, two passes | 55,738 | 71,431 | 1.020 |
| solve_project, joint solve | 64,985 | 85,090 | 1.021 |

### What we learned

1. **"Flipped vs seed" was the wrong measure.** The seed already has 5,168
   folds from the noisy coarse -> fine mapping, and removing them is the goal.
   Every method removes 55-80% of them, but most create many more new ones.
2. **Folds come from the uniform Laplacian's resting state, not the
   projection.** At rest, the Laplacian evens out vertex *counts*. Every coarse
   face has 1,024 triangles while their areas differ ~2,000x, so large faces get
   squeezed. In the explicit run, the early steps remove noise and the late
   steps create folds.
3. **The projection is correct.** Its BVHs (over the fine MAT) match brute force
   exactly; the local projection gave identical quality; projecting onto a
   level-4 subdivided fine mesh would give the same points.
4. **Damage is densest next to seams.** Relaxing everything at once in small
   steps reduces it everywhere. Solving everything at once in one linear system
   makes it worse at the seams, because the sheets then lean on the seam
   positions before projection.
5. **One-shot linear solves don't suit this case.** solve_project places sheet
   vertices off the curved MAT and then piles them up when projecting back.
6. **Per-coarse-face relaxation** cuts degenerate triangles but doesn't reduce
   folds, and it barely evens out the spacing.

### Recommended next step

Run the **explicit relaxation with the no-new-folds rule**
(`--coarse_subdiv_relax_method explicit --coarse_subdiv_relax_no_new_folds`),
implemented but never run. Folds can then only go down: it keeps the early
unfolding (<= 5,168) and blocks the later squeezing. Then see how far it evens
out the spacing. If spacing stays poor, the next lever is the Laplacian weights
(area or cotangent weights instead of uniform), which goes after the root cause
in point 2.

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

### Distance from the seams

Folded / degenerate rate by ring distance of a triangle from the nearest
seam/boundary/junction vertex (`test_scripts/seam_distance_folds.py`):

| ring | 0 | 1-2 | 3-5 | 6-10 | 11+ |
|---|---|---|---|---|---|
| triangles | 23,688 | 47,424 | 71,256 | 119,080 | 148,152 |
| seed | 1.29% / 0% | 1.32% / 0% | 1.28% / 0% | 1.29% / 0% | 1.21% / 0% |
| Newton | 4.66% / 4.81% | 4.82% / 3.70% | 4.44% / 3.48% | 4.10% / 2.71% | 2.66% / 1.35% |
| explicit | 4.41% / 3.59% | 3.88% / 2.06% | 3.55% / 1.82% | 2.90% / 0.98% | 1.67% / 0.32% |

The relaxation's damage is densest at the seams (Newton: 1.7x the folds and
3.6x the degenerate triangles of the far interior), but not confined to them.
Relaxing everything at once lowers it in every band, least at the seam itself.

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
| `--coarse_subdiv_relax_joint_solve` | solve_project: one LU solve of curves and sheets on the directed graph |
| (always) | relaxation graph at the seed as `laplacian_graph/{sheets,curves,junctions}_coarse_subdiv_<stem>.ply`, coloured by structure id; projector BVH check; projection jump counter |

The options are collected in `CoarseSubdivRelaxConfig` (`coarse_subdiv_relax.h`).

## 5. Not done yet

- **No-new-folds** rule: implemented (Newton and explicit) but no run finished.
  With the explicit method it should keep the early unfolding (folds can only go
  down, so the result is at most the seed's 5,168) and block the later squeezing.
  Next experiment.
- **Joint Newton pass**: implemented, the run was stopped before it finished.
- **Regression check** of the default Newton path after the projector
  consolidation (expected byte-identical to `base_newton`): not run yet. Newton
  with local projection on reproduces the same quality, which suggests the shared
  projector behaves the same.
- Long run-folder names exceed the Windows 260-character path limit and make the
  end-of-run `[SANITY]` check fail to read the JSON (it does not use the
  long-path form); keep run names short.
