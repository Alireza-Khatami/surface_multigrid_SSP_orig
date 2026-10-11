# Coarse-to-fine (multilevel) relaxation

Plan, implementation notes and results for relaxing a coarser sampling first and refining the
relaxed mesh up to the final sample count. Script: `relaxation_scripts_python/relax_multilevel.py`.
Background on why the explicit step is slow and how momentum helps:
`momentum_and_implicit_explained_14h12_2026-10-11.md`.

## The idea

The explicit step is slow because a large-scale imbalance only spreads one ring of neighbours
per step: a pattern k vertices wide needs about k² steps. On a coarser sampling the same
pattern is only a few vertices wide, so it settles in a handful of steps, and each step is also
cheaper because there are fewer vertices.

## The ladder

N = final sample count (277,700 for setup B). Each level has about 4× the samples of the one
before, so edge lengths halve.

1. **Level 0, about N/16 samples.** Build the sampling and relax it. A pattern is 4× narrower
   in vertices, so it needs about 16× fewer steps, and each step costs about 16× less. This level
   fixes the large-scale balance (which parts of a sheet are too dense or too sparse).
2. **Refine to about N/4.** Split edges of the **relaxed** level-0 mesh:
   - old vertices keep their relaxed positions;
   - each new vertex starts at the midpoint of its edge and is projected onto its own structure
     (sheet, or seam / boundary curve);
   - Delaunay flips, seam and boundary edges never flipped.
3. **Relax level 1 briefly.** Only medium-scale unevenness is left.
4. **Refine to N and relax briefly again.**

Coarse levels do the large scale, fine levels only the local detail. Each level can use
momentum, so the two speed-ups combine.

## Catches

- **Carriers of new vertices come from projection.** In the single-level pipeline every sample
  sits at an exact (dyadic) subdivision point of its coarse face. A midpoint of two vertices that
  already moved is not such a point; where it sits on the fine MAT has to be found by projecting
  it. Samples already slide off their original triangles during relaxation, so the `.sdt`
  output can still represent this, but it must be checked with `check_relaxed_sdt.py`.
- **The gain may be modest now.** Momentum already reaches good quality in about 300 iterations
  (28 s). Multilevel attacks the same large-scale problem, so it helps mostly where momentum
  alone stalls. It might also change the quality the runs settle at, not only the speed.
- **More code than momentum.** It needs a "refine a relaxed mesh" step and a driver that loops
  over the levels.

## Implementation

Script `relaxation_scripts_python/relax_multilevel.py`. It takes every `run_relax.py` flag,
plus:
- `--levels L` (default 3);
- `--level_ratio r` (default 4);
- `--level_iters a,b,c` (iterations per level, coarse to fine).

The final sample count is `--equal_area_samples N`; level l gets N / r^(L-1-l) samples.

**Per level, existing code only:**
- `run_relax.build_c2f`: equal-area split to that level's count, Delaunay flips, c2f walk;
- `CoarseRelaxSession` and `make_relaxer`: struct ids, seeds, graph, fold reference, and the
  explicit / momentum / implicit relaxer.

The relaxer already accepts a start state (`state=`), so the prolonged positions go in there.

**Why the levels nest.** The equal-area split always splits the largest face first and is
deterministic. So splitting to N/4 passes through exactly the N/16 mesh on the way. The first
n₀ vertices of level l are level l−1's vertices, in the same order, at the same coarse
positions. The script asserts this, and also that their struct sets are equal.

**The prolongation, the only new part:**
- old vertices take level l−1's relaxed state: position, fine face, curve edge, barycentrics;
- new vertices, in creation order: midpoint of the parent edge, projected onto the vertex's own
  structure with the existing projector. A parent is either an old vertex or an earlier new one.
- vertices that are fixed anyway (unmapped, junction, no target) keep their seed. There were
  none in the runs.

To know each new vertex's parent edge, `equal_area_refine.refine_equal_area` now records
`R.splitParents`. That is the only change to existing code, and it doesn't change the refinement.

**Flips are per level.** The Delaunay flips run on each level's mesh in coarse space, as in the
single-level run. They don't change the vertices, only the connectivity, so the prolongation is
unaffected.

**Outputs** are the same as `run_relax.py`, from the final level:
- the relaxed OBJ, `.sdt`, sliding stats, optional radius check, `results.json` (with a
  `multilevel` block per level) and `experiment_config.txt`;
- every level, in `levels/level<l>/`, the stages in order:
  - `00`–`04`, as in `run_relax.py`'s `relax_input/`: coarse mesh, equal-area split, before
    and after the flips, the subdivided mesh on the coarse geometry, the same at its c2f
    positions on the fine MAT, and the relaxation input;
  - `05_subdivided_relaxed` (level > 0): the relaxed level below, subdivided. New vertices sit
    at the midpoint of their parents, before the projection onto their structure;
  - `06_start`: the relaxation input, the same positions as `04`;
  - `07_relaxed`: after the level's iterations.
  `05`–`07` have vertex properties `role`, `set_id`, `free`, `fine_face`, `is_new` (made on this
  level) and face properties `coarse_face`, `folded` (against the fold reference; the count is
  checked against the relaxer's). About 216 MB per 3-level run at 277,700 samples.

**What the `.sdt` calls the seed.** `sub_V_seed` and the seed carriers are the final level's c2f
seeds, where the samples were made. They are not the start state prolonged from the coarser
level, which is in `relax_input/04_relax_input_*.ply`. The c2f seeds are also written as
`relax_input/04s_c2f_seeds_*.ply`. `check_relaxed_sdt.py` checks `sub_V_seed` and the sliding
against that file when it is present.

**Checks.** Every level runs the relaxer's consistency checks: seed off structure, fixed moved,
position vs interpolation, barycentrics, off own structure. The run fails if any of them fails.

Experiment set: `python run_experiments.py --set multilevel --suffix _valid`. One run with setup B:
`bash relaxation_scripts_python/run_multilevel.sh [NAME] [LEVEL_ITERS] [LEVELS] [extra flags]`
(runs `check_relaxed_sdt.py` afterwards). Results: `relaxation_multilevel_results_16h13_2026-10-11.md`.

## Results

**Setup.** All runs use setup B on `src_qslim200_valid` (ABC 00040057):
- equal-area split to 277,700 samples, Delaunay flips;
- uniform weights, two-sided graph, global projection, no new folds;
- momentum β 0.9 unless marked explicit.

Levels with ratio 4: 3 levels = 17,356 / 69,425 / 277,700 samples; 4 levels add 4,339 first.
Quality is the final mesh. Folds and degenerate triangles are counted against the final level's
c2f seeds, which have 595 folded.

**Timing.**
- *relax s* is the time spent in relaxation steps, all levels together.
- *total s* is wall time, including building the 278k mesh and the c2f map. That build is
  about 24 s for every run, single-level or not.
- In the multilevel runs, each coarser level adds about 2–7 s of setup.

| run | iterations | relax s | total s | edge CV | min angle p1 / p5 / median | folded | degenerate |
|---|---|---|---|---|---|---|---|
| single, momentum (`py_eqs278k_flip_momentum_b09_300_valid`) | 300 | 20.2 | 41 | **0.2126** | 24.09 / 34.45 / 49.91 | **303** | **7** |
| single, momentum (`py_eqs278k_flip_momentum_b09_2k_valid`) | 2000 | 172.5 | 196 | 0.2158 | 24.55 / 34.66 / 51.30 | 303 | 5 |
| single, explicit (`py_eqs278k_flip_nofold_global_2k_valid`) | 2000 | 420.6 | 448 | 0.2121 | 23.66 / 34.28 / 49.62 | 301 | 4 |
| multilevel 3, momentum (`py_ml3_momentum_300_100_100_valid`) | 300,100,100 | **9.2** | 42 | 0.2165 | 24.25 / 34.62 / 51.13 | 334 | 16 |
| multilevel 3, momentum (`py_ml3_momentum_300_100_30_valid`) | 300,100,30 | **5.7** | 38 | 0.2168 | 23.64 / 33.44 / 50.94 | 337 | 14 |
| multilevel 3, momentum (`py_ml3_momentum_300_100_300_valid`) | 300,100,300 | 18.2 | 51 | 0.2166 | **24.52 / 34.75 / 51.33** | 333 | 15 |
| multilevel 4, momentum (`py_ml4_momentum_300_100_100_100_valid`) | 300,100,100,100 | 8.9 | 42 | 0.2177 | 24.01 / 34.27 / 51.22 | 345 | 13 |
| multilevel 3, explicit (`py_ml3_explicit_1000_300_300_valid`) | 1000,300,300 | 20.5 | 53 | 0.2148 | 23.94 / 34.16 / 50.77 | 334 | 12 |

Per level, `py_ml3_momentum_300_100_100_valid`:

| level | samples | edge CV start → end | folded start → end |
|---|---|---|---|
| 0 | 17,356 | 0.306 → 0.233 | 51 → 16 |
| 1 | 69,425 | 0.265 → 0.221 | 158 → 90 |
| 2 | 277,700 | 0.245 → 0.217 | 484 → 334 |

The prolonged start of the final level is already much more even than its c2f seeds:
edge CV 0.245 against 0.317, and 484 folded against 595.

**What it shows:**
- **It is faster, but only in the relaxation itself.**
  - 300/100/100 reaches, in 9.2 s of relaxation, about the end state that single-level
    momentum reaches after 2000 iterations: edge CV 0.216, median angle 51.1–51.3. That run
    needed 172.5 s, so this is about 19× faster.
  - Against single-level momentum at 300 iterations (20.2 s) it is about 2× faster.
  - Wall time is the same (about 40 s), because building the 278k mesh and its c2f map
    dominates. To gain wall time, that build would have to get faster.
- **The quality is a different trade, not strictly better.**
  - Better: the angles (median 51.1 vs 49.9, p1 24.25 vs 24.09).
  - Worse: edge CV (0.2165 vs 0.2126), folds (334 vs 303) and degenerate triangles (16 vs 7).
  - The angles-for-evenness part matches the long single-level momentum runs: past ~300
    iterations their angles keep improving while edge CV creeps up. Multilevel lands in that
    regime directly.
  - The folds do not match: every single-level run ends at about 301–303 folds, every
    multilevel run at about 333–345.
- **More final iterations do not remove the extra folds.** 300 final-level iterations instead
  of 100 leave 333 instead of 334. **Unverified explanation:** folds that survive on a coarse
  level (16 on level 0) are carried up and split into several fine triangles each. The
  no-new-folds rule can still undo them, but they behave like existing folds, not new ones.
  Checking this would mean locating the final folds against the level-0 folds.
- **A 4th level does not help.** It is slightly worse on every measure than 3 levels.
- **Explicit steps** with multilevel give the lowest edge CV of the multilevel runs (0.2148),
  but they need about 2× the relaxation time of momentum.
- **Checks pass.** All runs pass the relaxer's consistency checks on every level, and
  `check_relaxed_sdt.py` passes on `py_ml3_momentum_300_100_100_valid` (19 / 19).

**Unified ranking** (`rank_experiments.py`), which compares end states:
- single-level momentum 2000 and 7000 are #1 and #2;
- single-level momentum 300 is #4;
- the best multilevel run (300/100/300) is #5; the others are #7–#9, pulled down by the
  extra folds and degenerate triangles.

**Recommendation.** It doesn't replace single-level momentum at 300 iterations as the default
yet. Multilevel is worth it when relaxation time matters more than the last ~30 folds. For
example, the viewer could get a good-angle mesh in about 6 s of relaxation (300/100/30). The
extra folds are the thing to fix first.
