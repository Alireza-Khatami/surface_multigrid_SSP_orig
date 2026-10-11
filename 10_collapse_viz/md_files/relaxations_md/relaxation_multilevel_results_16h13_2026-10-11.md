# Multilevel relaxation: results

Results of the coarse-to-fine (multilevel) relaxation on ABC 00040057. The method, catches and
implementation are in `relaxation_multilevel_15h43_2026-10-11.md`.

## How to run

```
cd 10_collapse_viz/relaxation_scripts_python
bash run_multilevel.sh [NAME] [LEVEL_ITERS] [LEVELS] [extra flags...]
bash run_multilevel.sh                                              # 3 levels, 300,100,100
bash run_multilevel.sh py_ml3_momentum_3000x3_valid 3000,3000,3000  # 3000 iterations on each level
```

`run_multilevel.sh` runs `relax_multilevel.py` with setup B, then `check_relaxed_sdt.py` on the
result. Any extra flags go to `relax_multilevel.py` / `run_relax.py`; for example
`--coarse_subdiv_relax_method explicit` switches to explicit steps. The whole comparison set is
`python run_experiments.py --set multilevel --suffix _valid`.

**Setup B, all runs:**
- input: bundle `src_qslim200_valid` (validity checks on) and the ABC 00040057 `.ma_struct`;
- sampling: equal-area split to 277,700 samples, Delaunay flips;
- relaxation: uniform weights, two-sided graph, global projection, no new folds, momentum β 0.9
  unless marked explicit.

Levels use ratio 4. 3 levels = 17,356 / 69,425 / 277,700 samples; 4 levels add 4,339 first.
Folds and degenerate triangles are counted against the final level's c2f seeds, which have 595
folded.

## Multilevel vs single level

*relax s* is the time spent in relaxation steps, all levels together. *total s* is wall time;
it includes about 24 s to build the 278k mesh and its c2f map, which every run pays.

| run (folder in `output/relaxation_experiments/`) | iterations | relax s | total s | edge CV | min angle p1 / p5 / median | folded | degenerate |
|---|---|---|---|---|---|---|---|
| single, momentum `py_eqs278k_flip_momentum_b09_300_valid` | 300 | 20.2 | 41 | **0.2126** | 24.09 / 34.45 / 49.91 | **303** | **7** |
| single, momentum `py_eqs278k_flip_momentum_b09_2k_valid` | 2000 | 172.5 | 196 | 0.2158 | 24.55 / 34.66 / 51.30 | 303 | 5 |
| single, explicit `py_eqs278k_flip_nofold_global_2k_valid` | 2000 | 420.6 | 448 | 0.2121 | 23.66 / 34.28 / 49.62 | 301 | 4 |
| 3 levels `py_ml3_momentum_300_100_100_valid` | 300,100,100 | **9.2** | 42 | 0.2165 | 24.25 / 34.62 / 51.13 | 334 | 16 |
| 3 levels `py_ml3_momentum_300_100_30_valid` | 300,100,30 | **5.7** | 38 | 0.2168 | 23.64 / 33.44 / 50.94 | 337 | 14 |
| 3 levels `py_ml3_momentum_300_100_300_valid` | 300,100,300 | 18.2 | 51 | 0.2166 | **24.52 / 34.75 / 51.33** | 333 | 15 |
| 4 levels `py_ml4_momentum_300_100_100_100_valid` | 300,100,100,100 | 8.9 | 42 | 0.2177 | 24.01 / 34.27 / 51.22 | 345 | 13 |
| 3 levels, explicit `py_ml3_explicit_1000_300_300_valid` | 1000,300,300 | 20.5 | 53 | 0.2148 | 23.94 / 34.16 / 50.77 | 334 | 12 |

**Findings:**
- **Faster relaxation, same wall time.** 300/100/100 reaches about the end state of 2000
  single-level momentum iterations in 9.2 s instead of 172.5 s. Against momentum at 300
  iterations it is about 2× faster. Wall time doesn't change, because the 278k build dominates.
- **Better angles, worse evenness and folds.**
  - Angles: median 51.1 vs 49.9.
  - Edge CV: 0.2165 vs 0.2126.
  - Folds: about 333–345 in every multilevel run, against about 301–303 in every
    single-level run.
  - Degenerate triangles: 12–16 vs 4–7.
- **More final-level iterations don't remove the extra folds:** 300 final iterations leave 333,
  against 334 after 100. Unverified explanation: folds left on a coarse level are carried up and
  split into several fine triangles each.
- **A 4th level is slightly worse** on every measure.
- **Unified ranking:** the best multilevel run (300/100/300) is #5. Single-level momentum at 2000
  and 7000 iterations is #1 and #2, and at 300 iterations #4.

## Per-level stages (`py_ml3_momentum_300_100_100_valid`, with exports)

Each level's meshes are in `levels/level<l>/` (the stages are listed in the method doc):
- `03_subdiv_at_fine`: the level's c2f seeds, as a fresh single-level start would be;
- `06_start`: what the level actually starts from (= `04_relax_input`);
- `07_relaxed`: after the level's iterations.

| level | samples | iterations | relax s | edge CV seeds / start → end | min angle p1 start → end | median start → end | folded seeds / start → end | degenerate |
|---|---|---|---|---|---|---|---|---|
| 0 | 17,356 | 300 | 1.0 | 0.306 / 0.306 → 0.233 | 10.37 → 22.02 | 39.32 → 49.35 | 51 / 51 → 16 | 1 |
| 1 | 69,425 | 100 | 1.2 | 0.310 / 0.265 → 0.221 | 15.90 → 23.49 | 48.08 → 50.73 | 185 / 158 → 90 | 4 |
| 2 | 277,700 | 100 | 6.9 | 0.317 / 0.245 → 0.217 | 18.84 → 24.25 | 50.09 → 51.13 | 595 / 484 → 334 | 16 |

**The step between levels (`05_subdivided_relaxed` → `06_start`):**

| step | old vertices | new vertices | projection distance mean / max (× diag) |
|---|---|---|---|
| level 0 → 1 | 17,356 | 52,069 | 1.8e-5 / 2.5e-3 |
| level 1 → 2 | 69,425 | 208,275 | 4.0e-6 / 9.9e-4 |

New vertices are projected from their parents' midpoint onto their own structure. No new vertex
had to stay at its seed (unmapped, junction or no target).

**Observations:**
- **The carried-up start is much better than fresh seeds.** On the final level it has edge CV
  0.245 against 0.317 for the c2f seeds, and 484 folds against 595. This is the large-scale
  balance the coarse levels pass on.
- **Folds go down on every level but don't reach the single-level count:** 51 → 16 on level 0,
  158 → 90 on level 1, 484 → 334 on level 2.
- **Checked on a small run (20,000 samples):** `04` = `06`, the old vertices in `05` and `06` are
  exactly the level below's `07`, and new vertices differ between `05` and `06` only by the
  projection.
- **`.sdt` check:** `check_relaxed_sdt.py` passes.

## 3000 iterations on each level

Run: `bash run_multilevel.sh py_ml3_momentum_3000x3_valid 3000,3000,3000`. Folder
`py_ml3_momentum_3000x3_valid`.

| level | samples | relax s | edge CV start → end | min angle p1 / p5 / median at end | folded start → end | degenerate | adaptive restarts | last max move (× diag) |
|---|---|---|---|---|---|---|---|---|
| 0 | 17,356 | 8.8 | 0.306 → 0.233 | 21.96 / 30.74 / 49.48 | 51 → 16 | 0 | 28 | 1.1e-7 |
| 1 | 69,425 | 33.5 | 0.266 → 0.223 | 23.47 / 33.36 / 51.11 | 162 → 91 | 3 | 41 | 3.7e-7 |
| 2 | 277,700 | 143.5 | 0.246 → 0.218 | 24.48 / 34.79 / 51.94 | 485 → 339 | 14 | 27 | 3.3e-6 |

Total: 185.9 s of relaxation, 219 s wall time. The `.sdt` check passes. 60.2% of the samples
ended on another fine triangle than their seed.

**Against the other runs** (final mesh):

| run | iterations | relax s | edge CV | min angle p1 / p5 / median | folded | degenerate |
|---|---|---|---|---|---|---|
| 3 levels | 300,100,100 | 9.2 | 0.2165 | 24.25 / 34.62 / 51.13 | 334 | 16 |
| **3 levels** | **3000,3000,3000** | **185.9** | 0.2178 | 24.48 / 34.79 / **51.94** | 339 | 14 |
| single-level momentum | 300 | 20.2 | **0.2126** | 24.09 / 34.45 / 49.91 | **303** | 7 |
| single-level momentum (`py_eqs278k_flip_momentum_b09_7k_valid`) | 7000 | 346.7 | 0.2172 | 24.53 / 34.66 / 51.7 | 302 | 5 |

**Observations:**
- **Ten to thirty times more iterations change little.**
  - Median angle 51.13 → 51.94 and p1 24.25 → 24.48.
  - Edge CV creeps up 0.2165 → 0.2178, the same drift the long single-level runs show.
  - Folds 334 → 339 and degenerate triangles 16 → 14.
  - The end state is close to single-level momentum at 7000 iterations, except for about 37
    more folds and 9 more degenerate triangles.
- **The folds are stuck on every level.**
  - Level 0 ends at 16 folds after 300 iterations and after 3000.
  - Level 1 ends at 90 vs 91; level 2 at 334 vs 339.
  - More iterations don't remove them, which fits the explanation that folds left on a coarse
    level are carried up and kept. Still unverified: it needs locating the final folds against
    the level-0 folds (the `folded` face property in `levels/level<l>/07_relaxed`).
- **No level converges** to the 1e-7 × diagonal stop rule, although level 0 comes close
  (1.1e-7).
- **The adaptive restart triggers here (27–41 times per level).** It never triggered in the single-level
  2000- and 7000-iteration runs. Each level starts from an already-balanced state, so momentum
  overshoots earlier.
- **Conclusion:** 3000 iterations per level isn't worth it. 300,100,100 gives nearly the same
  mesh in 9 s instead of 186 s.
