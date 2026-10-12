# Momentum relaxation: weights, directed graph and folds

Momentum (β 0.9) on setup B with different neighbour weights and both relaxation graphs, with and
without the no-new-folds rule. Momentum background: `momentum_and_implicit_explained_14h12_2026-10-11.md`.

## Setup

**Setup B:**
- input: bundle `src_qslim200_valid` (validity checks on) and the ABC 00040057 `.ma_struct`;
- sampling: equal-area split to 277,700 samples, Delaunay flips;
- relaxation: global projection, momentum β 0.9 with adaptive restart.

**What varies:**
- **Weights:** uniform, cotan or mean-value. Cotan and mean-value are computed from the coarse
  geometry.
- **Graph:**
  - **two-sided** (`symmetric`): every vertex is pulled by all its mesh neighbours;
  - **directed**: a curve sample is pulled only by curve or junction neighbours on its own seam
    or boundary, sheet samples by any neighbour on one of their sheets, junctions never move.

  Each sample is projected back onto its own structure after every step, whichever graph is used.
- **Folds:** with the no-new-folds rule, a move that would fold a triangle is held back. Without
  it ("folds allowed"), every move is taken.

Folds are counted against the c2f seeds, which have 595 folded triangles. *relax s* is the time
spent in relaxation steps. All runs are in `output/relaxation_experiments/`.

**How to run:**
- the whole set: `python run_experiments.py --set momentum_weights --suffix _valid`;
- one run: `--only <name>`.

## Default graph changed: directed

**Since 2026-10-11, the Python default graph is directed.** Before, it was two-sided, which is
also the C++ default.

| flag | effect |
|---|---|
| (none) | directed graph |
| `--explicit_symmetric_graph` | two-sided graph (new, Python-only) |
| `--explicit_directed_graph` | directed graph (still accepted) |
| `--coarse_subdiv_relax_method implicit` | two-sided graph unless `--explicit_directed_graph`; the implicit step needs a symmetric system |

To keep earlier results reproducible:
- **`run_experiments.py`** adds `--explicit_symmetric_graph` to every entry that names neither
  graph, since those sets were written when two-sided was the default;
- **`verify_all.sh`** does the same, because it compares against C++ runs, which use the
  two-sided graph;
- **all earlier result tables** (setup B, momentum, multilevel) are two-sided.

## Results: 300 iterations

| run | weights | graph | folds | relax s | edge CV | min angle p1 / p5 / median | folded | degenerate | restarts |
|---|---|---|---|---|---|---|---|---|---|
| `py_eqs278k_flip_momentum_b09_300_valid` | uniform | two-sided | no new | **20.2** | **0.2126** | **24.09 / 34.45 / 49.91** | **303** | **7** | 0 |
| `py_eqs278k_flip_momentum_b09_cotan_300_valid` | cotan | two-sided | no new | 21.0 | 0.2301 | 17.99 / 26.77 / 43.31 | 329 | 8 | 0 |
| `py_eqs278k_flip_momentum_b09_meanvalue_300_valid` | mean-value | two-sided | no new | 27.1 | 0.2289 | 17.55 / 26.96 / 43.34 | 323 | 8 | 0 |
| `py_eqs278k_flip_momentum_b09_dir_300_valid` | uniform | directed | no new | 86.4 | 0.2223 | 19.47 / 29.45 / 46.69 | 326 | 9 | 208 |
| `py_eqs278k_flip_momentum_b09_dir_allowfolds_300_valid` | uniform | directed | allowed | 17.6 | 0.2234 | 19.19 / 29.34 / 46.71 | 444 | 10 | 205 |

**References** (two-sided, uniform weights):

| run | method | iterations | relax s | edge CV | min angle p1 / p5 / median | folded | degenerate |
|---|---|---|---|---|---|---|---|
| `py_eqs278k_flip_cotan_nofold_global_2k_valid` (cotan weights) | explicit | 2000 | about 420 | 0.2300 | 17.89 / 26.72 / 43.32 | 329 | 6 |
| `py_eqs278k_flip_momentum_b09_7k_valid` | momentum | 7000 | 346.7 | 0.2172 | 24.53 / 34.66 / 51.74 | 302 | 5 |

All `.sdt` checks (`check_relaxed_sdt.py`) pass on the two-sided weight runs.

## Observations

- **Cotan and mean-value weights are worse than uniform.**
  - Their min angles are about 6° lower, edge CV is about 0.23 against 0.21, and they leave
    20–26 more folds.
  - Momentum isn't the problem: cotan with momentum at 300 iterations reaches the same end state
    as explicit cotan at 2000 iterations.
  - **Likely cause (not tested):** both weight sets come from the coarse geometry, so they
    preserve the samples' current layout, which is the uneven c2f seed spacing, instead of evening
    it out. Their mean move is about a third of uniform's (0.0012 against 0.0033 × diag).
- **The directed graph is worse than two-sided.**
  - Edge CV is 0.222 against 0.213, min angle p1 19.5 against 24.1, and there are more folds.
  - **Momentum and the directed graph don't fit.** The adaptive restart fires about 205–208
    times in 300 iterations, against 0 on the two-sided graph. A directed graph has no energy the
    step minimizes, so the energy the restart watches rises even when the step is fine
    (1.27652e6 at iteration 100 → 1.27689e6 at 300). Each restart wipes the momentum, so the run
    is close to plain explicit steps and stalls early: the max move is already about 1e-5 × diag
    by iteration 100.
- **Allowing folds removes the slowdown but not the quality gap.**
  - Directed with no new folds took 86 s for 300 iterations. Every one of the ~208 restarts also
    runs the fold check on the rejected move.
  - With folds allowed it takes 17.6 s. Evenness and angles are the same (edge CV 0.2234), but
    there are 444 folds instead of 326.
- **A directed, no-new-folds run at 5000 iterations was stopped** at iteration 3400. It was
  stuck at 326 folds, with moves of about 5e-6 × diag.

## Results: 5000 iterations, directed graph, folds allowed

The folders of the fold-allowed runs are named `..._dir_allowfolds_...` (renamed from `..._dir_folds_...`).

| run | weights | iterations | relax s | edge CV | min angle p1 / p5 / median | folded | degenerate | restarts |
|---|---|---|---|---|---|---|---|---|
| `py_eqs278k_flip_momentum_b09_dir_allowfolds_300_valid` | uniform | 300 | 17.6 | 0.2234 | 19.19 / 29.34 / 46.71 | 444 | 10 | 205 |
| `py_eqs278k_flip_momentum_b09_dir_allowfolds_5k_valid` | uniform | 5000 | 234.8 | 0.2421 | 16.66 / 25.97 / 45.69 | 437 | 10 | 4905 |
| `py_eqs278k_flip_momentum_b09_dir_allowfolds_cotan_5k_valid` | cotan | 5000 | 236.0 | **0.8800** | **0.00 / 2.10 / 30.35** | **4642** | **4241** | 4985 |
| mean-value, 5000 iterations | mean-value | stopped at 2700 | — | — | — | **37,025** at 2700 | 14,070 at 2700 | — |

- **More iterations make the directed graph worse.** Uniform at 5000 iterations has edge CV
  0.242, against 0.223 at 300, and lower angles. The restart fires in almost every iteration (4905
  of 5000), so momentum does nothing and the run drifts.
- **Cotan and mean-value diverge** without the no-new-folds rule. Cotan ends with 4642 folded and
  4241 degenerate triangles. Mean-value had 37,025 folded triangles at iteration 2700 and was
  stopped; its partial folder was removed. Cotan and mean-value weights can be negative or very
  uneven, so without the fold check the step can turn triangles over, and the folds then grow.
- **Conclusion:** the directed graph, with or without the fold rule, isn't worth pursuing with
  momentum. The two-sided graph stays the best.

## Step size λ = 1 against 0.5

The step moves each sample a fraction λ toward its neighbours' average. All runs above use λ 0.5.
Here: two-sided graph, uniform weights, no new folds, 300 iterations, setup B
(`python run_experiments.py --set lambda --suffix _valid`).

| run | method | λ | relax s | edge CV | min angle p1 / p5 / median | folded | degenerate | moves held back |
|---|---|---|---|---|---|---|---|---|
| `py_eqs278k_flip_momentum_b09_300_valid` | momentum β 0.9 | 0.5 | 20.2 | **0.2126** | 24.09 / 34.45 / 49.91 | 303 | 7 | 59,911 |
| `py_eqs278k_flip_sym_nofold_momentum_b09_lam1_300_valid` | momentum β 0.9 | 1 | 23.0 | 0.2136 | **24.50 / 34.67 / 50.67** | **294** | 10 | 62,421 |
| `py_eqs278k_flip_sym_nofold_explicit_lam05_300_valid` | explicit | 0.5 | 19.9 | 0.2154 | 21.15 / 32.08 / 47.87 | 322 | **4** | 46,333 |
| `py_eqs278k_flip_sym_nofold_explicit_lam1_300_valid` | explicit | 1 | 19.4 | 0.2128 | 21.99 / 33.29 / 48.70 | 309 | 9 | 51,691 |

All three new runs pass `check_relaxed_sdt.py`. Momentum λ 1 needed 0 adaptive restarts.

The first momentum λ 1 run took 82 s. Its setup steps, which don't depend on λ, were also 3.5×
slower than usual, so the machine was busy, most likely with the stopped mean-value run still
exiting. The table gives the rerun, which produced the same mesh in 23.0 s.

**Observations:**
- **Explicit steps gain clearly from λ 1.** In the same 300 iterations, edge CV goes from 0.2154
  to 0.2128 and folds from 322 to 309, with better angles. Explicit λ 1 at 300 iterations is about
  as even as momentum λ 0.5 (0.2128 against 0.2126), but its angles are about 2° lower.
- **Momentum gains a little from λ 1.**
  - Better: the angles (p1 24.50, median 50.67) and the fewest folds of any run so far (294).
  - Slightly worse: edge CV (0.2136 against 0.2126) and degenerate triangles (10 against 7).
  - It is similar to momentum λ 0.5 run longer (2000 iterations: edge CV 0.2158, median 51.3,
    303 folds), but in 300 iterations.
- **No instability.** λ 1 doesn't oscillate here, with or without momentum. Momentum is stable
  up to about 2(1 + β), and the no-new-folds rule holds back the moves that would fold a triangle
  (slightly more of them at λ 1).
- **Cost.** About the same time per iteration; the fold check and the projection didn't get more
  expensive.

## Recommendation (so far)

Single-level momentum with **uniform weights on the two-sided graph** (`--explicit_symmetric_graph`)
is still the best setting: 300 iterations, about 20 s of relaxation. With λ 0.5 it has the most
even edges (edge CV 0.2126). With `--explicit_lambda 1` it has better angles and the fewest folds
(294), for slightly less even edges (0.2136); which one is better depends on what matters more.

```
python run_relax.py \
    --output_dir ../output/relaxation_experiments/py_eqs278k_flip_momentum_b09_300_valid \
    --coarse_subdiv_relax_method momentum --momentum_beta 0.9 \
    --coarse_subdiv_relax_no_new_folds --explicit_symmetric_graph \
    --explicit_global_proj --equal_area_samples 277700 --delaunay_flips \
    --explicit_max_iter 300 --snapshot_iters 30,100,300
```

With the new default, the best setting needs `--explicit_symmetric_graph` explicitly.
