# Faster relaxation: momentum and the implicit step

Code (all in `relaxation_scripts_python/`):

| file | role |
|---|---|
| `relax_explicit.py` | the explicit relaxer (ported from C++, byte-identical to it). Setup, free set, projection, no-new-folds hold-back, commit, checks, report. |
| `relax_momentum.py` | `MomentumRelaxer`: the explicit relaxer with heavy-ball momentum. Also holds `project_targets`. |
| `relax_implicit.py` | `ImplicitRelaxer`: the explicit relaxer with a backward-Euler (implicit) step. |

To use them, pass `--coarse_subdiv_relax_method momentum` or `implicit` to `run_relax.py`. The
two methods are Python-only; the C++ has no counterpart.

## Why the explicit relaxation is slow

One explicit iteration does this for every free vertex i:

    y_i = x_i + λ (mean_i(x) − x_i)          mean_i(x) = Σ_j w_ij x_j / Σ_j w_ij
    x_i ← Π_i(y_i)                           Π_i = closest point on the vertex's own sheet / curve

followed by the no-new-folds hold-back and the commit.

The neighbour mean is taken at the **old** positions, so information travels one ring of
neighbours per iteration.
- **Fine wrinkles** (a few rings wide) settle in a few iterations.
- **A large-scale imbalance** across the whole mesh (more points on one side than the other)
  has to diffuse across the mesh. The iterations it needs grow with the **square** of the
  mesh's width in vertices.

On ABC 00040093 (277,700 samples, 3000 iterations) this shows as a slow drift. Between
iterations 2000 and 3000:
- almost every vertex still moves (272,003 of 277,700 by more than 1e-4 × diagonal);
- each step moves at most about 4e-6 × diagonal, so the "largest move < 1e-7 × diagonal" stop
  is never reached;
- the energy keeps falling slowly.

## Method 1: momentum (`relax_momentum.py`)

Add the last move to the step:

    y_i = x_i + λ (mean_i(x) − x_i) + β (x_i − x_i^prev)
    x_i ← Π_i(y_i)

x^prev is where the vertex started the previous iteration, so x − x^prev is the last committed
move (the "velocity"). β ∈ [0, 1) is how much of it is kept, 0.9 by default.

- **What it buys:** a large-scale drift keeps accumulating speed instead of restarting from
  zero every step. For an averaging iteration like this one, the iterations it needs drop from
  about width² to about width (the heavy-ball / Chebyshev result for linear iterations).
- **Held-back vertices:** a vertex held back by the no-new-folds rule did not move
  (x = x^prev), so its velocity is 0 and it starts again from rest.
- **Adaptive restart** (on by default, `--momentum_no_restart` turns it off): if a step raises
  the energy Σ w_ij |x_i − x_j|², every velocity is set to 0 for the next step. This stops
  momentum from overshooting and oscillating.
- **Cost:** about the same as an explicit step, plus one energy evaluation for the restart test.

`--momentum_beta B` sets β (0 gives back the explicit method).

## Method 2: the implicit step (`relax_implicit.py`)

Take the neighbour mean at the **new** positions:

    y_i − x_i = τ (mean_i(y) − y_i)          for every free vertex i

Multiplying by d_i = Σ_j w_ij gives one linear system for all free vertices together:

    (1 + τ) d_i y_i − τ Σ_{j free} w_ij y_j  =  d_i x_i + τ Σ_{j fixed} w_ij x_j

Fixed vertices (junctions, pinned, unmapped) keep y = x and appear on the right-hand side. Then,
as in the explicit step:

    x_i ← Π_i(y_i)       + no-new-folds hold-back + commit

**What it buys.** Write the shape error in Laplacian eigenmodes, where a mode with eigenvalue μ
is a pattern of spatial frequency ~√μ.
- **Explicit step:** a mode shrinks by (1 − λμ) per step, which is barely 1 for the small μ of
  large-scale modes. That is the slow drift.
- **Implicit step:** a mode shrinks by 1 / (1 + τμ), and τ can be large without the step
  becoming unstable. A large-scale mode with τμ ≈ 1 halves every step instead of every few
  thousand steps.

**τ** is the step length:

| τ | behaviour |
|---|---|
| small | close to the explicit step |
| 10 (default) | strong large-scale smoothing; targets stay near the surface |
| → ∞ | one solve of L y = 0, the C++ `solve_project`, which tore the mesh on this problem (about 47k flipped triangles) because the target ignores the surface completely |

Iterating with a finite τ and projecting every step keeps the targets close to the surface.

**Solving the system.** For the two-sided (symmetric) graph with symmetric weights (uniform or
cotan), the matrix is symmetric positive definite.
- It is solved per coordinate with **conjugate gradients**: matrix-free, Jacobi-preconditioned,
  warm-started at x, all in numba (no scipy).
- With τ = 10 that takes about 32 CG iterations per step on 277,700 vertices, about 0.4 s per
  step (vs about 0.06 s for an explicit step).
- The directed graph is not symmetric, so the implicit method refuses it.

`--implicit_tau T` sets τ. `--implicit_cg_tol` (1e-8) and `--implicit_cg_max_iter` (1000)
control CG, and a warning is logged if CG stops above 1000 × tol.

## How the code is shared (nothing is defined twice)

`ExplicitRelaxer.step()` is split into two parts:

1. **`_propose(trace)`**: makes every free vertex's candidate (nX, nFace, nEdge, nBary, move)
   and the projection statistics. The explicit version calls the original `step_kernel`, so
   the explicit method stays byte-identical to the C++. This was re-verified after the split:
   `verify_relax.py` on `src_qslim200_valid`, 1000 iterations, all identical.
2. **The rest of `step()`**: projection statistics, no-new-folds hold-back loop (`_new_folds`,
   `_hold_back`), commit (`_commit`), logging, snapshots, the convergence test. It is shared by
   all three methods.

The two new classes override only `_propose`:

| method | `_propose` |
|---|---|
| explicit | `step_kernel`: y = x + λ(mean − x), projected (original) |
| momentum | `_momentum_targets` (y with the momentum term), then `project_targets` |
| implicit | `_cg` solves for y, then `project_targets` (imported from `relax_momentum.py`) |

Everything else comes from the existing modules unchanged: setup (free vertices, pinning,
projector, weights), graph, fold reference, `finish()` checks, report, `.sdt` export, radius
validation.

## Results (ABC 00040057, bundle `src_qslim200_valid`, setup B)

The setup is B for every run:
- equal-area split to 277,700 samples, then Delaunay flips;
- uniform weights, two-sided graph, no new folds, global projection, λ = 0.5.

Quality was measured on each run's snapshots. Folded counts come from the run log. Times are
wall-clock from the start of the relaxation.

| method | iteration | time | edge CV | min angle p1 / p5 / median | folded |
|---|---|---|---|---|---|
| explicit (B) | 100 | 11 s | 0.2235 | 19.5 / 29.2 / 46.2 | 337 |
| explicit (B) | 1000 | 130 s | 0.2121 | 22.6 / 33.9 / 49.2 | 308 |
| explicit (B) | 2000 | 414 s | 0.2121 | 23.7 / 34.3 / 49.6 | 301 |
| **momentum β 0.9** | 30 | – | 0.2222 | 19.5 / 31.2 / 47.5 | – |
| **momentum β 0.9** | **100** | **14 s** | 0.2127 | 22.5 / 33.8 / 49.2 | 303 |
| **momentum β 0.9** | **300** | **28 s** | 0.2126 | **24.1 / 34.5 / 49.9** | 303 |
| momentum β 0.9 | 2000 | 172 s | 0.2158 | 24.6 / 34.7 / 51.3 | 303 |
| implicit τ 10 | 100 | 86 s | 0.4620 | 8.2 / 23.3 / 46.1 | 291 |
| implicit τ 10 | 300 | 164 s | 0.4636 | 8.0 / 23.3 / 46.7 | 282 |
| implicit τ 100 | 300 | 301 s | 0.8440 | 0.0 / 5.7 / 38.6 | 476 |

### Momentum: about 10× faster
- **Iteration 100 (14 s)** reaches what the explicit method reaches at iteration 1000 (130 s).
- **Iteration 300 (28 s)** is better than the explicit method at iteration 2000 (414 s) on
  every angle measure, with the same edge CV (0.213 vs 0.212) and about the same folds (303 vs
  301).
- **There were 0 adaptive restarts:** the energy never went up, so β = 0.9 is not overshooting
  here.
- **After iteration 300** the angles still improve slowly (median 49.9° → 51.3° at 2000), while
  the edge CV creeps up (0.2126 → 0.2158). Edge-length evenness and angles trade off, as in the
  explicit run.
- **Recommendation:** momentum with β = 0.9 and about 300 iterations.

### Implicit step: worse, not faster
The edge CV goes **up** from the seed (0.32 → 0.46 at τ 10, 0.84 at τ 100), and the smallest
angles collapse (p1 8° at τ 10, 0° at τ 100). The larger τ, the worse.

A likely reason, not checked separately: the implicit step solves the averaging in **3D**. A
large step pulls each vertex towards its neighbours along straight 3D chords that cut across
the curved MAT sheets; this is the same failure as the C++ `solve_project` (τ → ∞). Projecting
those targets back onto the sheets bunches vertices up where the sheet curves.

The run log shows the no-new-folds rule fighting the step: moves held back
- 1.28M in 300 iterations at τ 10,
- 28.4M in 300 iterations at τ 100,
- vs 0.39M in 2000 iterations for momentum.

Each implicit step also costs more: 0.4 s at τ 10 (32 CG iterations) and 1 s at τ 100 (108)
vs about 0.06–0.09 s.

Making an implicit step work here would need the averaging in the surface's tangent plane (a
per-vertex tangential solve), not in 3D. That is not done.

Run folders in `output/relaxation_experiments/`:
- `py_eqs278k_flip_momentum_b09_2k_valid`
- `py_eqs278k_flip_implicit_t10_300_valid`
- `py_eqs278k_flip_implicit_t100_300_valid`
- explicit: `py_eqs278k_flip_nofold_global_2k_valid`

Table: `py_experiments_results_accel_valid.md`.

## Hold-back (no-new-folds) off (added 14h12, 2026-10-11)

These runs repeat the setup with the no-new-folds rule switched off. The seed has 595 folded
triangles in every run. Times are wall-clock and noisy: other runs shared the machine.

| method | hold-back | iteration | edge CV | min angle p1 / p5 / median | folded | degenerate |
|---|---|---|---|---|---|---|
| explicit | on | 1000 | 0.2121 | 22.6 / 33.9 / 49.2 | 308 | 6 |
| explicit | off | 1000 | 0.2120 | 22.8 / 34.0 / 49.2 | 420 | 17 |
| explicit | off | 2000 | 0.2121 | 23.8 / 34.5 / 49.6 | 392 | 22 |
| momentum β 0.9 | on | 300 | 0.2126 | 24.1 / 34.5 / 49.9 | 303 | 7 |
| momentum β 0.9 | off | 300 | 0.2125 | 24.3 / 34.6 / 50.0 | 390 | 23 |
| momentum β 0.9 | on | 2000 | 0.2158 | 24.6 / 34.7 / 51.3 | 303 | 5 |
| momentum β 0.9 | off | 2000 | 0.2156 | 24.8 / 34.8 / 51.3 | 392 | 18 |
| implicit τ 10 | on | 300 | 0.4636 | 8.0 / 23.3 / 46.7 | 282 | 2085 |
| implicit τ 10 | off | 300 | 0.4607 | 8.7 / 25.3 / 47.0 | **2140** | 3083 |

What the hold-back does:
- **It barely changes edge evenness or angles.** With it off, edge CV is the same within 0.0003
  and the angles are 0.1–0.2° better. The few moves it refuses cost almost nothing in quality.
- **It keeps folds and degenerate triangles down.**
  - Explicit and momentum: with it on, folds go 595 → about 303 and stay there. With it off,
    folds go 595 → about 390, and there are 3–4× more degenerate triangles (18–23 vs 5–7).
  - Both methods still remove about a third of the seed's folds without it.
- **The implicit step is bad with or without it.** Without the hold-back, its edge CV is still
  0.46 and its folds jump to 2140. So the hold-back is not what made the implicit step fail; the
  3D averaging across curved sheets is (see `momentum_and_implicit_explained_14h12_2026-10-11.md`).

Recommendation: keep the hold-back on. The best setup stays momentum β 0.9 with the hold-back,
about 300–1000 iterations.

Runs, all in `output/relaxation_experiments/`:
- `py_eqs278k_flip_nohold_global_2k_valid`
- `py_eqs278k_flip_momentum_b09_nohold_2k_valid`
- `py_eqs278k_flip_implicit_t10_nohold_300_valid`
