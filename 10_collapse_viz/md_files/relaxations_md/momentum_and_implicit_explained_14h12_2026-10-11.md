# Momentum and implicit relaxation, explained

This explains the two faster relaxation methods in plain terms, starting from the explicit
method they both build on. Code is in `relaxation_scripts_python/` (`relax_explicit.py`,
`relax_momentum.py`, `relax_implicit.py`); measured results are in
`relaxation_momentum_implicit_13h57_2026-10-11.md`.

## The setting

The samples are the vertices of a triangle mesh lying on the fine MAT. Each vertex i belongs
to one structure, a sheet or a curve (seam / boundary). Π_i(p) is the closest point to p on
that structure.

Each vertex has graph neighbours j with weights w_ij ≥ 0 (uniform: all 1). Its **neighbour
mean** is

    m_i(x) = Σ_j w_ij x_j / Σ_j w_ij

The relaxation wants every free vertex to sit at the mean of its neighbours, i.e. to make the
mesh as even as the surface allows. The **energy** E = ½ Σ_edges w_ij |x_i − x_j|² measures
how far from that it is. Lower is more even.

## 1. Explicit (the starting point)

    y_i = x_i + λ (m_i(x) − x_i)        move a fraction λ (0.5) towards the mean
    x_i ← Π_i(y_i)                      put it back on its structure

Every vertex uses its neighbours' **old** positions, so in one step a vertex only "hears" from
its direct neighbours.

**Why it is slow.** Split the mesh's unevenness into patterns of different sizes:
- **Small patterns** (a few vertices wide, e.g. one bunched-up spot) are smoothed in a few
  steps, because the neighbours see them directly.
- **Large patterns** (one half of a sheet denser than the other) must be passed along vertex by
  vertex. A pattern k vertices wide needs about k² steps.

On 277,700 vertices that is thousands of steps. The slow drift seen in the runs is exactly
that: almost every vertex keeps moving a tiny bit, all in the same direction, for thousands of
steps.

## 2. Momentum

    y_i = x_i + λ (m_i(x) − x_i) + β (x_i − x_i^prev)
    x_i ← Π_i(y_i)

x_i − x_i^prev is the vertex's move in the last step, its "velocity". β (0.9) is how much of it
is kept.

**Intuition.** Think of each vertex as a heavy ball rolling downhill on the energy instead of a
point that only ever takes a fixed fraction of the remaining way.
- **Large, slow patterns:** the pull points the same way step after step, so the velocity
  builds up and the vertex covers ground much faster.
- **Small, fast patterns:** the pull changes sign from step to step, so the velocity cancels
  out and nothing overshoots wildly.

**How much faster.** For this kind of averaging, a pattern k vertices wide needs about k steps
instead of k². In the runs, momentum reaches in 100 steps what the explicit method reaches in
1000.

**Safety rules in the code:**
- **Held-back vertices start from rest.** If the no-new-folds rule held a vertex back, it did
  not move, so its velocity is 0. It does not "remember" a move that was refused.
- **Adaptive restart.** If a whole step raises the energy, all velocities are set to 0. This
  stops the balls from rolling past the bottom and oscillating. In the runs on ABC 00040057 it
  never triggered: the energy went down every step.

**What it does not fix.** The total energy goes down every step, but the largest single move
kept bouncing between about 1e-5 and 1e-4 × diagonal instead of shrinking. So the
"largest move < 1e-7 × diagonal" stop is never reached, even after 7000 steps. A small set of
vertices seems to keep swinging; which ones has not been checked.

## 3. Implicit (backward Euler)

Use the neighbours' **new** positions in the averaging:

    y_i = x_i + τ (m_i(y) − y_i)        for all free vertices at the same time
    x_i ← Π_i(y_i)

Because y appears on both sides, this is one linear system for all free vertices at once.
Multiplying by d_i = Σ_j w_ij gives

    (1 + τ) d_i y_i − τ Σ_j w_ij y_j = d_i x_i

where fixed vertices keep y_j = x_j and their terms go to the right-hand side. It is solved with
conjugate gradients: an iterative solver for symmetric systems, about 32 rounds per step at
τ = 10, all in numba.

**Intuition.** Solving the system lets every vertex hear from **every** other vertex in one
step, not only from its neighbours. A pattern of any size is damped by the factor
1 / (1 + τ·μ), where μ is small for large patterns and large for small ones. With τ = 10, even
a large pattern shrinks substantially every step. In principle this is the fastest way to remove
the large-scale drift.

**τ is the step length:**
- small τ behaves like the explicit step;
- τ → ∞ solves "every vertex at its neighbour mean" in one go. That is the C++
  `solve_project`.

**Why it failed here.** The averaging is done in **3D space**, not on the surface. The MAT
sheets are curved. A large implicit step pulls each vertex towards its neighbours along a
straight line through space, cutting across the bend of the sheet.

Projecting back onto the sheet then moves vertices sideways, unevenly: they bunch up where the
sheet curves. On ABC 00040057 the mesh got **worse** instead of better:
- edge CV went from 0.32 to 0.46 at τ = 10, and to 0.84 at τ = 100;
- the smallest angles collapsed;
- the no-new-folds rule refused millions of moves.

The C++ `solve_project` failed for the same reason, with even larger steps. Explicit and
momentum steps are small enough that each step stays close to the surface, so the projection
only nudges.

**What would fix it:** doing the implicit averaging in each vertex's tangent plane (sliding
along the surface) instead of in 3D. That is not implemented.

## The no-new-folds hold-back, and the test with it off

All three methods share the hold-back. After the candidates Π_i(y_i) are made, any triangle
that faced the right way and would flip has its moving corners reset to where they were. This
repeats until no new fold appears. It keeps the mesh from folding, but it fights large steps:
- in 300 implicit steps at τ = 10 it refused 1.28M moves;
- at τ = 100 it refused 28.4M;
- over 7000 momentum steps it refused 1.39M.

The test with the hold-back **off** asks two questions:
1. How much of each method's behaviour is the hold-back?
2. Does the implicit step look better or worse when its moves are not refused?

Runs, all in `output/relaxation_experiments/`:
- `py_eqs278k_flip_nohold_global_2k_valid` (explicit, 2000 steps)
- `py_eqs278k_flip_momentum_b09_nohold_2k_valid` (momentum, 2000 steps)
- `py_eqs278k_flip_implicit_t10_nohold_300_valid` (implicit τ 10, 300 steps)

The comparison is in `relaxation_momentum_implicit_13h57_2026-10-11.md`.
