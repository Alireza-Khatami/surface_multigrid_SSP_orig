# Why the relaxation runs curves first, then sheets

Context: `subdiv_sample_tracker/subdiv_relax.cpp`, Newton solver. See
`subdiv_relax_plan.md` and `subdiv_relax_results.md`.

The solver makes two passes. First it relaxes the curve vertices (seams and
boundaries) until they stop changing. Then it relaxes the sheet vertices, with
the curves held at their final positions. Everything could be relaxed at once,
and that would not change what counts as a correct answer. Two passes are used
because they are cheaper and keep the math clean.

## Sequential gives the same kind of answer

The graph rule makes the edges one-way: sheets feel curves, but curves never
feel sheets. So the curve rows of the Laplacian contain only curve and junction
vertices, and the whole system is block-triangular:

```
[ L_cc   0   ] [x_c]      curves: depend only on curves (+ fixed junctions)
[ L_sc  L_ss ] [x_s]      sheets: depend on sheets and on curves
```

A state is a fixed point of the joint iteration exactly when:

- the curves are at a curve fixed point, and
- the sheets are at a sheet fixed point given those curves.

Relaxing curves first and then sheets reaches a point in that same set, so the
ordering loses nothing. The plain Jacobi mode (`RelaxSolver::Jacobi`) already
updates everything at once, and it passes the same stopping test.

## Why the passes are separate anyway

### 1. Each pass has a real energy; the joint problem does not

Within the curve pass, and within the sheet pass with the curves held fixed,
the edges are two-way. So each pass minimises a proper energy, the sum of
squared edge lengths `E = 1/2 sum |x_i - x_j|^2`. That gives two things:

- **A symmetric positive-definite system.** The fast sparse solver (LDLT) needs
  this.
- **An energy that must go down on every accepted step.** This is what stops the
  cycling at sheet creases (the Armijo line search).

Solved jointly, the one-way sheet<-curve coupling makes the matrix
non-symmetric, as in the block form above. That has two costs:

- It needs a general sparse solver (LU) instead of LDLT.
- There is no single energy that the joint step is guaranteed to lower. A merit
  function would have to be made up, for example the sum of the two energies. A
  step that lowers that sum can still push the curves in a direction that
  raises the curve energy. So the guarantee that the iteration cannot cycle
  would be lost.

### 2. Cost

The two passes have very different sizes. Figures from level 5 on ABC 00040057
(1.4M samples):

| pass | unknowns | iterations | time per iteration |
|---|---|---|---|
| curves | about 40k | 104 | about 0.02 s |
| sheets | about 2.7M | 42 | about 14 s |

- **Sequential.** The sheets start from the final curve positions and need 42
  expensive solves.
- **Joint.** Every iteration pays for the big sheet solve while the curves are
  still moving, so it needs at least the curves' 104 iterations. That is about
  2.5x slower. Much of that sheet work is also wasted, because it fits the
  sheets to rims that are about to move.

## Caveat

The fixed point is not unique (see `subdiv_relax_results.md`). A joint solver
could therefore end on a different, equally valid stopping point than the
sequential one. It would not be more correct.

## If everything should be relaxed at once

A joint mode can be added. Because the matrix is block-triangular, the simplest
correct version still solves the curve block and then the sheet block inside
each iteration. That is the same sequential scheme, just interleaved per
iteration instead of per pass.

# Why not just apply the Laplacian graph to the vertices

Applying the Laplacian directly is already in the code: it is the Jacobi mode
(`RelaxSolver::Jacobi`), `x <- Pi(x + 0.5 L x)` repeated. It gives a correct
result, and the plain step is still what decides the answer. The Newton solver
only stops when one plain Laplacian step moves nothing. Newton is used only
because the plain version needs far too many steps to reach that point.

## Why plain steps are slow

One Laplacian step moves each vertex toward the average of its neighbours, so
information travels one edge per step. Smoothing behaves like heat diffusion:

- small wiggles die out in a few steps;
- a large-scale error, such as a whole sheet sitting slightly off balance,
  needs many steps, and the number grows with the square of the sheet's width
  in edges.

Each subdivision level doubles that width, so it needs about 4x more steps, on
4x more vertices, for about 16x the time per level.

Measured on ABC 00040057 with Jacobi, run until one step moves under
1e-12 x diag:

| level | samples | Jacobi steps | Jacobi time | Newton time |
|---|---|---|---|---|
| 0 | 1.5k | 9,853 | 5 s | 0.04 s |
| 1 | 5.7k | 37,596 | 39 s | 0.12 s |
| 4 | 350k | about 2.4M (est.) | about 44 h (est.) | 68 s |
| 5 | 1.4M | about 10M (est.) | weeks (est.) | 561 s |

Levels 4 and 5 are extrapolated at about 4x steps and 16x time per level. Jacobi
was only run to the end at levels 0 and 1.

## What the linear solve buys

Solving `A u = rhs` finds each pass's equilibrium, given the current tangent
planes, in one step. So the whole sheet balances at once instead of creeping
one edge per step. The solver iterates only because the surface bends: the
frames change as the vertices move. That is 42 solves at level 5 instead of
millions of plain steps.

## When plain steps are the right tool

If the exact resting state is not needed, only smoother samples, then a fixed
number of plain steps is simpler and cheap: for example 20-50 steps. That
smooths the local unevenness left by subdivision, but it will not reach "stops
changing". The requirement here is to relax until nothing changes, and plain
steps cannot get there at these sizes.

A middle option is to speed up the plain step with multigrid over the nested
subdivision levels, or with Chebyshev acceleration. That keeps the "apply the
Laplacian" structure without the linear solve, but it is more work to make exact
with the projection. It is worth doing only if the solver's memory becomes the
limit at levels 6-7.
