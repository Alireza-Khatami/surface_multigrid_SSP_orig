# Relaxing the subdivided samples: problems met, tests, fixes

Test case throughout: ABC 00040057 (MAT: 1,506 vertices, 2,715 faces, 68
structures, only 6 junction vertices). Release builds.
Related: `subdiv_relax_plan.md`, `subdiv_relax_results.md`,
`subdiv_relax_sequential_passes.md`.

## 0. Goal

Midpoint subdivision copies the triangle quality of the original MAT, so
samples cluster where the MAT has small faces. The goal is to move the
subdivided vertices so they are spread more evenly, while:

- the geometry stays exactly the MAT;
- every vertex stays on its own structure;
- connectivity is unchanged.

Method: a directed Laplacian graph plus iterative smoothing, with projection
back onto the MAT after every step.

| vertex | pulled by |
|---|---|
| junction | nothing (fixed) |
| seam / boundary | seam or boundary neighbours sharing its structure id, and junctions |
| sheet | any neighbour sharing its sheet id |

Stopping rule: repeat until one more step moves nothing (1e-12 x bbox diagonal).

## 1. Build the graph and check it

**Did.** Built the graph from the subdivided mesh edges with the rule above.
Roles are assigned by priority: junction > seam/boundary > sheet.

**Test.** An independent re-derivation of every graph entry, straight from the
`.ma_struct` struct types, plus:

- two-way sheet-sheet and seam-seam pairs;
- empty junction rows;
- no entries that are not mesh edges.

**Result.** 0 mismatches at every level (2,019,517 entries at level 4). The graph
was correct from the start.

## 2. Plain smoothing (average, then project) works but is far too slow

**Did.** Plain iteration `x <- Pi(x + 0.5 L x)`, where Pi is the closest point on
the vertex's own structure. It uses one BVH per structure id with an exact
point-triangle or point-segment distance.

**Test.** Ran it to the stopping rule at levels 0 and 1.

**Result.**

| level | samples | steps to converge | time |
|---|---|---|---|
| 0 | 1.5k | 9,853 | 5 s |
| 1 | 5.7k | 37,596 | 39 s |

**Why.** Smoothing is diffusion: information moves one edge per step, so the
number of steps grows with the square of the sheet's width in edges. That is
about 4x more steps per level on 4x more vertices, so 16x the time per level.
Level 4 would take an estimated 44 hours, and level 5 weeks.

**Fix.** Solve for the balance directly. This is a Newton step: in each
vertex's tangent plane (face plane for sheets, edge direction for seams), solve
the sparse linear system `deg_i u_i - sum_j T_i^T T_j u_j = T_i^T sum_j (x_j - x_i)`,
take the step, project, and repeat. Seams are solved first, then sheets with
the seams fixed. The seams do not depend on the sheets, so this ordering loses
nothing.

## 3. Newton on sheets never converged: a vertex bounced forever

**Test.** Level 0. Seams converged in 9 iterations, but the sheet pass hit the
1000-iteration cap with a constant max move of 7e-5 x diag.

**Diagnosis.**

- Printed the worst vertices, then traced the worst one (vertex 1114) every
  iteration.
- It alternated between faces 2692 and 1435 with a period of 6 iterations.
- It sat next to a *valley crease* between the two faces. In each face's
  tangent plane, the balance point lies just across the edge. So the step
  overshoots into the other face, and from that face's plane it overshoots
  back.
- The true resting point is on the crease itself.

**First fix attempt: per-vertex damping.** A vertex whose move reverses
direction gets a smaller step.

- **Failed.** Moves at rounding level (about 1e-17) flip sign at random, so
  healthy vertices were damped to a standstill.
- Restricting it to reversals above the tolerance still left a longer cycle
  (period 6 to 30).
- **Dropped:** it is a heuristic with no guarantee.

**Fix: energy line search.** Relaxation is equivalent to minimising the graph's
Dirichlet energy `E = 1/2 sum |x_i - x_j|^2`, with each vertex constrained to its
own structure. The linear system above is that energy's Hessian in the tangent
planes. So:

- keep the Newton direction;
- accept a step only if `E` decreases (Armijo backtracking).

A cycle would need the energy to return to an earlier value, so cycling is now
impossible.

**Result.** The sheet pass stopped. The tangential residual was 1e-12 at every
vertex inside a face.

## 4. "Stopped" was not the same as "converged"

**Test.** Added the literal definition as a check: from the result, take one
plain step `x <- Pi(x + 0.5 L x)` and measure the largest move.

**Result.** 0.0055 x diag at vertex 1162, so it was *not* a fixed point. The
residual was tiny only because the check skipped vertices sitting exactly on a
polyline kink or crease.

**Diagnosis.** A vertex exactly at a kink (seam) or on a crease (sheet) uses
the frame of whichever edge or face the projection happened to return. So
Newton cannot move it onto the neighbouring edge or face that its pull points
into. The plain step can, because it uses the full 3D pull.

**Fix, part 1: alternation.** Alternate Newton rounds with one plain step. The
plain step moves stuck vertices onto the right element, and Newton then
continues. The pass ends only when a plain step moves nothing, which is the
definition itself.

**Result.** At level 0, one plain step from the result moves 1.8e-16. It is a
true fixed point.

## 5. Newton and plain smoothing ended at different answers

**Test.** Compared the Newton result with the plain-smoothing result at levels
0 and 1.

**Result.** Up to 0.015 x diag apart, even though both passed the fixed-point
test.

**Diagnosis 1: sliding loops.**

- Counted connected components per role.
- 26 of the 38 seam components (1,092 of 1,249 seam vertices) are closed loops
  with no junction. The mesh has only 6 junction vertices.
- Such a loop can slide around itself, so its resting position is not unique.

**Fix.** Pin the lowest-index vertex of every group that nothing fixed pulls
on. That is an original MAT vertex, and it makes each loop's system well-posed.

**Result.** The difference remained (0.0147 x diag), so there was a second
cause.

**Diagnosis 2: the frame choice.** The hypothesis was that Newton got stuck at
kinks early and settled around them.

**Fix, part 2: active-set frames.** At a kink, use the incident seam edge the
pull points along. On a crease, use the incident face the pull enters, or else
slide along the crease edge. If the pull points into none of them, hold the
vertex for that iteration.

**A testing pitfall found here.** The first run showed *no change*. A leftover
test process from an earlier stopped run was locking the executable, so the
link had failed silently and the old binary ran. After killing it and
rebuilding:

- Newton converges in 0.04 s at level 0;
- only 513 vertices still differ (mean 1.4e-4 x diag);
- Newton's result now has *lower* energy than plain smoothing's.

## 6. The resting state is not unique, and that is a property of the problem

**Test 1.** Started Newton from the plain-smoothing result.

**Result.** It moved it by only 3e-10, so the plain-smoothing result is also a
true fixed point. Plain smoothing had genuinely finished; it was not a
convergence failure.

**Test 2.** Ran plain smoothing itself with step 0.25 instead of 0.5.

**Result.** It ended 0.0084 x diag from the step-0.5 result (476 vertices).

**Conclusion.**

- Seam polylines have a small kink at every original vertex, and sheets have
  creases, so the constrained energy has several local minima. A typical case
  is a loop shifted by one polyline segment.
- Even the literal "average, then project" answer depends on the step size.
- "Agree with plain smoothing" is therefore not a valid correctness test.

**Adopted test.** Every result must pass the definition: one plain step moves
no vertex more than 1e-10 x diag. Plus the structure checks:

- junctions bit-identical to the seed;
- position equals barycentric interpolation;
- valid barycentrics;
- each face or edge belongs to the vertex's own structure.

Among the three fixed points found, Newton's has the lowest energy.

## 7. Scaling up and the full pipeline

**Tests.**

- Levels 1–5 in the standalone test.
- The full decimation pipeline at levels 4 and 5.
- `--no_subdiv_relax` byte-compared against the pre-relaxation output.
- Two independent Python verifiers:
  - `.sdt` version 2: struct sets, and every vertex on its own structure;
  - `.slg` graph: full rebuild from the `.ma_struct`.

**Results.**

| | level 4 (350k) | level 5 (1.4M) |
|---|---|---|
| relaxation time | 68 s | 561 s |
| one plain step from the result | 2.9e-16 x diag | 3.5e-16 x diag |
| consistency checks | all 0 | all 0 |
| tracker counters through 1168 collapses | all 0 | all 0 |
| Python verifiers | pass | pass |

- With `--no_subdiv_relax`, the `.sdt` is byte-identical to the old output.
- The seed OBJ is byte-identical to the old fine OBJ.

## 8. Remaining limitations

- **Worst angles get worse.** At level 5:
  - the median per-triangle min angle improves (40.7 -> 41.9 deg);
  - the worst 1% get worse (16.1 -> 11.6 deg);
  - 13 triangles become degenerate and 30 flip.

  Cause: on a sheet whose rim is concave, some sheet vertices are pulled out of
  the sheet, so the projection stops them on the rim next to a seam vertex.
  This follows from the rule itself; it is not a bug.
- **The global edge-length spread barely changes** (CV 0.368 -> 0.366).
  Samples cannot move between sheets, and every original face keeps the same
  number of samples, so the rule evens out spacing mostly locally.
- **Levels 6–7 need a faster sheet solve.** Cost grows about 8x per level
  (the sparse LDLT). Options are multigrid over the nested subdivision levels,
  or plain smoothing for a fixed number of steps if the exact resting state is
  not required.

## Summary of the chain

1. Plain smoothing is correct but diffusion-slow, so switch to Newton in the
   tangent planes.
2. Newton cycled at a valley crease; damping failed, so add an energy line
   search, which makes cycling impossible.
3. It stopped, but not at a fixed point (kinks and creases), so alternate
   with plain steps and use the plain step as the stopping test.
4. Newton and plain smoothing disagreed:
   - closed seam loops slide, so pin one vertex each;
   - kink and crease frames were wrong, so use active-set frames.
5. They still disagreed. Tests showed several valid fixed points, and even
   plain smoothing depends on its step size. So the test became the fixed-point
   definition plus the structure checks, not agreement between methods.
6. It scales to level 5 with every check at 0. The open issues are rim
   crowding and the cost at levels 6–7.
