# Experiment: "solve L x = 0, then project" vs the Newton solver

Date: 2026-09-27. Mesh: ABC 00040057, level 3 (88,124 samples, 173,760 faces),
qslim to 200 faces, Release. All runs are in `output/relaxation_experiments/`,
one folder per run, with a `.log` next to each folder.

## The methods

- **Newton** (`--subdiv_relax_method newton`, the default;
  `subdiv_relax.cpp`). Vertices move only along their own sheet or seam
  (tangent-plane solves with a line search). It stops when one plain
  relaxation step moves nothing.
- **Solve + project** (`--subdiv_relax_method solve_project`;
  `subdiv_relax_solve_project.cpp`, a copy of `subdiv_relax.cpp` with only the
  solver replaced). One 3D sparse solve of `L x = 0` per pass (curves, then
  sheets with the curves fixed), with the fixed vertices as boundary values.
  Then every vertex is projected onto its own structure. No iteration.
- **Curve anchors** (`--subdiv_relax_curve_anchors N`, solve + project only).
  In every connected group of seam/boundary vertices, N vertices are fixed at
  their seed positions. They are chosen by farthest-point sampling on seed arc
  length, so they come out evenly spaced along the group.

All runs use the same graph, projection, loop pinning (one vertex per closed
seam loop with no junction) and checks.

## Files in each run folder

Folder names are kept short because the mesh stem is long. Windows paths over
260 characters failed to write. The tracker now also writes with the Windows
long-path prefix, so an over-long path can no longer fail.

| file | content |
|---|---|
| `subdiv_fine_seed_<stem>.obj` | subdivided MAT before relaxation |
| `subdiv_fine_relaxed_<method>_<stem>.obj` | after relaxation (`<method>` = `newton` or `solve_project`) |
| `subdiv_fine_relaxed_solve_project_with_anchors_<stem>.ply` | anchored runs only: the relaxed mesh (grey) plus a sphere at every anchor, one color per seam/boundary struct id (38 structs); PLY for vertex colors |
| `subdiv_deformed_relaxed_<method>_<stem>.obj` | relaxed samples at their tracked positions on the decimated mesh |
| `subdiv_graph_<stem>.slg`, `subdiv_<stem>.sdt` | graph and full tracker output |

Files named `subdiv_fine_<stem>.obj` or `subdiv_deformed_<stem>.obj`, with no
`relaxed_` tag, are left over from the first runs, before the naming change.
They are relaxed too, with the same method as their folder.

## Results

| level 3 | Newton | solve + project | + 10 anchors per seam | + 50 anchors per seam |
|---|---|---|---|---|
| folder | `l3_newton` | `l3_sp` | `l3_sp_a10` | `l3_sp_a50` |
| extra fixed curve vertices (anchors, incl. the 26 loop pins) | 0 | 0 | 354 (380) | 1,686 (1,712) |
| time | 2.1 s | 0.1 s | 0.1 s | 0.1 s |
| one more plain step moves (0 = at rest) | 3e-16 x diag | 0.044 x diag | 0.0084 x diag | 0.0057 x diag |
| 3D solution off the MAT before projection, curves max | – | 0.117 x diag | 0.057 x diag | 0.022 x diag |
| largest vertex move | 0.036 x diag | 0.84 x diag | 0.15 x diag | 0.024 x diag |
| edge-length CV (seed 0.368) | 0.373 | 2.58 | 1.23 | **0.360** |
| smallest angle, 1st percentile (seed 16.1 deg) | 9.6 | 0 | 0 | **19.5** |
| smallest angle, 5th percentile (seed 19.8 deg) | 17.8 | 0 | 0.5 | **26.1** |
| median smallest angle (seed 40.7 deg) | 41.7 | 0 | 38.9 | **43.2** |
| degenerate triangles | **1** | 79,998 | 3,452 | 14 |
| flipped vs seed | **0** | 71,342 | 7,102 | 12 |
| every vertex on its own structure | yes | yes | yes | yes |
| tracker counters through 1168 collapses | all 0 | all 0 | all 0 | all 0 |

"Smallest angle" is each triangle's smallest angle; the table gives its
percentiles over all triangles.

## What each step showed

1. **Plain solve + project fails badly.** `L x = 0` makes each free vertex the
   plain 3D average of its neighbours, and only the fixed vertices hold
   anything in place.
   - 26 of the 38 seam groups are closed loops whose only fixed vertex is the
     pinned one. The only solution is the whole loop at that point.
   - Open seams become straight chords between their junctions.
   - The sheets inherit the collapsed rims. Most triangles degenerate.
2. **10 anchors per seam.** The loops no longer collapse. But each stretch
   between anchors is about 200 vertices long, and the solve puts them evenly
   on the straight chord between the two anchors. Where the seam curves, that
   chord is up to 0.057 x diag away, so projection bunches the points. There
   are still 3,452 degenerate triangles.
3. **50 anchors per seam.** Stretches are about 5x shorter, and the chord error
   drops to 0.022 x diag. The result is the best of all runs on spacing and
   angles: edge CV 0.360, 1st percentile 19.5 deg, median 43.2 deg. It is
   better than the seed and better than Newton on these numbers. The costs:
   - It is **not a resting state**: one more plain step still moves up to
     0.0057 x diag.
   - 1,686 of about 10,000 seam vertices (17%) are frozen at their seed
     positions.
   - It has 14 degenerate and 12 flipped triangles, against Newton's 1 and 0.

## Why Newton scores worse on the worst angles

Newton reaches the exact resting state of the rule. On sheets with concave
rims, that resting state pulls some sheet vertices onto the rim, next to a seam
vertex. This is the rim-crowding effect in `subdiv_relax_results.md`, and it
lowers the 1st percentile of the smallest angle (16.1 -> 9.6 deg). Solve +
project does not converge, and its curves stay close to the seed because 17%
of them are anchored, so it does not reach that crowded state.

## Open questions for the next step

- Is the goal the exact resting state (Newton), or the best sample quality
  (anchored solve + project looks better here)? These are different targets on
  this mesh.
- The anchor count is a free parameter. 10 is too few; 50 works at level 3.
  The right number likely depends on the level and the seam curvature (for
  example one anchor every k seam vertices), which is worth testing at levels
  4-5.
- A combination: anchored solve + project as the starting point, then Newton
  with the anchors kept fixed. That would give a true resting state of the
  anchored problem.

## Adaptive anchors (Douglas-Peucker), 2026-09-27

**Why.** A fixed count per seam is wrong: short seams and straight seams need
fewer anchors than long, curved ones. Between two fixed vertices, the 3D solve
puts the seam vertices on the straight chord joining them. So an anchor is
needed exactly where the seam leaves that chord by more than a tolerance.

**Method** (`--subdiv_relax_anchor_tol t`, the default for solve_project,
`t = 1e-3`; `--subdiv_relax_curve_anchors N` still forces a fixed count).
Every seam/boundary group is split into chains:
- Chain ends are junctions, branch vertices (3+ seam neighbours), free ends, or
  one vertex per closed loop.
- Branch vertices, free ends and the loop vertex are anchored.
- Along each chain, Douglas-Peucker on the seed positions: the vertex farthest
  from the current chord becomes an anchor if it is more than `t x diag` away,
  then recurse on both halves.

This guarantees the chord error is below `t` everywhere. The measured
off-the-MAT distance of the curve solve confirms it: 0.0097 / 0.0030 / 0.00099
x diag for `t` = 1e-2 / 3e-3 / 1e-3.

On this mesh there are 38 chains: 26 closed loops and 12 open seams between
junctions. At `t = 1e-2`, only 32 of the 38 structures get an anchor. The
straight open seams need none, because their junction ends already hold them.

| level 3 | Newton | fixed 50 per seam | adaptive 1e-2 | adaptive 3e-3 | adaptive 1e-3 |
|---|---|---|---|---|---|
| folder | `l3_newton` | `l3_sp_a50` | `l3_sp_tol1e-2` | `l3_sp_tol3e-3` | `l3_sp_tol1e-3` |
| anchors (incl. loop vertices) | 0 | 1,712 | 202 | 375 | 558 |
| one more plain step moves (x diag) | 3e-16 | 0.0057 | 0.0020 | 0.00044 | 0.00028 |
| edge-length CV (seed 0.368) | 0.373 | 0.360 | 0.360 | **0.354** | **0.354** |
| smallest angle, 1st percentile (seed 16.1 deg) | 9.6 | **19.5** | 16.8 | 17.9 | 17.9 |
| smallest angle, 5th percentile (seed 19.8 deg) | 17.8 | **26.1** | 22.1 | 23.2 | 23.2 |
| median smallest angle (seed 40.7 deg) | 41.7 | **43.2** | 41.8 | 42.4 | 42.5 |
| degenerate triangles | **1** | 14 | 15 | 2 | 2 |
| flipped vs seed | **0** | 12 | 17 | 5 | 5 |
| tracker counters | all 0 | all 0 | all 0 | all 0 | all 0 |

**Reading.**
- **Adaptive 3e-3 is the best trade-off.** It uses 4.6x fewer anchors than
  fixed 50 (375 vs 1,712) and has the best edge-length spread. It has far fewer
  bad triangles (2 degenerate / 5 flipped vs 14 / 12), and it is 13x closer to
  a resting state.
- **Tightening to 1e-3 changes almost nothing**, so the result has converged in
  `t`.
- **Fixed 50 keeps the best angle percentiles.** It freezes 4.6x more seam
  vertices at their seed positions, and the seed angles near seams were
  already fair. It pays for that with more degenerate and flipped triangles.
