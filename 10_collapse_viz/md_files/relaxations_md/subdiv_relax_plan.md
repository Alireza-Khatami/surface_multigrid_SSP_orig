# Plan: structure-aware relaxation of the subdivided samples

Status: implemented (2026-09-27). Where the implementation differs from the plan below, "Implementation notes" at the end wins.

## Goal

Midpoint subdivision copies the triangle quality of the original MAT. Sliver
faces and very uneven face sizes become clusters and gaps of samples. We want to
move the subdivided vertices so they are spread more evenly, while:

- the geometry stays exactly the original MAT (`gVO` / `gFO`);
- every vertex stays in its own structure (sheet, seam, boundary, junction);
- the subdivided connectivity (`M.F`) is unchanged, since the Laplacian users need it.

Method: iterative Laplacian smoothing on a special **directed** graph, with a
projection back onto the original MAT after every iteration.

## Where it runs

Inside `subdiv_tracker_init`, exactly after subdivision and before seeding:

```
gM = build_subdiv_mesh(gVO, gFO, nTarget);
build_struct_sets(gM, gFO, ms, gPal, gSet);      // struct IDs come from the dyadic carriers
subdiv_relax(gM, gVO, gFO, ms, gPal, gSet);   // NEW: runs to convergence,: moves V, rewrites fineFace/fineBary
... seeding (gFace / gBC / gBF / gBucket) reads fineFace/fineBary as today
```

The struct IDs are computed **before** relaxation, from the exact carriers, and
are never recomputed. Each vertex keeps its identity and its IDs. The
relaxation only changes where the vertex sits inside its own structure.

No iteration count and no step-size flag: the relaxation runs **until the
positions stop changing** (section 3). The only flag is an off switch,
`--no_subdiv_relax`, which reproduces today's output byte for byte (needed for
comparisons and the regression check).

## 1. Vertex roles

Each subdivided vertex gets exactly one role from its struct-set type mask
(`StructPalette::typeMask`), by priority:

```
JUNCTION  if mask has STRUCT_MASK_JUNCTION
CURVE     else if mask has STRUCT_MASK_SEAM or STRUCT_MASK_BOUNDARY
SHEET     else (sheet only, or no structure at all)
```

Priority is needed because a seam vertex's set also contains the sheet IDs of
the faces around it, and a junction vertex's set contains everything around it.

**Seam and boundary are one class ("CURVE").** Per the discussion, the
structure type should not make a meaningful difference: seams and boundaries
follow the same rule and are told apart only by their IDs. (If you meant
something else by "the structure doesn't have any meaningful difference",
this is the one place to change.)

## 2. The graph (directed, uniform weights)

Candidate edges are the undirected edges of the subdivided mesh `M.F`. Each
undirected edge {i, j} gives up to two directed edges, and `i ← j` means
**j pulls on i** (j is in row i of the Laplacian). "Shares an ID" means that
the two palette sets have a non-empty intersection, restricted to IDs of the
given type.

| row vertex i | i ← j is kept when |
|---|---|
| JUNCTION | never: the row is empty and i never moves |
| CURVE | j is JUNCTION or CURVE, and i and j share at least one seam/boundary ID |
| SHEET | j is any role, and i and j share at least one sheet ID |

Consequences:

- Sheet → sheet edges are two-way.
- A sheet vertex feels the curves and junctions around it, but they don't feel it.
- A curve vertex is pulled only along its own curve, and a junction pulls it
  but is not pulled back.
- **"Shares at least one ID" makes vertices with several IDs a non-issue.**
  Say a vertex sits where two seams meet but is not marked junction. It
  connects to both seams, so no special case is needed.
- For a SHEET row, "shares a sheet ID" is what stops a sheet vertex from being
  pulled by a sheet vertex of a *different* sheet. That can only happen with
  inconsistent data (an unmarked seam), and it is counted (see Checks).
- Without a .ma_struct every vertex is SHEET with an empty set. The graph then
  falls back to the plain mesh adjacency, with all edges two-way.

Laplacian row (uniform "umbrella" weights):

```
(L x)_i = (1/|N(i)|) · Σ_{j∈N(i)} x_j  −  x_i        (0 if N(i) is empty)
```

Uniform weights are the right choice here. They relax toward even spacing,
while cotan weights mostly move points across the surface (curvature flow),
which the projection would cancel.

Storage: CSR, with `rowOffsets` (Vs+1, int64) and `cols` (int32). Weights are
implicit (1/deg). At 10M vertices with about 6 neighbours each, that is about
240 MB.

## 3. The iteration

Jacobi style, where every vertex uses the previous iteration's positions,
repeated **until the positions no longer change**:

```
repeat:
    y_i = x_i + λ (L x)_i            for all i
    x_i' = Π_i(y_i)                   projection, see below
    δ = max_i |x_i' − x_i|
    x = x'
until δ ≤ tol · bbox_diagonal
```

**Why λ is not a parameter.** At convergence, x = Π(x + λ L x), which holds
exactly when L x has no component along the vertex's own structure (along
the sheet for sheet vertices, along the curve for curve vertices). That
condition does not contain λ, so λ only sets the speed, not the answer. It
is fixed internally at 0.5. That is the largest safe value, because with λ = 1
Jacobi can oscillate back and forth on bipartite parts of the graph.

**Stopping test.** Floating point never reaches an exact zero change, so
"stopped changing" means that the largest move in one iteration is below
`tol = 1e-12` relative to the bounding-box diagonal, about the rounding level
of the projection. A hard cap of 1e6 iterations, reported loudly if hit,
guards against a component that never settles. The log prints the iteration
count and δ every 100 iterations.

**Problem: plain Jacobi is far too slow at full size.** Smoothing is
diffusion. The slowest error mode decays per iteration by roughly
1 − c / h², where h is the graph diameter in hops. At 7 levels a sheet can be
thousands of hops across, which means millions of iterations over 10M
vertices.

**Fix: coarse-to-fine over the nested levels.** The subdivision levels are
nested, so we can:

1. Converge at level 0 (the original MAT vertices, with the same graph rule
   applied to level-0 edges).
2. Go to level k+1. Each new vertex starts at the midpoint of its two relaxed
   parents on its level-k edge (every new vertex is an edge midpoint of the
   level before), projected onto its own structure.
3. Converge at level k+1. Only the local error is left, which needs a few
   dozen iterations, not millions.
4. Repeat up to the final level.

The answer is the same fixed point, since the stopping test is applied on the
final level's full graph. The coarse levels only give a good starting guess.
The iteration count per level is logged, so a per-level blow-up is visible.

**What the fixed point looks like.** Inside a sheet it is the uniform-weight
("Tutte") arrangement pinned by the relaxed rims. On a curve it is even
spacing between junctions. A closed curve loop or a closed sheet with no
junction still has an equilibrium (even spacing around the loop), but it can
slide along itself. The stopping test is on the per-iteration move, so it
still terminates.

## 4. Projection back onto the MAT, per role

A plain closest-point projection onto the whole MAT is **wrong**. The MAT is
non-manifold, so near a seam a sheet vertex would snap onto a neighbouring
sheet. Projection uses the same "own structure" rule as the graph:

| role | projected onto | result stored |
|---|---|---|
| JUNCTION | nothing (it did not move) | unchanged |
| CURVE | the polyline of all original MAT edges carrying any of the vertex's seam/boundary IDs | closest point; fine face = lowest-index face incident to that edge, bary on it (same rule as EDGE carriers today) |
| SHEET | the faces of all sheets whose ID the vertex carries | closest point; fine face + bary directly |

Implementation: one `igl::AABB` per structure ID, built once, over its faces
(sheets, 3-column F) or its edges (curves, 2-column F). A vertex with several
IDs queries each of its structures and keeps the nearest point. The AABBs cost
memory only once, and querying 10M vertices per iteration is fine in Release.

After the last iteration, `M.V`, `M.fineFace` and `M.fineBary` are overwritten.
Seeding then uses them unchanged, and the decimation tracking is not touched.

## 5. What changes in the outputs

- `.sdt`: `sub_V`, `fine_face` and `fine_bary` hold the relaxed values. The
  dyadic carrier arrays (`carrier_type`, `carrier_index`) describe the
  **seed** positions before relaxation, and are kept because the struct IDs
  were derived from them. The header records whether relaxation ran, the iteration count per level and the final δ,
  in the reserved fields, and the version goes to 2.
- The levels stay nested **as connectivity**. After relaxation, the vertices of
  a coarser level are no longer exact midpoints of anything. They are still the
  same vertex ids, and `subdiv_level_faces` still works.
- **Both versions of the initial subdivided mesh are exported, for
  comparison.** They share the same connectivity and vertex ids, so they can
  be diffed vertex by vertex. Both honour `--subdiv_obj_max_verts` (the same
  nested level is written for both):
  - `subdiv_fine_<stem>.obj`: after relaxation, the version the tracker seeds
    from. It keeps today's name, so existing consumers get the relaxed samples.
  - `subdiv_fine_seed_<stem>.obj`: before relaxation, the exact midpoint
    subdivision. It is byte-identical to today's `subdiv_fine_<stem>.obj`.
  - The seed positions are also stored in the `.sdt` as an extra array,
    `sub_V_seed` (f64, Vs×3), so the full-resolution comparison does not
    depend on the OBJ cap. `n_arrays` goes from 14 to 15.
  - The run log prints a before/after summary:
    - subdivided edge-length coefficient of variation;
    - min angle (min, p1, p5, median);
    - flipped-triangle count;
    - max and mean per-vertex displacement.
  - With `--no_subdiv_relax` only `subdiv_fine_<stem>.obj` is written, exactly
    as today.

## 6. The graph file and its viewer

The goal of the viewer is only to make sure the graph is built correctly, so
the viewer **just parses the graph file** and displays it. It does not rebuild
anything.

Graph file `subdiv_graph_<stem>.slg`, written right after the graph is built:

```
header: magic "SUBDIVG\0", version, Vs, nnz
V        f64  Vs×3    seed positions (before relaxation)
role     u8   Vs      0 SHEET, 1 CURVE, 2 JUNCTION
set_id   i32  Vs      palette index (same palette as the .sdt)
rowOffs  i64  Vs+1
cols     i32  nnz
+ the palette (offsets, ids, typeMask)
```

Viewer, a C++ polyscope tool `subdiv_sample_tracker/subdiv_graph_viewer.cpp`
with its own target, which reads only the .slg:

- A point cloud colored by role (sheet / curve / junction), with the same radii
  as the tracker viewer.
- A curve network with one segment per undirected pair, colored by kind:
  - two-way sheet↔sheet;
  - one-way sheet←curve;
  - one-way sheet←junction;
  - two-way curve↔curve;
  - one-way curve←junction.
  Kinds can be toggled on and off.
- One-way edges are shown as a half segment from the pulled vertex toward the
  puller, so the direction is visible without arrows.
- A vertex query panel: type or click a vertex id and see its role, its IDs,
  its row (who pulls it) and its column (whom it pulls). The neighbours are
  highlighted.
- Display cap: like the tracker viewer, above about 500k vertices it shows a
  coarser nested level. At a coarser level the graph is re-derived only for
  display, from the same rule applied to that level's edges, and this is
  labelled clearly.

## 7. Checks (at every step, on ABC 00040057, Release)

Graph:

- An independent Python verifier (scratchpad, like `verify_sdt.py`) reads the
  .slg plus the .ma_struct, rebuilds roles and edges from scratch, and must
  match the file exactly.
- Every JUNCTION row is empty.
- Every CURVE row contains only CURVE/JUNCTION columns that share an ID.
- Counters are logged:
  - CURVE vertices with 0 neighbours (a stranded curve piece: it will not move,
    which is fine, but should be rare);
  - SHEET vertices with 0 neighbours;
  - sheet-edges dropped because the two vertices share no sheet ID (expected 0).

Relaxation:

- `--no_subdiv_relax` gives a **byte-identical** .sdt to today.
- Convergence: final δ ≤ tol on the full graph. Running one more iteration from the result moves nothing beyond tol.
- Junction positions are bit-identical before and after.
- Every CURVE vertex lies on its own curve, and every SHEET vertex on its own
  sheet, with distance ≤ 1e-12 × bbox diagonal. `fine_bary` reproduces `sub_V`.
- Struct IDs are unchanged, since they are not recomputed and the fine face
  lies in the vertex's own structure.
- Quality before and after:
  - coefficient of variation of subdivided edge lengths;
  - min angle histogram;
  - count of subdivided triangles whose normal flipped against their original
    face normal.
- The tracker's error counters stay 0 through the decimation.

## Open points

- Non-convex sheets. The uniform-weight fixed point is flip-free only for a
  planar sheet with a convex rim. On MAT sheets that are concave or curved,
  some subdivided triangles may fold or crowd against the rim at
  convergence. The flip count in the checks will show whether this happens.
  If it does, the fallback is to relax only the tangential part (project the
  Laplacian step onto the tangent plane before the closest-point
  projection), which keeps the same fixed point but moves more gently.
- A sheet vertex whose closest point on its own sheet is on the sheet's rim
  stays on the rim. That is correct, and it is not moved onto the curve's role.

## Implementation notes (2026-09-27)

Code: `subdiv_sample_tracker/subdiv_relax.{h,cpp}` (graph, projection, solver,
checks), called from `subdiv_tracker_init`. The test program is
`tests/subdiv_relax_test.cpp` (CMake option `SUBDIV_TESTS`). The viewer is
`subdiv_graph_viewer.cpp`, its own target.

**Flag.** Only `--no_subdiv_relax` exists (relaxation is on by default). With it
the `.sdt` is byte-identical to the pre-relaxation code, and no seed OBJ or
graph file is written.

**Solver.** The fixed point of `x <- Pi(x + lambda L x)` is a stationary point
of the graph's Dirichlet energy `E = 1/2 sum |x_i - x_j|^2` with every vertex
constrained to its own structure. Plain iteration (Jacobi) is correct but too
slow: level 1 took 37,596 iterations (39 s), and each level multiplies both
the vertex count and the iteration count by about 4. The solver used instead:

1. **Newton.** Per iteration, one linear solve (sparse LDLT) of the
   tangent-plane Hessian `deg_i u_i - sum_j T_i^T T_j u_j = T_i^T sum_j (x_j - x_i)`.
   Curves are solved first, then sheets with the curves held fixed. `T_i` is
   the vertex's current edge direction or face plane.
2. **Energy line search (Armijo).** Every accepted step must lower `E`, so the
   iteration cannot cycle. Without it, a vertex at a valley crease between two
   faces of a sheet bounced between their tangent planes forever.
3. **Active-set frames.** For a vertex exactly on a polyline kink or a sheet
   crease, `T_i` is the incident edge or face its pull actually points into.
   If the pull points into none of them, the vertex is held for that
   iteration.
4. **Plain step as the stopping test.** Newton rounds alternate with one plain
   step `x <- Pi(x + 0.5 L x)`. The pass ends only when that plain step moves no
   vertex more than `1e-12 x diag`, which is the plan's own definition of
   "stopped changing".

Coarse-to-fine initialisation (section 3) was not needed.

**Checks run on every relaxation**, and the tracker aborts if any fails:
- one plain step from the result moves no vertex more than `1e-10 x diag`;
- fixed and junction vertices are bit-identical to the seed;
- `V` equals the interpolation of `(fineFace, fineBary)` bit for bit;
- barycentrics are valid;
- every vertex's face or edge is in its own structure.

**Pinning.** A connected group of free vertices of one role that nothing
fixed pulls on can slide as a whole. On ABC 00040057 that is 26 of the 38 curve
components: closed seam loops with no junction. There are only 6 junction
vertices on this mesh. Each such group gets its lowest-index vertex pinned
(an original MAT vertex).

**The fixed point is not unique.** Curves are polylines with a small kink at
every original vertex, and sheets have creases. Several resting positions
exist, typically a loop shifted by one polyline segment. Evidence at level 0:
- Jacobi with `lambda = 0.5` and `lambda = 0.25` end `0.0084 x diag` apart
  (476 vertices).
- Newton reaches a third fixed point, with the lowest energy of the three.

All three are verified fixed points: Newton started from Jacobi's result moves
it by only `3e-10`. So the tests check the definition (a plain step moves
nothing), not agreement between solvers.

**Results on ABC 00040057** (Release, level 4 = 350,016 vertices, qslim to 200
faces):

| | value |
|---|---|
| relaxation time | 68 s (curves 58 iterations, sheets 39) |
| one plain step from the result | max move 2.9e-16 x diag |
| tangential residual `|T^T L x| / edge` | sheet 6.9e-13, curve 4.2e-13 |
| consistency checks | all 0 |
| tracker counters through 1168 collapses | all 0; placement diff 0 |
| `verify_sdt.py` (v2) / `verify_slg.py` (independent rebuild of the graph) | pass / pass, 2,019,517 entries |

Quality before and after at level 4:

| | seed | relaxed |
|---|---|---|
| edge-length CV | 0.3676 | 0.3682 |
| min angle p1 | 16.1 deg | 10.9 deg |
| min angle p5 | 19.8 deg | 18.8 deg |
| median min angle | 40.7 deg | 41.9 deg |
| degenerate triangles | 0 | 7 |
| flipped vs seed | – | 11 |

The degenerate and flipped triangles are the rim-crowding case from "Open
points". On a sheet with a concave rim, a sheet vertex's pull points out of
the sheet, so the projection stops it on the rim, right next to a curve
vertex. The global edge-length spread does not improve. Samples cannot move
between sheets, and every original face keeps the same number of
subdivided faces, so the rule mostly evens out spacing locally (the median
angle improves, while the worst angles get worse).

**Formats.**
- `.sdt`: version 2 when relaxed. The header's `relaxed` field is the old
  `reserved`, `n_arrays` is 15, and array 14 is `sub_V_seed`. `carrier_*`
  describe the seed.
- `.slg`: 8 arrays, layout in `subdiv_relax.cpp`.
- `subdiv_fine_seed_<stem>.obj` is byte-identical to the unrelaxed run's
  `subdiv_fine_<stem>.obj`.

Level 5 (1,395,080 vertices): relaxation took 561 s (curves 104 iterations,
sheets 42 at about 14 s each). One plain step from the result moves at most
3.5e-16 x diag. All checks are 0. The tracker counters are 0 through 1168
collapses, and both verifiers pass (8,209,197 graph entries). Quality: edge
CV 0.3676 -> 0.3663, min angle p1 16.1 -> 11.6, median 40.7 -> 41.9 deg,
13 degenerate triangles, 30 flipped. Cost grows about 8x per level (the
sparse LDLT per sheet iteration), so levels 6-7 (5.6M / 22M) would need a
faster sheet solve.
