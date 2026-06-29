# SSP Non-Manifold Extension — Methodology Changes

## Background

SSP (Subdivision Surface Parameterization) decimates a mesh by collapsing edges one at a time.
For each collapse it records a joint UV flattening of the one-ring neighbourhood, which is
later used to lift query points between coarse and fine levels.

The original implementation was designed for closed or boundary-manifold meshes.
A manifold mesh has the property that every edge is shared by exactly 1 (boundary) or 2 (interior) faces.
Non-manifold meshes — meshes with **seam edges** shared by 3 or more faces — caused hard crashes.

---

## Change 1 — Sheet Partitioning

### Original
No concept of sheets existed. The entire mesh was treated as one connected manifold domain.
The algorithm assumed every edge was shared by at most 2 faces.

### New
Before any processing begins, the original mesh is partitioned into **manifold sheets** using BFS.

The BFS crosses only edges shared by exactly 2 faces (manifold interior edges).
It stops at:
- **Non-manifold edges** (3+ incident faces) — these become sheet boundaries.
- **Boundary edges** (1 incident face) — nothing to cross.

Each face is assigned a **sheet ID**. Faces on opposite sides of a seam edge belong to different sheets.

This partitioning is computed once at startup on the **original mesh**, before the infinity extension.

---

## Change 2 — One-Ring Collection

### Original
`igl::circulation` walked the face fan around each edge endpoint using the EF/EI (edge-flap) data structure.
EF stores, for each edge, the two faces on either side. On non-manifold edges EF has only two slots,
so the third (or more) incident face was silently dropped or caused an out-of-bounds access.

### New
A **vertex-face adjacency list (VF)** is built at startup and maintained throughout the decimation.
The one-ring of each endpoint is collected by directly scanning `VF[vertex]`.
This naturally handles any number of incident faces — the list simply has more entries.

The VF list includes both real faces and infinity faces (added by `connect_boundary_to_infinity`).
Null faces (previously collapsed) are filtered out lazily at collection time.

After each collapse, `VF[absorbed_vertex]` is merged into `VF[surviving_vertex]` and cleared,
keeping the structure consistent for subsequent collapses.

---

## Change 3 — Per-Sheet UV Computation

### Original
One flat UV parameterization was computed for the entire one-ring of the collapsing edge.
The one-ring was treated as a single connected neighbourhood, which is only valid on manifold meshes.

### New
After collecting the full one-ring, faces are **bucketed by sheet ID**.
For each sheet that has faces on both sides of the collapsing edge, an independent UV parameterization
is computed. The existing `joint_lscm` routine is called once per sheet.

Each sheet's result is stored as a `SheetData` entry. A single collapse may produce multiple
`SheetData` entries (one per touched sheet). For manifold meshes the result is always one entry,
so there is no change in behaviour for the common case.

The routing key stored with each `SheetData` is the **global sheet ID**, which is used at query
time to look up the correct UV data for a given face.

---

## Change 4 — Ordered Fan Walk (fan_walk_local)

### Original
`igl::circulation` returned neighbour vertices in face-adjacency winding order automatically,
as a byproduct of the EF/EI walk. The last vertex in the returned list was always the
other endpoint of the collapsing edge — an invariant required by `joint_lscm`.

### New
The VF-based one-ring collection returns an **unordered set** of neighbour vertices.
A dedicated fan walk reconstructs the winding order within each sheet's face list.

The walk starts at a boundary vertex (count-1 neighbour, which may be the infinity vertex
on an open boundary) and steps face-by-face around the fan until reaching the target endpoint.
The endpoint always ends up last in the returned list, preserving the `joint_lscm` invariant.

For interior (closed-fan) vertices the walk starts at the CCW neighbour of the target endpoint
and visits all faces exactly once.

---

## Change 5 — Boundary Signal to joint_lscm

### Original
The infinity vertex added by `connect_boundary_to_infinity` was included in the global one-ring.
`joint_lscm` detected boundary by looking for the infinity vertex (local index `-1`) in the
neighbour list. This worked because `igl::circulation` included infinity faces in its walk.

### New
The same mechanism is preserved, but it now operates per-sheet.

For each sheet, a **walk face list** is constructed: the sheet's real faces plus any adjacent
infinity faces. An infinity face is considered adjacent to sheet `si` if its non-center,
non-infinity vertex appears in the sheet's real face set.

The fan walk over this list naturally encounters the infinity vertex at the open end of the fan.
`to_local(infVtx)` returns `-1`, which is inserted into the ordered neighbour list at the
correct position. `joint_lscm` sees the `-1` sentinel and applies the boundary constraint,
exactly as it did for manifold meshes.

No changes were made to `joint_lscm`.

---

## Change 6 — Topology Update

### Original
After the UV computation, the mesh topology was updated using EF/EI.
The update used EF to find the two **flap faces** (faces containing both endpoints of the
collapsing edge), killed their shared edges, and remapped the vertex `d → s` in the
surviving faces. This relied on each edge having exactly 2 incident faces (one per EF slot).

### New
EF/EI are treated as **initialization-only** after the first collapse.

The topology update uses a **two-pass VF scan** over the face list of the absorbed vertex:

- **Pass 1** — Scan for flap faces (faces containing both `s` and `d`).
  For each flap: redirect `EMAP` entries from the killed edge to the kept edge,
  kill the edge, and null the face. Works for any number of flap faces (≥ 2 for non-manifold).

- **Pass 2** — Scan surviving faces.
  For each face still containing `d`, update `F` and `E` entries to replace `d` with `s`.

`EMAP` and `E` are maintained correctly throughout. EF and EI are not updated after
initialization; no code path reads them after the first collapse.

---

## Change 7 — Query Routing

### Original
At query time, `decInfo[collapse_index]` held a single flat UV record.
The lookup was direct: given a face index, find the collapse that covers it, read the UV.

### New
`decInfo[collapse_index]` holds a list of `SheetData` entries.
At query time, the face's sheet ID is looked up from the same `faceSheetID` array computed
at startup. The query then selects the `SheetData` whose `global_sheet_id` matches,
and performs the UV interpolation using that sheet's UV data.

For manifold meshes (one sheet) the lookup always finds `sheets[0]`, which is identical
in behaviour to the original.

---

## What Was Not Changed

| Component | Reason |
|-----------|--------|
| `joint_lscm` | Correct and complete; the `-1` sentinel mechanism already handles boundaries |
| `get_collapse_onering_faces` | Already correctly filters null and infinity faces; called per-sheet |
| `edge_collapse_is_valid` | Takes the full one-ring; topological validity is mesh-global, not per-sheet |
| `connect_boundary_to_infinity` | Still called once globally; infinity faces provide the boundary signal |
| `SSP_random_collapse_edge` | Uses EF/EI internally; still manifold-only; updated only for struct compatibility |
| Cost function (`shortest_edge_and_midpoint`) | Geometry-only; no EF/EI dependency |
