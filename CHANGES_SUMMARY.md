# SSP Non-Manifold Edge-Collapse: Changes Summary

This document records all changes made to enable the SSP edge-collapse pipeline to run on
**non-manifold meshes** (meshes with seam edges shared by 3+ faces, or boundary edges that
become non-manifold after `connect_boundary_to_infinity`).

---

## Files Changed

| File | Nature of change |
|---|---|
| `src/single_collapse_data.h` | New `SheetData` struct; `single_collapse_data` restructured |
| `src/SSP_collapse_edge.h` | Signature updates for new optional parameters |
| `src/SSP_collapse_edge.cpp` | Core changes: VF-based one-ring, sheet-aware UV, VF topology update |
| `src/SSP_midpoint.cpp` | Non-manifold guard removed; VF + sheet init added |
| `src/SSP_decimate.cpp` | Non-manifold guard removed |
| `src/SSP_random_collapse_edge.cpp` | Stores UV data into `SheetData` instead of flat fields |
| `src/query_coarse_to_fine.cpp` | Accesses UV data via `sheets[0]` |
| `src/query_fine_to_coarse.cpp` | Accesses UV data via `sheets[0]` |
| `10_collapse_viz/main.cpp` | Adds VF/sheet/seam init; guards `d.sheets[0]` access |

Two new source files were also added (untracked, not reverted by git):
- `src/partition_into_sheets.cpp/.h`
- `src/partition_onering_by_sheet.cpp/.h`

---

## Why `igl::edge_flaps` / `igl::circulation` breaks on non-manifold meshes

`edge_flaps` stores each edge's two incident faces in a fixed 2-slot matrix `EF(e, 0..1)`.
For a manifold mesh each undirected edge has exactly one `u→v` face (slot 0) and one `v→u`
face (slot 1). `circulation` hops between faces using those slots.

Non-manifold topology breaks both slots in two distinct ways:

**Case A — EMPTY SLOT (2-face same-winding edge)**
Both incident faces happen to wind the same direction (`u→v`). Both write to slot 0; slot 1
is left at `EF(e,1) = -1`. `circulation` exits through the `-1` slot → `F(-1, *)` → crash.

**Case B — LOST FACE (seam edge with 3+ faces)**
Three faces share edge `(u,v)`: two wind `u→v`, one winds `v→u`.
```
EF(e,0) = f2   ← last u→v write overwrites f0 (f0 is LOST)
EF(e,1) = f1   ← the v→u face
```
When `circulation` arrives at `f0` and tries to cross `(u,v)`, neither slot contains `f0`.
The walker is lost; the assertion `"e should touch ff"` fires.

---

## Three approaches considered for fixing one-ring collection

### Approach A — Variable-length per-edge face list (not implemented)

Replace the `Nx2` EF matrix with `vector<vector<pair<int,int>>>` storing ALL faces per edge.
Requires rewriting `circulation` and all downstream users of `EF`/`EI`.
- **Pro:** Principled, complete, handles any topology.
- **Con:** Heavy refactor; breaks the existing igl interface contract throughout the codebase.

### Approach B — Sheet-aware per-vertex face list ✓ (CHOSEN)

Replace `circulation` with a direct scan of a maintained VF adjacency list:
for vertex `v`, collect all live faces `f` where any corner equals `v`. No EF traversal.
Group those faces by sheet ID (from `partition_into_sheets`) to get per-sheet disk one-rings.
Each sheet runs its own `joint_lscm`.
- **Pro:** No EF dependency at all; handles Case A and Case B naturally; integrates cleanly
  with the existing sheet-partitioning infrastructure; VF is cheap to maintain (merge lists
  on collapse, filter nulls lazily).
- **Con:** VF lists accumulate stale null-face entries over time (they are filtered on each
  use, never compacted); sheet partitioning adds a dependency on `faceSheetID`.

### Approach C — Patch `circulation` to exit early on EF=-1 (not implemented)

Keep everything as-is; add `if (nf < 0) break;` (or `continue`) inside `circulation`
whenever an EF slot is -1.
- **Pro:** Minimal change; unblocks progress immediately.
- **Con:** The one-ring result is silently incomplete near seam edges. For a seam vertex the
  walk terminates before it has seen all sheets, so the UV flattening uses an incomplete
  one-ring. Not obviously better than crashing — the output is wrong rather than absent.

**Decision:** Approach B was chosen because it is the only option that gives a *correct*
one-ring for non-manifold vertices, and the infrastructure (`partition_into_sheets`,
`partition_onering_by_sheet`) was already in place.

---

## Problem 1 — Non-manifold guard blocked all non-manifold meshes

**Files:** `src/SSP_decimate.cpp`, `src/SSP_midpoint.cpp`, `10_collapse_viz/main.cpp`

**Original behaviour:** Both `SSP_decimate` and the internal `SSP_midpoint` overload called
`igl::is_edge_manifold` and returned false (or aborted) for any non-manifold input.
`main.cpp` similarly aborted with `"not edge-manifold – aborting"`.

**Fix:** Removed all three manifold guards. Non-manifold edges are now handled by the
VF-based one-ring approach (see Problem 2) instead of being rejected.

---

## Problem 2 — `igl::circulation` crashes on non-manifold edges (EF = -1)

**File:** `src/SSP_collapse_edge.cpp` (outer overload, ~line 511)

**Original behaviour:** The outer overload called `igl::circulation(e, true/false, F, EMAP, EF, EI, ...)`
to collect the one-ring faces/vertices of each endpoint. For non-manifold edges,
`EF(e, 1) = -1` (EMPTY SLOT), causing `circulation` to access `F(-1, *)` and crash.

**Fix (Approach B):** Added an optional `std::vector<std::vector<int>> * VF` parameter.
When provided, the one-ring is collected by iterating `(*VF)[v]` directly and filtering
null faces — no EF traversal at all:
```cpp
collect_onering(E(e,1), Nsf, Nsv);   // vj's faces
collect_onering(E(e,0), Ndf, Ndv);   // vi's faces
```
After a successful collapse, `(*VF)[sv]` absorbs `(*VF)[dv]`; `(*VF)[dv]` is cleared.
Null faces accumulate in VF lists but are filtered at each call.

---

## Problem 3 — Seam edges have 3+ faces; single one-ring UV flattening breaks

**File:** `src/SSP_collapse_edge.cpp` (inner overload), `src/single_collapse_data.h`

**Original behaviour:** The inner overload treated the entire one-ring as a single disk and
ran one `joint_lscm` call over all faces. For a seam edge the one-ring is not a disk — it
is multiple sheets that share only the edge endpoints. Passing all faces at once makes the
local mesh non-manifold and LSCM fails or produces garbage UVs.

**Fix — `SheetData` + per-sheet LSCM:**

1. **`single_collapse_data.h`:** Added `SheetData` struct holding per-sheet `b`, `subsetVIdx`,
   `UV_pre/post`, `FUV_pre/post`, `FIdx_pre/post`. `single_collapse_data` now stores
   `std::vector<SheetData> sheets` instead of flat fields.

2. **`partition_into_sheets` / `partition_onering_by_sheet`:** New functions (untracked files)
   that assign a sheet ID to every face and split the one-ring face lists `Nsf`/`Ndf` by sheet.

3. **Inner overload:** Loops over sheets; for each sheet where both `vi` and `vj` have faces,
   runs `get_collapse_onering_faces` + `remove_unreferenced_lessF` + `joint_lscm` independently.
   Results are pushed into `data.sheets`. Sheets where one endpoint is absent are skipped.

4. **`query_coarse_to_fine.cpp` / `query_fine_to_coarse.cpp`:** Updated to read UV data from
   `decInfo[dIdx].sheets[0]` instead of the old flat fields.

5. **`SSP_random_collapse_edge.cpp`:** Updated to pack its UV result into a `SheetData` and
   push it into `data.sheets`.

---

## Problem 4 — Topology update crashed on non-manifold edges (EF/EI-based)

**File:** `src/SSP_collapse_edge.cpp` (inner overload, topology update block)

**Root cause sequence (three sub-crashes):**

### 4a — EMAP(-1) out-of-bounds
```
EMAP(f + m*((v+sign*1+3)%3))   with f=-1, v=-1
```
For a non-manifold edge `EF(e,1) = -1`; the side-loop with `side=1` ran with `f = EF(e,1) = -1`
→ negative index into EMAP.

### 4b — Wrong EF-based flip for seam faces
```
assert(E(e1, flip1) == d || E(e1, flip1) == s)   // failed: flip1=0, E(e1,0)=other vertex
```
`flip1 = (EF(e1,0)==f) ? 1 : 0`. For a seam face not stored in `EF[e1]`, this gives the
wrong column → the wrong slot of E is read → assert fires.

### 4c — `a_e2` uninitialized
When side=1 was skipped (EF(e,1)<0), `a_e2` was never assigned; the outer overload then did
`EQ(a_e2) = -1` with `a_e2 = garbage`.

**Fix — VF-based two-pass topology update (replaced entire ~50-line EF/EI block):**

- **Pass 1:** Iterate `nV2Fd` (d's face list). For each face containing both `s` and `d` (flap
  face): identify `e_kill` (edge opposite s-corner) and `e_keep` (edge opposite d-corner);
  redirect any sibling face's EMAP from `e_kill` to `e_keep`; kill the flap face; record
  `a_e1`/`a_f1`, `a_e2`/`a_f2`.
- **Pass 2:** Iterate `nV2Fd` again. For each surviving face containing `d`: use `e_side_of`
  (reads E directly, no EF) to find which column holds `d`, replace with `s`; update F(f,c)=s.
- `kill_edge(e)` called once at the end.
- `EF`/`EI` are left stale intentionally — both cost functions (`shortest_edge_and_midpoint`
  and QEM) explicitly ignore them.
- Outer overload guards: `if(e1 >= 0) EQ(e1) = -1; if(e2 >= 0) EQ(e2) = -1;`

---

## Problem 5 — `d.sheets[0]` out-of-bounds in main.cpp

**File:** `10_collapse_viz/main.cpp` (~line 185)

**Original behaviour:** `do_next_step` accessed `d.sheets[0]` unconditionally after a
successful collapse. When all sheets were skipped (both seam endpoints absent on every sheet)
the collapse still returned `true` but `data.sheets` was empty → vector OOB crash.

**Fix:** Wrapped the snapshot update in `if (!d.sheets.empty()) { ... }`.

---

## Seam-vertex detection (bonus fix)

Using `EF(e,1) == -1` to detect seam vertices was wrong:
`connect_boundary_to_infinity` creates "infinity fan" edges whose two triangles both go
in the same directed half-edge direction, leaving `EF=-1` on every boundary fan edge and
falsely flagging every regular boundary vertex as a seam vertex.

**Fix (both `main.cpp` and `SSP_midpoint.cpp`):** Count face incidence on edges of the
*original* mesh (`FO` / `OF`, before `connect_boundary_to_infinity`). An edge with ≥ 3
incident faces is a true seam; both endpoints are marked `seamVertex=1`.

---

## Diagnostic logging added (temporary)

Several `std::cerr` blocks were added during debugging (controlled by static counters):
- `[BOGUS ONE-RING]` — fires when `Nsv.size() < 2 || Ndv.size() < 2`
- `[sheets]` — logs sheet breakdown (vi, vj, nSheets, per-sheet sizes)
- `[b_si FAIL]` — fires when vi or vj not found in a sheet's `subsetVIdx`
- `[joint_lscm #N]` — logs inputs to each LSCM call

These should be removed once the pipeline is confirmed stable.
