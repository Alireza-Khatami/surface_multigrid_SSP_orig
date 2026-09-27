# Plan: subdivided-mesh tracker (replaces face + edge sample trackers)

Status: implemented (2026-09-27). The sections below are the original plan;
where the implementation differs, "Implementation notes" at the end wins.

## Goal

Replace `face_sample_tracker` and `edge_sample_tracker` with one tracker that:

1. subdivides the initial MAT surface (`gVO` / `gFO`) until it reaches the requested
   number of samples,
2. keeps the subdivided mesh's connectivity (needed for Laplacian operators),
3. knows the structure IDs of every subdivided vertex,
4. tracks every subdivided vertex through the decimation onto the coarse mesh.

Target scale: up to ~10M subdivided vertices.

**The subdivided mesh is never decimated.** It is only a set of samples with fixed
connectivity. Decimation still runs on the original fine mesh (`gV` / `gF`) exactly as
today; the subdivided vertices just ride along as barycentric points on it. The
subdivided mesh's vertices, faces and fine-mesh coordinates never change after
seeding; only each vertex's current position on the decimating mesh is updated.

## Decisions (resolved)

| # | Question | Decision |
|---|---|---|
| 1 | How to hit the sample count | Uniform midpoint subdivision, one level at a time, until vertex count ≥ N |
| 2 | Face-flip / vertex-watch debug tools | Keep them: port them onto the new tracker, in their own file under `subdiv_sample_tracker/` |
| 3 | Python viewers in `../11_correspond_viz/` | Out of scope for now |
| 4 | Output format | Raw little-endian binary with a header (one file), layout in section 6 |
| 5 | Old trackers | Not deleted: moved to a legacy folder and excluded from the build (section 9) |
| 6 | Viewer | A C++ polyscope viewer for the subdivided mesh, in `subdiv_sample_tracker/` (section 8) |
| 7 | Decimating the subdivided mesh | Never: it is only samples (see Goal) |

---

## 1. Folder layout

New folder `10_collapse_viz/subdiv_sample_tracker/`:

| File | Contents |
|---|---|
| `subdiv_mesh.{h,cpp}` | Midpoint subdivision, per-vertex carrier, `.ma_struct` element loader, struct-set palette |
| `subdiv_tracker.{h,cpp}` | Seeding, per-collapse update, binary export, deformed-mesh export |
| `subdiv_tracker_viz.{h,cpp}` | Polyscope display + ImGui panel for the subdivided mesh (section 8) |
| `debug_trackers.{h,cpp}` | Face-flip tracker + vertex-watch tracker, ported to read from `subdiv_tracker` |

`CMakeLists.txt` adds these four `.cpp` files and drops `face_sample_tracker.cpp` /
`edge_sample_tracker.cpp` (the files themselves are moved, not deleted; section 9).

## 2. Subdivision (once, after `init_ssp` + `load_matstruct`)

- **Uniform midpoint 1→4 subdivision, no smoothing.** Repeat whole levels until
  `#vertices ≥ N` (`--n_subdiv_samples N`). Can overshoot by up to ~4× since each
  level roughly quadruples the count. `N ≤ |gVO|` means level 0 (the mesh itself).
  Loop-style smoothing would move vertices off the MAT surface; midpoint subdivision
  keeps every vertex exactly on an original triangle.
- **Seams stay connected.** Midpoints are keyed by undirected edge, so the 3+ sheets
  meeting at a non-manifold seam share one midpoint vertex. The Laplacian sees the
  sheets joined there, the same way `gFO` joins them. Face winding is inherited from
  `gFO` (already consistently oriented by `orient_faces_consistently`).
- **Index stability:** subdivided vertices `0 .. |gVO|-1` are the original vertices in
  their original order; new vertices are appended after them. So an original vertex id
  is also its subdivided id (used by the debug trackers and by any consumer).
- **Every subdivided vertex records its carrier** on the original mesh:
  - `VERTEX` (0): an original vertex (index into `gVO`)
  - `EDGE` (1): lies on an original edge (index into the original edge list)
  - `FACE` (2): lies inside an original face (index into `gFO`)

  plus its exact position as `(gFO face, barycentric)`. Tracking and structure IDs are
  both derived from this. For an `EDGE` carrier the barycentric has exactly one zero,
  and the two non-zero entries are the linear coordinates `(1−t, t)` along that edge.

## 3. Structure IDs

A small loader reads the `.ma_struct` file into:
- face → sheet struct ID (type 0)
- edge → seam / boundary struct ID (types 1, 2)
- vertex → struct ID set (existing `load_matstruct`, includes junctions, type 3)

Rules (consistent with `load_matstruct` for original vertices):

| Carrier | Structure IDs |
|---|---|
| `VERTEX` v | `gVertexStructIDs[v]` unchanged (the only case with junction IDs) |
| `EDGE` e | seam/boundary ID of e (if any) ∪ sheet IDs of all faces incident to e |
| `FACE` f | sheet ID of f |

At startup the loader verifies that the `.ma` face/edge lists match `gFO` (same count,
same corners) and aborts if not. Today this correspondence is assumed, never checked.

## 4. Storing structure IDs at 10M-vertex scale

Problem: seam/junction vertices can carry 6–10 IDs. A per-vertex `std::set<int>` costs
~40–50 bytes per ID plus overhead, so 10M vertices would reach several GB in memory,
and writing variable-length ID lists as text would add hundreds of MB to the output.

The key observation: **the set of distinct ID sets is tiny.** Every subdivided vertex
takes its IDs from its carrier, and there are only as many distinct sets as there are
distinct (seam/sheet/junction) combinations in the original MAT: typically tens to a
few thousand, regardless of subdivision depth. Almost all 10M vertices are `FACE`
carriers with exactly one sheet ID.

Approach: **interned set table + one int per vertex.**

- Build a **struct-set palette**: a table of unique sorted ID sets,
  `palette[k] = {id_0, id_1, ...}`, stored once (CSR: `offsets[]` + `ids[]`).
- Each subdivided vertex stores a single `int32 struct_set_id` indexing into it.
- Memory / disk: 10M × 4 B = **40 MB**, plus a palette of a few KB.
- Each palette entry also gets a precomputed type mask
  (bit 0 sheet, bit 1 seam, bit 2 boundary, bit 3 junction), so "is vertex i on a
  seam?" is one array lookup.

In C++ memory: the same layout (`std::vector<int32_t>` per vertex + a flat palette),
never a per-vertex `std::set`.

## 5. Tracking

- Seed each subdivided vertex on its `(gFO face, barycentric)`. `EDGE` / `VERTEX`
  carriers on seams sit on the first incident face (same policy as today's edge
  tracker, relying on the seam UV pinning in `joint_lscm_pinned.cpp` to keep sheets
  consistent).
- After each successful `SSP_collapse_edge`: the same per-sheet UV_pre → UV_post
  barycentric cast as the old trackers.

**Scaling problem to fix:** the old trackers do a d→s vertex fix-up over *all* samples
after every collapse, which is O(N_samples) per collapse. At 10M samples × ~100k+
collapses that is ~10¹² operations. The new tracker avoids it by not storing global
corner IDs per sample:
- store only `cur_FIdx` + barycentric in **`gF` row corner order**;
- at cast time, reorder the `FUV_post` barycentric to match `gF.row(new_FIdx)`
  (mapping d→s for the subset indices);
- corners are read from `gF.row(cur_FIdx)` when needed. Faces that are not recast
  (`non_active_faces`) get d→s rewritten in `gF` itself, so they stay valid for free.

Per-collapse cost then depends only on the samples in the collapse's one-ring.

Per-sample memory: `int32 face + 3 × float64 bary` ≈ 28 B → ~280 MB at 10M.

## 6. Output: one raw binary file `subdiv_<stem>.sdt`

Written in `export_final_outputs`, via the shared `CoarseFaceLookup`.
All values little-endian. Each array starts on an 8-byte boundary (zero padding).
Arrays appear in the fixed order below; the header gives every count, so a reader can
compute every offset (the header also stores them explicitly for convenience).

### Header (fixed size)

| Field | Type | Meaning |
|---|---|---|
| `magic` | `char[8]` | `"SUBDIVT\0"` |
| `version` | `uint32` | `1` |
| `header_bytes` | `uint32` | size of this header |
| `n_levels` | `uint32` | subdivision levels applied |
| `n_samples_requested` | `uint32` | the `N` given on the command line |
| `n_sub_verts` | `uint64` | Vs: subdivided vertices |
| `n_sub_faces` | `uint64` | Fs: subdivided faces |
| `n_fine_verts` | `uint64` | `gVO.rows()` |
| `n_fine_faces` | `uint64` | `gFO.rows()` |
| `n_coarse_verts` | `uint64` | `cmc.Vbase.rows()` (compact coarse mesh) |
| `n_coarse_faces` | `uint64` | `cmc.Fout.rows()` |
| `n_palette` | `uint64` | P: distinct struct-ID sets |
| `n_palette_ids` | `uint64` | total IDs across all palette entries |
| `offsets[11]` | `uint64` | byte offset of each array below, in order |

### Arrays (in order)

| # | Array | Type / shape | Meaning |
|---|---|---|---|
| 0 | `sub_V` | `float64 [Vs×3]` | subdivided vertex positions on the fine mesh |
| 1 | `sub_F` | `int32 [Fs×3]` | **subdivided face connectivity** (Laplacian input) |
| 2 | `carrier_type` | `uint8 [Vs]` | 0 = VERTEX, 1 = EDGE, 2 = FACE |
| 3 | `carrier_index` | `int32 [Vs]` | gVO vertex / original edge / gFO face index |
| 4 | `fine_face` | `int32 [Vs]` | gFO face the vertex lies on |
| 5 | `fine_bary` | `float64 [Vs×3]` | barycentric in `gFO.row(fine_face)` corner order (linear `(1−t,t)` for EDGE carriers) |
| 6 | `coarse_face` | `int32 [Vs]` | face in the compact coarse mesh (`cmc.Fout`, = `simplified_<stem>.obj` face order) |
| 7 | `coarse_bary` | `float64 [Vs×3]` | barycentric in `cmc.Fout.row(coarse_face)` corner order |
| 8 | `struct_set_id` | `int32 [Vs]` | index into the palette |
| 9 | `palette_offsets` | `int32 [P+1]` | CSR offsets into `palette_ids` |
| 10 | `palette_ids` + `palette_type_mask` | `int32 [n_palette_ids]` then `uint8 [P]` | the ID sets and their type masks |

The coarse mesh itself is not duplicated: it is `simplified_<stem>.obj`, and the header's
coarse counts let a reader verify it has the matching file.

Size at 10M vertices (≈20M faces): ~240 MB V + ~240 MB F + ~280 MB fine coords +
~280 MB coarse coords + ~90 MB carrier/struct ≈ **1.1 GB**. Dropping `sub_V`
(derivable from `fine_face` + `fine_bary`) would save 240 MB; kept for convenience.

Any reader (e.g. Python with `numpy.frombuffer`) can load arrays directly at the header
offsets; no parsing needed. Writing a reader is out of scope for now.

### Other outputs

- `subdiv_deformed_<stem>.obj`: subdivided connectivity with every vertex at its coarse
  position (replaces `deformed_fine_mesh_`). For large meshes this can be skipped via
  a flag, since the `.sdt` already has everything needed to build it.

## 7. Debug trackers (`subdiv_sample_tracker/debug_trackers.{h,cpp}`)

Same public API and behavior as today, reading positions from `subdiv_tracker`:
- **Face flip tracker** (`--track_face_flip <gFO face>`): tracks the face's 3 original
  vertices, i.e. subdivided vertices with the same ids (index stability, section 2).
- **Vertex watch** (UI input, fine vertex id): same id mapping; watches the coarse face
  currently holding that vertex.

`subdiv_tracker` exposes a small accessor for this: current coarse position and current
corner vertices of subdivided vertex i.

## 8. Viewer (`subdiv_sample_tracker/subdiv_tracker_viz.{h,cpp}`)

A C++ polyscope viewer inside the existing app, replacing the old
`sample_tracker_show()` / `sample_tracker_show_vertices()` / `edge_sample_tracker_show()`
calls in `visualizer.cpp`. Compiled only under `C2F_VIZ_DIAGNOSTIC`, like the old
display code.

What it shows:
- **Subdivided mesh on the fine surface**: polyscope surface mesh (`sub_V`, `sub_F`),
  registered once at init since it never changes.
- **Subdivided mesh on the current coarse surface**: same connectivity, each vertex at
  its current tracked position (barycentric on `gV` / `gF`). Updated after collapses.
  This shows how the samples deform as decimation proceeds.
- **Colour quantities** on both: struct type mask (sheet / seam / boundary / junction),
  struct set id, and carrier type (vertex / edge / face).
- **Displacement vectors** (fine → coarse) as an optional vector quantity.

ImGui panel:
- counts (levels, vertices, faces, palette size) and on/off toggles per structure;
- a "refresh every K collapses" setting, and a "refresh now" button;
- a vertex-id input that highlights one subdivided vertex and prints its carrier,
  struct IDs, and current coarse face + barycentric.

At 10M vertices, re-uploading the deformed mesh every collapse would dominate run time,
so by default it refreshes only on demand, every K collapses, or when decimation
stops. For display a max-vertex cap can show a coarser subdivision level instead of
the full one (the levels are nested, so the first `|V_level|` vertices of the full
mesh are exactly the lower level's vertices).

## 9. Old trackers: moved, not deleted

- **Move** `face_sample_tracker.{h,cpp}`, `edge_sample_tracker.{h,cpp}` and
  `edge_sample_tracker_viz.py` into `10_collapse_viz/legacy_sample_trackers/`
  (with `git mv`, so history follows them). Nothing is deleted.
- **Exclude from the build:** remove their entries from `CMakeLists.txt`, and remove
  their `#include`s and calls from `main.cpp` and `visualizer.cpp`.
- CLI: `--n_samples_total` → `--n_subdiv_samples`; remove `--trace_vertices`,
  `--n_edge_samples_per_struct`, `--edge_sample_monte_carlo`. `--track_face_flip` stays.
- `visualizer.cpp`: display calls switch to `subdiv_tracker_viz`; the canonical-view
  ring-sample display is dropped; face-flip and vertex-watch panels stay, pointed at
  `debug_trackers`.

---

## Implementation notes

### Files

| File | Contents |
|---|---|
| `subdiv_sample_tracker/subdiv_mesh.{h,cpp}` | Subdivision, carriers, `levelVerts`, `subdiv_level_faces()` |
| `subdiv_sample_tracker/subdiv_struct_ids.{h,cpp}` | `.ma_struct` loader (validated against `gFO`) + struct-set palette (split out of `subdiv_mesh`) |
| `subdiv_sample_tracker/subdiv_tracker.{h,cpp}` | Seeding, per-collapse cast, `.sdt` writer, deformed OBJ |
| `subdiv_sample_tracker/subdiv_tracker_viz.{h,cpp}` | Polyscope display + panel (GUI builds only) |
| `subdiv_sample_tracker/debug_trackers.{h,cpp}` | Face-flip + vertex-watch, same function names as before |
| `subdiv_sample_tracker/tests/subdiv_mesh_test.cpp` | Standalone checks, CMake option `SUBDIV_TESTS` (off by default) |
| `legacy_sample_trackers/` | Old `face_sample_tracker`, `edge_sample_tracker`, `edge_sample_tracker_viz.py`, `load_matstruct_edges` (moved with `git mv`, not built) |

CLI: `--n_subdiv_samples N`. `--track_face_flip` needs it. `--n_samples_total`, `--trace_vertices`,
`--n_edge_samples_per_struct`, `--edge_sample_monte_carlo` are gone (unknown flags are ignored,
so old launch configs still start, but with the tracker off).

### Differences from the plan

- **Duplicate faces.** A midpoint on an original edge is shared by every face around that
  edge. A midpoint inside an original face belongs to that face alone, so distinct faces on
  the same three vertices get distinct interior vertices.
- **Corner ids.** Each sample stores its three corner ids. Ids absorbed by later collapses are
  resolved through a survivor redirect (`d -> s`, path halving) instead of rewriting every
  sample after each collapse. Each cast matches the sample's corners against the
  `FUV_pre` corners by id and would count a mismatch.
- **`.sdt` layout.** 14 arrays, 224-byte header. It adds `sub_face_orig` (fine face of each
  sub face) and `orig_edges` (so the EDGE `carrier_index` is self-contained), and
  `n_samples_requested` is `uint64`. The exact layout is in the comment above
  `subdiv_tracker_save()` in `subdiv_tracker.cpp`.
- **Deformed OBJ.** Written only up to 2M vertices.
- **Overshoot.** Uniform levels jump by ~4x. On the ABC test mesh N = 10M gives 22.3M
  vertices (7 levels), a 2.57 GB `.sdt`, and 109 s / 5.2 GB peak RAM for the whole run.

### Verification (ABC 00040057, qslim to 200 faces, 1168 collapses)

- `subdiv_mesh_test` at 0/1/4/7 levels:
  - positions recomputed from carriers are exact;
  - sub faces tile each fine face (area error ≤ 4e-13) with no flips;
  - every sub edge has the valence its carriers predict;
  - each level built on its own equals the final mesh's prefix;
  - original-vertex struct sets equal `load_matstruct`.
- Run-time counters, all 0 up to 22.3M samples: corner mismatch, non-identity corner order,
  per-sheet `(s,d)` mismatch, face in two sheets, stranded samples, queries outside the
  post ring. At the end every sample's corners equal its `gF` face's corners, and the gV
  position equals the compact-mesh position.
- Level 0 against the old face tracker: all 1506 vertices bit-identical.
- An independent Python reader of `.sdt` recomputed all struct sets from `.ma_struct` and
  checked the layout, fine positions and coarse barycentrics.
- The shared prefix of a 4-level and a 5-level run is bit-identical.
