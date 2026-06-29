# Non-Manifold SSP Extension

This document lists every function/struct added or modified to extend the SSP pipeline to handle non-manifold meshes with seam edges (edges shared by 3 or more faces).

---

## New files

### `src/partition_into_sheets.h` / `src/partition_into_sheets.cpp`

**Function:** `partition_into_sheets(F, faceSheetID, numSheets)`

Assigns each face a sheet ID by flood-filling face adjacency and stopping at seam edges. An edge is a seam edge when it is shared by anything other than exactly 2 faces. The result is that each connected component separated by seam edges becomes one sheet.

- Builds a `std::map<Edge, std::vector<int>> edgeFaces` from the face list.
- Only propagates across edges with exactly 2 incident faces.
- BFS assigns `faceSheetID(f)` for every face `f` and returns the count `numSheets`.

### `src/partition_onering_by_sheet.h` / `src/partition_onering_by_sheet.cpp`

**Function:** `partition_onering_by_sheet(Nsf, Ndf, faceSheetID, sheets_Nsf, sheets_Ndf)`

Splits the flat per-vertex face circulation lists `Nsf` (from vi) and `Ndf` (from vj) into per-sheet sublists using the precomputed `faceSheetID`. For a regular (non-seam) edge, all faces are on one sheet so the output vectors have size 1 — callers are uniform across both cases.

---

## Modified files

### `src/single_collapse_data.h`

Added the **`SheetData`** struct:

```cpp
struct SheetData {
  Eigen::VectorXi b;           // local indices of vi, vj within this sheet
  Eigen::VectorXi subsetVIdx;  // global vertex indices for this sheet
  Eigen::MatrixXd UV_pre, UV_post;
  Eigen::MatrixXi FUV_pre, FUV_post;
  Eigen::VectorXi FIdx_pre, FIdx_post;
};
```

Changed **`single_collapse_data`**: the flat UV/index fields (`b`, `subsetVIdx`, `UV_pre`, `UV_post`, `FUV_pre`, `FUV_post`, `FIdx_pre`, `FIdx_post`) are replaced by `std::vector<SheetData> sheets`. The 3-D geometry fields (`V_pre`, `V_post`, `Nsv`, `Ndv`) remain on `single_collapse_data` directly (shared across sheets; vertex space is global).

### `src/SSP_collapse_edge.h`

- Added `#include <partition_into_sheets.h>` and `#include <partition_onering_by_sheet.h>`.
- Added `const Eigen::VectorXi & faceSheetID = Eigen::VectorXi()` as trailing default parameter to both overloads (inner and outer). When empty (manifold meshes or callers that don't supply it), behaviour is identical to the original single-path code.

### `src/SSP_collapse_edge.cpp`

- **Inner overload**: added `faceSheetID` to the implementation signature.
  - Replaced the old single-path UV flattening block with a per-sheet loop:
    1. `partition_onering_by_sheet` (or fallback to one sheet when `faceSheetID` is empty)
    2. Per-sheet: `get_collapse_onering_faces` → `remove_unreferenced_lessF` → `b_si` → `V_post_si` → `get_post_faces` → `joint_lscm`
    3. `FIdx_onering_pre` is accumulated across sheets so the outer function's `decIM` update covers all pre-collapse faces.
    4. `data.V_pre / V_post / Nsv / Ndv` are filled from sheet 0 (shared 3-D vertex space).
    5. Each sheet's UV data is stored in a `SheetData` pushed to `data.sheets`.
- **Outer overload**: added `faceSheetID` to the implementation signature and passes it to the inner call.

### `src/SSP_midpoint.cpp`

- Added `#include <partition_into_sheets.h>`.
- **1st overload**: removed the early-return `is_edge_manifold` check that rejected non-manifold meshes.
- **4th overload (main decimation loop)**:
  - Removed the `is_edge_manifold` check.
  - Added `partition_into_sheets(OF, faceSheetID, numSheets)` call after `edge_flaps`.
  - Passes `faceSheetID` to every `SSP_collapse_edge` call in the loop.

### `src/SSP_decimate.cpp`

Removed the `is_vertex_manifold` / `is_edge_manifold` guard that returned `false` for non-manifold input meshes.

### `src/query_coarse_to_fine.cpp`

All per-collapse data accesses now go through `sheets[0]`:
- `decInfo[dIdx].subsetVIdx` → `sd.subsetVIdx`
- `decInfo[dIdx].b` → `sd.b`
- `decInfo[dIdx].UV_post` → `sd.UV_post`
- `decInfo[dIdx].UV_pre` → `sd.UV_pre`
- `decInfo[dIdx].FUV_pre` → `sd.FUV_pre`
- `decInfo[dIdx].FIdx_pre` → `sd.FIdx_pre`

where `sd` is a `const SheetData & sd = decInfo[dIdx].sheets[0]`.

### `10_collapse_viz/main.cpp`

Updated `do_next_step()` to populate the `gSnap` display snapshot via `d.sheets[0]` instead of flat fields on `single_collapse_data`.

---

## Design notes

- **Backward compatibility**: all new `faceSheetID` parameters default to `Eigen::VectorXi()`. When empty, the per-sheet loop degenerates to the old single-path behaviour (one sheet = all one-ring faces), so all existing manifold-mesh callers continue to work unchanged.
- **Seam-edge collapses**: `igl::edge_collapse_is_valid` will reject seam edges (the link condition fails for non-manifold edges), so they are skipped gracefully. Full seam-edge support is future work.
- **`edge_flaps` limitation**: IGL's `edge_flaps` stores only 2 faces per edge; for seam edges the third (or more) face is not represented. Circulations near seam vertices may therefore be incomplete. The per-sheet partitioning mitigates this for edges that ARE collapsed, but a complete solution would require replacing `edge_flaps` with a half-edge structure that supports non-manifold topology.
