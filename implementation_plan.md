# Non-Manifold SSP — Implementation Plan

All file paths are relative to `external/surf_subgrid_SSP_orig/`.

---

## Overview of what changes

| # | File | Change type | What |
|---|------|-------------|------|
| 1 | `src/partition_into_sheets.cpp/.h` | **New** | Decompose a non-manifold mesh into sheets by flood-fill at seam edges |
| 2 | `src/partition_onering_by_sheet.cpp/.h` | **New** | Given a flat list of one-ring faces, split them into per-sheet subsets |
| 3 | `src/single_collapse_data.h` | **Modify** | Replace flat UV/face fields with `std::vector<SheetData> sheets` |
| 4 | `src/SSP_collapse_edge.cpp` | **Modify** | Detect seam vs regular edge; call `joint_lscm_case2` per sheet for seam edges |
| 5 | `src/query_coarse_to_fine.cpp/.h` | **Modify** | Route each query vertex to the correct sheet's UV map |
| 6 | `src/SSP_midpoint.cpp` | **Modify** | Skip monolithic manifold check; connect boundary-to-infinity per sheet |
| 7 | `08_subdiv_remesh/main.cpp` | **Modify** | Accept non-manifold input; pass sheet metadata through the pipeline |

---

## Step 1 — `src/partition_into_sheets.cpp/.h` (new)

**Purpose:** given a non-manifold mesh `(V, F)`, assign each face a sheet ID by
flood-filling edge adjacency and stopping at non-manifold edges (edges shared by 3+
faces).

```cpp
void partition_into_sheets(
    const Eigen::MatrixXi & F,
    Eigen::VectorXi & faceSheetID,    // output: sheet index per face
    int & numSheets);                 // output: total number of sheets
```

**Algorithm:**
1. Build a face-adjacency map. For each edge, record all faces that share it.
2. Mark an edge as a "seam edge" if it is shared by 3 or more faces.
3. BFS/flood-fill across non-seam edges; each connected component is one sheet.
4. Write the component index to `faceSheetID`.

**Why needed:** all downstream changes depend on knowing which sheet each face belongs
to. This runs once on the input mesh before any collapse loop.

---

## Step 2 — `src/partition_onering_by_sheet.cpp/.h` (new)

**Purpose:** given the flat face lists `Nsf` and `Ndf` from `igl::circulation`, split
them into per-sheet subsets using the precomputed `faceSheetID` from Step 1.

```cpp
void partition_onering_by_sheet(
    const std::vector<int> & Nsf,
    const std::vector<int> & Ndf,
    const Eigen::VectorXi & faceSheetID,
    std::vector<std::vector<int>> & sheets_Nsf,   // output: Nsf faces per sheet
    std::vector<std::vector<int>> & sheets_Ndf);  // output: Ndf faces per sheet
```

For a regular (non-seam) edge all faces land in a single sheet; the function still
returns a size-1 vector so callers are uniform.

---

## Step 3 — `src/single_collapse_data.h` (modify)

**Current:**
```cpp
struct single_collapse_data {
    Eigen::VectorXi b, FIdx_pre, FIdx_post, subsetVIdx;
    Eigen::MatrixXd UV_pre, UV_post, V_pre, V_post;
    Eigen::MatrixXi FUV_pre, FUV_post;
    std::vector<int> Nsv, Ndv;
};
```

**After:**
```cpp
struct SheetData {
    Eigen::MatrixXd UV_pre, UV_post;
    Eigen::MatrixXi FUV_pre, FUV_post;
    Eigen::VectorXi FIdx_pre, FIdx_post;
    Eigen::VectorXi subsetVIdx;
    Eigen::VectorXi b;            // local indices of vi, vj in this sheet
};

struct single_collapse_data {
    std::vector<SheetData> sheets; // one entry per sheet (size 1 for regular edges)
    Eigen::MatrixXd V_pre, V_post; // shared 3D geometry (same vi/vj across sheets)
    std::vector<int> Nsv, Ndv;    // kept for reference
};
```

The `query_coarse_to_fine` loop indexes into `sheets[sheetID]` for the UV transport,
so the per-sheet split is transparent to the rest of the pipeline.

---

## Step 4 — `src/SSP_collapse_edge.cpp` (modify)

This is the main change. The inner `SSP_collapse_edge(e, p, Nsv, Nsf, ...)` function
(lines 17–378) is modified as follows.

**Current flow (one path):**
```
get_collapse_onering_faces → V_pre / FUV_pre (all faces)
joint_lscm (once)
record data
```

**New flow (per-sheet):**
```
partition_onering_by_sheet(Nsf, Ndf, faceSheetID)
    → sheets_Nsf[], sheets_Ndf[]

for each sheet i:
    get_collapse_onering_faces(V, F, vi, vj, sheets_Nsf[i], sheets_Ndf[i])
        → V_pre_i, FUV_pre_i, V_post_i, FUV_post_i
    remove_unreferenced_lessF → compact local indices
    joint_lscm_case2(...)    ← always Case 2: (vi,vj) is boundary in each sheet
    if not valid → return false

record data.sheets[i] = {UV_pre_i, UV_post_i, FUV_pre_i, FUV_post_i, ...}
```

**Signature change:** `faceSheetID` is threaded in as a new `const Eigen::VectorXi &`
parameter to both overloads of `SSP_collapse_edge`.

**Why always Case 2:** for every sheet, `(vi, vj)` sits on that sheet's boundary (the
seam edge is the straight edge of the fan). The joint LSCM boundary-edge case handles
this exactly. For a regular (single-sheet) edge the function reduces to the existing
Case 0/1/2 dispatch — use the original `joint_lscm` with its existing case detection
when `sheets.size() == 1` and the edge is not a seam edge.

---

## Step 5 — `src/query_coarse_to_fine.cpp/.h` (modify)

**Current:** for each query vertex `q`, look up `decInfo[dIdx].UV_post` directly
(flat struct).

**After:** look up `decInfo[dIdx].sheets[sheetID].UV_post`, where `sheetID` is
determined by which sheet `FIdx[q]` belongs to.

Requires threading `faceSheetID` into `query_coarse_to_fine`:

```cpp
void query_coarse_to_fine(
    const std::vector<single_collapse_data> & decInfo,
    const Eigen::VectorXi & IM,
    const std::vector<std::vector<int>> & decIM,
    const Eigen::VectorXi & IMF,
    const Eigen::VectorXi & faceSheetID,   // ← new
    Eigen::MatrixXd & BC,
    Eigen::MatrixXi & BF,
    Eigen::VectorXi & FIdx);
```

Inside the parallel loop, before the UV transport step:
```cpp
int sheetID = faceSheetID(FIdx(qIdx));
const SheetData & sd = decInfo[dIdx].sheets[sheetID];
// use sd.UV_post, sd.UV_pre, sd.FUV_pre, sd.subsetVIdx
```

---

## Step 6 — `src/SSP_midpoint.cpp` (modify)

**Current problem:** the manifold check on lines 38–45 rejects all non-manifold meshes
before any collapse happens.

**Change:**
- Replace the single `connect_boundary_to_infinity(V, F, VO, FO)` + global manifold
  check with a per-sheet boundary connection.
- For each sheet `i` (from Step 1), connect only that sheet's boundary edges to a
  virtual vertex. Each sheet is then a closed manifold.
- Build one combined `(FO, E, EMAP, EF, EI)` from the union of all augmented sheets.
- The combined mesh is edge-manifold even though the original input is not, because
  seam edges are now sheet-boundary edges that connect to per-sheet virtual vertices.
- Thread `faceSheetID` through to `SSP_collapse_edge` and `query_coarse_to_fine`.

**Signature change:** add `const Eigen::VectorXi & faceSheetID` output parameter (or
compute it internally and pass downstream).

---

## Step 7 — `08_subdiv_remesh/main.cpp` (modify)

- Remove or guard the early `is_vertex_manifold` / `is_edge_manifold` abort (currently
  at the top of the file before Phase 2).
- Call `partition_into_sheets` to get `faceSheetID` right after loading the mesh.
- Pass `faceSheetID` into `SSP_midpoint` (or whichever decimation entry point is used).
- Pass `faceSheetID` into `query_coarse_to_fine`.

---

## Dependency order for implementation

```
Step 1  partition_into_sheets          (no deps on other new code)
Step 2  partition_onering_by_sheet     (depends on Step 1 output)
Step 3  single_collapse_data.h         (data layout, no code deps)
Step 4  SSP_collapse_edge              (depends on Steps 2, 3)
Step 5  query_coarse_to_fine           (depends on Step 3)
Step 6  SSP_midpoint                   (depends on Steps 1, 4, 5)
Step 7  main.cpp                       (depends on Steps 1, 6)
```

---

## What is NOT changing

- `src/joint_lscm.cpp` — no changes needed. Case 2 already handles boundary edges.
- `src/get_post_faces.cpp` — no changes; called per-sheet from Step 4.
- `src/remove_unreferenced_lessF.cpp` — no changes; called per-sheet from Step 4.
- `src/compute_barycentric.cpp` — no changes; called per-sheet from Step 5.
- `src/SSP_qslim.cpp`, `src/SSP_vertexRemoval.cpp` — same collapse-edge interface,
  pick up the changes automatically once `SSP_collapse_edge` is updated.
