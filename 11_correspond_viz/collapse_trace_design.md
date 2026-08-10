# Collapse Trace Visualization — Design Notes

## What the feature does

Click a fine vertex → find every SSP collapse step that touched that vertex's
neighborhood → show a step-by-step replay.

From the bundle data:

- `decIM[face_idx]` = sorted list of collapse indices that touched that fine face
- `decInfo[collapse_idx]` → `SheetData` → `UV_pre`, `UV_post`, `FUV_pre` = the
  local UV parameterization domain *before* and *after* that collapse
- `subsetVIdx` = which global vertices are in the sheet

For a clicked fine vertex with ID `v`:
1. Find all fine faces containing `v` → call `decIM[face]` for each → union of
   collapse indices
2. For each collapse step in that set, find the sheet that includes `v` via
   `subsetVIdx`
3. In that sheet's UV domain, locate where `v` sits in `UV_pre`, draw an arrow
   to `UV_post` — that's the displacement in UV space
4. The "canonical view" = render the UV domain as a flat 2D mesh inside
   Polyscope (Z=0 plane), swapped out at each step with Forward / Back buttons

---

## Implementation options

| Option | Tradeoff |
|---|---|
| **3D trajectory polyline** (simplest) | Trace the 3D position of the clicked point across all collapse steps as a single polyline on the current mesh. No step navigation needed — the entire path is visible at once. Fast to implement. |
| **UV domain flat-mesh view** (what was described) | Render `UV_pre` triangulation as a flat 3D mesh (Z-offset plane) inside Polyscope. Arrows show `UV_pre → UV_post` displacement. Forward / Back buttons step through collapses. Most informative for understanding LSCM distortion. |
| **2D matplotlib popup** | Pop open a matplotlib window (non-blocking) for the UV plot. Cleaner 2D rendering but lives outside Polyscope. |
| **Heatmap on fine mesh** (global view) | Color every fine face by how many collapses touched it, or by which collapse touched it most recently. No navigation — gives the whole picture at once. Good complement to the step-by-step view. |

**Recommended**: implement both:
- The **3D trajectory polyline** immediately on click for quick context.
- The **UV flat-mesh view** with Forward / Back for the deep dive.

Skip matplotlib — staying inside Polyscope is cleaner. The heatmap is a cheap
bonus once the other two are in place.

---

## Vertex destruction — what actually happens

In a standard edge collapse `(u, v) → u` in 3D topology, vertex `v` is destroyed
and the collapse faces are deleted. But the SSP bundle does NOT store it that way.

**Key observation**: both `UV_pre` and `UV_post` have the same number of rows
(`uvRows`). The collapse is encoded as a **UV domain deformation**, not a vertex
deletion:

- `UV_pre[local_v]`  = where `v` sits in UV space **before** the collapse
- `UV_post[local_v]` = where `v` sits **after** — now coinciding with `u`'s UV
  position (they merged)

So no vertex is missing from the stored arrays. The collapse is a warp: `v`
slides to `u`'s UV location. This is exactly what `query_coarse_to_fine`
exploits — it evaluates a coarse-state UV point and finds where it lands in
the finer state.

**What CAN disappear**: the collapse faces (triangles containing both `u` and
`v`) are removed from the 3D topology and will not appear in any later `decIM`
entries. Flap faces (containing `v` but not `u`) survive and continue to
accumulate collapse indices.

**Direction**: `query_coarse_to_fine` walks backward (coarse → fine). For the
collapse trace feature we walk forward (fine → coarse): `decIM[face]` is
already sorted in collapse order, so we iterate forward through it.

---

## Key data paths in c2f_query.py

```
Bundle.decIM        list[list[int]]     per original fine face → collapse indices
Bundle.decInfo      list[CollapseData]  per collapse index
  CollapseData.sheets  list[SheetData]
    SheetData.subsetVIdx   (svSz,)  int32  global vertex indices in this sheet
    SheetData.UV_pre       (uvRows, 2)     UV coords before collapse
    SheetData.UV_post      (uvRows, 2)     UV coords after collapse
    SheetData.FUV_pre      (fuvRows, 3)   triangulation (local UV indices)
    SheetData.FIdx_pre     (fuvRows,)     global fine face index per UV triangle
```
