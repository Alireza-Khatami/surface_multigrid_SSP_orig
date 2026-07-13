# `query_coarse_to_fine` Debug Log

## Failing assertion

```
query_coarse_to_fine.cpp : 104
assert(v1_vec.size() == 1);
```

`v1_vec` is the result of searching for `f(1)` (the second vertex of the current query face `BF.row(qIdx)`) inside `sd.subsetVIdx` (the one-ring vertex set of the routed SheetData). The assert means that vertex was not found exactly once — most likely **not found at all** (`size == 0`).

---

## Variable glossary (query_coarse_to_fine context)

| Variable | Type | Meaning |
|---|---|---|
| `decInfo` | `vector<single_collapse_data>` | Full history of every SSP collapse in chronological order. `decInfo[k]` is the k-th collapse. |
| `decInfo[k].sheets` | `vector<SheetData>` | One entry per sheet of the collapsed edge. Manifold edge → 1 sheet. Non-manifold seam → 2+ sheets. |
| `sd` (SheetData) | struct | The one-ring data for **one sheet** of collapse `k`. |
| `sd.global_sheet_id` | int | Which partition-into-sheets sheet this SheetData belongs to. |
| `sd.subsetVIdx` | `VectorXi` | Local→global map: `subsetVIdx(j)` = global `gV` index of the j-th vertex in the one-ring. This is the set the assert searches. |
| `sd.UV_pre` | `MatrixXd` | UV coordinates of every one-ring vertex **before** the collapse. Indexed by local index (same rows as subsetVIdx). |
| `sd.UV_post` | `MatrixXd` | UV coordinates of every one-ring vertex **after** the collapse. Same local indexing. |
| `sd.FUV_pre` | `MatrixXi` | Face connectivity in UV_pre space (local indices). Used to find which UV_pre triangle the query lands in. |
| `sd.FIdx_pre` | `VectorXi` | `FIdx_pre(t)` = global face index in `gF` for the t-th triangle of `FUV_pre`. Updated into `FIdx` after each backward step. |
| `decIM` | `vector<vector<int>>` | `decIM[f]` = list of collapse indices `k` such that face `f` was **in the one-ring** of collapse `k`. Indexed by global face index. |
| `IM` | `VectorXi` | Vertex index map: coarse mesh vertex `i` → original fine mesh vertex `IM(i)`. Used once at the start to convert `BF`. Identity in SSP (no renumbering). |
| `IMF` | `VectorXi` | Face index map: coarse mesh face `i` → original fine mesh face `IMF(i)`. Used once at the start to convert `FIdx`. Identity in SSP. |
| `faceSheetID` | `VectorXi` | `faceSheetID(f)` = which sheet face `f` belongs to (from `partition_into_sheets`). Used every iteration to route to the correct `SheetData`. |
| `BC` | `MatrixXd` (N×3) | Query barycentric coordinates. `BC.row(q)` = weights within `BF.row(q)`. Updated each backward step. |
| `BF` | `MatrixXi` (N×3) | Query face vertices (global gV indices). `BF.row(q)` = the 3 vertex indices of the face the query currently lives in. Updated each backward step. |
| `FIdx` | `VectorXi` (N) | Current face index (global gF index) for each query. Updated each backward step via `sd.FIdx_pre`. |

### What one backward step does

```
current state: BF.row(q) = (v0, v1, v2),  BC.row(q) = (b0, b1, b2),  FIdx(q) = f
              → query point = b0*V[v0] + b1*V[v1] + b2*V[v2]  (on coarse mesh)

1. Look up decIM[f] → find most recent collapse dIdx that touched f
2. Route to sheet:   sid = faceSheetID(f)  →  find SheetData sd with sd.global_sheet_id == sid
3. Map query into sd.UV_post:
       queryUV = b0*UV_post[local(v0)] + b1*UV_post[local(v1)] + b2*UV_post[local(v2)]
   (requires v0,v1,v2 all present in sd.subsetVIdx  ← ASSERT CHECKS THIS)
4. Find queryUV in sd.UV_pre triangles via compute_barycentric
5. Update: BF ← sd.subsetVIdx(sd.FUV_pre[best_tri]),
           BC ← new barycentric coords,
           FIdx ← sd.FIdx_pre[best_tri]
Repeat until no more collapse touches the current face.
```

---

## Hypotheses

### Hypothesis A — `faceSheetID` is stale

**Status:** ❌ RULED OUT

Diagnostic output confirmed `faceSheetID(fi) == routed_sheet` in ALL 5 failures. Sheet routing is correct.

---

### Hypothesis B — stale vertex indices in BF (confirmed root cause)

**Status:** ✅ CONFIRMED

The diagnostic shows 5 failures, all with the same pattern:

```
FAIL q598  fi=1082  faceSheetID=5  routed_sheet=5  dIdx=144  BF=(808,810,645)  missing: v2=645
           decIM[fi]: 144
           sheet 5 has v0=1 v1=1 v2=0
FAIL q621  fi=1202  faceSheetID=14 routed_sheet=14 dIdx=615  BF=(906,858,298)  missing: v1=858
FAIL q646  fi=1277  faceSheetID=6  routed_sheet=6  dIdx=550  BF=(978,952,954)  missing: v1=952
FAIL q684  fi=1432  faceSheetID=4  routed_sheet=4  dIdx=134  BF=(1111,1100,1106) missing: v2=1106
```

Two vertices of the face ARE in the one-ring of collapse `dIdx`, but one is not.

**Root cause chain:**

1. BF is initialized from `gF` (current coarse mesh state), which reflects ALL vertex remappings that happened across the full decimation.
2. `subsetVIdx` at collapse `dIdx` reflects the vertex state **as it was at that moment** — not the final coarse state.
3. Between `dIdx` and the end of decimation, some vertex `X` in face `fi` gets absorbed into its smaller-index survivor `S` (e.g. `X=952` becomes `S=645`... or vice versa). This remapping happens in **Pass 2** of a collapse `M > dIdx` that touches `fi`.
4. If the sheet that `fi` belongs to is **non-active** for collapse `M` (because the surviving vertex `S` has no incident faces on that sheet), then `M` is NOT added to `decIM[fi]`. `fi`'s entry in `decIM` ends at `dIdx`.
5. The backward walk picks `dIdx` as the last active collapse. It expects `BF` to contain the vertex indices **at the post-collapse state of `dIdx`**, but BF was initialized with the final coarse indices — which include the remapping from `X → S` by collapse `M`.

**Why two vertices match and one doesn't:**  
The surviving vertex `S` of collapse `M` (the one that absorbs `X`) is also present in the one-ring at `dIdx` (it's a neighbor vertex). So `S` appears in `subsetVIdx` — but as a neighbor, not as the vertex at face `fi`'s position 2. The vertex that WAS at position 2 of face `fi` at time `dIdx` was `X` (which was later replaced by `S`). Since BF shows `S` instead of `X`, the lookup fails.

---

### Hypothesis C — cross-sheet vertex leak

**Status:** ⬜ not tested (irrelevant — assert fires on the first step, before any backward step completes)

---

## Logging / validation plan

Add a pre-call validator in `coarse_fine_viz.cpp` that:
1. For each query `i`, finds the first collapse in `decIM[FIdx(i)]`
2. Routes to the sheet using `faceSheetID(FIdx(i))`
3. Checks whether `BF(i,0..2)` are all in the routed `sd.subsetVIdx`
4. For the first 5 failures, prints: `fi`, `sid`, `routed sheet id`, missing vertex, full `decIM[fi]` list, and sheet IDs available at that collapse

```
[c2f diag] q<i>: fi=<f>  sid=<sid>  routed_sheet=<X>  dIdx=<k>
           BF=(v0,v1,v2)  missing in subsetVIdx: v1=<Y>
           faceSheetID(fi)=<Z>  decIM[fi] collapses: k1 k2 k3 ...
           sheets at dIdx: sheetA sheetB ...
```

---

## Findings

- **272 queries checked, 5 failures (of first 756 total).**
- In every failure: `faceSheetID(fi) == routed_sheet` → routing is correct (Hypothesis A ruled out).
- In every failure: two of the three BF vertices ARE in `subsetVIdx`; only ONE is missing.
- The missing vertex is always at position 1 or 2 (never position 0 = the coarse vertex being queried).
- `decIM[fi]` has only 1–4 entries. `dIdx` is the last one. No collapse after `dIdx` is tracked.
- Confirmed: non-active-sheet collapses after `dIdx` remapped a vertex in `fi` without updating `decIM[fi]`.

---

## Fix

**Initialize BF from `sd.FIdx_post` instead of current `gF`.**

For each query (coarse vertex `vi`, face `fi`):
1. Find the last active collapse `dIdx` in `decIM[fi]`.
2. Route to `SheetData sd` via `faceSheetID(fi)`.
3. Find `fi` in `sd.FIdx_post` → row `r`.
4. Read the vertex state of `fi` at collapse `dIdx` time:
   `gv0 = subsetVIdx(FUV_post(r,0))`, etc.
5. Rotate so `vi` is first. Set `BC = (1,0,0)`.

At collapse `dIdx`, face `fi`'s vertices are `(…, X, …)` where `X` is the vertex that was **later** (non-actively) remapped to the surviving `S`. `X` IS in `subsetVIdx` of `dIdx` ✓. The current gF would show `S` (wrong). FIdx_post shows `X` (correct).

Implemented in `coarse_fine_viz.cpp` inside `coarse_fine_compute_and_save()`.

---

## Variable glossary (query_coarse_to_fine context)

  The backward walk treats the query as a point that lives inside a triangle on the current (coarse) mesh, and traces it back one collapse at a
  time:

  - BF.row(q) — the 3 global vertex indices (gV row numbers) of the triangle the query currently lives inside. Starts as the coarse face, gets
  overwritten each step to move to a finer triangle.
  - BC.row(q) — barycentric weights within that triangle. Starts as (1,0,0) for a vertex query.
  - FIdx(q) — the global face index (gF row) of that triangle. Used to look up decIM and faceSheetID.
  - sd.subsetVIdx — the one-ring vertex set for the routed sheet: maps local one-ring indices to global gV indices. The assert checks that
  BF(q,1) appears exactly once here.
  - sd.UV_pre / UV_post — UV coordinates before/after the collapse, indexed by the same local indices as subsetVIdx. The backward step maps the
  query from UV_post into UV_pre to find the finer-mesh triangle.
  - decIM[f] — which collapse steps claimed face f is in their one-ring. The function picks the latest one < current dIdx.
  - faceSheetID(f) — which sheet face f belongs to; used to select which SheetData to route into. If this is stale, the wrong sheet gets picked
  → its subsetVIdx won't contain the query's vertices → assert fires.
