# SheetData: what is saved and how, traced from source

---

## 1. Setup (before any collapse)

**`SSP_midpoint.cpp:110–157`**

```
igl::connect_boundary_to_infinity(V, F, VO, FO)
```
Appends a single "infinity vertex" (`infVtx = V.rows()-1`) and one infinity triangle for every
boundary edge.  All faces with index `>= numOrigFaces` contain `infVtx`.

```
partition_into_sheets(OF.topRows(numOrigFaces), faceSheetID, numSheets)
```
BFS over the **original** (non-infinity) faces only.  Two faces are in the same sheet if they
share a manifold interior edge (exactly 2 incident faces).  Boundary edges (1 face) and
non-manifold / seam edges (3+ faces) act as sheet walls.  Every original face gets an integer
`faceSheetID(f)`.

```
igl::vertex_triangle_adjacency(..., VF_sspm, ...)
```
`VF[v]` = list of all face indices incident to vertex `v`.  This list grows as collapses merge
`VF[d]` into `VF[s]` (lazy; null faces are filtered at query time).

---

## 2. Per-collapse outer loop

**`SSP_collapse_edge` (outer overload) — `SSP_collapse_edge.cpp:593–761`**

For each edge `e` popped from the priority queue:

```
s = min(E(e,0), E(e,1))   // surviving vertex (smaller global index)
d = max(E(e,0), E(e,1))   // dying / absorbed vertex
vi = s,  vj = d
```

Collect one-rings via VF adjacency (**real + infinity faces; null faces filtered**):

```
collect_onering(E(e,1), Nsf, Nsv_verts)   // all faces of E(e,1)
collect_onering(E(e,0), Ndf, Ndv_verts)   // all faces of E(e,0)
```

Note on naming vs. eflip:
- If `eflip=0` (E(e,0) < E(e,1)):  s = E(e,0),  d = E(e,1),  Nsf = faces of d,  Ndf = faces of s
- If `eflip=1` (E(e,0) > E(e,1)):  s = E(e,1),  d = E(e,0),  Nsf = faces of s,  Ndf = faces of d

Regardless of eflip, **Pass 2 operates on `nV2Fd = (!eflip ? Nsf : Ndf)`**, which is always
"all faces of d".

---

## 3. Per-collapse inner loop

**`SSP_collapse_edge` (inner overload) — `SSP_collapse_edge.cpp:14–585`**

### 3a. Sheet bucketing

```cpp
// SSP_collapse_edge.cpp:194–208
for (int f : Nsf) {
    if (null_face(f) || f >= numOrigFaces) continue;   // skip null + infinity faces
    sheets_Nsf[faceSheetID(f)].push_back(f);
}
for (int f : Ndf) {
    if (null_face(f) || f >= numOrigFaces) continue;
    sheets_Ndf[faceSheetID(f)].push_back(f);
}

// Active sheets: only those where BOTH Nsf and Ndf have real faces
for (auto & kv : sheets_Nsf)
    if (sheets_Ndf.count(kv.first) && !sheets_Ndf[kv.first].empty())
        active_sheets.insert(kv.first);
```

**Active sheet** = a sheet that has at least one real (non-null, non-infinity) face containing
E(e,1) AND at least one containing E(e,0).  In other words: both endpoints of the collapsed
edge must appear on that sheet.

---

### 3b. Per-sheet UV loop (for each `sid` in `active_sheets`)

**`SSP_collapse_edge.cpp:266–503`** — this entire block runs **before** any modification to
`V` or `F`.  All reads of `F` and `V` capture state at exact collapse moment.

#### Step 1 — build pre-collapse one-ring for this sheet

**`get_collapse_onering_faces.cpp:1–134`** called as:
```cpp
get_collapse_onering_faces(V, F, vi, vj, Nsf_si, Ndf_si,
                           FIdx_pre_si, FIdx_post_si,
                           F_ring_pre_si, F_ring_post_si)
```

Inside `get_collapse_onering_faces`:

1. Iterate `Nsf_si ∪ Ndf_si` (real faces of this sheet for each endpoint).  
   For each face: skip if null, skip if any vertex has `isinf(V(...,0))`, skip if it contains
   neither `vi` nor `vj`.  Collect surviving global face row indices into
   `FIdx_ring_pre_duplicated`.

2. `igl::unique(tmp, FIdx_ring_pre)` — sort and deduplicate.  
   **`FIdx_pre` = sorted, deduplicated global `gF` row indices of all one-ring real faces for
   this sheet, as they exist in `gF` at this exact moment (before Pass 1 and Pass 2).**

3. `igl::slice(F, FIdx_ring_pre, 1, F_ring_pre)` — read those rows out of current `gF`.  
   **`F_ring_pre` = the actual `[v0, v1, v2]` rows from current `gF`** (contains global vertex
   indices, including the dying vertex `d`).

#### Step 2 — compute FIdx_post and F_ring_post

Still inside `get_collapse_onering_faces`, immediately after Step 1:

```cpp
// get_collapse_onering_faces.cpp:88–91
get_post_faces(F_ring_pre, vi, vj, f_keep, F_ring_post);
igl::slice(FIdx_ring_pre, f_keep, 1, FIdx_ring_post);
```

Inside `get_post_faces` (`get_post_faces.cpp`):

```cpp
// Keep a face only if it does NOT contain BOTH vi and vj simultaneously.
// (A face containing both = a flap face = topologically killed by the collapse.)
for (int ii = 0; ii < F_pre.rows(); ii++) {
    if ((v0!=vi || v1!=vi || v2!=vi) ||   // i.e., NOT (all three != vi) → contains vi
        (v0!=vj || v1!=vj || v2!=vj))     // and NOT contains vj
    // The actual logic: keep if NOT (contains vi) OR NOT (contains vj)
    // = keep if it does NOT have both vi and vj = not a flap
        f_keep_vec.push_back(ii);
}
// In kept faces, replace vj → vi (global-index d→s remap in local copy)
for (r, c): if F_post(r,c) == vj → F_post(r,c) = vi;
```

So:
- **`FIdx_post` = global `gF` row indices of the faces that survive the collapse (flap faces
  excluded).**  The number of flap faces = 2 for a manifold edge, >2 for a seam edge.
- **`F_ring_post` = `F_ring_pre` after removing flap rows and replacing `vj→vi` (global
  indices)**.  This is not stored in SheetData; only `FIdx_post` is.
- `f_keep` is the local row index vector (indices into `FIdx_ring_pre`) of surviving faces.

**`FIdx_post` is derived purely from `FIdx_pre` and the face topology at collapse time — it is
NOT read from `gF` after Pass 1 or Pass 2.**

#### Step 3 — compact local mesh (subsetVIdx)

Back in `SSP_collapse_edge`:

```cpp
// SSP_collapse_edge.cpp:284–288
remove_unreferenced_lessF(V, F_ring_pre_si, V_pre_si, FUV_pre_si, IM_tmp, subsetVIdx_si);
```

Inside `remove_unreferenced_lessF`:

1. Flatten all entries of `F_ring_pre` into one vector, sort, take unique values.  
   **`subsetVIdx` = sorted ascending list of global vertex indices that appear in the one-ring
   faces of this sheet, as read from `gF` at this exact moment.**

2. `IM_tmp[global_v] = local_idx` — inverse map.

3. `FUV_pre_si` = `F_ring_pre_si` with every global index replaced by its local index via
   `IM_tmp`.  **Row `r` of `FUV_pre_si` is the local-index version of the face at global row
   `FIdx_pre(r)`.  Column order is identical — same c=0,1,2 layout.**

4. `V_pre_si` = `igl::slice(V, subsetVIdx, 1)` — 3-D positions of the one-ring vertices.

#### Step 4 — find b (local indices of vi and vj)

```cpp
// SSP_collapse_edge.cpp:293–298
for (int ii = 0; ii < subsetVIdx_si.size(); ii++) {
    if      (subsetVIdx_si(ii) == vi) b_si(0) = ii;
    else if (subsetVIdx_si(ii) == vj) b_si(1) = ii;
}
```

**`b(0)` = local index (into `subsetVIdx`) of `s` (surviving vertex).**  
**`b(1)` = local index (into `subsetVIdx`) of `d` (dying vertex).**  
Because `subsetVIdx` is sorted ascending and `vi = s < d = vj` globally, `b(0) < b(1)` always.

#### Step 5 — build post-collapse local mesh

```cpp
// SSP_collapse_edge.cpp:322–326
MatrixXd V_post_si = V_pre_si;
V_post_si.row(b_si(0)) = p;        // move s to the new collapsed position p
get_post_faces(FUV_pre_si, b_si(0), b_si(1), FUV_pre_keep_si, FUV_post_si);
```

`get_post_faces` is called again but now with **local** indices:
- Removes flap rows from `FUV_pre_si` (rows containing both `b(0)` and `b(1)`).
- In surviving rows replaces `b(1) → b(0)` (local d→s remap).

**`FUV_post` = local-index face matrix for the post-collapse one-ring, with flap faces removed
and d→s applied.**

#### Step 6 — fan_walk and joint_lscm

`fan_walk_local` walks the face fan around each endpoint to build winding-order neighbor lists
(`Nsv_local`, `Ndv_local`) for `joint_lscm`.

```cpp
joint_lscm(V_pre_si, FUV_pre_si, V_post_si, FUV_post_si,
           b_si(0), b_si(1), Nsv_local, Ndv_local,
           UV_pre_si, UV_post_si)
```

**`UV_pre`** = LSCM parameterization of the one-ring on the pre-collapse 3-D geometry.  
**`UV_post`** = LSCM parameterization on the post-collapse 3-D geometry.  
Both are `(nSubset × 2)` matrices indexed by local index (same row = same entry of `subsetVIdx`).

---

### 3c. SheetData struct — complete field trace

```cpp
// SSP_collapse_edge.cpp:479–490
sd.global_sheet_id = sid;        // integer label from partition_into_sheets
sd.b          = b_si;            // local indices of [s, d]
sd.subsetVIdx = subsetVIdx_si;   // global indices of one-ring vertices (sorted asc)
sd.UV_pre     = UV_pre_si;       // LSCM UV, pre-collapse, local-indexed
sd.UV_post    = UV_post_si;      // LSCM UV, post-collapse, local-indexed
sd.FUV_pre    = FUV_pre_si;      // face connectivity, pre-collapse, LOCAL indices
sd.FUV_post   = FUV_post_si;     // face connectivity, post-collapse, LOCAL indices
sd.FIdx_pre   = FIdx_pre_si;     // global gF row of each FUV_pre row (same ordering)
sd.FIdx_post  = FIdx_post_si;    // global gF row of each FUV_post row
```

| Field | Type | What it holds | When valid |
|---|---|---|---|
| `global_sheet_id` | `int` | Sheet label from `partition_into_sheets` | Always |
| `b(0)` | local idx | Local index of s (surviving) in `subsetVIdx` | Always |
| `b(1)` | local idx | Local index of d (dying) in `subsetVIdx` | Always |
| `subsetVIdx` | `VectorXi` | Global `gV` row indices of all one-ring vertices; sorted ascending | Captured before Pass 1/2 |
| `UV_pre` | `MatrixXd (N×2)` | LSCM UV of each one-ring vertex on pre-collapse geometry | Captured before Pass 1/2 |
| `UV_post` | `MatrixXd (N×2)` | LSCM UV of each one-ring vertex on post-collapse geometry | Computed from `V_post_si` (V_pre with s moved to p) |
| `FUV_pre` | `MatrixXi (F×3)` | One-ring face connectivity in **local** indices (row r = face at `FIdx_pre(r)`) | Captured before Pass 1/2 |
| `FUV_post` | `MatrixXi (F'×3)` | Surviving faces in local indices with d→s applied (`F' = F - numFlapFaces`) | Derived from FUV_pre |
| `FIdx_pre` | `VectorXi (F)` | Global `gF` row of each `FUV_pre` row | Captured before Pass 1/2 |
| `FIdx_post` | `VectorXi (F')` | Global `gF` row of each `FUV_post` row | Subset of `FIdx_pre` |

**Critical invariant:** `FUV_pre(r, c)` and `FIdx_pre(r)` describe the **same face** at row `r`.
Column `c` gives the local index; `FIdx_pre(r)` gives the global `gF` row.  This is the key
used by the backward walk fix in `query_coarse_to_fine.cpp`.

---

### 3d. FIdx_combined and decIM update

After all active sheets are processed:

```cpp
// SSP_collapse_edge.cpp:500–501, 509–510
for (int ii = 0; ii < FIdx_pre_si.size(); ii++)
    FIdx_combined.insert(FIdx_pre_si(ii));   // union across all active sheets

// outer overload, SSP_collapse_edge.cpp:714–715
for (int ii = 0; ii < FIdx_onering_pre.size(); ii++)
    decIM[FIdx_onering_pre(ii)].push_back((int)decInfo.size() - 1);
```

**`decIM[f]` = list of collapse indices (dIdx) where global face `f` appeared in the one-ring
of an active sheet.**  Only active-sheet faces are inserted.  Non-active-sheet faces of d that
get rewritten by Pass 2 are NOT inserted into `decIM`.

---

## 4. Topology update (AFTER all SheetData is captured)

```cpp
// SSP_collapse_edge.cpp:519–580
V.row(s) = p;  V.row(d) = p;       // move both endpoints to new position

// Pass 1: kill flap faces (those containing both s and d)
for (int f : nV2Fd) {               // nV2Fd = all faces of d
    if (contains s and d) {
        null out F(f, *);           // F row → [NULL, NULL, NULL]
        kill the edge opposite s;
        flap_count++;
    }
}
data.numFlapFaces = flap_count;    // 2 = manifold interior, >2 = seam edge

// Pass 2: remap d → s in ALL surviving faces of d
for (int f : nV2Fd) {
    if (!null_face(f))
        replace d with s in F(f, *)
}
kill_edge(e);
```

**Pass 2 operates on ALL faces of d, across ALL sheets — not just active-sheet faces.**  
This is what causes stale BF entries for non-active-sheet faces (the bug fixed in
`query_coarse_to_fine.cpp`).

---

## 5. Seam edges: decimation across multiple sheets

A **seam edge** is an edge shared by ≥ 3 original faces (= one non-manifold edge that borders
multiple sheets).

`partition_into_sheets` stops BFS at non-manifold edges, so each of those 3+ incident faces is
on a different sheet.

When the seam edge `(s, d)` is collapsed:

1. **Sheet bucketing:** every sheet that has real faces of BOTH s and d is added to
   `active_sheets`.  For a seam edge with 3 incident faces on 3 sheets, if all 3 sheets have
   faces of both s and d, all 3 are active.

2. **UV loop runs once per active sheet** — independent `get_collapse_onering_faces` +
   `joint_lscm` per sheet → 3 separate `SheetData` objects stored in `data.sheets`.

3. **Topology update runs once, globally** — Pass 1 kills all flap faces across all sheets
   (`data.numFlapFaces > 2` for a seam edge), Pass 2 rewrites d→s in all faces of d across
   all sheets.

4. **`decIM` update covers the union of `FIdx_pre` from all active sheets.**

So: yes, for a seam edge, decimation (UV computation) is done per-sheet, but the geometry
update (vertex and face table modification) is done once, globally.

---

## 6. Non-active sheets (the stale-vertex bug)

A sheet is **non-active** for a given collapse if the collapsed edge's two vertices don't both
appear on that sheet.  This can happen when:

- d has faces on sheet B (e.g. inherited via VF merge from a previous collapse), but s has no
  face on sheet B.
- Or: the collapse corresponds to a non-seam edge on one sheet and the other sheet's face
  distribution just doesn't bring both endpoints together.

For non-active sheets:
- No `SheetData` is created → `decIM` is NOT updated for those faces.
- **Pass 2 still rewrites `d→s` in those faces** (because `nV2Fd` = all faces of d).

Effect on backward walk: `gF[f]` now contains s where it used to have d.  When the backward
walk initializes `BF` from the coarse `gF` and then searches `subsetVIdx` for that value,
it finds s instead of d — either missing entirely (Type A) or finding the wrong vertex (Type B).
Fix: use `FIdx_pre` column alignment instead of searching `subsetVIdx` by value.





● Good question. The answer is that the backward walk never undoes anything — the time-capsule structure makes recovery automatic. Let me trace it
  precisely.

  ---
  What each collapse does to global vertex space

  Every d→s remap in Pass 2 overwrites entries of gF. Vertex index d still exists in gV (both gV[s] and gV[d] are moved to position p), but gF no longer
  references d in the one-ring faces — they all say s now.

  The SheetData captured at that collapse still has d in subsetVIdx, because it was captured before Pass 2 ran.

  ---
  How the backward walk recovers this

  The walk in query_coarse_to_fine.cpp runs backwards through dIdx values. At each step:

  BF(qIdx, 0) = sd.subsetVIdx(sd.FUV_pre(idxToFUV, 0));
  BF(qIdx, 1) = sd.subsetVIdx(sd.FUV_pre(idxToFUV, 1));
  BF(qIdx, 2) = sd.subsetVIdx(sd.FUV_pre(idxToFUV, 2));
  FIdx(qIdx)  = sd.FIdx_pre(idxToFUV);

  BF is overwritten at every step. The vertices written into BF come from subsetVIdx frozen at collapse time dIdx — which means they are whichever global
  indices were live at that moment, including d before it was absorbed.

  Concrete example with 3 collapses:

  dIdx=0: collapse (2→s, 3→d). subsetVIdx_0 = [1,2,3,4]   — both 2 and 3 present
  dIdx=1: collapse (1→s, 2→d). subsetVIdx_1 = [0,1,2,4]   — 2 still there (3 already gone)
  dIdx=2: collapse (0→s, 1→d). subsetVIdx_2 = [0,1,4]     — 1 still there (2,3 already gone)

  The walk starts from the coarse face (BF = global indices after all collapses), then steps backward:

  - At dIdx=2: subsetVIdx_2(FUV_pre_2(...)) → BF gets e.g. [0, 1, 4]
  - At dIdx=1: subsetVIdx_1(FUV_pre_1(...)) → BF might get [0, 2, 4] — vertex 2 reappears because at collapse 1, vertex 2 was still in the one-ring as a
  distinct vertex (it hadn't been absorbed yet)
  - At dIdx=0: subsetVIdx_0(FUV_pre_0(...)) → BF might get [1, 3, 4] — vertex 3 reappears, because subsetVIdx_0 was captured before collapse 0 killed it

  At the earliest collapse where queryFIdx was in the one-ring, BF naturally contains fine-mesh global indices, including vertices that were later absorbed.
  No explicit "undo" is needed — stepping backwards through subsetVIdx time-capsules reverses the remapping automatically.

  ---
  Why decIM is the key link

  The loop condition is:

  vector<int> dIdxList = decIM[queryFIdx];
  // find the largest dIdx in dIdxList that is < current dIdx

  decIM[f] records every collapse where face f was in an active-sheet one-ring. So when dIdx=1 updates FIdx(qIdx) to some face g, the walk next checks
  decIM[g] to find an earlier collapse where g appeared. At that earlier collapse, subsetVIdx has g's vertices as they were at that moment — which includes
  any vertex that was later absorbed.

  The whole chain terminates when decIM[queryFIdx] has no entry older than the current dIdx. At that point BF holds barycentric coordinates on a face that
  existed at the START of decimation — i.e., a fine mesh face — and BF values are fine-mesh global vertex indices.

✻ Brewed for 1m 39s

> what do we save to idxToFUV during the decimation ?

● idxToFUV is not saved during decimation at all. It is computed on-the-fly during the backward walk in query_coarse_to_fine.cpp. Here is the exact code
  that sets it:

  // query_coarse_to_fine.cpp:121–125
  Eigen::MatrixXd B;
  compute_barycentric(queryUV, sd.UV_pre, sd.FUV_pre, B);

  VectorXd distToValid = -B.rowwise().minCoeff();
  double minD = 1.0;
  int idxToFUV;
  for (int bb = 0; bb < distToValid.size(); bb++) {
      if (distToValid(bb) < minD) {
          minD = distToValid(bb);
          idxToFUV = bb;
      }
  }

  What B is: compute_barycentric takes queryUV (a 2D point in UV_post space) and tests it against every face in sd.UV_pre / sd.FUV_pre. It returns one row
  per face, each row being the barycentric coordinates [b0, b1, b2] of queryUV with respect to that face's UV triangle.

  What distToValid is: -B.rowwise().minCoeff() — for each face row, take the most negative barycentric coordinate and negate it. If all coords are ≥ 0 the
  point is inside that triangle (dist = 0). The larger the value, the further outside.

  What idxToFUV is: the row index (into FUV_pre) of the face where queryUV is most "inside" — i.e. the face in the pre-collapse UV layout that best contains
  the query point.

  Then it is used immediately:

  BC.row(qIdx) = B.row(idxToFUV);
  BF(qIdx, 0)  = sd.subsetVIdx(sd.FUV_pre(idxToFUV, 0));
  BF(qIdx, 1)  = sd.subsetVIdx(sd.FUV_pre(idxToFUV, 1));
  BF(qIdx, 2)  = sd.subsetVIdx(sd.FUV_pre(idxToFUV, 2));
  FIdx(qIdx)   = sd.FIdx_pre(idxToFUV);

  It is a local variable that lives only for that one iteration of the while loop — pointing at which row of FUV_pre the query UV landed in, so the walk
  knows which pre-collapse face to step into next.
