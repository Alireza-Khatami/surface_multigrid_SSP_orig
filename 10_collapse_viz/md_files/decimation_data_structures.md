# SSP Decimation: Data Structures and Correspondence Pipeline

Treating `joint_lscm` as a black box: you hand it a pre-collapse patch, a
post-collapse patch, and the winding-order walks around both endpoints of the
collapsed edge. It returns two UV parameterizations — one for the patch before
the collapse (`UV_pre`), one after (`UV_post`) — aligned so that a point at
UV coordinate `(u,v)` in `UV_post` corresponds to the same material point at
`(u,v)` in `UV_pre`.

---

## 1 — What Happens at Each Collapse

The whole mesh is stored in `V` (vertices, rows = vertex indices) and `F`
(faces, rows = face indices). Decimation proceeds by repeatedly picking an
edge `(s, d)` where `s` is the surviving vertex and `d` is the dying vertex.

### Before the collapse — snapshot

For each *active sheet* of this edge (a sheet where both endpoints have
real faces), the code:

1. **Collects the one-ring** — all real faces that touch either `s` or `d`
   on this sheet. These are split into:
   - `Nsf` faces (one-ring of `s`)
   - `Ndf` faces (one-ring of `d`)
   - Together they form `FIdx_pre`: the sorted list of global face row numbers
     that will exist just before the collapse fires.

2. **Compacts the patch** — `remove_unreferenced_lessF` extracts only the
   vertices referenced by those faces into a small local mesh
   (`V_pre`, `FUV_pre`). It also produces `subsetVIdx`: a sorted list of
   global vertex indices, where `subsetVIdx[i]` is the global vertex for
   local index `i`.

3. **Builds fan walks** — `fan_walk_local` traces the CCW ring of neighbors
   around `s` (in `Nsf` faces) and around `d` (in `Ndf` faces), producing
   `Nsv_local` and `Ndv_local` as sequences of *local* vertex indices
   (infVtx → `-1` sentinel for open boundaries).

4. **Calls joint_lscm** — receives `V_pre/FUV_pre` (pre patch), the
   post-collapse version of the same patch (`V_post/FUV_post`, which has
   flap faces removed and d-corners replaced by s), `b` (local indices of
   s and d), and the two fan walks. Returns `UV_pre` and `UV_post`.

5. **Stores `SheetData`** — saves every field listed in §3 below for this
   sheet into `data.sheets`.

6. **Accumulates `FIdx_combined`** — a running set that is the union of
   `FIdx_pre` across all active sheets for this collapse.

### The collapse itself — topology update

**Pass 1 — kill flap faces.** A flap face is a triangle that contains *both*
`s` and `d`. After collapsing `(s,d)`, those triangles degenerate to a line.
For each flap face:
- The edge `e_kill` (opposite the `s`-corner, spanning `d` and the third
  vertex) is redirected: all EMAP entries in the one-ring that pointed to
  `e_kill` now point to the surviving edge `e_keep` (opposite the `d`-corner).
- The face row is nulled out: `F(f, 0/1/2) = IGL_COLLAPSE_EDGE_NULL`.

**Pass 2 — remap d → s.** For every surviving face that still carries `d` in
one of its columns, `F(f, col) = s`. The two edge entries adjacent to that
corner are also updated to replace `d` with `s` in the `E` matrix.

After both passes, `V.row(s)` is set to the new collapsed position `p`, and
the edge `e` itself is killed.

### After the collapse — bookkeeping

```
decInfo.push_back(data);                         // record this collapse
int dIdx = (int)decInfo.size() - 1;

for (int f : FIdx_onering_pre)                   // FIdx_combined as a sorted vector
    decIM[f].push_back(dIdx);                    // "face f was in the one-ring of collapse dIdx"
```

This is the only write site for both `decInfo` and `decIM`.

---

## 2 — The Two Bookkeeping Tables

### `decInfo` — the collapse journal

```cpp
std::vector<single_collapse_data>   decInfo;
// decInfo[dIdx]  = everything about the dIdx-th collapse
```

Indexed by collapse number. Appended once per successful collapse.

### `decIM` — the face-to-collapse index

```cpp
std::vector<std::vector<int>>       decIM;
// decIM[faceRow]  = sorted list of dIdx values for every collapse
//                   whose active-sheet one-ring included faceRow
```

Indexed by global face row. Each inner list is in ascending order because
collapses are appended in chronological order.

**Key property:** if `decIM[f] = [a, b, c, ...]`, it means face `f` was a
live active-sheet one-ring member at collapses `a`, `b`, `c`, ... (oldest
first). Any collapse NOT in this list either did not touch `f` at all, or
touched it only on a *non-active* sheet (Pass 2 remapped `f`'s `d`-corner
to `s` in `F`, but no `decIM` entry was written).

---

## 3 — `SheetData` Fields

One `SheetData` is stored per active sheet per collapse.

| Field | Type | Size | Content |
|---|---|---|---|
| `global_sheet_id` | `int` | — | Which sheet this data belongs to. Always 0 on manifold meshes. |
| `subsetVIdx` | `VectorXi` | (nLocalV,) | Global vertex indices of the compact patch, sorted ascending. Maps **local → global**. |
| `b` | `VectorXi` | (2,) | Local indices of the two edge endpoints. `b(0)` = local index of `s` (surviving); `b(1)` = local index of `d` (absorbed). |
| `FIdx_pre` | `VectorXi` | (nFacesPre,) | Global face row indices of the **pre-collapse** one-ring for this sheet. Row `r` here corresponds to row `r` of `FUV_pre`. |
| `FUV_pre` | `MatrixXi` | (nFacesPre × 3) | Face connectivity in **local** index space for the pre-collapse patch. `FUV_pre(r, c)` is a local vertex index (index into `subsetVIdx`). |
| `UV_pre` | `MatrixXd` | (nLocalV × 2) | LSCM UV coords of the **pre-collapse** patch. `UV_pre.row(i)` is the UV of `subsetVIdx(i)`. |
| `UV_post` | `MatrixXd` | (nLocalV × 2) | LSCM UV coords of the **post-collapse** patch. Same vertex layout as `UV_pre`. |
| `FIdx_post` | `VectorXi` | (nFacesPost,) | Global face row indices of the **surviving** faces (flap faces excluded). |
| `FUV_post` | `MatrixXi` | (nFacesPost × 3) | Face connectivity in local index space for the post-collapse patch. d-corners replaced by `b(0)`. |

**Invariant:** `FIdx_pre.size() == FUV_pre.rows()` and `FIdx_post.size() ==
FUV_post.rows()`. The row correspondence is exact — `FIdx_pre(r)` is the
global face for `FUV_pre.row(r)`.

---

## 4 — `single_collapse_data` Fields

```cpp
struct single_collapse_data {
    std::vector<SheetData> sheets;   // one per active sheet
    Eigen::MatrixXd V_pre, V_post;  // 3-D geometry snapshot (first successful sheet)
    std::vector<int> Nsv, Ndv;      // fan walks in local indices (first sheet)
    int numFlapFaces = 0;           // 2 for manifold interior, >2 for seam
};
```

`V_pre` and `V_post` are stored only for visualization; the backward walk
never uses them. The walk uses only `sheets[*].UV_pre`, `UV_post`, `FUV_pre`,
`FIdx_pre`, `subsetVIdx`.

---

## 5 — Initialization Before the Backward Walk

Before `query_coarse_to_fine` runs, `BF`, `BC`, and `FIdx` are built in
`coarse_fine_viz.cpp`. One query row per live coarse vertex:

```
BC(i, *)  = (1, 0, 0)     -- barycentric coords: vertex itself sits at corner 0
BF(i, *)  = (vi, v1, v2)  -- global vertex indices of some coarse face incident on vi
FIdx(i)   = fi             -- global row of that coarse face
```

To find the right face for each coarse vertex `vi`:
1. Look up any live face `fi` in `gVF[vi]` (vertex-face adjacency of the
   coarse mesh).
2. Find the most recent entry in `gDecIM[fi]` that is `< nDec` (total
   collapses performed so far). Call it `dIdx`.
3. If no such entry exists, `fi` was never in any active-sheet one-ring:
   read `gF.row(fi)` directly (the face still has fine-mesh global vertex
   indices because no collapse ever touched it), rotate so `vi` is column 0.
4. If `dIdx` exists, look up `decInfo[dIdx].sheets[*]` for the sheet whose
   `FIdx_post` contains `fi`, read that post-collapse face's vertex triplet
   via `subsetVIdx(FUV_post(r, *))`.

The identity maps `IM = 0..V.rows()-1` and `IMF = 0..F.rows()-1` are passed
alongside — SSP never renumbers rows in place, so both maps are no-ops inside
`query_coarse_to_fine`.

---

## 6 — The Backward Walk (`query_coarse_to_fine`)

Starting state for query `i`: a point at barycentric coordinates `BC(i,*)` inside
coarse face `BF(i,*)`, where `FIdx(i)` is that face's global row in `gF`.

The walk replays the decimation **in reverse** (newest collapse first,
oldest last):

```
dIdx = decInfo.size()   // one past the end: "newer than all collapses"

loop:
    f = FIdx(i)
    find largest entry in decIM[f] that is < dIdx  → new dIdx
    if none found → EXIT (f is already a fine-mesh face; BF/BC are the answer)

    sd = decInfo[dIdx].sheets[ sheet matching faceSheetID(f) ]

    find row r in sd.FIdx_pre where sd.FIdx_pre(r) == f
    v0,v1,v2 = sd.FUV_pre(r, 0/1/2)           -- local UV corner indices

    queryUV = BC(i,0)*UV_post.row(v0)
            + BC(i,1)*UV_post.row(v1)
            + BC(i,2)*UV_post.row(v2)          -- project into UV_post

    B = compute_barycentric(queryUV, UV_pre, FUV_pre)
                                                -- find (face, BC) in UV_pre
    choose the row of B with best "inside" score → idxToFUV

    FIdx(i)   = sd.FIdx_pre(idxToFUV)          -- advance to pre-collapse face
    BF(i, *)  = subsetVIdx( FUV_pre(idxToFUV, *) )  -- global vertices
    BC(i, *)  = B.row(idxToFUV)                -- refined barycentric coords
```

**Why this works:** `UV_post` and `UV_pre` are aligned by `joint_lscm` — the
same material point has the same `(u,v)` in both. Projecting through
`UV_post` and then lifting the result back through `UV_pre` moves the query
one step back in time: from the post-collapse representation to the
pre-collapse representation, while preserving position on the surface.

**Termination:** when no entry in `decIM[f] < dIdx` exists, face `f` was
never in any active-sheet one-ring before this point, which means `f`'s
vertex indices in `BF` have never been remapped by Pass 2 for any earlier
collapse. They are therefore the **original fine-mesh vertex indices**. The
current `BC` gives the fine-mesh barycentric position.

---

## 7 — Summary: Data Flow Diagram

```
DECIMATION (forward, collapse 0 → N-1)
─────────────────────────────────────
For each collapse dIdx:
  ┌── Per active sheet ──────────────────────────────────────┐
  │  FIdx_pre  ──────────────────────────────┐               │
  │  FUV_pre   ──┐                           │               │
  │  subsetVIdx──┤  → compact local patch    │               │
  │  b         ──┘                           │               │
  │                    ┌──────────────┐      │               │
  │  UV_pre  ←─────────┤  joint_lscm  │      │               │
  │  UV_post ←─────────┤  (black box) │      │               │
  │                    └──────────────┘      │               │
  │  SheetData { subsetVIdx, b,              │               │
  │              FIdx_pre, FUV_pre, UV_pre,  │               │
  │              FIdx_post, FUV_post, UV_post}               │
  └──────────────────────────────────────────┼───────────────┘
                                             │
  decInfo[dIdx] = { sheets: [...SheetData] } │
  for f in FIdx_combined:                    │
      decIM[f].push_back(dIdx) ◄─────────────┘

  Pass 1: null out flap faces in F
  Pass 2: replace d with s in all surviving faces of d

QUERY (backward, coarse → fine)
────────────────────────────────
BF(i) = coarse face vertices   BC(i) = (1,0,0)   FIdx(i) = coarse face row
  │
  ▼  (repeat until decIM[FIdx(i)] has no entry older than dIdx)
  │
  ├─ look up dIdx via decIM[FIdx(i)]
  ├─ fetch SheetData for that collapse + that face's sheet
  ├─ project BC through UV_post → queryUV
  ├─ lift queryUV through UV_pre → new (FIdx, BF, BC)
  │
  ▼
final FIdx = fine-mesh face row
final BF   = fine-mesh vertex indices (global)
final BC   = barycentric position within that fine-mesh triangle
```
