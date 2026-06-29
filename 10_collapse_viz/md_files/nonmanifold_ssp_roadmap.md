# Non-Manifold SSP — Full Implementation Roadmap

This document is the definitive reference for implementing (or continuing) non-manifold
SSP edge-collapse. It covers: the original manifold pipeline, every assumption it made,
what was changed, what is still broken, and the exact flow a non-manifold mesh must go
through to be handled correctly.

---

## 1. What SSP Is

SSP (Subdivision Surface Parameterization) decimates a mesh by collapsing edges one at a
time. For each accepted collapse `(vi, vj) → p`, it records a UV flattening of the
one-ring neighbourhood so that query points on the finer mesh can be lifted back to the
coarser mesh. The two outputs are:

- **`decInfo`** — one `single_collapse_data` per accepted collapse; stores the UV maps.
- **`decIM`** — index map from face index → list of `decInfo` indices; lets a query on
  face `f` find all collapses that affected `f` in chronological order.

---

## 2. Original Manifold Pipeline — Data Structures

### 2.1 Mesh representation

| Symbol | Type | Meaning |
|--------|------|---------|
| `V` | `MatrixXd (nV × 3)` | Vertex positions |
| `F` | `MatrixXi (nF × 3)` | Triangle vertex indices; null face = all `IGL_COLLAPSE_EDGE_NULL` |
| `E` | `MatrixXi (nE × 2)` | Undirected edge endpoints (sorted: `E(e,0) < E(e,1)` usually) |
| `EMAP` | `VectorXi (3×nF)` | `EMAP(c*nF + f)` = edge index of the edge opposite corner `c` of face `f` |
| `EF` | `MatrixXi (nE × 2)` | `EF(e,0/1)` = the face on each side of edge `e` (or `-1`) |
| `EI` | `MatrixXi (nE × 2)` | `EI(e,0/1)` = local corner of that face opposite the edge |
| `Q` | `min_heap<(cost,e,timestamp)>` | Priority queue of collapse candidates |
| `EQ` | `VectorXi (nE)` | Timestamp of last time edge `e` was enqueued |
| `C` | `MatrixXd (nE × 3)` | Proposed placement point `p` for each edge |

### 2.2 Infinity extension

Before any collapse, `igl::connect_boundary_to_infinity(VO, FO, V, F)` is called:
- Adds one extra vertex at `(±∞, ±∞, ±∞)` (the "infinity vertex"), index `V.rows()-1`.
- For every boundary edge `(a,b)` of FO, adds a triangle `(a, b, infVtx)` to F.
- Result: every edge now has exactly 2 incident faces → the mesh is formally closed.
- `edge_flaps(F, E, EMAP, EF, EI)` then runs cleanly (no `-1` EF slots for manifold inputs).

### 2.3 Collapse loop invariant (original)

After `edge_flaps` runs, the code assumes:
- Every undirected edge `e` has `EF(e,0) ≥ 0` and `EF(e,1) ≥ 0`.
- `igl::circulation(e, true/false, F, EMAP, EF, EI)` can walk the one-ring around
  `E(e,1)` / `E(e,0)` without hitting a `-1` face.

---

## 3. Original Manifold Pipeline — Call Flow

```
init:
  connect_boundary_to_infinity(VO, FO,  →  V, F)
  edge_flaps(F,  →  E, EMAP, EF, EI)
  foreach edge e: cost_and_placement(e)  →  Q

per-collapse (outer overload SSP_collapse_edge):
  pop lowest-cost e from Q
  circulation(e, true,  F, EMAP, EF, EI)  →  Nsf (faces of E(e,1)), Nsv (vertices, last=E(e,0))
  circulation(e, false, F, EMAP, EF, EI)  →  Ndf (faces of E(e,0)), Ndv (vertices, last=E(e,1))
  edge_collapse_is_valid(Nsv, Ndv)  →  reject if false
  call inner SSP_collapse_edge(e, p, Nsv, Nsf, Ndv, Ndf, ...)

inner SSP_collapse_edge:
  vi = s = min(E(e,0), E(e,1))   [surviving vertex, smaller global index]
  vj = d = max(E(e,0), E(e,1))   [absorbed vertex, larger global index]
  get_collapse_onering_faces(V, F, vi, vj, Nsf, Ndf,  →  FIdx_pre, FIdx_post, F_ring_pre, F_ring_post)
  remove_unreferenced_lessF(V, F_ring_pre,  →  V_pre, FUV_pre, IM, subsetVIdx)
  build b: local index of vi in subsetVIdx = b(0), vj = b(1)  [b(0) < b(1) guaranteed since vi<vj]
  V_post = V_pre with row b(0) replaced by p
  get_post_faces(FUV_pre, b(0), b(1),  →  FUV_post)
  joint_lscm(V_pre, FUV_pre, V_post, FUV_post, b(0), b(1), Nsv_local, Ndv_local,  →  UV_pre, UV_post)
  if not valid → return false
  store data: UV_pre, UV_post, FUV_pre, FUV_post, subsetVIdx, FIdx_pre, FIdx_post, b

topology update (VF-based, see §6.4):
  pass 1: kill flap faces, repair EMAP
  pass 2: remap d→s in F and E

outer:
  if collapsed: merge VF[dv] → VF[sv]; push data to decInfo; update decIM; requeue neighbours
```

---

## 4. `joint_lscm` — What It Expects (DO NOT MODIFY)

`joint_lscm` is an API. Its expectations are documented here so callers produce correct inputs.

### 4.1 Signature
```cpp
bool joint_lscm(
    const MatrixXd & V_pre,     // 3D positions (local mesh, infVtx excluded)
    const MatrixXi & FUV_pre,   // triangles of local mesh (pre-collapse)
    const MatrixXd & V_post,    // 3D positions post-collapse (same size as V_pre)
    const MatrixXi & FUV_post,  // triangles post-collapse (flap faces removed)
    const int vi,               // LOCAL index of surviving vertex (s) in V_pre
    const int vj,               // LOCAL index of absorbed vertex (d) in V_pre
    const vector<int> & Nsv,    // winding-order neighbours of E(e,1), last = E(e,0)
    const vector<int> & Ndv,    // winding-order neighbours of E(e,0), last = E(e,1)
    MatrixXd & UV_pre,          // OUTPUT: UV layout before collapse
    MatrixXd & UV_post);        // OUTPUT: UV layout after collapse
```

### 4.2 Critical ordering invariant

```
// As stated in joint_lscm.cpp lines 23-24:
// Nsv[end] = E(e,0)
// Ndv[end] = E(e,1)
```

`joint_lscm` unconditionally reads `Nsv[Nsv.size()-1]` and `Ndv[Ndv.size()-1]` on its
first line to identify which is `e0` and which is `e1`:

```cpp
int e0, e1;
if (Nsv[Nsv.size()-1] == vi)   // vi = b(0) = local index of s
    e0 = vi; e1 = vj;
else if (Ndv[Ndv.size()-1] == vi)
    e1 = vi; e0 = vj;
// if NEITHER matches → e0/e1 are UNINITIALIZED → crash on first use
```

**If `Nsv` is sorted by global index (as `std::set` produces), this check fails
unless the edge endpoint happens to be the largest-indexed neighbour.**

### 4.3 `infVIdx` convention inside `joint_lscm`

Inside `joint_lscm`, `infVIdx = -1`. A value of `-1` in `Nsv` or `Ndv` means "infinity
vertex" → the vertex is on a mesh boundary. `joint_lscm` uses this to switch between
LSCM cases:

- `onBd.sum() == 0` — both vi and vj interior: Case 0
- `onBd.sum() == 1` — one on boundary: Case 1
- `onBd.sum() == 2` — both on boundary: check for flap, otherwise Case 2

**The global infinity vertex has index `V.rows()-1`. The caller must map it to `-1`
before passing to `joint_lscm`.**

### 4.4 Full ordering requirement

`joint_lscm` uses the ORDER of all elements of `Nsv` and `Ndv`, not just the last:

```cpp
// Flap check (onBd.sum()==2):
int Nsv0 = Nsv[0];              // first neighbour of fan
int Nsv1 = Nsv[Nsv.size()-2];   // second-to-last (neighbour just before the edge endpoint)
// → Nsv[0] == Ndv[0] == infVIdx means both fans start at the infinity side (not a flap)

// bdLoop construction (onBd.sum()==0 or 1):
for (ii=0; ii<Ndv.size()-2; ii++)
    bdLoop(ii) = Ndv[Ndv.size()-3-ii];   // Ndv reversed (excluding last 2)
for (ii=0; ii<Nsv.size()-2; ii++)
    bdLoop(ii+Ndv.size()-2) = Nsv[ii+1]; // Nsv forward (excluding first and last)
```

The two lists must be in compatible **winding order** so that the boundary loop they form
is non-self-intersecting.

### 4.5 Core invariant — what each sheet's inputs must satisfy

Every call to `joint_lscm` must satisfy ALL five of the following properties.
Everything in the implementation is ultimately about preserving these five.

1. **All real faces belong to exactly one sheet.** No face from a different sheet may
   appear in `FUV_pre` or `FUV_post`.
2. **Both collapse endpoints (`vi`, `vj`) appear in the sheet's local vertex set.**
   If either is absent, `b_si` is `-1` and the call must be skipped entirely.
3. **`Nsv` and `Ndv` are ordered exactly as `joint_lscm` expects** — full winding order
   with the correct endpoint last, not sorted by global index.
4. **Boundary information is preserved.** If the center vertex is on a mesh boundary,
   the infinity vertex must appear (as `-1`) in the neighbour list; `joint_lscm` uses
   this to choose its LSCM case.
5. **No face from another sheet may appear** — including infinity faces that happened to
   receive the same sheet ID as a real face due to an upstream routing bug.

---

## 5. Non-Manifold Mesh — What Breaks

### 5.0 Architectural distinction: routing vs boundary detection

Two independent mechanisms serve two independent purposes and must not be conflated:

| Mechanism | Question it answers | Data |
|-----------|--------------------|----|
| `partition_onering_by_sheet` | Which sheet does this face belong to? | `faceSheetID(f)` |
| `fan_walk_local` `-1` sentinel | Is this vertex on a mesh boundary? | `infVIdx` in the face fan |

`partition_onering_by_sheet` is a **topological routing** step — it partitions faces by
connectivity. It should never see infinity faces.

`fan_walk_local` boundary detection relies on infinity faces being present in its face
list — they are what makes `result` contain a `-1` entry, which `joint_lscm` interprets
as "boundary vertex".

Bug 1 collapses these two roles: infinity faces enter the routing step because
`partition_into_sheets` was called on the extended mesh. Fixing Bug 1 means restoring
the clean separation — infinity faces must be invisible to routing but visible to the
fan walk.

### 5.1 `edge_flaps` EF failure modes

When a mesh has non-manifold edges (seam edges shared by 3+ faces), `edge_flaps` stores
only 2 of the 3+ incident faces in the 2-slot `EF` matrix:

**Case A — Same-winding duplicate:**  
Two faces both go `u→v` (same directed half-edge). Both write to slot 0; slot 1 = `-1`.

**Case B — Three-face seam:**  
Three faces: two go `u→v`, one goes `v→u`. Slot 0 is overwritten by the last `u→v`
writer (first face LOST). Slot 1 = the `v→u` face.

### 5.2 `igl::circulation` crash

`circulation` hops between faces using `EF` slots. With `EF(e,1) = -1`:
```
nf = EF(e, nside)  // = -1
rv = F(nf, nv)     // = F(-1, *)  → CRASH (Eigen bounds assertion)
```

With 3 faces on a seam edge, the walker reaches a face `f0` that's NOT stored in
`EF[e1]`, and the assert `"e should touch ff"` fires.

### 5.3 `connect_boundary_to_infinity` complicates seam detection

The infinity extension is called BEFORE seam detection. After it runs, every original
boundary edge gets an infinity triangle that makes it look like a 2-face manifold edge.
However, the two faces on the "fan" edge adjacent to the infinity vertex both wind the
same direction (both `u→v`), leaving `EF=-1` on those edges.

**Using `EF(e,1)==-1` to detect seam edges incorrectly flags every regular boundary
vertex as a seam vertex.**

Correct seam detection: count edge face incidence in the ORIGINAL mesh (before infinity
extension), flag edges with ≥ 3 incident faces.

---

## 6. Changes Made — Current State of the Code

### 6.1 Non-manifold guard removed

**Files:** `SSP_decimate.cpp`, `SSP_midpoint.cpp`, `10_collapse_viz/main.cpp`

The original `igl::is_edge_manifold` guards that rejected non-manifold input were removed.

### 6.2 `single_collapse_data.h` — new SheetData struct

**File:** `src/single_collapse_data.h`

```cpp
struct SheetData {
    Eigen::VectorXi b;           // b(0)=local vi, b(1)=local vj within this sheet
    Eigen::VectorXi subsetVIdx;  // global→local vertex index mapping (sorted ascending)
    Eigen::MatrixXd UV_pre, UV_post;
    Eigen::MatrixXi FUV_pre, FUV_post;
    Eigen::VectorXi FIdx_pre, FIdx_post;
};

struct single_collapse_data {
    std::vector<SheetData> sheets;  // one entry per sheet (size 1 for regular edges)
    Eigen::MatrixXd V_pre, V_post;  // 3D geometry (shared, first sheet)
    std::vector<int> Nsv, Ndv;      // local winding-order neighbour lists (first sheet)
};
```

**Key invariant:** `b(0) < b(1)` **when both endpoints are found** in `subsetVIdx`.
The guarantee rests on two conditions:
1. `subsetVIdx` is sorted ascending (by `remove_unreferenced_lessF`).
2. `vi < vj` always — enforced by the `eflip` logic in the outer overload (`s = min(E(e,0),E(e,1))`).

The guarantee does NOT hold when one endpoint is absent from `subsetVIdx` (that is Bug 1).
The `b_si(0) >= b_si(1)` branch in the guard is therefore theoretically unreachable when
both are found, but it is still checked defensively.

### 6.3 New files: `partition_into_sheets` and `partition_onering_by_sheet`

**`src/partition_into_sheets.cpp/.h`**

```cpp
void partition_into_sheets(
    const Eigen::MatrixXi & F,
    Eigen::VectorXi & faceSheetID,   // output: sheet ID per face (-1 unset)
    int & numSheets);
```

Algorithm: build edge→faces map; BFS flood-fill stopping only at edges with exactly ≠ 2
incident faces (seam edges with 3+ faces stop BFS; boundary edges with 1 face also stop).

**Critical detail:** When called on the EXTENDED mesh (after `connect_boundary_to_infinity`),
boundary edges now have 2 faces (1 real + 1 infinity triangle). BFS CROSSES these edges,
assigning infinity triangles the same sheet ID as their adjacent real face. This causes
Bug 1 (see §8.1).

When called on the ORIGINAL mesh (before infinity extension), boundary edges have 1 face
→ BFS stops → infinity faces would not be included → correct sheet IDs.

**Current code calls `partition_into_sheets` on the EXTENDED mesh** in both `main.cpp`
and `SSP_midpoint.cpp`. This is Bug 1's root cause.

---

**`src/partition_onering_by_sheet.cpp/.h`**

```cpp
void partition_onering_by_sheet(
    const std::vector<int> & Nsf,     // faces of E(e,1) from collect_onering
    const std::vector<int> & Ndf,     // faces of E(e,0) from collect_onering
    const Eigen::VectorXi & faceSheetID,
    std::vector<std::vector<int>> & sheets_Nsf,  // output: Nsf split per sheet
    std::vector<std::vector<int>> & sheets_Ndf); // output: Ndf split per sheet
```

Uses `faceSheetID(f)` to route each face to a per-sheet sub-list. Returns as many
entries as there are distinct sheet IDs in the one-ring. Empty entries = that sheet has
no faces on that side.

### 6.4 VF-based one-ring collection (`collect_onering`)

**File:** `src/SSP_collapse_edge.cpp`, outer overload, lines ~685–715

Replaces `igl::circulation`. Built as a lambda inside the outer overload:

```cpp
auto collect_onering = [&](const int v, vector<int> & faces, vector<int> & verts) {
    faces.clear(); verts.clear();
    std::set<int> vset;
    for (const int f : (*VF)[v]) {
        if (null_face(f)) continue;
        // NOTE: does NOT filter infinity faces — Bug 1
        faces.push_back(f);
        for (int c = 0; c < 3; c++)
            if (F(f,c) != v) vset.insert(F(f,c));
    }
    verts.assign(vset.begin(), vset.end());  // sorted ascending — Bug 2
};
collect_onering(E(e,1), Nsf, Nsv);   // Nsv = neighbours of E(e,1)
collect_onering(E(e,0), Ndf, Ndv);   // Ndv = neighbours of E(e,0)
```

**Bug 1:** Infinity faces are not filtered here → see §8.1.  
**Bug 2:** `std::set` sorts vertices by global index → Nsv/Ndv are in sorted order,
NOT winding order → `joint_lscm` crash → see §8.2 (now fixed by `fan_walk_local`).

### 6.5 Per-sheet UV loop in inner overload

**File:** `src/SSP_collapse_edge.cpp`, inner overload, lines ~85–450

The key loop (pseudo-code):

```cpp
vi = s;  vj = d;   // vi < vj, vi = surviving vertex
partition_onering_by_sheet(Nsf, Ndf, faceSheetID, sheets_Nsf, sheets_Ndf);

for each sheet si:
    if (sheets_Nsf[si].empty() || sheets_Ndf[si].empty()) continue;
    get_collapse_onering_faces(V, F, vi, vj, sheets_Nsf[si], sheets_Ndf[si],
                               FIdx_pre, FIdx_post, F_ring_pre, F_ring_post);
    if (!validEdge) return false;
    remove_unreferenced_lessF(V, F_ring_pre,  → V_pre, FUV_pre, IM, subsetVIdx);
    find b_si: b(0)=local(vi), b(1)=local(vj)  in subsetVIdx
    if b(0)<0 || b(1)<0: log [b_si FAIL] and continue  (diagnostic, see §8.1)
    if b(0) >= b(1): log [b_si FAIL] and continue      (should never happen when both found)
    V_post = V_pre with row b(0) replaced by p
    get_post_faces(FUV_pre, b(0), b(1),  → FUV_post)
    Nsv_local = fan_walk_local(E(e,1), E(e,0), sheets_Nsf[si], subsetVIdx)
    Ndv_local = fan_walk_local(E(e,0), E(e,1), sheets_Ndf[si], subsetVIdx)
    if Nsv_local.empty() || Ndv_local.empty(): log [fan_walk EMPTY] and ASSERT  (bug upstream)
    joint_lscm(V_pre, FUV_pre, V_post, FUV_post, b(0), b(1), Nsv_local, Ndv_local,
               → UV_pre, UV_post)
    if !valid: return false
    push SheetData into data.sheets
```

### 6.6 `fan_walk_local` — replaces `make_local_nv`

**File:** `src/SSP_collapse_edge.cpp`, inner overload, defined as lambda at lines ~122–240

`make_local_nv` (old): copied global Nsv/Ndv preserving their `std::set`-sorted order
and mapped each to a local index. Produced sorted-order local lists → wrong for
`joint_lscm`.

`fan_walk_local` (new): given `center` vertex, `end_vtx`, and the sheet's face list,
walks the face fan in adjacency order. Returns local indices with `local(end_vtx)` last.

```
Algorithm:
1. Filter null faces from face_list; keep infinity faces (they map to -1 = infVIdx in local space)
2. Count how many faces each neighbour vertex appears in:
   - count=1 → fan boundary vertex
   - count=2 → interior of fan
3. Find start_vtx: boundary vertex ≠ end_vtx
   - If none found (closed fan / interior vertex): pick CCW-adjacent neighbour of end_vtx
     from one of its two faces using (c+1)%3 convention
4. Walk: push current, find face containing (center, current) ≠ prev_face,
         advance to the other non-center vertex in that face, repeat until cur == end_vtx
5. Map result to local indices via svIdx (infVIdx → -1, not in svIdx → -1)
```

**Last-element invariant satisfied by construction:** `result.back() == local(end_vtx)`.

For the Nsv/Ndv pair passed to `joint_lscm`:
- `Nsv_local`: walk around `E(e,1)`, ending at `E(e,0)` → `Nsv_local.back() == local(E(e,0))`
- `Ndv_local`: walk around `E(e,0)`, ending at `E(e,1)` → `Ndv_local.back() == local(E(e,1))`

Both eflip=0 and eflip=1 are handled correctly because the walk uses `E(e,*)` directly
(not vi/vj which swap with eflip).

**Unvalidated:** The winding direction of the intermediate elements is not yet verified to
match the direction `igl::circulation` would have produced. `fan_walk_local` picks its
traversal direction implicitly from the face data (`find_nb_face`); if the adjacency graph
happens to traverse CCW when CW is needed (or vice versa), `bdLoop` will be wound
incorrectly even though `result.back()` is correct. This must be validated by checking
`bdLoop == igl::boundary_loop(FUV_pre)` on representative test cases before the fix is
considered confirmed.

### 6.7 VF-based two-pass topology update ✓ verified correct

**File:** `src/SSP_collapse_edge.cpp`, inner overload, lines ~526–615

Replaces the old EF/EI-based topology update. Works on `nV2Fd = (*eflip ? Nsf : Ndf)`
(faces of `d`).

**Pass 1 — kill flap faces:**
For each face f in nV2Fd containing both s AND d:
- Identify corners: cs (s-corner), cd (d-corner), ca (third vertex a)
- `e_kill = EMAP(f + m*cs)` = edge(d,a), opposite s-corner
- `e_keep = EMAP(f + m*cd)` = edge(s,a), opposite d-corner
- For every other face in nV2Fd containing vertex a: redirect `EMAP(fa + m*c) = e_keep`
  where that corner's EMAP pointed to `e_kill`
- `kill_edge(e_kill)`; null-out face f
- Record `a_e1, a_f1` (and `a_e2, a_f2` for second flap if exists)

**Pass 2 — remap d→s:**
For each surviving face f in nV2Fd (not null):
- Find corner c where `F(f,c) == d`
- Update edges: `e1 = EMAP(f + m*((c+1)%3))`: if `E(e1, side) == d`, set `E(e1,side) = s`
- Same for `e2 = EMAP(f + m*((c+2)%3))`
- Set `F(f,c) = s`

**EF/EI left stale intentionally** — cost functions (`shortest_edge_and_midpoint`, QEM)
do not read EF or EI.

Guards in outer overload: `if(e1 >= 0) EQ(e1) = -1; if(e2 >= 0) EQ(e2) = -1;`

### 6.8 Seam vertex detection

**Files:** `main.cpp` (lines ~122–137), `SSP_midpoint.cpp` (lines ~158–179)

Count edge face incidence on the ORIGINAL mesh FO (before infinity extension):

```cpp
for each face f in FO (skip if contains infVtx):
    for each edge (u,v) of f: edgeFaceCount[{min,max}]++
for each edge with count >= 3:
    seamVertex(u) = seamVertex(v) = 1
```

`seamVertex` is currently passed to `SSP_collapse_edge` outer overload but is NOT yet
used inside it. The intention was to skip collapses on seam edges via `pre_collapse`,
but this is not implemented.

### 6.9 VF maintenance after collapse

**File:** `src/SSP_collapse_edge.cpp`, outer overload, lines ~797–801

```cpp
if (collapsed) {
    (*VF)[sv].insert((*VF)[sv].end(), (*VF)[dv].begin(), (*VF)[dv].end());
    (*VF)[dv].clear();
}
```

Merges dv's face list into sv's. Stale (null) entries accumulate in VF[sv] and are
lazily filtered at the next `collect_onering` call. This is correct and intentional.

### 6.10 `query_coarse_to_fine` — current state (INCOMPLETE)

**File:** `src/query_coarse_to_fine.cpp`, line 76:

```cpp
const SheetData & sd = decInfo[dIdx].sheets[0];
```

**HARDCODED to `sheets[0]`**. For non-manifold meshes with multiple sheets per collapse,
this is wrong — a query on a face belonging to sheet 1 or 2 would incorrectly use sheet
0's UV map.

Correct implementation needs:
```cpp
int sheetID = /* which sheet does FIdx(qIdx) belong to? */;
const SheetData & sd = decInfo[dIdx].sheets[sheetID];
```

This requires threading the global `faceSheetID` vector into `query_coarse_to_fine` and
mapping `FIdx(qIdx)` to a local sheet index within that collapse's `sheets[]` array.

---

## 7. Input/Output Flow: Non-Manifold Mesh → SSP

```
NON-MANIFOLD INPUT MESH (VO, FO)
        │
        ▼
[1] partition_into_sheets(FO, faceSheetID, numSheets)
    ─ BFS on FO (BEFORE infinity extension)
    ─ Stops at edges with ≠ 2 incident faces (seams & boundaries)
    ─ Output: faceSheetID[f] ∈ {0..numSheets-1} for every face in FO

        │
        ▼
[2] connect_boundary_to_infinity(VO, FO,  →  V, F)
    ─ Appends infVtx = V.rows()-1 at (±∞, ±∞, ±∞)
    ─ Adds infinity triangles for every boundary edge
    ─ IMPORTANT: faceSheetID has numFO entries; the new infinity faces have
      indices [numFO .. F.rows()-1] and have NO entry in faceSheetID.
    ─ CURRENT BUG: partition_into_sheets is called on F AFTER this step,
      so faceSheetID covers infinity faces too and BFS crosses boundary edges.
    ─ CORRECT ORDER: [1] before [2].

        │
        ▼
[3] edge_flaps(F,  →  E, EMAP, EF, EI)
    ─ EF may have -1 slots for seam edges (3+ faces) and for infinity-fan edges
      (same-winding duplicates).
    ─ THESE EF SLOTS ARE NEVER TRUSTED after this point — all one-ring work
      uses VF instead.

        │
        ▼
[4] Build VF adjacency on F (including infinity faces)
    for f in 0..F.rows()-1:
        for c in 0..2: VF[F(f,c)].push_back(f)
    ─ VF[v] contains both real and infinity faces for boundary vertices.

        │
        ▼
[5] Compute seamVertex from FO (ORIGINAL, before step [2])
    Count edge face incidence; flag both endpoints of edges with ≥ 3 faces.
    ─ This correctly identifies vertices that lie on seam edges.
    ─ Currently computed but not yet used to skip collapses.

        │
        ▼
[6] Initialise priority queue Q with costs for all edges in E

        │
        ▼ (collapse loop)

[7] outer SSP_collapse_edge — pop lowest-cost e from Q
        │
        ├─ IF seamVertex[E(e,0)] or seamVertex[E(e,1)]:
        │    ─ Currently NOT skipped (seamVertex unused in collapse logic)
        │    ─ TODO: should skip or handle specially
        │
        ▼
[8] collect_onering(E(e,1), Nsf, Nsv)
    collect_onering(E(e,0), Ndf, Ndv)
    ─ Scans VF[v], filters null faces, keeps infinity faces (currently)
    ─ Builds verts via std::set → sorted order (only used as pool; fan_walk
      reconstructs the correct order per-sheet)

        │
        ▼
[9] inner SSP_collapse_edge — per-sheet UV loop
    partition_onering_by_sheet(Nsf, Ndf, faceSheetID,  →  sheets_Nsf, sheets_Ndf)
    ─ Uses faceSheetID(f). If f is an infinity face: faceSheetID(f) is currently
      set (Bug 1) → routes infinity faces into a sheet → b_si = -1 bug.
    ─ CORRECT: infinity faces should NOT appear in sheets_Nsf/sheets_Ndf at all.

    for each sheet si:
        get_collapse_onering_faces(...)  ← filters infinity faces (correct)
        remove_unreferenced_lessF(...)   ← compact local mesh
        find b_si                        ← b(0)=local(vi), b(1)=local(vj)
        fan_walk_local(E(e,1), E(e,0), sheets_Nsf[si], ...)  →  Nsv_local
        fan_walk_local(E(e,0), E(e,1), sheets_Ndf[si], ...)  →  Ndv_local
        joint_lscm(...)                  ← UV flattening

        │
        ▼
[10] VF-based two-pass topology update
     ─ Pass 1: kill flap faces, repair EMAP
     ─ Pass 2: remap d→s in F and E

        │
        ▼
[11] Merge VF[dv] → VF[sv]; push data to decInfo; update decIM

        │
        ▼ (repeat from [7] until stopping condition)

[12] query_coarse_to_fine(decInfo, IM, decIM, IMF, BC, BF, FIdx)
     ─ For each query vertex: walk decIM backward to find the collapse that
       affected it, apply UV transport.
     ─ CURRENT STATE: uses sheets[0] hardcoded → wrong for multi-sheet collapses.
     ─ NEEDED: route to the sheet matching FIdx(qIdx)'s sheet ID.
```

---

## 8. Known Bugs and Their Status

### 8.1 Bug 1 — b_si = -1 (infinity face routing)

**Symptom:** `b_si(0) = -1` (or `b_si(1) = -1`) at the sheet UV loop.

**What is established:**
- `collect_onering` includes infinity faces (confirmed by reading code).
- `partition_onering_by_sheet` receives them (confirmed by reading code).
- `get_collapse_onering_faces` filters them out (confirmed by reading code).
- Sometimes `b_si` becomes `-1` (crash observed).

**Leading hypothesis (needs confirmation by running the diagnostics):**

```
partition_into_sheets called on EXTENDED F (after connect_boundary_to_infinity)
  → BFS crosses boundary edges (now have 2 faces) → infinity faces get valid sheet IDs
  → VF[boundary_v] contains infinity faces
  → collect_onering includes infinity faces in Nsf/Ndf
  → partition_onering_by_sheet routes infinity faces into sheets_Ndf[si]
  → guard if(sheets_Ndf[si].empty()) PASSES (non-empty, but all infinity)
  → get_collapse_onering_faces FILTERS infinity faces → vi absent from F_ring_pre
  → remove_unreferenced_lessF: vi not in F_ring_pre → not in subsetVIdx_si
  → b_si(0) = -1
```

**The missing proof:** the `[b_si FAIL]` diagnostic must fire and print
`"HYPOTHESIS CONFIRMED: sheets_Ndf[si] is ALL INFINITY faces"`. Until the code is run
and that line appears, this remains a hypothesis. If the diagnostic prints
`"HYPOTHESIS NOT CONFIRMED: real faces present but vertex still missing"`, there is a
different root cause to investigate.

**Diagnostic logging added:** `[b_si FAIL]` block logs real vs infinity face counts in
`sheets_Ndf[si]` and `sheets_Nsf[si]`. If `ndf_real==0 && ndf_inf>0`: hypothesis confirmed.

**Correct fix (not yet applied):**
Call `partition_into_sheets(FO, ...)` on the ORIGINAL mesh BEFORE
`connect_boundary_to_infinity`. OR filter infinity faces inside `collect_onering`:
```cpp
if (isinf(V(F(f,0),0)) || isinf(V(F(f,1),0)) || isinf(V(F(f,2),0))) continue;
```
Both approaches prevent infinity faces from entering the sheet routing.

**BUT:** Infinity faces serve a purpose — their presence in the fan signals to
`joint_lscm` that the center vertex is on a boundary (the infinity vertex maps to
`-1 = infVIdx` in the local neighbour list). If infinity faces are filtered from
`collect_onering`, the boundary signal is lost.

**Resolution:** The fix depends on how `fan_walk_local` handles infinity faces.
`fan_walk_local` KEEPS infinity faces in its input (it only filters null faces),
so as long as the sheet's face list includes the infinity triangle, the boundary
signal propagates correctly. The issue is that the sheet routing assigns infinity
faces to the WRONG sheet when `partition_into_sheets` is run on the extended mesh.

**Cleanest fix:** Run `partition_into_sheets` on FO (original). For the extended mesh F,
map each face index: if `f < FO.rows()` use `faceSheetID(f)`, else assign a special
"infinity sheet" ID and skip it in `partition_onering_by_sheet`. This way infinity faces
are always excluded from sheet routing but `fan_walk_local` still sees them correctly
because it reads the per-sheet face list AFTER the guard has been skipped.

### 8.2 Bug 2 — `Nsv`/`Ndv` sorted order breaks `joint_lscm` (e0/e1 uninitialized)

**Symptom:** Crash in `joint_lscm` at `bdLoop(ii) = e0` with "e0 uninitialized".

**Root cause:** `collect_onering` uses `std::set` → sorted order. `joint_lscm` requires
`Nsv[last] = E(e,0)`. Sorted order places the largest global vertex index last, which is
not necessarily `E(e,0)`.

**Attempted fix:** `fan_walk_local` replaces `make_local_nv` (old calls are commented out).
`fan_walk_local` traverses the sheet's face fan in adjacency order, guaranteeing
`result.back() == local(end_vtx)`. The commented-out old calls are at lines ~330–335 of
`SSP_collapse_edge.cpp`.

**Status: Implemented. Needs validation against original manifold behaviour.**

The last-element invariant (`Nsv.back() == E(e,0)`) is satisfied by construction.
However, `joint_lscm` also uses the full sequence for `bdLoop` construction — it reverses
`Ndv[0..size-3]` and walks `Nsv[1..size-2]` forward. This assumes the entire traversal
is in the same winding direction as `igl::circulation` would have produced. `fan_walk_local`
picks its start vertex and traversal direction from the face adjacency graph; it is not
validated that this direction matches the original `circulation` convention. Correctness
requires checking `bdLoop == igl::boundary_loop(FUV_pre)` on representative cases.

### 8.3 Gap — `query_coarse_to_fine` hardcoded to `sheets[0]`

**File:** `src/query_coarse_to_fine.cpp`, line 76.

For non-manifold collapses with N sheets, only sheet 0's UV map is used for ALL query
vertices, regardless of which sheet they belong to.

**Required fix:**
1. Add `faceSheetID` parameter to `query_coarse_to_fine`.
2. Store `global_sheet_id` in each `SheetData` (or build a map from global sheet ID →
   local index within `data.sheets[]`).
3. At the query step: `sheetID = faceSheetID(FIdx(qIdx))`. Look up local sheet index.
   Use `decInfo[dIdx].sheets[localIdx]`.

This is a **correctness bug for multi-sheet collapses** but is invisible on manifold
meshes (where `sheets.size() == 1` always).

### 8.4 Gap — `seamVertex` not used to gate collapses

`seamVertex` is computed and passed to the outer `SSP_collapse_edge` but never consulted
inside it. For seam edges, the collapse currently proceeds and relies on the per-sheet UV
loop to handle them. Whether to SKIP seam collapses entirely or handle them with
per-sheet LSCM is a design decision not yet implemented.

---

## 9. Files Changed vs. Original — Summary

| File | Change |
|------|--------|
| `src/single_collapse_data.h` | New `SheetData`; `single_collapse_data` restructured |
| `src/SSP_collapse_edge.h` | Signature: added `faceSheetID`, `seamVertex`, `VF*` params |
| `src/SSP_collapse_edge.cpp` | VF-based `collect_onering`; per-sheet loop; `fan_walk_local`; VF topology update; diagnostic logging |
| `src/SSP_midpoint.cpp` | Removed manifold guard; added VF init; added `partition_into_sheets`; added `seamVertex` computation |
| `src/SSP_decimate.cpp` | Removed manifold guard |
| `src/SSP_random_collapse_edge.cpp` | Updated to write UV into `SheetData` |
| `src/query_coarse_to_fine.cpp` | Uses `sheets[0]` (incomplete — see §8.3) |
| `src/query_fine_to_coarse.cpp` | Uses `sheets[0]` (same incomplete state) |
| `10_collapse_viz/main.cpp` | VF init; `faceSheetID`; `seamVertex`; `sheets.empty()` guard |
| `src/partition_into_sheets.cpp/.h` | **New file** |
| `src/partition_onering_by_sheet.cpp/.h` | **New file** |

**Files NOT changed (used as-is):**

| File | Role |
|------|------|
| `src/joint_lscm.cpp` | UV flattening — API, called per sheet |
| `src/get_collapse_onering_faces.cpp` | Filters null/infinity faces; called per sheet |
| `src/remove_unreferenced_lessF.cpp` | Compact local mesh; assigns local indices by sorted global index |
| `src/get_post_faces.cpp` | Removes flap faces from FUV; called per sheet |
| `src/compute_barycentric.cpp` | Barycentric interpolation; called in query |

---

## 10. Diagnostic Logging Currently Active

All logging is controlled by static counters (fires only for first N occurrences):

| Tag | Location | Fires when |
|-----|----------|------------|
| `[collect_onering]` | outer overload | infinity face count > 0 in VF scan (Bug 1 evidence) |
| `[BOGUS ONE-RING]` | outer overload | `Nsv.size() < 2 \|\| Ndv.size() < 2` |
| `[sheets]` | inner overload | first 10 sheet breakdowns (vi, vj, per-sheet Nsf/Ndf sizes) |
| `[b_si FAIL]` | inner overload | `b_si(0)<0 \|\| b_si(1)<0 \|\| b_si(0)>=b_si(1)` — logs real vs infinity face counts |
| `[joint_lscm #N]` | inner overload | first 10 calls — logs local indices, face counts, Nsv_local/Ndv_local |
| `[fan_walk EMPTY]` | inner overload | `fan_walk_local` returns `{}` despite valid `b_si` — ASSERT after log |

---

## 11. Ordered Fix List for Next Session

Apply in this order (each depends on the previous):

### Fix A — Call `partition_into_sheets` on original mesh (Bug 1 root fix)

In `main.cpp` (`init_ssp`) and `SSP_midpoint.cpp`:

```cpp
// BEFORE connect_boundary_to_infinity:
int numSheets;
partition_into_sheets(FO, faceSheetID, numSheets);

// THEN:
igl::connect_boundary_to_infinity(VO, FO, gV, gF);
igl::edge_flaps(gF, gE, gEMAP, gEF, gEI);
```

`faceSheetID` will have `FO.rows()` entries. Infinity faces (indices `FO.rows()..F.rows()-1`)
have no entry in `faceSheetID`. In `partition_onering_by_sheet`, add a bounds check:

```cpp
for (int f : Nsf) {
    if (f < 0 || f >= faceSheetID.size()) continue;  // skip infinity faces
    sheets_Nsf[sheetToLocal[faceSheetID(f)]].push_back(f);
}
```

This cleanly excludes infinity faces from sheet routing while `fan_walk_local` in the
inner overload still sees them (it receives the per-sheet face list BEFORE they were
routed — wait, no: they won't be in sheets_Nsf anymore). 

Revisit: if infinity faces are excluded from `sheets_Nsf[si]`, they won't be in the
input to `fan_walk_local`, so the fan won't include `infVtx` as a neighbour, and
`joint_lscm` won't get the `-1` boundary signal.

> **Open design question:**
> `fan_walk_local` still requires infinity faces to recover boundary information.
> The cleanest mechanism for passing boundary information without allowing infinity faces
> to pollute sheet routing is not yet resolved. Options being considered:
>
> **Option A** (two face lists): build `sheets_Nsf_routing[si]` (no infinity faces, for
> `partition_onering_by_sheet`) and `sheets_Nsf_walk[si]` (real + adjacent infinity faces,
> for `fan_walk_local`).
>
> **Option B** (separate boundary flag): pass boundary-vertex status as an explicit boolean
> flag to the per-sheet UV loop and inject `-1` into the neighbour list synthetically,
> removing the dependency on infinity faces in `fan_walk_local`.
>
> Option A is more conservative (less restructuring); Option B is architecturally cleaner
> (eliminates the routing/boundary confusion at the source). Neither is implemented yet.

### Fix B — Confirm `fan_walk_local` correctness

Build and run. Observe log output:
- `[collect_onering]` should print (infinity faces ARE in VF — the walk will see them)
- `[b_si FAIL]` should NOT print after Fix A (infinity faces no longer pollute sheets)
- `[fan_walk EMPTY]` should NOT print (valid fans always exist when b_si passes)
- No crash in `joint_lscm`

### Fix C — `query_coarse_to_fine` sheet routing (Gap §8.3)

1. Add `global_sheet_id` field to `SheetData`.
2. When pushing `SheetData` in the per-sheet loop, set `sd.global_sheet_id = faceSheetID(sheets_Nsf[si][0])` (any face in this sheet gives the correct ID).
3. Add `const VectorXi & faceSheetID` parameter to `query_coarse_to_fine`.
4. In the query loop:
   ```cpp
   int gSID = faceSheetID(FIdx(qIdx));  // global sheet ID of query face
   const SheetData * sd_ptr = nullptr;
   for (auto & sd : decInfo[dIdx].sheets)
       if (sd.global_sheet_id == gSID) { sd_ptr = &sd; break; }
   if (!sd_ptr) continue;  // this collapse didn't touch this sheet
   const SheetData & sd = *sd_ptr;
   ```

### Fix D — Remove diagnostic logging

Once Fixes A–C are confirmed stable, remove all `[DIAG]` / static-counter logging blocks.

---

## 12. Assumptions That Must Remain True

These invariants the new code relies on and must not be broken:

1. **`s < d` always** (`s = min(E(e,0), E(e,1))`). Asserted at line 509.
   Guarantees `vi < vj` (global index). Combined with `subsetVIdx` being sorted ascending,
   this guarantees `b(0) < b(1)` **when both endpoints are found** in `subsetVIdx`.
   The guarantee does NOT hold when one endpoint is absent — that is Bug 1 territory.

2. **`subsetVIdx` is sorted ascending** — guaranteed by `remove_unreferenced_lessF`
   (it sorts all vertex indices from F_ring_pre before assigning local indices).

3. **`infVIdx = V.rows()-1`** (global) maps to `-1` (local) in `fan_walk_local`.
   `joint_lscm` defines `infVIdx = -1` internally.

4. **`IGL_COLLAPSE_EDGE_NULL`** is the sentinel for dead faces/edges (all corners equal
   to this value). It is 0 in practice. Null faces are filtered everywhere before
   processing.

5. **EF/EI are left stale** after each collapse — no code after the topology update
   should read EF or EI for routing decisions. Cost functions must not rely on EF/EI.

6. **VF is maintained**: after each accepted collapse, `VF[sv]` absorbs `VF[dv]`.
   The raw VF lists accumulate null entries (stale face indices); these are filtered
   on every `collect_onering` call.

7. **`faceSheetID` has exactly `FO.rows()` entries** (after Fix A). Any code that
   indexes `faceSheetID(f)` must bounds-check against `faceSheetID.size()`.
