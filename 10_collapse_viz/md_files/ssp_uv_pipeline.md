# SSP UV Pipeline: Coarse-to-Fine Correspondence

Exactly what the code does at each collapse, what gets saved, and how the
backward walk in `query_coarse_to_fine` uses it to map coarse points back to
the original mesh.

---

## 1. Per-collapse computation (`SSP_collapse_edge.cpp`)

Every call to `SSP_collapse_edge` collapses one edge **e = (s, d)** where

| symbol | meaning |
|---|---|
| `s` = `vi` | surviving vertex (smaller global index) |
| `d` = `vj` | absorbed vertex  (larger  global index) |
| `p`        | new 3-D position for both after the collapse |

### 1.1 Partition the one-ring by sheet

The faces around `vi` (`Nsf`) and around `vj` (`Ndf`) are each bucketed by
`faceSheetID(f)`.  Only **active sheets** — those where BOTH vi's side and
vj's side have at least one real (non-null, non-infinity) face — get UV
maps computed.

```
sheets_Nsf[sid] = real faces of vi's side belonging to sheet sid
sheets_Ndf[sid] = real faces of vj's side belonging to sheet sid
active_sheets   = sid where BOTH sides are non-empty
```

For a **manifold edge** there is always exactly one active sheet.
For a **non-manifold seam edge** there can be two or more.

---

### 1.2 For each active sheet `sid` — build the pre-collapse local mesh

```
get_collapse_onering_faces(V, F, vi, vj, Nsf_si, Ndf_si, ...)
    → FIdx_pre_si   : global gF indices of the one-ring faces (pre-collapse)
    → F_ring_pre_si : 3×3 face connectivity (still uses global vertex indices)

remove_unreferenced_lessF(V, F_ring_pre_si, ...)
    → V_pre_si     : (nLocal × 3) 3-D positions, compacted
    → FUV_pre_si   : (nFaces × 3) face connectivity in LOCAL vertex indices
    → subsetVIdx_si: (nLocal) LOCAL index → GLOBAL gV index, sorted ascending
```

`subsetVIdx_si` is the key bridge between local and global indexing.
`subsetVIdx_si(j) == gV_row_j` for every local vertex j.

The local indices of vi and vj are found by linear scan:
```
b_si(0) = local index of vi   (= smaller value since subsetVIdx is sorted)
b_si(1) = local index of vj
```
Invariant: `b_si(0) < b_si(1)` always.

---

### 1.3 Build the post-collapse local mesh

```
V_post_si        = copy of V_pre_si
V_post_si[b(0)]  = p          ← surviving vertex moved to new position

get_post_faces(FUV_pre_si, b(0), b(1), ...)
    → FUV_post_si : same one-ring but with flap faces removed and vj→vi remapped
    → FIdx_post_si: global gF indices of surviving post-collapse faces
```

The "flap faces" are those containing both local vi and local vj (the edge
being collapsed).  They die in the topology.  All other one-ring faces
survive with vj remapped to vi.

---

### 1.4 Build winding-order neighbour lists (`fan_walk_local`)

`joint_lscm` needs to know the ordered sequence of neighbours around each
endpoint of the collapsed edge, so it can construct boundary loops correctly.

```
Nsv_local = fan_walk_local(center=E(e,1), end_vtx=E(e,0), Nsf_walk, subsetVIdx_si)
Ndv_local = fan_walk_local(center=E(e,0), end_vtx=E(e,1), Ndf_walk, subsetVIdx_si)
```

Each returns a list of **local** vertex indices.  The infVtx (boundary
sentinel, added by `connect_boundary_to_infinity`) maps to `-1`.
The last element of each walk must be the local index of the end_vtx.

---

### 1.5 Run `joint_lscm`

```cpp
joint_lscm(V_pre_si, FUV_pre_si, V_post_si, FUV_post_si,
           b_si(0), b_si(1), Nsv_local, Ndv_local,
           UV_pre_si, UV_post_si)
```

LSCM is run **jointly** on the pre- and post-collapse one-rings so that
the two UV maps share the same boundary conditions and are therefore
directly comparable.

| output | shape | meaning |
|---|---|---|
| `UV_pre_si`  | nLocal × 2 | UV coordinate of local vertex j **before** the collapse |
| `UV_post_si` | nLocal × 2 | UV coordinate of local vertex j **after**  the collapse |

Both are indexed by the same local index j (same as `subsetVIdx_si`).
A point expressed in `UV_post` (post-collapse space) can be looked up in
the `UV_pre` triangles to find the corresponding finer-mesh location.

---

### 1.6 What gets saved (`SheetData` in `single_collapse_data`)

```
sd.global_sheet_id = sid          // which partition_into_sheets sheet
sd.b               = b_si         // local indices of (vi, vj)
sd.subsetVIdx      = subsetVIdx_si // local → global gV index
sd.UV_pre          = UV_pre_si    // pre-collapse UV  (nLocal × 2)
sd.UV_post         = UV_post_si   // post-collapse UV (nLocal × 2)
sd.FUV_pre         = FUV_pre_si   // pre-collapse faces  (nFaces × 3, local indices)
sd.FUV_post        = FUV_post_si  // post-collapse faces (nFaces × 3, local indices)
sd.FIdx_pre        = FIdx_pre_si  // global gF index of each FUV_pre row
sd.FIdx_post       = FIdx_post_si // global gF index of each FUV_post row
```

For a non-manifold edge, `single_collapse_data.sheets` has one `SheetData`
per active sheet.  For manifold edges, exactly one entry.

---

### 1.7 Update `decIM`

After the collapse, for every face in the **union** of all sheets'
`FIdx_pre_si`:

```cpp
decIM[f].push_back(current_collapse_index);
```

`decIM[f]` is a list of all collapse indices that had face `f` in their
pre-collapse one-ring.  This is the index structure the backward walk uses
to find which collapse to trace back through.

---

### 1.8 VF topology update (happens AFTER UV computation)

```
Pass 1: kill flap faces (contain both s and d) + their edges
Pass 2: remap d → s in all surviving faces of the one-ring
V.row(s) = p;  V.row(d) = p;          // both get the new position
VF[s] ← VF[s] ∪ VF[d];  VF[d].clear()
```

After this, vertex `d` is effectively gone from the topology.

---

## 2. What the stored data represents geometrically

```
UV_post   =  the "coarse" UV domain  — what the one-ring looks like
             AFTER the collapse, parameterised conformally.

UV_pre    =  the "fine" UV domain    — what the one-ring looked like
             BEFORE the collapse, parameterised conformally.

Both UV domains share the same local vertex set (subsetVIdx).
For a coarse-mesh point p expressed as a barycentric combination of
UV_post triangle vertices, the corresponding fine-mesh location is found
by looking up that same UV coordinate inside the UV_pre triangles.
```

This is the key insight: UV_post and UV_pre are a **bijective pair** over
the same one-ring, so a point in UV_post space has a unique preimage in
UV_pre space, and that preimage is closer to the original fine mesh.

---

## 3. Backward walk: `query_coarse_to_fine`

### Input (for N query points)

```
BF   (N×3 int)    : current triangle — 3 global gV vertex indices
BC   (N×3 double) : barycentric coordinates within that triangle
FIdx (N int)      : global gF index of that triangle
```

### Initialisation

```cpp
BF(r,c) = IM(BF(r,c));   // IM = identity in SSP
FIdx(i) = IMF(FIdx(i));  // IMF = identity in SSP
```

In SSP vertex and face indices are never renumbered, so IM and IMF are
both identity.

### One backward step (per query, repeated until no more collapses touch it)

```
queryFIdx = FIdx(q)

1. dList = decIM[queryFIdx]
   Find the largest collapse index dIdx in dList that is < current_dIdx.
   → this is the most recent collapse that involved face queryFIdx.

2. Sheet routing:
   sid     = faceSheetID(queryFIdx)   ← sheet of the current face
   sd_ptr  = SheetData in decInfo[dIdx].sheets where global_sheet_id == sid
   (fallback: sheets[0] if no exact match — manifold case)

3. Convert BF vertices to local indices:
   f = BF.row(q)    ← 3 global vertex indices
   Find v0, v1, v2 such that sd.subsetVIdx(v0) == f(0), etc.
   *** THIS IS WHERE THE ASSERT FIRES if any f(k) is not in subsetVIdx ***

4. Evaluate in UV_post:
   queryUV = BC(q,0)*UV_post.row(v0)
           + BC(q,1)*UV_post.row(v1)
           + BC(q,2)*UV_post.row(v2)

5. Find queryUV in UV_pre triangles (compute_barycentric):
   B       = barycentric coordinates of queryUV in each FUV_pre triangle
   idxToFUV = triangle with best (least negative) min barycentric weight

6. Update query state:
   BC(q)   = B.row(idxToFUV)            ← new barycentric coords
   BF(q,k) = sd.subsetVIdx(FUV_pre(idxToFUV, k))  for k=0,1,2
   FIdx(q) = sd.FIdx_pre(idxToFUV)      ← now points to a finer face

Repeat from step 1 using the updated FIdx(q) and BF(q).
```

### Termination

The loop breaks when `decIM[FIdx(q)]` has no collapse index smaller than
the current `dIdx`.  At that point `BF(q)` and `BC(q)` express the query
point as barycentric coordinates inside a face of the **original fine mesh**.

### Output

```
BF   : vertex indices in the original fine mesh (gVO rows)
BC   : barycentric weights — fine mesh position = BC(0)*gVO[BF(0)]
                                                 + BC(1)*gVO[BF(1)]
                                                 + BC(2)*gVO[BF(2)]
FIdx : face index in the original fine mesh
```

---

## 4. Assert failure context

**Location:** `query_coarse_to_fine.cpp:104`
```cpp
assert(v1_vec.size() == 1);  // f(1) not found in sd.subsetVIdx
```

This fires at step 3 of the backward walk above.
`f(1) = BF(q,1)` is the second global vertex of the current query triangle.
It must appear exactly once in `sd.subsetVIdx`.  If it does not:

- `size == 0` : the vertex is not part of this collapse's one-ring at all
  (wrong sheet routing, over-populated decIM, or stale faceSheetID)
- `size >= 2` : duplicate vertex in subsetVIdx (shouldn't happen normally)

See `c2f_debug.md` for the three hypotheses and diagnostic logging plan.
