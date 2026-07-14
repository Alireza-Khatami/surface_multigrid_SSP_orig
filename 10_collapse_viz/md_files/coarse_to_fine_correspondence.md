# Coarse-to-Fine Correspondence: How It Works

---

## 1 — What the correspondence IS

**Between:** every live coarse mesh vertex → a point on the **original** fine mesh.

The result for each coarse vertex is:
```
bc0, bc1, bc2,   fv0, fv1, fv2
```
meaning the 3-D point `bc0·gVO[fv0] + bc1·gVO[fv1] + bc2·gVO[fv2]` on the original
fine mesh surface. The arrows in the visualizer are exactly `fine_pos − coarse_pos`
(`coarse_fine_viz.cpp:353`).

---

## 2 — "1 level coarser" is wrong

> "barycentric coords of all the vertices 1 level coarser to the current level finer mesh"

| Part | Correct? | Reality |
|---|---|---|
| barycentric coords | ✅ | `(bc0, bc1, bc2)` within a fine mesh triangle |
| all the vertices | ✅ | one query per live coarse vertex (`coarseVerts`, lines 75–86) |
| 1 level coarser | ❌ | from the **fully decimated** coarse mesh all the way back to the **original** fine mesh, tracing through **every collapse** along the way |

There is no concept of "1 level" — SSP is a continuous sequence of collapses, not a
hierarchy. The backward walk chains as many steps as needed until it reaches a face
with no earlier history.

---

## 3 — What ONE collapse stores (the local piece)

Each collapse records, for its local one-ring patch, a `SheetData` containing:

```
UV_post   (nLocalV × 2)   UV coords of the post-collapse patch
UV_pre    (nLocalV × 2)   UV coords of the pre-collapse patch
FUV_post  (nFacesPost × 3) face connectivity in local UV space, post-collapse
FUV_pre   (nFacesPre  × 3) face connectivity in local UV space, pre-collapse
FIdx_post (nFacesPost,)   global face row indices, post-collapse
FIdx_pre  (nFacesPre,)    global face row indices, pre-collapse
subsetVIdx (nLocalV,)     global vertex indices of the compact patch
```

`joint_lscm` aligns `UV_pre` and `UV_post` so that **the same `(u,v)` coordinate
represents the same material point** in both maps.

The correspondence encoded by one collapse is:
```
given  (face in FIdx_post, bc_post)
→ project:  queryUV = bc0·UV_post[v0] + bc1·UV_post[v1] + bc2·UV_post[v2]
→ lift:     find (face in FIdx_pre, bc_pre) s.t. bc_pre maps to queryUV in UV_pre
→ result:   (face in FIdx_pre, bc_pre)   — one step earlier in time
```

---

## 4 — Initialization (where the first BC and face come from)

`coarse_fine_viz.cpp` lines 97–179, called before `query_coarse_to_fine`:

**For each live coarse vertex `vi`:**

**Step 1** — pick any live coarse face `fi` touching `vi`:
```cpp
for (int f : gVF[vi]) {          // vertex-face adjacency of current coarse mesh
    if (f >= nFO) continue;       // skip infinity faces
    if (gF(f,0) == NULL) continue;
    fi = f; break;
}
```

**Step 2** — find the most recent collapse that recorded `fi` in its one-ring:
```cpp
const auto & dList = gDecIM[fi];
for (int ii = dList.size()-1; ii >= 0; ii--)
    if (dList[ii] < nDec) { dIdx = dList[ii]; break; }
```

**Step 3a** — if `dIdx < 0` (no collapse ever touched `fi`):
```cpp
BF(i)   = gF.row(fi)    // vertices directly from current F
BC(i)   = (1, 0, 0)
FIdx(i) = fi
```

**Step 3b** — if `dIdx ≥ 0`: look up the `SheetData` for collapse `dIdx`, find `fi`
in `FIdx_post`, read its vertex triplet from `subsetVIdx(FUV_post(post_row, *))`:
```cpp
gv0 = sd.subsetVIdx(sd.FUV_post(post_row, 0))
gv1 = sd.subsetVIdx(sd.FUV_post(post_row, 1))
gv2 = sd.subsetVIdx(sd.FUV_post(post_row, 2))
// rotate so vi is at column 0
BF(i)   = (vi, ..., ...)
BC(i)   = (1, 0, 0)
FIdx(i) = fi
```

Step 3b is needed instead of reading `gF` directly because later Pass 2 remaps can
overwrite `fi`'s vertex columns in `gF` for collapses that didn't write a `decIM`
entry. `FIdx_post` from `dIdx` captures the vertex state at the moment that was
actually recorded — safe to start the walk from.

**Starting BC is always `(1, 0, 0)`** — the coarse vertex itself sits exactly at
corner 0 of the starting face.

---

## 5 — The backward walk: how BC is updated at every step

`query_coarse_to_fine.cpp`. At each step the state is `(FIdx, BC, BF)`.

### Step A — find the next collapse from `decIM`
```cpp
// find largest entry in decIM[FIdx(qIdx)] that is < current dIdx
dIdx = ...;
if (!found) break;   // EXIT: no more history → FIdx is a fine-mesh face
```

### Step B — get the UV corners of the current face in this collapse's pre-patch
```cpp
// find row pre_row in FIdx_pre where FIdx_pre(pre_row) == FIdx(qIdx)
int v0 = FUV_pre(pre_row, 0);   // local vertex indices (into subsetVIdx / UV rows)
int v1 = FUV_pre(pre_row, 1);
int v2 = FUV_pre(pre_row, 2);
```

### Step C — project current BC through `UV_post` → get `queryUV`
```cpp
queryUV = BC(0)*UV_post.row(v0)
        + BC(1)*UV_post.row(v1)
        + BC(2)*UV_post.row(v2);
```
This converts from face-space into the 2-D UV domain of the post-collapse patch.

### Step D — lift `queryUV` into `UV_pre` → new (face, BC)
```cpp
compute_barycentric(queryUV, UV_pre, FUV_pre, B);
// B is (nFacesPre × 3): B(r,c) = weight on column c of face r in UV_pre
idxToFUV = row of B with most-positive minimum coordinate;
```

### Step E — update state
```cpp
BC.row(qIdx)  = B.row(idxToFUV);                        // new barycentric coords
BF(qIdx, *)   = subsetVIdx(FUV_pre(idxToFUV, *));       // new global vertices
FIdx(qIdx)    = FIdx_pre(idxToFUV);                      // new face
```

BC is **computed fresh** at every step — never stored between steps. Only
`UV_pre`, `UV_post`, `FUV_pre`, `FIdx_pre`, `subsetVIdx` are read from storage.

---

## 6 — The column contract (why BC(c) always means corner c)

`BC(c)` is always the weight on **column c** of the current face. This is maintained
because:

- `FUV_pre` is built by `remove_unreferenced_lessF` reading `F.row(fi)` directly.
- SSP's Pass 2 replaces a vertex value at a specific column (`F(f,col) = s`) but
  **never reorders columns**.
- Therefore, column 0 of face `fi` in `F` is always the same geometric corner across
  all collapses — only the vertex index stored there changes as vertices get absorbed.

`compute_barycentric` returns weights relative to `FUV_pre`'s column ordering, so
`B(idxToFUV, 0)` is the weight on `FUV_pre(idxToFUV, 0)` and becomes `BC(0)` for
the next step — always referring to the same geometric corner.

```
Step N result:
  FIdx = fi,   FUV_pre(row, *) = [v0, v1, v2],   BC = [0.6, 0.3, 0.1]
               └col 0┘ └col 1┘ └col 2┘

Step N+1: look up fi in next collapse's FUV_pre → same column ordering
  queryUV = 0.6·UV_post[v0'] + 0.3·UV_post[v1'] + 0.1·UV_post[v2']
                ↑BC(0)               ↑BC(1)             ↑BC(2)
```

---

## 7 — Diagnostic findings: early walk exits and the real bug

### Finding 1 — early exits are correct (NOT a bug)

All 427 backward-walk queries exit early. The `[SHEET-ALL-FAILED]` diagnostic
confirmed that rejected collapses touch large-numbered faces (1952, 2435, 2482 ...)
which are non-manifold seam-duplicate faces. The backward walk exit faces (21, 64,
72, 92 ...) are small-numbered original faces that appear nowhere in the rejected
collapse one-rings.

**Early exits are correct feature-face behavior** — faces 21, 64, 72 are high-quality
feature triangles that were last touched by collapses in the 800–1078 range (the
final ~25% of decimation). The backward walk correctly exhausts their `decIM` history
and terminates with valid fine-mesh barycentric positions.

### Finding 2 — the real bug: 3-sheet seam collapses all rejected

Every `[SHEET-ALL-FAILED]` entry shows `active_sheets=3`. These are edges shared by 3
topological sheets simultaneously on the non-manifold mesh. All 3 sheets fail their
invariant checks so the collapse is rejected entirely — the seam region of the mesh
is left under-decimated.

The degenerate walk `[0,6,]  -1 at pos -1 / 2` comes from exactly this case: a
3-sheet seam vertex where `fan_walk_local` dead-ends after 2 steps because the face
adjacency within one sheet's sub-list doesn't connect all the way to `end_vtx`.

| Question | Answer |
|---|---|
| Are early backward-walk exits caused by the degenerate fan walk? | No — different faces entirely |
| Is the backward walk producing correct correspondences? | Yes — feature faces with late `decIM` entries, correctly terminated |
| Is there a real bug? | Yes — 3-sheet seam edges are being rejected wholesale |

### Root cause of the 3-sheet failure

`fan_walk_local` was designed for 1-sheet (manifold interior) and 2-sheet (seam)
edges. With 3 active sheets, the `real_si` face list passed to `get_walk_faces` for
each individual sheet is a strict subset of the full one-ring. The face adjacency
within that subset dead-ends before reaching `end_vtx`, producing the `[0,6,]` walk
and failing INV-F.

**Next step:** either handle the 3-sheet fan walk differently within
`fan_walk_local`, or detect the 3-active-sheets case before the per-sheet loop and
route to a dedicated fallback path.



o we do this for all the 3 vertices of the finer level [which encapsulate the current query poirnt , right ? ]( to find out each one belongs
  to what faces and triangle right ?)

● No — it's the opposite. There is only one query point per coarse vertex, and BF + BC together just describe where that single point lands.

  ---
  What BF and BC actually are

  At the end of the walk for coarse vertex vi:

  BF = (fv0, fv1, fv2)   ← the 3 corners of whichever fine-mesh triangle the point landed in
  BC = (bc0, bc1, bc2)   ← how far inside that triangle the point sits

  The actual 3-D position is reconstructed as one weighted sum:
  fine_pos = bc0·V[fv0] + bc1·V[fv1] + bc2·V[fv2]

  BF is not 3 separate queries — it's 3 corners of the container triangle. BC is not 3 separate results — it's 3 weights that locate one point
  inside that triangle.

  ---
  The walk traces one point, not three vertices

  At each step:
  queryUV = BC(0)·UV_post[v0] + BC(1)·UV_post[v1] + BC(2)·UV_post[v2]

  This computes one 2-D point — a weighted blend of the three UV corners. Everything from that point on is single-point arithmetic: one
  compute_barycentric call finding which pre-collapse face that one point falls in, outputting one new (FIdx, BC).

  BF and FIdx are just bookkeeping — they tell you which local UV patch to load at the next step. The actual quantity being propagated is the
  single point queryUV, re-expressed as new barycentric coords in a new face at each step.

  ---
  Summary

  ┌────────────────────────────────────────┬──────────────────────────────────────────────────────┐
  │          What you might think          │                What actually happens                 │
  ├────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ 3 separate walks, one per BF vertex    │ 1 walk, tracing 1 point                              │
  ├────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ BF = 3 query results                   │ BF = 3 corners of the landing triangle               │
  ├────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ BC = 3 separate correspondences        │ BC = 3 weights locating 1 point inside that triangle │
  ├────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ Find which face each vertex belongs to │ Find which triangle contains the single query point  │
  └────────────────────────────────────────┴──────────────────────────────────────────────────────┘

  The 3 vertices in BF only exist so the final 3-D position can be interpolated. They are a consequence of the walk's last step — whichever
  fine-mesh triangle the point happened to fall into.
