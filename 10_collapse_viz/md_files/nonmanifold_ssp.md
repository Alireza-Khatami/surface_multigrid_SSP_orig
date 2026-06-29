# SSP on Non-Manifold Meshes (MAT)

## Advisor guidance (summary)

SSP as implemented cannot handle non-manifold meshes. For a seam edge shared by sheets
S1, S2, S3, each sheet must be treated as a **separate manifold region** where `(vi, vj)`
sits on the **boundary** of that region — not in its interior.

> "you should regard s1, s2, and s3 as 3 different local regions. so it won't be a one ring
> neighborhood. the simplified edge and the new vertex will be on the boundary of those local
> regions not in the middle."

---

## How it differs from standard SSP

| | Standard SSP | Non-manifold SSP |
|---|---|---|
| One-ring | All faces around `(vi, vj)` — one connected disk | Split into per-sheet fans — each fan is a strip |
| `vi`, `vj` topology | May be interior to the one-ring | Always boundary of each sheet's fan |
| LSCM case | Case 0 (interior), 1, or 2 | Always **Case 2** (boundary edge) in every sheet |
| Accept condition | All UVs non-folded in the single joint solve | All UVs non-folded **in every sheet's** joint solve |

---

## What "boundary of each local region" means in practice

For a seam edge `(vi, vj)` shared by three sheets:

```
S1 fan:       S2 fan:       S3 fan:
 a             c             e
 |\ S1_faces   |\ S2_faces   |\ S3_faces
 | \           | \           | \
vi--vj        vi--vj        vi--vj
 | /           | /           | /
 |/ S1_faces   |/ S2_faces   |/ S3_faces
 b             d             f
```

Each fan is a strip of triangles with `vi` and `vj` on its two straight edges (boundary
vertices). After collapsing `(vi, vj)` to `p`, the point `p` remains on the boundary of
each fan and the two triangles that contained the edge disappear from each fan.

---

## Concrete change to `SSP_collapse_edge`

**Currently:**
```
gather full one-ring faces around (vi, vj)  →  one (V_pre, FUV_pre) + (V_post, FUV_post)
run joint_lscm once
```

**With non-manifold support:**
```
partition one-ring faces by sheet  →  {S1_faces, S2_faces, S3_faces}
for each sheet Si:
    build (V_pre_i, FUV_pre_i) and (V_post_i, FUV_post_i)
    (vi, vj) is a boundary edge of Si_faces  →  always Case 2 of joint_lscm
    run joint_lscm_case2 on Si
    if any Si fails  →  reject the collapse
```

---

## Implementation plan

### 1. Sheet detection

Given the face list from `igl::circulation`, group faces by which sheet they belong to.
Since each sheet is a 2-manifold, a flood-fill by edge-adjacency restricted to the
one-ring faces will find the connected components. Each component is one sheet.

### 2. Per-sheet geometry extraction

Run the same extraction that currently happens once:

```
get_collapse_onering_faces(V, F, vi, vj, Si_faces, ...)  →  V_pre_i, FUV_pre_i
remove_unreferenced_lessF(V, FUV_pre_i, ...)
```

done once per sheet fan instead of once total.

### 3. `joint_lscm_case2` always

Since `(vi, vj)` is always a boundary edge in each sheet's local region, the existing
Case 2 code handles it without modification. Call it once per sheet and require all
sheets to pass before accepting the collapse.

### 4. `decInfo` per-sheet storage

`single_collapse_data` currently holds one `(UV_pre, UV_post, FUV_pre, FUV_post,
subsetVIdx)`. With multiple sheets it becomes a vector of such tuples:

```cpp
struct single_collapse_data {
    std::vector<SheetData> sheets;   // one entry per sheet
    // SheetData: UV_pre, UV_post, FUV_pre, FUV_post, subsetVIdx, FIdx_pre, FIdx_post
};
```

The `query_coarse_to_fine` reverse traversal then routes each subdivided vertex to the
correct sheet based on which sheet's face it falls on.

---

## Three maps per collapse — is it a problem?

No. The reason is implicit in how `query_coarse_to_fine` works.

Each subdivided vertex carries a `FIdx` — the coarse face it sits on. Every coarse face
belongs to **exactly one sheet**. So when tracing a vertex backward through a collapse,
you already know which sheet it's on from its `FIdx`, and you use only that sheet's
`(UV_pre, UV_post)` pair. The other two maps are never consulted for that vertex.

```
subdivided vertex q
  └─ FIdx[q]  →  belongs to sheet S2
       └─ collapse dIdx touches FIdx[q]
            └─ use sheets[1].UV_post, sheets[1].UV_pre   (only S2's map)
```

The 3 maps are independent and never need to agree with each other in UV space — they
just all need to be fold-free for the collapse to be accepted. The only thing shared
across all 3 is the 3D position of the new vertex `p`, which is the same point in world
space regardless of which sheet you're looking at it from.

Having 3 maps per collapse adds storage and 3× the LSCM solves, but the reverse traversal
logic stays exactly the same — it just indexes into `sheets[sheet_id]` instead of a flat
struct.

---

## Open question for implementation

How are sheet memberships identified in the MAT mesh — is each face already tagged with a
sheet ID, or does it need to be inferred from face-adjacency at seam edges?


## 1'st crash 

● The crash is circulation reading an invalid EF entry for a seam edge. Let me see the exact circulation code and the EF values that cause it:

  Read 2 files (ctrl+o to expand)

● Now I understand the crash precisely. Line 119 is rv = F(nf, nv) where nf = EF(e, nside). For a non-manifold edge edge_flaps only stores 2 of the 3+ incident faces, so when circulation traverses a seam
  vertex's one-ring it can land on a face index of -1 (the edge_flaps default for unseen sides), crashing the Eigen bounds assertion.

  The fix: compute seam vertices (any vertex touching faces from multiple sheets) and skip collapses involving them before circulation is ever called.

  I'll add seamVertex as an optional parameter on the outer SSP_collapse_edge, compute it in SSP_midpoint and 10_collapse_viz/main.cpp from faceSheetID: