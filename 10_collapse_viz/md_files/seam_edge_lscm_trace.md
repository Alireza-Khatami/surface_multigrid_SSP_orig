# Seam Edge UV Parameterization: Code Trace

---

## Question
For non-manifold meshes, is the seam edge divided into sheets and treated as a boundary
when passed to LSCM for UV parameterization?

---

## Step 1 — Sheet bucketing (`SSP_collapse_edge.cpp:194–208`)

For each edge `e` with endpoints `E(e,0)` and `E(e,1)`:

```cpp
for (int f : Nsf) {   // all faces of E(e,1)
    if (null_face(f) || f >= numOrigFaces) continue;
    sheets_Nsf[faceSheetID(f)].push_back(f);
}
for (int f : Ndf) {   // all faces of E(e,0)
    if (null_face(f) || f >= numOrigFaces) continue;
    sheets_Ndf[faceSheetID(f)].push_back(f);
}
// active = sheets where BOTH have real faces
for (auto & kv : sheets_Nsf)
    if (sheets_Ndf.count(kv.first) && !sheets_Ndf[kv.first].empty())
        active_sheets.insert(kv.first);
```

For a seam edge shared by 3 sheets, all 3 end up in `active_sheets`.
The UV loop at line 266 runs **once per active sheet**.

---

## Step 2 — What the local mesh looks like per sheet

`FUV_pre_si` is built from **only the faces of sheet `sid`** in the one-ring.

For a seam edge (vi, vj) with e.g. 2 flap faces on sheet `sid`:
- Both flap faces are **present** in `FUV_pre_si` — the edge (vi, vj) is an **interior edge** of
  this sheet's local patch (faces on both sides within the sheet).
- Faces from other sheets are **entirely absent**.

After `get_post_faces`, the 2 flap faces are removed. In `FUV_post_si` the edge (vi, vj) is gone
and the surviving faces form the post-collapse ring.

The seam edge creates an **implicit** open side in the local patch due to the absence of
other-sheet faces — but this is never explicitly marked.

---

## Step 3 — Boundary detection (`joint_lscm.cpp:37–53`)

```cpp
int infVIdx = -1;   // local sentinel value for the infinity vertex

VectorXi onBd(2);
onBd.setZero();
if (Nsv[Nsv.size()-1] == vj)
{
    if (std::find(Nsv.begin(), Nsv.end(), infVIdx) != Nsv.end())
        onBd(0) = 1;   // E(e,1) is on real mesh boundary
    if (std::find(Ndv.begin(), Ndv.end(), infVIdx) != Ndv.end())
        onBd(1) = 1;   // E(e,0) is on real mesh boundary
}
```

`onBd(k) = 1` **only** when the value `-1` appears in `Nsv` or `Ndv`.
That `-1` comes from `fan_walk_local`'s `to_local` lambda (`SSP_collapse_edge.cpp:94–97`):

```cpp
auto to_local = [&](int gv) -> int {
    if (gv == infVtx) return -1;   // infVtx = V.rows()-1 (boundary-extension vertex)
    for (int i = 0; i < sz; i++)
        if (svIdx(i) == gv) return i;
    return -1;
};
```

`infVtx` only appears in the infinity triangles added by `igl::connect_boundary_to_infinity` —
one triangle per **real mesh boundary edge**. A seam edge is NOT a real boundary edge (it is
shared by 3+ original faces), so no infinity triangle is adjacent to it.

| Edge type | `infVtx` in walk? | `-1` in Nsv/Ndv? | `onBd` |
|---|---|---|---|
| Real mesh boundary | Yes — infinity face attached | Yes | 1 |
| Interior seam edge | No — no infinity face at seam | No | 0 |

The seam edge is **never detected** by this mechanism. `onBd` is purely about the real mesh
boundary, not about non-manifold seams.

---

## Step 4 — Case selection (`joint_lscm.cpp:214–225`)

```cpp
int whichCase;
if      (onBd.sum() == 0)  whichCase = 0;   // both vi, vj interior
else if (onBd.sum() == 1)  whichCase = 1;   // one on real boundary
else if (onBd.sum() == 2)  whichCase = 2;   // both on real boundary
```

For a seam edge in the mesh interior (no real boundary involvement):
`onBd.sum() == 0` → **Case 0**.

---

## Step 5 — What gets pinned in the LSCM solve (Case 0, `joint_lscm.cpp:584–629`)

```cpp
nVjoint = nV + 1;

// Joint vertex array = [V_pre rows ; V_post.row(vi)]
// The post-collapse position of vi is added as a NEW vertex at index nV.
Vjoint.row(nV) = V_post.row(vi);

// Fjoint_post: every reference to vi → nV (separate UV slot for post-vi)
for (r, c): if Fjoint_post(r,c) == vi → Fjoint_post(r,c) = nV;

// Pin only 2 vertices:
b_UV  << vi_pre, vj_pre, vi_pre + nVjoint, vj_pre + nVjoint;
bc_UV << 0,      1,      0,                0;
//        vi=(0,0)  vj=(1,0)   (y-coords both 0)
```

The energy minimized inside `flatten` (`joint_lscm.cpp:526`):

```cpp
Q = -LUV_pre + 2*A_pre  -LUV_post + 2*A_post;
```

This is the **joint LSCM energy** — simultaneously minimizing conformal distortion for the
pre-collapse mesh (`L_pre`, `A_pre`) and the post-collapse mesh (`L_post`, `A_post`) in a
single shared UV space. The seam edge is not a constraint; only vi and vj are pinned.

---

## Direct answer

| Claim | What the code does |
|---|---|
| "divide seam edge into sheets" | **Yes.** Each active sheet runs a completely separate UV computation with its own local mesh patch. |
| "treat the seam edge as boundary" | **Not explicitly.** `onBd` only fires when `infVtx` (-1) is in the fan walk, which only happens at real mesh boundaries — not at seam edges. |
| "pass boundary to LSCM" | **Not as a boundary constraint.** For an interior seam edge (Case 0), only vi and vj are pinned at `(0,0)` and `(1,0)`. LSCM has no knowledge that the edge was a seam. |

### What is actually happening

Each active sheet gets its own local mesh patch (`FUV_pre_si`) built from only that sheet's
faces. Because faces from other sheets are excluded, the seam edge creates an **implicit** open
side in the local patch. But this is never communicated to LSCM as a boundary condition.

LSCM simply sees a small mesh with 2 pinned vertices and minimizes conformal distortion jointly
for the pre- and post-collapse configurations. The concept of "this edge was a seam" does not
exist inside `joint_lscm`.

The seam edge participates in the UV computation only as the edge being collapsed — its two
endpoints vi and vj are the pinned boundary conditions, exactly as they would be for any
interior edge on a manifold mesh.
