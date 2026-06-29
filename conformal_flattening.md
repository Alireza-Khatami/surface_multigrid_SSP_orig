# Conformal Flattening in surf_subgrid_SSP_orig

## Role in SSP decimation

SSP (Successive Self-Parameterization) decimates a mesh by repeatedly collapsing edges. Each
candidate edge collapse involves two vertices `vi` and `vj`. Before committing a collapse, the
algorithm needs to check whether the collapse would introduce a fold-over — i.e., whether the
mesh locally remains bijective across the change.

Conformal flattening is the tool for that check. It maps both the **pre-collapse** one-ring and
the **post-collapse** one-ring into 2D and verifies that neither flattening has any flipped
triangles.

---

## What "before" and "after" mean

Every edge collapse changes a local neighbourhood of the mesh:

```
Before (pre)                      After (post)

   a --- b                           a --- b
   |\ F1 |                           |  \ |
   | \   |                           |   \|
   |  vi-vj        vi+vj merged      | vi'=p
   |   / |          into p -------->  |   /|
   | /   |                           |  / |
   | \ F2|                           | /  |
   c --- d                           c --- d
```

- **`V_pre`, `FUV_pre`** — the edge one-ring of `(vi, vj)` **before** the collapse:
  all faces that contain `vi` or `vj`.
- **`V_post`, `FUV_post`** — the vertex one-ring **after** the collapse:
  same geometry but `vi` moved to the new midpoint `p`, and `vj` and its two adjacent faces
  deleted.

Both meshes share the same vertex set; `FUV_post` is `FUV_pre` with `vj` removed from the
face list and `vi` moved to `p`.

---

## Why a *joint* LSCM

A single LSCM of `V_pre` would not tell you anything about `V_post`. Computing them
independently would also not guarantee that the two UV layouts share the same boundary — which
is required to compare them.

The **joint solve** flattens both meshes simultaneously into the same UV domain with a shared
boundary curve (the outer ring of neighbours). This gives two UV layouts whose boundaries
coincide by construction. A fold-over test on each layout then tells you:

- Does `V_pre` flatten without overlap? (baseline — should always pass for a valid input)
- Does `V_post` flatten without overlap? (the real test — rejects the collapse if not)

If either check fails, `joint_lscm` returns `false` and the collapse is skipped.

---

## Why LSCM is a conformal mapping

The LSCM energy for a single mesh is:

```
E(u) = u^T (-L + 2A) u
```

where:
- `u` is the stacked UV vector `[u₁…uₙ, v₁…vₙ]^T`
- `L` is the **cotangent Laplacian**: `L_ij = -½(cot α_ij + cot β_ij)` for adjacent vertices,
  diagonal entries are the negative row sum. It measures the harmonic (smooth) part of the map.
- `A` is the **vector area matrix**: a skew-symmetric matrix that couples the `u` and `v`
  components. For a triangle with vertices `(u1,v1), (u2,v2), (u3,v3)` it enforces
  `(u2-u1)(v3-v1) - (u3-u1)(v2-v1) > 0` (positive signed area = no flip).

Minimising `-L + 2A` is equivalent to minimizing the integral

```
∫_M |∂f/∂z̄|² dA
```

over the surface `M`, where `z̄` is the conjugate complex coordinate. Setting this to zero
gives the **Cauchy-Riemann equations** in weak form:

```
∂u/∂x = ∂v/∂y
∂u/∂y = -∂v/∂x
```

A map satisfying Cauchy-Riemann is conformal — it is locally angle-preserving. Lengths may
scale by some factor, but the shape of infinitesimal triangles is preserved.

The cotangent Laplacian (`-L`) pulls `u` and `v` toward harmonic functions (zero Laplacian),
minimising stretching. The area matrix (`+2A`) couples `u` and `v` so they satisfy the
Cauchy-Riemann constraint. Together they produce the least-squares best conformal map given
the boundary.

---

## Joint quadratic in `flatten()` (`src/joint_lscm.cpp:483`)

The code builds one quadratic that covers *both* pre and post meshes simultaneously:

```cpp
Q = -LUV_pre + 2*A_pre - LUV_post + 2*A_post
```

Both `LUV_pre` and `LUV_post` are block-diagonal: `[L, 0; 0, L]` in the `(u,v)` stacking.

The sign convention is:

| Term | Sign | Role |
|------|------|------|
| `-LUV_pre` | − | pre-mesh conformality (cotangent smoothness) |
| `+2*A_pre` | + | pre-mesh orientation (signed area coupling) |
| `-LUV_post` | − | post-mesh conformality |
| `+2*A_post` | + | post-mesh orientation |

The joint vertex set is `nVjoint = nV + 1`, where the extra vertex at index `nV` represents
`vi`'s new position after the collapse. `FUV_post` is reindexed so `vi → nV`, making both
face lists operate over the same `nVjoint`-vertex domain.

### Joint mesh construction (case 0: both interior, `joint_lscm_case0`)

```
nVjoint = nV + 1

Vjoint       = [V_pre rows 0..nV-1 ; V_post.row(vi)]  // vi's new position appended at nV
Fjoint_pre   = FUV_pre                                 // unchanged
Fjoint_post  = FUV_post with vi → nV                  // vi_post is a distinct index
```

Boundary pins (case 0):

```
b_UV  = [vi_pre, vj_pre, vi_pre + nVjoint, vj_pre + nVjoint]
bc_UV = [0,      1,      0,                0               ]
```

This fixes `vi → (0, 0)` and `vj → (1, 0)` in UV space. The rest of the boundary (the shared
outer ring) is determined by the LSCM solve, not by hard constraints.

---

## How the linear system is solved (`src/mqwf_dense.cpp`)

The problem is a **constrained quadratic**:

```
minimise  ½ x^T Q x + x^T b      subject to  x[k] = bc
```

With `b = 0` (RHS is zero), this reduces to `Q x = 0` with the known entries fixed.

**Precompute step** (`mqwf_dense_precompute`):
1. Partition indices into known (`k`) and unknown (`u`).
2. Slice `Q` into blocks `Quu`, `Quk`, `Qku`.
3. Factor `Quu` with **LDLT** (Cholesky-like for symmetric indefinite matrices):
   `data.Auu_pref = Quu.ldlt()`

**Solve step** (`mqwf_dense_solve`):
1. Build the reduced right-hand side:
   `RHS_reduced = -½ (Quk + Qku^T) * bc`
2. Solve the reduced system:
   `x[u] = Quu^{-1} * RHS_reduced`   (using the stored LDLT factor)
3. Write known values back:
   `x[k] = bc`

LDLT is chosen over plain LLT because `Q` is symmetric but may not be strictly positive
definite near degenerate configurations.

---

## Quasi-conformal error (`src/quasi_conformal_error.cpp`)

After flattening, each triangle's distortion is measured by the **quasi-conformal ratio**
`σ/γ`, where `σ ≥ γ > 0` are the singular values of the Jacobian of the map from 3D to UV:

```
Ss = (q1*(t2-t3) + q2*(t3-t1) + q3*(t1-t2)) / (2A)   // ∂position/∂s
St = (q1*(s3-s2) + q2*(s1-s3) + q3*(s2-s1)) / (2A)   // ∂position/∂t

a = Ss·Ss,  b = Ss·St,  c = St·St

σ = sqrt((a+c + sqrt((a-c)²+4b²)) / 2)
γ = sqrt((a+c - sqrt((a-c)²+4b²)) / 2)

error = σ / γ
```

- `σ/γ = 1` is a perfect conformal map (circle maps to circle).
- `σ/γ → ∞` signals a near-degenerate or folded triangle.

This is the metric from *"Texture Mapping Progressive Meshes"* (Sander et al. 2001). It is used
upstream to rank or reject candidate collapses based on how much distortion they introduce.

---

---

## Boundary edge handling

Boundary vertices are detected via the sentinel value `infVIdx = -1` in the neighbour lists
`Nsv` and `Ndv`. If `Nsv` contains `-1`, then `vi` is on the mesh boundary; similarly for `vj`
via `Ndv`. This gives three cases.

---

### Flap rejection (pre-check, all cases)

Before any case is tried, a **flap** is detected:

> A flap is when both `vi` and `vj` are on the mesh boundary **and** they share two boundary
> edges (not just one). Topologically the one-ring looks like a leaf — two boundary edges meet
> at the far corner. Collapsing such an edge would merge two separate boundary segments, making
> the result non-manifold.

```
Boundary:  ...--a--vi--b...       but also  ...--a--vj--b...
                    collapsed →  non-manifold pinch
```

If this is detected, `joint_lscm` returns `false` immediately and the collapse is skipped.
Detection code (`joint_lscm.cpp:60`):

```cpp
if (onBd.sum() == 2)           // both on boundary
{
    if (Nsv0==Ndv0 && Nsv0==infVIdx) isFlap = false;
    else if (Nsv1==Ndv1 && Nsv1==infVIdx) isFlap = false;
    else isFlap = true;         // they share boundary on both sides → flap
}
if (isFlap) return false;
```

---

### Case 0 — both interior

Handled by `joint_lscm_case0`.

The shared outer boundary of the one-ring is the natural constraint. Because both `vi` and `vj`
are interior, the post-collapse mesh still has the same outer boundary ring. The only thing that
changes topologically is that `vi` moves to a new position and the two faces that contained edge
`(vi,vj)` disappear.

**Joint mesh construction:**

```
nVjoint = nV + 1
Vjoint  = [V_pre; V_post.row(vi)]   ← vi's new position appended at index nV
Fjoint_pre  = FUV_pre                ← unchanged
Fjoint_post = FUV_post, vi → nV     ← post-mesh references the new vertex instead of vi
```

**Pins (4 constraints):**

```
vi_pre  → (0, 0)    u=0, v=0
vj_pre  → (1, 0)    u=1, v=0
```

The 4 constraints come from stacking u and v: `b_UV = [vi, vj, vi+nVjoint, vj+nVjoint]`,
`bc_UV = [0, 1, 0, 0]`. The outer ring is left free and solved by LSCM.

---

### Case 1 — one of vi, vj is on the mesh boundary

Handled by `joint_lscm_case1`.

Let `v_bd` = whichever of `vi`, `vj` is on the boundary. The edge being collapsed has one
endpoint inside and one on the boundary. After the collapse, the boundary vertex `v_bd`
moves to the new midpoint position.

**Key difference from case 0 — no extra vertex is needed:**

```
nVjoint = nV          ← same vertex count, no appended vertex
Vjoint_post.row(v_bd) = V_post.row(vi)   ← move v_bd in-place to the new midpoint
Fjoint_post: vi → v_bd                    ← post-mesh routes through v_bd, not nV
```

Because `v_bd` is on the boundary, its UV position will be on the boundary of the UV domain
anyway, so moving it in-place (rather than appending a new index) avoids ambiguity about which
side of the boundary the merged vertex lives on.

**Pins (same 4 constraints as case 0):**

```
vi_pre → (0, 0),  vj_pre → (1, 0)
```

---

### Case 2 — both on the mesh boundary, non-flap

Handled by `joint_lscm_case2`.

This is the most complex case. Both endpoints lie on the mesh boundary, meaning `(vi, vj)` is
itself a **boundary edge**. After the collapse, `vj` is removed from the boundary and `vi` moves
to the new midpoint. The post-mesh boundary loop is therefore shorter (missing `vj`).

Because the two boundaries (pre and post) now differ in length, pinning only `vi` and `vj` is
not enough to produce a well-conditioned system. The code tries **three constraint strategies**
and picks the one with the lowest total quasi-conformal error.

#### Strategy A — snap to vi (`case2_constraint3_snap1(..., snapIdx=vi, ...)`)

The post-mesh's merged vertex is pinned to **vi's own pre-UV position**:

```
nVjoint = nV   (no new vertex; post-mesh routes vi → vi)
Vjoint_post.row(vi) = V_post.row(vi)
```

**Pins (5 constraints):**

```
vi_pre → (0, 0),  vj_pre → (1, 0)
vi_post → u=0, v=0   (v-component pinned to 0: stays on boundary line)
vk  → v=0            (vk is the boundary neighbour of vi that is not vj)
```

`vk` is found by walking the `bdLoop` to find which boundary vertex sits next to `snapIdx` on
the side away from `(vi,vj)`. Pinning `vk` to `v=0` keeps the post-collapse boundary straight
(all on the horizontal axis in UV space).

#### Strategy B — snap to vj (`case2_constraint3_snap1(..., snapIdx=vj, ...)`)

Same construction, but the merged vertex is mapped to **vj's pre-UV position** instead:

```
Vjoint_post.row(vj) = V_post.row(vi)
Fjoint_post: vi → vj
```

The `vk` used for the extra pin is now the boundary neighbour of `vj` on the away side.

#### Strategy C — no snap (`case2_constraint4`)

Adds an extra vertex at `nV` (like case 0) and constrains a larger portion of the boundary:

```
nVjoint = nV + 1
vi_post = nV
```

Instead of snapping to either pre-vertex, it pins most of the pre-boundary to `v=0` (on the
horizontal axis), **except** for the two boundary neighbours of `vi` in the post-mesh (which
are left free). This gives the LSCM more room to place the merged vertex wherever it minimises
distortion.

**Pins:**

```
vi_pre  → (0, 0),  vj_pre → (1, 0),  vi_post → v=0
all other pre-boundary vertices except vi's two post-neighbours → v=0
```

#### Choosing the best strategy

```cpp
objVal_snap_vi = ||error_pre_vi|| + ||error_post_vi||
objVal_snap_vj = ||error_pre_vj|| + ||error_post_vj||
objVal_no_snap = ||error_pre_n||  + ||error_post_n||
```

The strategy with the smallest total quasi-conformal error (σ/γ norm) is kept. If the winner
still produces flipped triangles, `check_valid_UV_lscm` will catch it and the whole collapse
is rejected.

---

### Summary of how boundary topology affects the joint mesh

| Case | Both boundary? | Edge on boundary? | Extra vertex? | Boundary pins |
|------|----------------|-------------------|---------------|---------------|
| 0 | No  | No  | Yes (`nV+1`) | 4 (just vi, vj) |
| 1 | No  | No  | No (`nV`) | 4 (just vi, vj) |
| 2 | Yes | Yes | Depends on strategy | 5+ (vi, vj, vk, partial boundary) |

---

## Call chain summary

```
SSP_collapse_edge()                     src/SSP_collapse_edge.cpp
  │
  ├─ get_collapse_onering_faces()       build V_pre/F_pre and V_post/F_post one-rings
  │
  ├─ remove_unreferenced_lessF()        reindex to local vertex numbering
  │
  ├─ get_post_faces()                   remove vj faces, produce FUV_post
  │
  ├─ joint_lscm()                       src/joint_lscm.cpp
  │    ├─ build boundary loop (bdLoop)
  │    ├─ case 0/1/2 dispatch
  │    │    ├─ build Vjoint / Fjoint (joint vertex set nV+1)
  │    │    ├─ pin vi→(0,0), vj→(1,0)  (or boundary-adjusted pins for cases 1/2)
  │    │    └─ flatten()               solve joint LSCM via mqwf_dense
  │    └─ check_valid_UV_lscm()        reject if NaN, flipped UV tri, or fold-over
  │
  └─ (if valid) commit the collapse
```
