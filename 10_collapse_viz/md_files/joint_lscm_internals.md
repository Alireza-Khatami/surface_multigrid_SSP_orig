# Joint LSCM — Internals, Matrix Construction, and the Case 1 Problem

## 1. What Joint LSCM Solves

Joint LSCM parameterises two meshes simultaneously: the **pre-collapse** one-ring
and the **post-collapse** one-ring. The key constraint is that vi's UV position
before and after the collapse must be consistent — the map must be conformal on
both.

The system is built by assembling a single quadratic energy over a **joint vertex
set** and solving it once with fixed boundary pins.

---

## 2. The Joint Vertex Set (Vjoint)

```
nV       = V_pre.rows()           (one-ring vertices)
nVjoint  = nV + 1                 (adds one extra slot for vi's post-collapse 3D position)

Vjoint rows 0..nV-1  = V_pre      (pre-collapse 3D positions)
Vjoint row  nV       = V_post.row(vi)   (vi's new 3D position after collapse)
```

The post-collapse face matrix `Fjoint_post` is identical to `FUV_post` except
every occurrence of `vi` is remapped to index `nV` — so the pre and post meshes
share all vertices except vi's post position gets its own slot.

**Exception in Case 1**: `nVjoint = nV` (no extra slot). The boundary vertex
`v_bd` plays the role of vi's post position directly (explained in §5).

---

## 3. The Quadratic Energy (LSCM)

LSCM minimises the **conformal energy**: angle distortion integrated over all
faces. For a triangulated mesh the discrete form uses two matrices:

### Cotangent Laplacian  `L`  (nV × nV)

```
L[i,j] = -(cot α_ij + cot β_ij) / 2   for edge (i,j)
L[i,i] = -sum of off-diagonals
```

where α_ij and β_ij are the angles opposite to edge (i,j) in the two incident
triangles. Built from 3D vertex positions via `cotmatrix_dense(V, F, L)`.

**Dimension**: nVjoint × nVjoint

### Block-diagonal UV Laplacian  `LUV`  (2·nVjoint × 2·nVjoint)

To handle 2D UV coordinates the Laplacian is duplicated:

```
LUV = [ L   0 ]
      [ 0   L ]
```

One block for the U coordinate, one for V.  
Built separately for pre (`LUV_pre`) and post (`LUV_post`).

### Vector Area Matrix  `A`  (2·nVjoint × 2·nVjoint)

`A` encodes the **signed area** of UV triangles. It couples U and V coordinates
across triangle vertices so that the conformal constraint (Cauchy–Riemann
equations) is enforced per face.

Built via `vector_area_matrix_size(F, nVjoint, A)`.

### Combined Quadratic  `Q`

```
Q = -LUV_pre + 2·A_pre - LUV_post + 2·A_post
```

**Dimension**: 2·nVjoint × 2·nVjoint

The energy being minimised is:

```
E(u) = u^T Q u    subject to  u[b] = bc
```

where `u` is the stacked UV vector `[U_0, U_1, …, U_{nVjoint-1}, V_0, V_1, …]`
(first block = U, second block = V).

The term `−LUV` comes from the Dirichlet conformal energy (angle preservation).
The term `+2A` comes from the area constraint that prevents degenerate solutions.
Combining pre and post in one system forces UV_pre and UV_post to be jointly
conformal — the pre map and post map are solved together, not independently.

---

## 4. Pins and Solve

Pins fix two UV coordinates as hard constraints. The system is solved via
`mqwf_dense` (dense minimum quadratic with fixed boundary):

```
min  u^T Q u   s.t.  u[b_UV] = bc_UV
```

After solving, `UVjoint_flat` is a vector of length `2·nVjoint`:

```
UVjoint_flat = [ U_0, U_1, …, U_{nVjoint-1},   ← first block
                 V_0, V_1, …, V_{nVjoint-1} ]   ← second block
```

### Column Swap on Reshape

The reshape loop **reverses the column order**:

```cpp
for (unsigned i = 0; i < UVjoint.cols(); ++i)
    UVjoint.col(UVjoint.cols() - i - 1) = UVjoint_flat.block(nVjoint * i, 0, nVjoint, 1);
```

For a 2-column matrix (`cols() = 2`):
- `i=0`: `col(1)` ← first block  → **col(1) = U**
- `i=1`: `col(0)` ← second block → **col(0) = V**

So the stored matrix has **col(0) = V, col(1) = U** — columns are swapped
relative to the natural (U, V) order. All code that reads UV_pre must account
for this: `act_U = UV_pre(v, 1)`, `act_V = UV_pre(v, 0)`.

---

## 5. Case Selection

`onBd(0)` = 1 if vi is on the mesh boundary (detected via presence of `infVIdx`
in its one-ring neighbor list). `onBd(1)` = 1 for vj.

| `onBd.sum()` | Case | Description |
|---|---|---|
| 0 | **Case 0** | Both vi, vj interior |
| 1 | **Case 1** | Exactly one endpoint on boundary |
| 2 | **Case 2 (DC)** | Both endpoints on boundary edge |

---

## 6. Case 0 Pins

```
b_UV  = [vi,  vj,  vi+nVjoint,  vj+nVjoint]
bc_UV = [ 0,   1,           0,           0]
```

Decoded with the column swap:
- `vi`  → flat index for U-block → stored in col(1) → **vi pinned to U=0, V=0**
- `vj`  → flat index for U-block → stored in col(1) → **vj pinned to U=1, V=0**
- `vi+nVjoint`, `vj+nVjoint` → V-block → **both at V=0**

---

## 7. Case 1 — One Boundary Vertex

### Joint Mesh Differences from Case 0

- `nVjoint = nV` (no extra slot, because the boundary vertex does not get a
  new 3D position — it stays on the boundary)
- `v_bd` = whichever of vi/vj is on the boundary
- `Vjoint_post.row(v_bd) = V_post.row(vi)` — the boundary vertex takes vi's
  post-collapse 3D position in the post mesh
- Post faces: every occurrence of `vi` remapped to `v_bd` (not to `nV`)

### Pins (identical formula to Case 0)

```
b_UV  = [vi,  vj,  vi+nVjoint,  vj+nVjoint]
bc_UV = [ 0,   1,           0,           0]
```

With the column swap:
- **vi pinned to U=0, V=0**
- **vj pinned to U=1, V=0**

These pins are assigned purely by **index order** — vi always gets (0,0), vj
always gets (1,0) — regardless of which one is the mesh boundary vertex.

---

## 8. The Case 1 Problem

### What the topology actually is

When one endpoint is on the mesh boundary, its one-ring is an **open fan** —
a disc with a gap (missing faces on the boundary side). The correct conformal
map for an open fan would pin the boundary vertex **on the UV boundary arc**,
where it topologically belongs.

### What Case 1 actually does

Case 1 pins both collapse endpoints at interior positions `(0,0)` and `(1,0)`.
The LSCM then minimises conformal energy on the open-fan faces, with the outer
ring of vertices forming the UV boundary arc. The boundary vertex ends up at
one of the inner pins, **not on the arc**.

This is the fundamental topological mismatch: Case 1 applies a closed-disc
interior pinning scheme to an open-fan topology.

### Why orientation flips between collapses

The pin assignment follows index order:
- If the boundary vertex is **vi** → it gets `(0,0)`, the interior vertex gets
  `(1,0)`. The open fan gap anchors at `(0,0)`.
- If the boundary vertex is **vj** → it gets `(1,0)`, the interior vertex gets
  `(0,0)`. The gap anchors at `(1,0)`.

The gap flips sides between the two sub-cases, producing a mirrored UV layout.
This is why consecutive Case 1 collapses can look like the boundary/interior
assignment swapped — the orientation is determined by index order, not topology.

### When does it cause distortion or fold-overs?

The severity depends on where the boundary vertex's inner pin sits relative to
the arc in U:

| `[ARC_POSITION]` | Meaning | Risk |
|---|---|---|
| `ALL_HIGHER` | All arc verts have U > boundary pin | Pin at low end of arc — geometrically consistent |
| `ALL_LOWER`  | All arc verts have U < boundary pin | Pin at high end of arc — geometrically consistent |
| `MIXED`      | Arc verts on both sides of pin in U | Pin in interior of arc — **fold-over / high distortion** |

`MIXED` is the dangerous case. The open fan wraps around the pin from both
sides, forcing the LSCM to squeeze faces around an inner point that has no
support on one side. The result is flipped triangles or high quasi-conformal
error.

### Why Double Cover (Case 2) fixes this

The double cover reflects the open one-ring across the boundary arc to form a
closed, symmetric disc. After the reflection:
- The boundary vertices sit **on the reflection axis** (the seam, y=0) — their
  correct topological position
- vi and vj become free variables, placed conformally by the LSCM
- The pinning fixes the arc endpoints `B_glued[0]` and `B_glued[1]` at
  `(-1, 0)` and `(+1, 0)` on the seam line
- The system is topologically consistent: no gap, no mismatch

This is why DC always produces a geometrically valid UV when the B-arc has
enough vertices, while Case 1 can fail even when the 3D geometry looks fine.

---

## See Also

- `src/joint_lscm.cpp` line 1127 — `flatten()` energy assembly and solve
- `src/joint_lscm.cpp` line 1297 — `joint_lscm_case1()` implementation
- `src/joint_lscm.cpp` line 70  — `joint_lscm_double_cover()` implementation
- `10_collapse_viz/md_files/case1_uv_boundary_interior_swap.md` — orientation
  swap explained in terms of sub-cases A and B
