# C2F Bundle Format — Data Structures, Reading, and Correspondence Query

## 1. File Format (binary, little-endian, extension `.c2f`)

All multi-byte fields are little-endian. Two format versions exist:

| Magic       | Version | SSP query data? |
|-------------|---------|-----------------|
| `0xC2F50001`| v1      | No              |
| `0xC2F50002`| v2      | Yes             |

### 1.1 Common header (v1 and v2)

```
uint32  magic          0xC2F50001 or 0xC2F50002
uint32  NC             compact coarse vertex count
uint32  FC             compact coarse face count
uint32  NF             fine (original) vertex count
uint32  FF             fine (original) face count
```

### 1.2 Coarse mesh

```
NC × 3  double   coarseV     3D positions of compact coarse mesh vertices
FC × 3  uint32   coarseF     triangles (compact vertex indices, 0-based)
```

### 1.3 Fine (original) mesh

```
NF × 3  double   fineV       3D positions of original fine mesh vertices
FF × 3  uint32   fineF       triangles (fine vertex indices, 0-based)
```

### 1.4 Correspondence table

One record per compact coarse vertex (NC records total):

```
per-vertex:
  double  bc0, bc1, bc2    barycentric weights (sum to 1)
  uint32  fv0, fv1, fv2    fine vertex indices of the carrier triangle
```

The 3D correspondence position is:

```
fine_pos = bc0 * fineV[fv0] + bc1 * fineV[fv1] + bc2 * fineV[fv2]
corrVec  = fine_pos - coarseV[i]            # displacement vector
```

### 1.5 SSP query data (v2 only)

All signed integers are `int32`. The block starts immediately after the
correspondence table.

```
uint32  nV_total    total vertex count of the SSP working mesh
                    (fine mesh + 1 infinity vertex added by connect_boundary_to_infinity)
uint32  nF_decIM    total face count tracked by decIM
                    (= numOrigFaces, i.e. fine faces only, before the infinity faces)
uint32  nFO         number of original (non-infinity) fine faces
                    (= numOrigFaces = nF_decIM in practice)

int32[NC]           vtxMap     compact→global vertex index (into SSP V matrix)
int32[FC]           faceMap    compact→global face  index (into SSP F matrix)
int32[nFO]          faceSheetID  sheet ID for each original fine face

--- decIM ---
For each of the nF_decIM original faces f:
  uint32  cnt                number of collapses that touched face f
  int32[cnt]  collapse_ids   sorted ascending, indices into decInfo

--- decInfo ---
uint32  nDec        total number of SSP edge collapses recorded

For each collapse d in [0, nDec):
  uint32  nSheets    number of topological sheets active for this collapse

  For each sheet s in [0, nSheets):
    int32   sid          global_sheet_id
    uint32  svSz         number of vertices in the local one-ring mesh
    int32[svSz]  subsetVIdx   global vertex indices of the local mesh (sorted ascending)

    uint32  uvRows       number of UV vertices (= svSz)
    double[uvRows × 2]  UV_pre   UV coords before collapse (row-major, u then v)
    double[uvRows × 2]  UV_post  UV coords after  collapse

    uint32  fuvRows      number of faces in the local mesh (pre-collapse)
    int32[fuvRows × 3]  FUV_pre   local vertex indices of each UV face
    int32[fuvRows]      FIdx_pre  global original face index for each UV face
```

---

## 2. Data Structures

### Bundle (top-level container)

| Field         | Shape      | dtype   | Description                                     |
|---------------|------------|---------|-------------------------------------------------|
| `coarseV`     | `(NC, 3)`  | float64 | Compact coarse 3D vertex positions              |
| `coarseF`     | `(FC, 3)`  | int32   | Compact coarse triangles                        |
| `fineV`       | `(NF, 3)`  | float64 | Original fine 3D vertex positions               |
| `fineF`       | `(FF, 3)`  | int32   | Original fine triangles                         |
| `corrVec`     | `(NC, 3)`  | float64 | `fine_pos - coarseV[i]` displacement per vertex |
| `vtxMap`      | `(NC,)`    | int32   | compact→global vertex index                     |
| `faceMap`     | `(FC,)`    | int32   | compact→global face index                       |
| `faceSheetID` | `(nFO,)`   | int32   | sheet ID per original face                      |
| `decIM`       | list of lists | int | per-face sorted collapse indices                |
| `decInfo`     | list       | —       | one `CollapseData` per SSP collapse             |

### CollapseData (one per edge collapse)

```python
CollapseData.sheets : List[SheetData]   # one per active topological sheet
```

For manifold meshes, `len(sheets) == 1`. For non-manifold seam edges,
`len(sheets) >= 2`.

### SheetData (one per sheet per collapse)

| Field            | Shape         | dtype   | Description                                      |
|------------------|---------------|---------|--------------------------------------------------|
| `global_sheet_id`| scalar        | int     | Sheet index from `partition_into_sheets`         |
| `subsetVIdx`     | `(svSz,)`     | int32   | Global vertex indices of the local mesh (sorted) |
| `UV_pre`         | `(uvRows, 2)` | float64 | UV coordinates **before** this collapse          |
| `UV_post`        | `(uvRows, 2)` | float64 | UV coordinates **after** this collapse           |
| `FUV_pre`        | `(fuvRows, 3)`| int32   | UV face topology (local vertex indices)          |
| `FIdx_pre`       | `(fuvRows,)`  | int32   | Global original face index per UV face           |

The local vertex index in `FUV_pre` is an index into `UV_pre`/`UV_post` and
also into `subsetVIdx`. So the global 3D vertex for `FUV_pre[f, c]` is:

```python
global_vtx = subsetVIdx[FUV_pre[f, c]]
```

---

## 3. Code Assumptions

1. **Compact mesh vs. global mesh**: the "compact" coarse mesh stored in the
   bundle has vertex count `NC` and face count `FC`. These are re-indexed
   starting from 0. The SSP algorithm internally works with a larger global
   mesh that includes an extra *infinity vertex* appended by
   `igl::connect_boundary_to_infinity`. The `vtxMap` and `faceMap` arrays
   translate compact indices back to global SSP indices.

2. **Infinity vertex**: the SSP working mesh always has `nV_total` vertices
   where `V[nV_total - 1]` is the infinity vertex. It is used to close open
   boundaries into a watertight surface so that `igl::edge_collapse_is_valid`
   works. It is excluded from the compact coarse output.

3. **Dead faces (null faces)**: when a face is killed during a collapse its
   three vertex entries in `F` are all set to `IGL_COLLAPSE_EDGE_NULL = 0`.
   Such faces are never stored in the bundle. The `decIM` list for such a face
   will be empty.

4. **decIM is indexed by original face**: `decIM[f]` is valid only for
   `f < nFO` (original fine faces). Infinity faces (`f >= nFO`) are not
   tracked. The list is sorted ascending, so the most recent collapse for face
   `f` is `decIM[f][-1]`.

5. **IM / IMF are identity**: in the visualizer, `IM = arange(nV_total)` and
   `IMF = arange(nF_total)`. The decimation never renumbers vertices or faces
   — it only nullifies them. So global vertex and face indices are stable
   throughout the entire SSP sequence.

6. **Sheet routing in the query**: when a query face's sheet ID doesn't match
   any sheet stored for a given collapse, the code falls back to
   `sheets[0]`. This is a best-effort fallback; it can introduce small UV
   errors at non-manifold seam edges.

7. **UV domain is per-collapse-per-sheet**: `UV_pre` and `UV_post` are valid
   only within the local one-ring around the collapsed edge. They are produced
   by `joint_lscm` and are NOT a global parameterization.

8. **Barycentric clamping**: the UV query point may land slightly outside the
   pre-collapse UV triangle due to numerical noise. The code picks the face
   with the least negative barycentric coordinate and clamps any negative
   component to 0 before renormalizing. This is a projection-to-nearest
   heuristic.

9. **Walk is backwards in time**: the query starts from a coarse face
   (`FIdx = faceMap[coarse_face]`, i.e. global) and walks backwards through
   `decIM[FIdx]` in descending-index order, stopping at the collapse with the
   largest index smaller than the current `dIdx`. Each step moves the point
   to the pre-collapse face in the local UV domain.

10. **Sheet IDs never change**: `faceSheetID` is computed once from the
    original fine mesh topology and never updated. A face's sheet ID is
    constant for its entire lifetime; dead faces simply stop being queried.

---

## 4. Correspondence Query Algorithm

Given a query point defined by barycentric coordinates `BC` on a coarse face
`coarseF[cf]`, the goal is to find the corresponding point on the fine mesh.

**Inputs:**
- `BC`: `(N, 3)` barycentric weights inside the coarse face
- `BF`: `(N, 3)` global vertex indices (initially from `vtxMap` for the three
  coarse face corners, same for every sample in the same face)
- `FIdx`: `(N,)` global original face index (initially `faceMap[cf]`, same
  for every sample in the same face)

**For each query `q`:**

```
dIdx ← len(decInfo)   # start at "present", walk backwards

loop:
    face ← FIdx[q]
    d_list ← decIM[face]               # collapses that touched this face
    find the largest d in d_list with d < dIdx  →  new dIdx
    if not found: break

    sid  ← faceSheetID[face]
    sd   ← collapse d's SheetData for sheet sid
    r    ← row in sd.FIdx_pre where sd.FIdx_pre[r] == face

    # Project current BC through UV_post to get the query UV point
    uv_query ← BC[q,0]*UV_post[FUV_pre[r,0]]
              + BC[q,1]*UV_post[FUV_pre[r,1]]
              + BC[q,2]*UV_post[FUV_pre[r,2]]

    # Find the best pre-collapse face in UV_pre
    B ← compute_barycentric(uv_query, UV_pre, FUV_pre)   # (fuvRows, 3)
    best ← argmin of -B.min(axis=1)                       # most inside face

    # Clamp + renormalize
    b ← max(0, B[best]);  b /= b.sum()

    BC[q]   ← b
    BF[q]   ← [subsetVIdx[FUV_pre[best,0]],
               subsetVIdx[FUV_pre[best,1]],
               subsetVIdx[FUV_pre[best,2]]]
    FIdx[q] ← FIdx_pre[best]

# Final fine-mesh 3D position
finePos ← BC[q,0]*fineV[BF[q,0]] + BC[q,1]*fineV[BF[q,1]] + BC[q,2]*fineV[BF[q,2]]
```

---

## 5. 2D Barycentric Coordinate Formula

For query point `p ∈ ℝ²` and triangle `(a, b, c) ∈ ℝ²`:

```
v0 = b - a
v1 = c - a
v2 = p - a

d00 = dot(v0, v0)
d01 = dot(v0, v1)
d11 = dot(v1, v1)
d20 = dot(v2, v0)
d21 = dot(v2, v1)

denom = d00*d11 - d01*d01
vv = (d11*d20 - d01*d21) / denom   → weight for b
ww = (d00*d21 - d01*d20) / denom   → weight for c
uu = 1 - vv - ww                   → weight for a

BC = (uu, vv, ww)
```

Applied batch-wise to all faces simultaneously.
