# SSP Pipeline — Step by Step

Entry point: `08_subdiv_remesh/main.cpp`

```
Input:  mesh_path, target_faces (tarF), num_subdivisions
Output: output_s0.obj … output_s{num_subdivs}.obj
```

---

## Phase 1 — Load and validate

### 1. Load mesh
```cpp
igl::read_triangle_mesh(mesh_path, VO, FO)
```
Produces the original fine mesh `(VO, FO)`: `|V|×3` vertex positions, `|F|×3` face indices.

### 2. Manifold check
```cpp
igl::is_vertex_manifold(FO, BI)
igl::is_edge_manifold(FO)
```
`SSP_decimate.cpp:20`

Two conditions must hold:

- **Vertex manifold**: the one-ring of faces around every vertex is a single connected disk (or fan for boundary vertices). Fails at "bowtie" vertices where two separate fans share only one vertex.
- **Edge manifold**: every edge is shared by at most 2 faces. Fails on medial-axis-style meshes where 3+ surface sheets meet at an edge.

If either fails: print `"input mesh is not manifold"` and abort. See `conformal_flattening.md` for why both conditions are required.

---

## Phase 2 — Build half-edge structure

### 3. Connect boundary to infinity
```cpp
igl::connect_boundary_to_infinity(V, F, VO_aug, FO_aug)
```
`SSP_midpoint.cpp:31`

Adds a virtual vertex at index `nV` (position `[0,0,∞]`). Every boundary edge gets a new triangle connecting it to this vertex. This turns an open mesh into a topologically closed one so the same collapse code works for both open and closed inputs.

Side effect: non-manifold boundary vertices create non-manifold edges in the augmented mesh — caught by the second edge-manifold check below.

### 4. Build edge-flip tables
```cpp
igl::edge_flaps(FO_aug, E, EMAP, EF, EI)
```
Builds three tables over all `|E|` unique edges:
- `E` `|E|×2` — the two vertex indices of each edge
- `EMAP` `|2F|×1` — maps each half-edge `(face, corner)` to its edge index in `E`
- `EF` `|E|×2` — which two faces are on each side of edge `e`
- `EI` `|E|×2` — which corner index within each of those faces

These tables are the backbone for all subsequent traversals.

### 5. Re-validate edge manifold on augmented mesh
```cpp
is_edge_manifold(FO_aug, E.rows(), EMAP, BF, BE)
```
`SSP_midpoint.cpp:41`

Repeated after augmentation because connecting to infinity can expose non-manifold boundary vertices as non-manifold edges (see step 2 above).

### 6. Initialise cost priority queue
```cpp
cost_and_placement(e, V, F, E, EMAP, EF, EI, cost, p)   // for each edge e
Q.emplace(cost, e, 0)
```
`SSP_midpoint.cpp:170–181`

The cost function used in `dec_type=1` (midpoint) is `shortest_edge_and_midpoint`: cost = edge length, placement `p` = edge midpoint. All `|E|` edges are evaluated in parallel and pushed into a min-heap `Q`.

---

## Phase 3 — Successive edge collapses

Runs until `|F| == tarF` or the heap is exhausted.

### Decimation strategy (`dec_type`)

The strategy is selected in `main.cpp:144` and dispatched in `SSP_decimate.cpp:25`:

| `dec_type` | Function | Cost | Placement |
|------------|----------|------|-----------|
| `0` | `SSP_qslim` | Sum of squared distances to adjacent face planes (QEM) | Optimal point minimising that quadric |
| `1` *(default)* | `SSP_midpoint` | Edge length | Midpoint of the edge |
| `2` | `SSP_vertexRemoval` | Edge length | Midpoint of the edge |

**Midpoint (`dec_type=1`)** collapses the shortest edge first and places the merged vertex at the
midpoint. The result is a spatially uniform decimation — it tends to produce evenly-sized
triangles but ignores surface curvature and sharp features.

**QEM (`dec_type=0`, `SSP_qslim`)** assigns each vertex a *quadric* `Q_v`: the sum of outer
products of all adjacent face planes `(n, d)` as `Q_v = Σ [n; d][n; d]^T`. The cost of
collapsing edge `(vi, vj)` is evaluated at the optimal point `p*` that minimises
`p^T (Q_vi + Q_vj) p`. This favours collapsing edges in flat regions first and naturally
preserves sharp creases and high-curvature features because those have high quadric cost.

To switch from midpoint to QEM, change one line in `main.cpp`:

```cpp
int dec_type = 0;  // was 1 (midpoint), 0 = QEM (qslim)
```

The joint LSCM validity check (step 11 below) is identical in both paths — QEM only changes
which edge is chosen and where the merged vertex is placed. If the chosen placement causes a
UV fold-over, the collapse is still rejected and the next cheapest edge is tried.

---

### 7. Pop cheapest edge
```cpp
std::get<0>(Q.top())   // cost
e = std::get<1>(Q.top())
```
`SSP_midpoint.cpp:188`

Takes the edge with minimum cost from the heap.

### 8. Get 1-ring neighbourhoods
```cpp
igl::circulation(e, true,  F, EMAP, EF, EI, Nsv, Nsf)
igl::circulation(e, false, F, EMAP, EF, EI, Ndv, Ndf)
```
Inside `SSP_collapse_edge.cpp` (called from the overloaded `SSP_collapse_edge`).

`Nsv` = ordered ring of vertex neighbours of `vi` (source), `Ndv` = same for `vj` (destination). `Nsf`/`Ndf` = corresponding face rings. The `infVIdx` sentinel (`= nV_aug - 1`) appears in these lists wherever the ring touches the mesh boundary.

### 9. Link condition (topological validity)
```cpp
igl::edge_collapse_is_valid(Nsv, Ndv)
```
`SSP_collapse_edge.cpp:57`

Checks the **link condition**: the intersection of `Nsv` and `Ndv` must equal the union of shared faces' third vertices. Ensures the collapse does not create a non-manifold result (e.g. collapsing the base edge of a tetrahedron would merge two boundary triangles into one face).

If invalid: mark edge cost as `∞`, skip.

### 10. Extract local geometry
```cpp
get_collapse_onering_faces(V, F, vi, vj, Nsf, Ndf,
    FIdx_onering_pre, FIdx_onering_post,
    F_onering_pre,   F_onering_post)
remove_unreferenced_lessF(V, F_onering_pre, V_pre, FUV_pre, IM, subsetVIdx)
```
`SSP_collapse_edge.cpp:88–103`

Extracts two local meshes in a compact local index space:

- **pre**: all faces touching `vi` or `vj`, with their current geometry — `(V_pre, FUV_pre)`
- **post**: same faces after collapsing `vj` into `vi` at position `p`, with vj's two adjacent faces removed — `(V_post, FUV_post)`

`subsetVIdx` maps local indices back to global ones.

### 11. Joint LSCM (conformal flattening)
```cpp
joint_lscm(V_pre, FUV_pre, V_post, FUV_post,
           b(0), b(1), Nsv_local, Ndv_local,
           UV_pre, UV_post)
```
`SSP_collapse_edge.cpp:183`

Flattens both the pre and post one-rings into 2D using LSCM, with a shared outer boundary curve. See `conformal_flattening.md` for the full derivation.

Returns `false` (collapse rejected) if:
- Edge is a topological flap
- 3D triangle quality of post-mesh is too low (< 0.3 normalised quality)
- The LSCM solve produces NaN
- Any UV triangle in pre or post has negative or zero signed area (fold-over)

### 12. Record collapse data
```cpp
data.b          = b              // local indices of vi, vj
data.subsetVIdx = subsetVIdx     // local → global vertex map
data.UV_pre     = UV_pre         // pre-collapse UV layout
data.UV_post    = UV_post        // post-collapse UV layout
data.FUV_pre    = FUV_pre        // pre-collapse face list (local)
data.FUV_post   = FUV_post       // post-collapse face list (local)
data.FIdx_pre   = FIdx_onering_pre    // global face indices, pre
data.FIdx_post  = FIdx_onering_post   // global face indices, post

decInfo.push_back(data)
decIM[fIdx].push_back(decInfo.size()-1)   // face → list of collapses that touched it
```
`SSP_collapse_edge.cpp:247–261`

`decInfo` is the complete history of all accepted collapses. `decIM` is an index that maps each face to the list of collapse records that affected it — needed for the reverse traversal in Phase 5.

### 13. Perform the collapse
```cpp
V.row(s) = p            // move source vertex to midpoint
// delete vj's faces from F (set to IGL_COLLAPSE_EDGE_NULL)
// re-evaluate costs of all edges in the new 1-ring
```
`SSP_collapse_edge.cpp:275–` and igl collapse internals.

Updates `V`, `F`, `E`, `EMAP`, `EF`, `EI` in place. Affected edges get new costs pushed onto `Q` with an incremented "timestamp" — stale entries in `Q` are discarded lazily when popped.

### 14. Stopping condition
```cpp
max_faces_stopping_condition(m, orig_m, max_m)
```
Stops when the number of live faces (non-null rows of `F`) reaches `tarF`. The output coarse mesh is `(V_coarse, F_coarse)`.

---

## Phase 4 — Loop subdivision on coarse mesh

### 15. Upsample and extract barycentric coordinates
```cpp
loop_upsample_barycentric(V, F, num_subdivs, BC, BF, FIdx, SF)
```
`08_subdiv_remesh/main.cpp:156`

Internally applies Loop subdivision `num_subdivs` times via the sparse operator `S` (assembled by `igl::upsample`). For `k` rounds, `S` is composed: `S_total = S_k * … * S_1`, so `V_subdivided = S_total * V_coarse`.

For every row `v` of `S_total` (every subdivided vertex), the non-zero columns identify the 1–3 vertices of the coarse face that `v` is expressed in, and the non-zero values are the barycentric weights. These are stored as:

- `BC` `nV_sub × 3` — barycentric weights
- `BF` `nV_sub × 3` — the three coarse vertex indices (global) for each sub-vertex
- `FIdx` `nV_sub` — which coarse face each sub-vertex belongs to
- `SF` — the subdivided face list (connectivity only)

---

## Phase 5 — Coarse-to-fine query (reverse collapse traversal)

### 16. Propagate barycentric coordinates back to original mesh
```cpp
query_coarse_to_fine(decInfo, IM, decIM, IMF, BC, BF, FIdx)
```
`08_subdiv_remesh/main.cpp:157`, implemented in `src/query_coarse_to_fine.cpp`

The coarse mesh was decimated from `VO`. Each subdivided vertex knows its barycentric coordinates on the *coarse* mesh, but we need them on the *original fine* mesh `VO`.

For each subdivided vertex `q` (parallelised):

```
while there are still collapses to replay for FIdx[q]:
    find the latest collapse dIdx that touched face FIdx[q]
    
    look up UV_post for that collapse:
        queryUV = BC[q,0]*UV_post[v0] + BC[q,1]*UV_post[v1] + BC[q,2]*UV_post[v2]
    
    find which pre-collapse UV triangle contains queryUV:
        compute_barycentric(queryUV, UV_pre, FUV_pre, B)
    
    update BC[q], BF[q], FIdx[q] to the pre-collapse face
```

At the end of this traversal, `(BC, BF)` express every subdivided vertex as a barycentric combination of three vertices in the *original* mesh `VO`. The UV maps stored in `decInfo` act as the "transport" between successive levels of resolution.

---

## Phase 6 — Reconstruct and output

### 17. Compute subdivided positions on original mesh
```cpp
SV[ii] = BC[ii,0]*VO[BF[ii,0]] + BC[ii,1]*VO[BF[ii,1]] + BC[ii,2]*VO[BF[ii,2]]
```
`08_subdiv_remesh/main.cpp:162–165`

Every subdivided vertex is reconstructed from the original fine mesh using the barycentric coordinates computed in Phase 5. The result is a high-resolution mesh whose vertex positions come from `VO` but whose connectivity comes from Loop subdivision of the coarse mesh.

### 18. Write hierarchical output
```cpp
for iter in 0 .. num_subdivs:
    igl::upsample(V, F, NV, NF, iter)     // coarse mesh upsampled iter times
    NV = SV[0 : |NV|]                     // replace positions with fine-mesh projections
    igl::writeOBJ("output_s{iter}.obj", NV, NF)
```
`08_subdiv_remesh/main.cpp:168–177`

Outputs `num_subdivs + 1` meshes:
- `output_s0.obj` — the coarse mesh itself (positions projected back onto `VO`)
- `output_s1.obj` — one Loop subdivision level
- …
- `output_s{N}.obj` — the fully subdivided mesh

Each level uses Loop connectivity but vertex positions derived from the original fine mesh, giving a clean subdivision hierarchy that approximates `VO`.

---

## Data flow summary

```
VO, FO  (original fine mesh)
  │
  ├─[Phase 1-2] validate + build half-edge structure (E, EMAP, EF, EI)
  │
  ├─[Phase 3]  successive collapses ──────────────────────> decInfo[]
  │             vi,vj → p (midpoint)                         UV_pre, UV_post per collapse
  │             reject if non-manifold or UV fold-over        subsetVIdx, FUV_pre, FUV_post
  │
  └──> V_coarse, F_coarse   (tarF faces)
         │
         ├─[Phase 4] loop_upsample_barycentric
         │           BC, BF, FIdx  (barycentric coords on coarse mesh)
         │
         ├─[Phase 5] query_coarse_to_fine
         │           replay decInfo in reverse, transport BC through UV maps
         │           BC, BF  now on original VO
         │
         └─[Phase 6] SV = barycentric interpolation on VO
                     write output_s{0..N}.obj
```
