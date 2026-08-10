# Fine Face Flip Diagnosis

Faces flip in the deformed fine mesh when the normal of the deformed triangle (corners = coarse correspondence positions) is opposite to the original fine face normal.

---

## 1. Coarse Face Boundary Straddling (most likely)

The deformed mesh replaces each fine vertex with its correspondence on the coarse mesh. A fine face flips when its 3 corners map to coarse positions that span across a **coarse face boundary** — the correspondence pulls vertices in opposite directions.

**Clue to check**: Are flipped fine faces concentrated near the boundaries between coarse faces? If yes, this is the primary culprit — not LSCM.

---

## 2. SSP Sheet Boundaries

The pipeline uses `faceSheetID` for SSP queries. Near sheet seam edges (logged in `seam_diag_*.txt`), the subgrid has cuts. Fine faces that straddle a seam edge can get correspondences that jump discontinuously across the cut.

**Clue to check**: Overlay `seam_edge_costs_*.txt` or seam diag data on the deformed mesh — do flipped faces cluster on or adjacent to seam edges?

---

## 3. LSCM / Joint Parameterization Distortion

If the parameterization used to compute barycentric coordinates has high distortion, the computed coarse positions can be accurate in UV but wrong in 3D when the coarse face is itself distorted.

**Clue to check**: Are flipped faces on highly non-planar coarse faces (high dihedral angle between a coarse face and its neighbors)? High curvature regions of the coarse mesh make flat-UV parameterizations inaccurate in 3D.

---

## 4. High Collapse Ratio Regions (QEM)

Unlikely to directly cause flips, but when many fine vertices collapse toward a single coarse vertex, all surrounding fine faces point their correspondences toward one spot — the angular spread can make some triangles invert.

**Clue to check**: Are flipped faces in regions where many fine faces map to a single coarse face?

---

## Quickest First Check

In `fine_deform_viz.py`, after identifying the flipped face set, also visualize which **coarse faces** those flipped fine faces correspond to (via `_tracker_vtx_coarse_fids`):

- Cluster on a handful of coarse faces or near coarse edges → **boundary straddling**
- Spread along seam lines → **SSP sheet cuts**
- Scattered at high-curvature coarse regions → **LSCM distortion**
