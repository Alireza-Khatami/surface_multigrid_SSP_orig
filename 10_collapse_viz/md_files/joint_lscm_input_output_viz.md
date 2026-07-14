What joint_lscm receives / produces (per sheet, in SSP_collapse_edge.cpp:350-354)

  Inputs:  V_pre_si, FUV_pre_si   (pre-collapse 3D geometry + faces)
           V_post_si, FUV_post_si (post-collapse: vi moved to p, flap faces removed)
           b_si(0/1), Nsv_local, Ndv_local

  Outputs: UV_pre_si, UV_post_si  (per-vertex UV, same row count as V_pre_si)

  These are stored verbatim into SheetData at lines 362–366.

  ---
  What refresh_snap copies into gSnap (visualizer.cpp:100–116)

  It reads decInfo.back() → d.sheets[0] only:

  ┌────────────────┬───────────────────────────┬────────────────────────────────┐
  │  gSnap field   │    Source in SheetData    │ Same as joint_lscm arg/output? │
  ├────────────────┼───────────────────────────┼────────────────────────────────┤
  │ gSnap.V_pre    │ d.V_pre → V_pre_si        │ yes — exact pre-collapse 3D    │
  ├────────────────┼───────────────────────────┼────────────────────────────────┤
  │ gSnap.FUV_pre  │ sd.FUV_pre → FUV_pre_si   │ yes                            │
  ├────────────────┼───────────────────────────┼────────────────────────────────┤
  │ gSnap.FUV_post │ sd.FUV_post → FUV_post_si │ yes                            │
  ├────────────────┼───────────────────────────┼────────────────────────────────┤
  │ gSnap.UV_pre   │ sd.UV_pre → UV_pre_si     │ yes — direct joint_lscm output │
  ├────────────────┼───────────────────────────┼────────────────────────────────┤
  │ gSnap.UV_post  │ sd.UV_post → UV_post_si   │ yes — direct joint_lscm output │
  ├────────────────┼───────────────────────────┼────────────────────────────────┤
  │ gSnap.b        │ sd.b → b_si               │ yes — same local indices       │
  └────────────────┴───────────────────────────┴────────────────────────────────┘

  ---
  What polyscope actually renders (visualizer.cpp:228–282)

  ┌───────────────┬───────────────────────────────┬────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────┐
  │   Structure   │           Vertices            │             Faces              │                                         Match?                                         │
  ├───────────────┼───────────────────────────────┼────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ one_ring_pre  │ gSnap.V_pre (= V_pre_si)      │ gSnap.FUV_pre (= FUV_pre_si)   │ exact joint_lscm input                                                                 │
  ├───────────────┼───────────────────────────────┼────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ one_ring_post │ gSnap.V_pre (= V_pre_si)      │ gSnap.FUV_post (= FUV_post_si) │ faces match; 3D positions use pre-collapse V_pre_si, not V_post_si (vi not moved to p) │
  ├───────────────┼───────────────────────────────┼────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ uv_pre        │ gSnap.UV_pre projected to 3D  │ gSnap.FUV_pre                  │ exact joint_lscm UV output                                                             │
  ├───────────────┼───────────────────────────────┼────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ uv_post       │ gSnap.UV_post projected to 3D │ gSnap.FUV_post                 │ exact joint_lscm UV output                                                             │
  └───────────────┴───────────────────────────────┴────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  The one subtle discrepancy

  one_ring_post shows the correct face topology (flap faces removed) but the 3D vertex positions are V_pre_si — the pre-collapse positions where vi is still at its old location. joint_lscm actually received
  V_post_si where V_post_si.row(b_si(0)) = p (vi already moved to the collapse point). So the post one-ring in 3D is topologically correct but geometrically shows pre-collapse positions for vi/vj.

  This is intentional for visualization — it lets you see which faces disappear without the ring jumping to the collapsed position — but it means the 3D one-ring post is not geometrically what joint_lscm
  computed the post UV from.

  The UV panels (uv_pre, uv_post) are exact joint_lscm outputs with no transformation other than scaling and embedding into 3D space for display.

  One more caveat: for multi-sheet (non-manifold) collapses, the visualizer only shows sheets[0]. If there are 2 active sheets, only the first sheet's UV and one-ring are displayed.