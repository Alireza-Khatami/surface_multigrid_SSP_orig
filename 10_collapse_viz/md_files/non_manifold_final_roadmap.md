
 ---
  Non-Manifold SSP Implementation Roadmap

  Architecture

  Global mesh layer          Sheet layer
  ──────────────────         ────────────────────────────────
  topology (F, E, EMAP)      given: sheet faces + collapse edge
  VF maintenance             │
  sheet ID lookup            ↓
  collapse scheduling        ordered fan walk (fan_walk_local)
                             compact local mesh
                             joint_lscm
                             SheetData

  EF and EI are initialization-only after the first collapse.
  EMAP, E, F, VF are live and maintained throughout.

  ---
  New Files

  src/partition_into_sheets.h / .cpp

  BFS on the original mesh (before infinity extension). Stops crossing any edge where the number of incident faces ≠ 2. Assigns each face a sheet ID (0-indexed). Returns faceSheetID (VectorXi, size = original
  face count) and numSheets.

  Called once at startup on the original mesh.

  ---
  Modified Files

  1. src/single_collapse_data.h

  Add SheetData struct:
  global_sheet_id, b, subsetVIdx, UV_pre, UV_post, FUV_pre, FUV_post, FIdx_pre, FIdx_post

  Restructure single_collapse_data:
  sheets   : vector<SheetData>   (one per touched sheet)
  V_pre    : MatrixXd            (3D geometry, from first sheet, for display)
  V_post   : MatrixXd
  Nsv, Ndv : vector<int>         (winding-order neighbours, first sheet)

  ---
  2. src/SSP_collapse_edge.h

  Inner overload — add two parameters at end:
  const Eigen::VectorXi & faceSheetID
  Eigen::VectorXi       & EQ

  Outer overload — add two parameters at end:
  std::vector<std::vector<int>> * VF
  const Eigen::VectorXi        & faceSheetID

  Remove #include <igl/circulation.h> (no longer used).
  Add #include "partition_into_sheets.h".

  ---
  3. src/SSP_collapse_edge.cpp — core rewrite

  Inner overload changes:

  ┌─────────────────────────────────┬──────────────────────────────────────────────────┐
  │               Old               │                       New                        │
  ├─────────────────────────────────┼──────────────────────────────────────────────────┤
  │ igl::circulation for neighbours │ — (neighbours arrive from outer overload via VF) │
  ├─────────────────────────────────┼──────────────────────────────────────────────────┤
  │ flat get_collapse_onering_faces │ called per-sheet                                 │
  ├─────────────────────────────────┼──────────────────────────────────────────────────┤
  │ flat remove_unreferenced_lessF  │ called per-sheet                                 │
  ├─────────────────────────────────┼──────────────────────────────────────────────────┤
  │ sorted std::set for Nsv/Ndv     │ fan_walk_local lambda (winding order)            │
  ├─────────────────────────────────┼──────────────────────────────────────────────────┤
  │ flat joint_lscm                 │ called per-sheet                                 │
  ├─────────────────────────────────┼──────────────────────────────────────────────────┤
  │ flat data.b, data.UV_pre, …     │ fills SheetData, pushed into data.sheets         │
  ├─────────────────────────────────┼──────────────────────────────────────────────────┤
  │ EF/EI topology update loop      │ VF-based two-pass update                         │
  └─────────────────────────────────┴──────────────────────────────────────────────────┘

  New fan_walk_local lambda (inside inner overload):
  Walks the face fan around a center vertex ending at end_vtx, in face-adjacency order. Returns local indices with end_vtx last (required by joint_lscm). Infinity vertex maps to -1.

  New VF-based topology update (replaces EF/EI loop):
  - Pass 1 — kill flap faces: for each face in nV2Fd containing both s and d, redirect EMAP from e_kill → e_keep, null the face, mark EQ(e_kill) = -1
  - Pass 2 — remap d→s: for surviving faces in nV2Fd, update F(f,c), E(e,*) entries

  Initialize a_e1 = a_e2 = a_f1 = a_f2 = -1.

  Sheet bucketing (replaces partition_onering_by_sheet): bucket VF[s] and VF[d] directly by faceSheetID(f) on the fly — no separate routine needed.

  Outer overload changes:

  ┌───────────────────────────────┬───────────────────────────────────────────┐
  │              Old              │                    New                    │
  ├───────────────────────────────┼───────────────────────────────────────────┤
  │ igl::circulation              │ collect_onering lambda scanning (*VF)[v]  │
  ├───────────────────────────────┼───────────────────────────────────────────┤
  │ EQ(e1) = -1; EQ(e2) = -1      │ guarded: if (e1 >= 0) EQ(e1) = -1;        │
  ├───────────────────────────────┼───────────────────────────────────────────┤
  │ —                             │ after collapse: merge (*VF)[d] → (*VF)[s] │
  ├───────────────────────────────┼───────────────────────────────────────────┤
  │ —                             │ skip null/killed edges in re-enqueue loop │
  ├───────────────────────────────┼───────────────────────────────────────────┤
  │ inner call: no faceSheetID/EQ │ passes both                               │
  └───────────────────────────────┴───────────────────────────────────────────┘

  Add safety guard after timestamp check: if E(e,0) == IGL_COLLAPSE_EDGE_NULL, mark EQ(e) = -1 and continue.

  Diagnostic logging (all behind static counters, fire for first N occurrences):

  ┌───────────────────┬──────────────────────────────────────────────────────────────────┐
  │        Tag        │                            Fires when                            │
  ├───────────────────┼──────────────────────────────────────────────────────────────────┤
  │ [collect_onering] │ infinity face count > 0 in VF scan                               │
  ├───────────────────┼──────────────────────────────────────────────────────────────────┤
  │ [BOGUS ONE-RING]  │ Nsv.size() < 2 || Ndv.size() < 2                                 │
  ├───────────────────┼──────────────────────────────────────────────────────────────────┤
  │ [sheets]          │ first 10 collapses — prints sheet breakdown                      │
  ├───────────────────┼──────────────────────────────────────────────────────────────────┤
  │ [b_si FAIL]       │ b_si(0) < 0 || b_si(1) < 0 — prints real vs infinity face counts │
  ├───────────────────┼──────────────────────────────────────────────────────────────────┤
  │ [fan_walk EMPTY]  │ fan_walk_local returns empty despite valid b_si                  │
  ├───────────────────┼──────────────────────────────────────────────────────────────────┤
  │ [joint_lscm #N]   │ first 10 calls — prints local indices                            │
  └───────────────────┴──────────────────────────────────────────────────────────────────┘

  ---
  4. src/SSP_midpoint.cpp

  - Remove is_edge_manifold guard in top-level overload (lines ~38-45)
  - Remove is_edge_manifold guard in bottom overload (lines ~148-155)
  - Bottom overload: detect numOrigFaces from extended OF (count faces not containing infVtx = V.rows()-1), call partition_into_sheets(OF.topRows(numOrigFaces), faceSheetID, numSheets)
  - Bottom overload: initialize VF from F using igl::vertex_triangle_adjacency
  - Pass &VF and faceSheetID to SSP_collapse_edge call

  ---
  5. src/SSP_random_collapse_edge.cpp

  No algorithmic changes. Update only for single_collapse_data struct compatibility:
  - Create a local SheetData sd, fill it with all existing fields (b, subsetVIdx, UV_pre, etc.)
  - Push sd into data.sheets
  - Set data.V_pre, data.V_post, data.Nsv, data.Ndv at top level

  ---
  6. 10_collapse_viz/main.cpp

  New globals:
  VectorXi gFaceSheetID;
  int gNumSheets = 1;
  vector<vector<int>> gVF;

  init_ssp changes (order matters):
  1. partition_into_sheets(FO, gFaceSheetID, gNumSheets)   ← on ORIGINAL mesh
  2. connect_boundary_to_infinity(VO, FO, gV, gF)
  3. edge_flaps(gF, gE, gEMAP, gEF, gEI)
  4. remove is_edge_manifold guard
  5. igl::vertex_triangle_adjacency(gF, gV.rows(), gVF)
  6. compute seamVertex (optional display aid)
  7. init queue as before

  do_next_step changes:
  Pass &gVF and gFaceSheetID to SSP_collapse_edge.

  Update display reads from d.sheets[0].*:
  d.sheets[0].b, d.sheets[0].subsetVIdx,
  d.sheets[0].FUV_pre/post, d.sheets[0].UV_pre/post,
  d.V_pre (top-level, unchanged)

  ---
  7. src/query_coarse_to_fine.h / .cpp

  Add const Eigen::VectorXi & faceSheetID parameter.

  Inside the query loop, route to the correct sheet:
  int sheetID = faceSheetID(FIdx(qIdx));
  const SheetData & sd = find_sheet(decInfo[dIdx], sheetID);
  // use sd.subsetVIdx, sd.UV_post, sd.FUV_post, sd.FIdx_post

  ---
  8. src/query_fine_to_coarse.h / .cpp

  Same change as above. Use sd.UV_pre, sd.FUV_pre, sd.FIdx_pre.

  ---
  Implementation Order

  1. single_collapse_data.h          (struct changes — everything depends on this)
  2. partition_into_sheets.h / .cpp  (new utility)
  3. SSP_collapse_edge.h             (signature updates)
  4. SSP_collapse_edge.cpp           (core rewrite)
  5. SSP_random_collapse_edge.cpp    (struct compat only)
  6. SSP_midpoint.cpp                (remove guards, add VF)
  7. main.cpp                        (startup + display)
  8. query_coarse_to_fine.h / .cpp   (sheet routing)
  9. query_fine_to_coarse.h / .cpp   (sheet routing)