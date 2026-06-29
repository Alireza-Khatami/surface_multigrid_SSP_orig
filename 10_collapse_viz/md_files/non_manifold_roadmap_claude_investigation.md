1. Bug 1 — hypothesis vs confirmed root cause

  Confirmed you're right. main.cpp:113 calls partition_into_sheets(gF, ...) AFTER connect_boundary_to_infinity at line 104. The b_si diagnostic at SSP_collapse_edge.cpp:295-303 has the "HYPOTHESIS CONFIRMED"
  print — but it is gated on ndf_real==0 && ndf_inf>0 and has never been executed. The word "root cause" in the roadmap is premature. "Leading hypothesis" is the right framing.

  ---
  2. fan_walk_local — implemented, not validated

  Confirmed. The function is in place (lines 127–233) but has never been compiled or run. More importantly, looking at joint_lscm.cpp lines 128–141:

  bdLoop(ii) = Ndv[Ndv.size() - 3 - ii];   // reversed inner, 0..size-3
  bdLoop(ii+Ndv.size()-2) = Nsv[ii+1];      // forward inner, 1..size-2

  fan_walk_local makes no attempt to match the winding direction that igl::circulation would have produced. It finds a start_vtx from adjacency, but there are two possible winding directions and the code picks
  one implicitly. This is not validated. "Implemented, needs validation" is the correct status.

  ---
  3. Routing vs boundary detection mixing

  Confirmed. collect_onering (line 697) deliberately keeps infinity faces:
  // keep infinity faces — they produce infVIdx=-1 ...
  Those faces then flow into partition_onering_by_sheet, which routes them into a sheet by faceSheetID. That means boundary detection (does -1 appear in the fan?) and sheet routing (which sheet owns this
  face?) are tangled at the same data. The architecture mixes them.

  ---
  4. Fix A — open design question confirmed

  Confirmed. fan_walk_local at line 133–141 explicitly keeps infinity faces to produce the -1 sentinel for joint_lscm. If partition_into_sheets is moved to run on FO, the faceSheetID vector will be sized
  FO.rows() — but collect_onering pulls faces from VF[v] which still includes extended-mesh infinity faces (indices ≥ FO.rows()). So partition_onering_by_sheet would skip them via the existing bounds check (f
  >= nF), but fan_walk_local still needs them in its face list for boundary signaling. The two mechanisms require different handling of the same faces. The architecture is unresolved.

  ---
  5. Core invariant — missing from the document

  Cannot confirm from code — this is a documentation gap, not a code issue. But from reading joint_lscm.cpp, the five properties you listed are exactly what the function requires: both endpoints in the local
  mesh, correct neighbor ordering, and boundary signaling via -1. None of these are written down in a single place anywhere in the codebase.

  ---
  6. b_si guarantee — needs precise framing

  Confirmed. The guarantee chain is:
  - main.cpp:47-50: eflip = E(e,0)>E(e,1), then s = min, d = max → s < d
  - SSP_collapse_edge.cpp:71-72: vi = s; vj = d; → vi < vj always
  - subsetVIdx_si is sorted ascending (from remove_unreferenced_lessF)
  - Therefore b_si(0) < b_si(1) iff both endpoints are found

  The check at line 266 (b_si(0) < 0 || b_si(1) < 0 || b_si(0) >= b_si(1)) treats all three failure modes identically. The third condition (b_si(0) >= b_si(1)) can only fire if vi >= vj in global index, which
  is ruled out by the eflip logic — unless a vertex was remapped during a prior collapse. That edge case is not documented.

  ---
  7. Two-pass topology update — confirmed correct

  Pass 1 (lines 553–590) kills flap faces and repairs EMAP. Pass 2 (lines 593–610) remaps d→s in F and E. The ordering is correct: EMAP is repaired before E is updated.

  ---
  Point 8 was cut off — your message ended at "Your observation about". If you want to finish that thought I can respond to it.