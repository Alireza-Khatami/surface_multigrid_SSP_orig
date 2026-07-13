

[fan_walk EMPTY] sid=0 vi=666 vj=1425  Nsf_walk=3 Ndf_walk=4  Nsv_local=0 Ndv_local=5
Assertion failed: false && "fan_walk_local returned empty Γ?? upstream topology bug", file C:\Users\alirz\Projects\Graphics\Neural QMAT\external\surf_subgrid_SSP_orig\src\SSP_collapse_edge.cpp, line 300
Debug Error!s





The likely source is fan_walk_local for open fans. For an open fan (boundary vertex), the walk searches for a start vertex by counting how many times each neighbour appears across all incident faces —
  count==1 means a boundary end. But there are two count-1 neighbours in an open fan: the real geometry boundary vertex and infVtx. If the walk picks infVtx as the start (instead of the real boundary vertex),
  it traverses the fan in the reverse winding order. The sequence that arrives at joint_lscm is then wound backwards, and LSCM produces a reflected/flipped UV for those faces. This is consistent — it happens
  every time a boundary edge is collapsed — which matches the "constant" character you're seeing. The fix is to explicitly exclude infVtx from being chosen as start_vtx in fan_walk_locals



  fix :
  ---
  Root cause: stale VF lists + active_sheets admitting sheets with no shared face

  The lazy VF merge

  Every time vertex d (absorbed) is collapsed into s (surviving), the outer overload does:

  (*VF)[sv].insert((*VF)[sv].end(), (*VF)[dv].begin(), (*VF)[dv].end());
  (*VF)[dv].clear();

  Null faces (the two flap triangles killed during the collapse) are never removed from VF[sv]. They accumulate. collect_onering filters them out lazily on the next call, but the raw VF list grows and stays dirty.

  active_sheets reads those stale lists

  for (auto & kv : sheets_Nsf)
      if (sheets_Ndf.count(kv.first) && !sheets_Ndf[kv.first].empty())
          active_sheets.insert(kv.first);

  A sheet is accepted as active if:
  - E(e,1) has ≥ 1 live real face in that sheet (sheets_Nsf[sid] non-empty), AND
  - E(e,0) has ≥ 1 live real face in that sheet (sheets_Ndf[sid] non-empty).

  It never checks whether any face in the sheet contains both endpoints simultaneously. After enough collapses, the VF list of E(e,1) can contain faces from a sheet that E(e,0) once shared — but those faces have since been killed or
  remapped. E(e,0) has its own live faces in that sheet from a different neighborhood. No single face spans the collapse edge in that sheet.

  fan_walk_local fails in three distinct modes

  Given face lists where end_vtx is absent or the topology is wrong, fan_walk_local fails:

  Mode 1 — dead-end (INV-F): The walk starts at a real boundary vertex, steps face by face, but end_vtx is never found in any remaining face. The walk stops at whatever vertex was last. Nsv.back() ≠ loc_e0 or Ndv.back() ≠ loc_e1.

  Mode 2 — INF-AS-START (INV-H): The count-1 boundary scan finds only infVtx (count = 1, no other boundary vertex other than end_vtx). infVtx is picked as start_vtx. The result is [-1, end_vtx] — it accidentally passes INV-F and INV-G
  but starts at the infinity vertex. Inside joint_lscm this produces a boundary loop where vi and vj are not adjacent, causing assert(vk != -1) at joint_lscm.cpp:924.

  Mode 3 — empty (INV-A): end_vtx is completely absent from the face list; the closed-fan fallback also fails. Returns {}. The old assert(false) fired here.

  What the diagnostics confirmed

  The [sheets] log showed entries with end_vtx_E0_present=0 — the end vertex was simply absent from the Nsf face list for that sheet. The [fan_walk EMPTY] log showed end_in_Nsf=0. The [pre-jlscm] log showed INV-F and INV-H failures for
  the cases that slipped past the empty-walk guard.

  The fix applied (guard, not cure)

  The six continue guards in the pre-joint_lscm check (INV-A through INV-H) are a guard, not a cure. They prevent malformed inputs from reaching joint_lscm, so those collapses are silently skipped (the edge gets re-queued as
  non-collapsible). The real fix would be to correct active_sheets to only admit sheets where a face containing both E(e,0) and E(e,1) exists, which eliminates the bad walks before they're attempted. That was the change pending at the
  time you asked for the diagnostic phase.
