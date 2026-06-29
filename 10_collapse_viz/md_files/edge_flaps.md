What was going on

  edge_flaps and its two slots

  IGL's edge_flaps builds EF and EI by scanning every directed half-edge in the face list. For each undirected edge it has two slots:

  - EF(e, 0) / EI(e, 0) — filled when it sees the half-edge going u→v
  - EF(e, 1) / EI(e, 1) — filled when it sees the half-edge going v→u

  Both slots start at -1 (never written).

  The inconsistently-wound edge

  Your non-manifold mesh had at least one edge where both incident faces wound it in the same direction — say both contributed v→u. So edge_flaps wrote EF(e, 1) twice (second write overwrote first) and EF(e,
  0) was never touched — stayed at -1.

  This doesn't show up in a face-count. The edge looks like it has exactly 2 faces, so edge-count = 2 = "normal manifold edge". The three previous guard strategies all missed it:

  ┌───────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │  Guard strategy   │                                                        Why it missed                                                        │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ faceSheetID-based │ connect_boundary_to_infinity adds faces that bridge the sheets → everything collapses to 1 sheet → no seam vertices flagged │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Edge-count-based  │ count = 2 → looks manifold → not flagged                                                                                    │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ EF-scan           │ catches it: EF(e, 0) == -1 is the exact crash condition                                                                     │
  └───────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  The crash path

  When igl::circulation traversed the one-ring of a nearby vertex, its inner step() lambda did:

  const int nside = EF(e,0)==ff ? 1 : 0;   // EF(e,0)==-1, never equals ff → nside=0
  const int nv    = EI(e, nside);           // EI(e,0) = -1
  nf              = EF(e, nside);           // EF(e,0) = -1
  rv              = F(nf, nv);             // F(-1, -1) → Eigen bounds assert → crash

  Because EF(e, 0) = -1, the face-matching check EF(e,0)==ff always fails, nside falls back to 0, and then both nf and nv are read from the -1 slots. F(-1, ...) is the out-of-bounds crash.

  Why the EF scan fixes it

  The scan marks both endpoints of any edge with EF(e,0)==-1 || EF(e,1)==-1 as seam vertices. The guard then blocks any collapse whose source or destination is flagged, so circulation is never called for those
  vertices, and the bad EF entries are never reached.

  ---
  Do we need to undo anything related to connect_boundary_to_infinity?

  No. We never modified connect_boundary_to_infinity itself — that call is untouched. What we learned during debugging is that calling partition_into_sheets after the infinity extension gives a misleading
  sheet partition (infinity faces bridge the seams into one big sheet), so we switched the seam-vertex detection to the EF scan. The partition_into_sheets call is still there and still useful: for non-seam
  collapses the faceSheetID is consistent, and when you eventually want to collapse across seam edges the sheet info will be ready. Nothing needs reverting.
