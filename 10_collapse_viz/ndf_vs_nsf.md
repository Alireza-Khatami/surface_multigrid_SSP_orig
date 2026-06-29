

  1. Link condition check (edge_collapse_is_valid)
  The validity check requires the vertex neighbor lists of each endpoint separately (Nsv, Ndv) to compute their intersection. The link condition
  says: |Nsv ∩ Ndv| == 2 (exactly the two "wing" vertices). This is inherently a per-endpoint concept — you can't derive it from the union.

  2. UV flattening (joint_lscm)
  The LSCM solve takes Nsv_local and Ndv_local as separate input — they encode which local vertices are neighbors of vi vs. vj. The solver uses
  this to set up the boundary constraints differently for the two endpoints. A merged union loses the information about which vertex each
  neighbor belongs to.

  3. Mesh topology update (nV2Fd)
  Inside the inner collapse, only d's faces need their vertex reference remapped from d → s:
  const std::vector<int> & nV2Fd = (!eflip ? Nsf : Ndf);  // always = faces of d
  If you only had the union you'd have to re-scan all faces to find which ones actually contain d.

  ---
  The union (Nsf ∪ Ndf) is only needed for two things:
  - Building F_ring_pre in get_collapse_onering_faces (which faces to flatten)
  - Updating edge costs after the collapse (all affected edges)

  So the philosophy is: Nsf and Ndf are the same ring, viewed from each side. They overlap on the two flap faces. You need the union for "what
  changes", but you need the split for "who owns what" — validity, UV boundary conditions, and which references to remap.