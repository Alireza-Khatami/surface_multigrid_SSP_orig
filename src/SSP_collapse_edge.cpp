#include "SSP_collapse_edge.h"
#include <igl/edge_collapse_is_valid.h>
#include <always_try_never_care.h>
#include <vector>
#include <map>
#include <set>
#include <cstdio>

// ============================================================
// Inner overload
// Performs per-sheet UV computation and VF-based topology update.
// EF/EI are left stale after the first collapse; only EMAP and E are maintained.
// ============================================================
bool SSP_collapse_edge(
  const int e,
  const Eigen::RowVectorXd & p,
  /*const*/ std::vector<int> & Nsv,
  const std::vector<int> & Nsf,
  /*const*/ std::vector<int> & Ndv,
  const std::vector<int> & Ndf,
  Eigen::MatrixXd & V,
  Eigen::MatrixXi & F,
  Eigen::MatrixXi & E,
  Eigen::VectorXi & EMAP,
  Eigen::MatrixXi & EF,
  Eigen::MatrixXi & EI,
  int & a_e1,
  int & a_e2,
  int & a_f1,
  int & a_f2,
  std::vector<single_collapse_data> & decInfo,
  std::vector<std::vector<int>> & decIM,
  single_collapse_data & data,
  Eigen::VectorXi & FIdx_onering_pre,
  const Eigen::VectorXi & faceSheetID,
  Eigen::VectorXi & EQ)
{
  using namespace Eigen;
  using namespace std;
  using namespace igl;

  if ((decInfo.size()+1) % 100000 == 0)
    fprintf(stderr, "#collapses: %zu\n", decInfo.size()+1);

  const int eflip = E(e,0) > E(e,1);
  const int s = eflip ? E(e,1) : E(e,0);   // surviving vertex (smaller index)
  const int d = eflip ? E(e,0) : E(e,1);   // absorbed vertex  (larger  index)
  const int vi = s;
  const int vj = d;

  assert(s < d && "s should be less than d");

  // Validity check — uses full one-ring (unordered, order doesn't matter here)
  {
    vector<int> Nsv_alec = Nsv;
    vector<int> Ndv_alec = Ndv;
    if (!igl::edge_collapse_is_valid(Nsv_alec, Ndv_alec))
      return false;
  }

  const int m             = F.rows();
  const int infVtx        = (int)V.rows() - 1;
  const int numOrigFaces  = (int)faceSheetID.size();

  // ---- helpers ----

  auto null_face = [&](int f) -> bool {
    return F(f,0) == IGL_COLLAPSE_EDGE_NULL &&
           F(f,1) == IGL_COLLAPSE_EDGE_NULL &&
           F(f,2) == IGL_COLLAPSE_EDGE_NULL;
  };

  // fan_walk_local
  // Walks the face fan around 'center', ending at 'end_vtx', in face-adjacency order.
  // face_list = real faces of this sheet + adjacent infinity faces (for boundary detection).
  // Returns local indices via svIdx; infVtx maps to -1 (joint_lscm boundary sentinel).
  // Last element is always local(end_vtx) — required by joint_lscm.
  auto fan_walk_local = [&](
      const int center,
      const int end_vtx,
      const vector<int> & face_list,
      const VectorXi & svIdx) -> vector<int>
  {
    // Filter null faces; keep infinity faces (boundary detection)
    vector<int> faces;
    faces.reserve(face_list.size());
    for (int f : face_list)
      if (!null_face(f)) faces.push_back(f);
    if (faces.empty()) return {};

    const int sz = svIdx.size();
    auto to_local = [&](int gv) -> int {
      if (gv == infVtx) return -1;
      for (int i = 0; i < sz; i++)
        if (svIdx(i) == gv) return i;
      return -1;
    };

    // Count face appearances per neighbor to detect boundary vertices (count == 1)
    map<int,int> nb_count;
    for (int f : faces)
      for (int c = 0; c < 3; c++)
        if (F(f,c) != center) nb_count[F(f,c)]++;

    // Start vertex: boundary vertex != end_vtx; for closed fan pick CCW neighbor of end_vtx
    int start_vtx = -1;
    for (auto & kv : nb_count)
      if (kv.second == 1 && kv.first != end_vtx) { start_vtx = kv.first; break; }

    // [fan_walk DEBUG] fire whenever infVtx appears in the face list (= open fan).
    // No static cap — we need to see every occurrence to catch the pattern.
    if (nb_count.count(infVtx)) {
      const char * method = (start_vtx != -1) ? "count-1" : "closed-fallback";
      fprintf(stderr,
        "[fan_walk OPEN] center=%d end=%d  infVtx_count=%d  "
        "start_method=%s  start=%d%s\n",
        center, end_vtx, nb_count.at(infVtx), method, start_vtx,
        (start_vtx == infVtx) ? " *** INF-AS-START ***" : "");
      // Show the full count-1 set so we can see which real boundary vtx (if any) was masked
      fprintf(stderr, "  nb count-1 (excl end=%d): [", end_vtx);
      for (auto & kv : nb_count)
        if (kv.second == 1 && kv.first != end_vtx)
          fprintf(stderr, "%d%s ", kv.first, kv.first == infVtx ? "(INF)" : "");
      fprintf(stderr, "]\n");
    }

    if (start_vtx == -1) {
      // Closed fan: pick third vertex of any face containing (center, end_vtx)
      for (int f : faces) {
        bool hc = false, he = false;
        for (int c = 0; c < 3; c++) {
          if (F(f,c) == center)  hc = true;
          if (F(f,c) == end_vtx) he = true;
        }
        if (!hc || !he) continue;
        for (int c = 0; c < 3; c++) {
          int v = F(f,c);
          if (v != center && v != end_vtx) { start_vtx = v; break; }
        }
        if (start_vtx != -1) break;
      }
    }
    if (start_vtx == -1) return {};

    // Walk: start_vtx → … → end_vtx (one step per face)
    vector<int> walk;
    walk.push_back(start_vtx);
    int prev_f = -1, cur = start_vtx;
    const int max_steps = (int)faces.size() + 2;

    for (int step = 0; step < max_steps && cur != end_vtx; step++) {
      int next_v = -1, next_f = -1;
      for (int f : faces) {
        if (f == prev_f) continue;
        bool hc = false, hcur = false;
        for (int c = 0; c < 3; c++) {
          if (F(f,c) == center) hc   = true;
          if (F(f,c) == cur)    hcur = true;
        }
        if (!hc || !hcur) continue;
        for (int c = 0; c < 3; c++) {
          int v = F(f,c);
          if (v != center && v != cur) { next_v = v; break; }
        }
        next_f = f;
        break;
      }
      if (next_f == -1) break;  // boundary dead-end
      prev_f = next_f;
      cur    = next_v;
      walk.push_back(cur);
    }

    vector<int> result;
    result.reserve(walk.size());
    for (int gv : walk) result.push_back(to_local(gv));

    // [fan_walk DEBUG] print final walk for every open fan so we can see
    // whether -1 lands at position 0 (open end found), middle, or near end.
    if (nb_count.count(infVtx)) {
      fprintf(stderr, "  walk result (local): [");
      for (int v : result) fprintf(stderr, "%d,", v);
      int inf_pos = -1;
      for (int i = 0; i < (int)result.size(); i++)
        if (result[i] == -1) { inf_pos = i; break; }
      fprintf(stderr, "]  -1 at pos %d / %d\n", inf_pos, (int)result.size());
    }

    return result;
  };
  // ---- end fan_walk_local ----

  // ---- Bucket Nsf/Ndf by sheet (real faces only) ----
  map<int, vector<int>> sheets_Nsf, sheets_Ndf;
  for (int f : Nsf) {
    if (null_face(f) || f >= numOrigFaces) continue;
    sheets_Nsf[faceSheetID(f)].push_back(f);
  }
  for (int f : Ndf) {
    if (null_face(f) || f >= numOrigFaces) continue;
    sheets_Ndf[faceSheetID(f)].push_back(f);
  }

  // Active sheets: only those where BOTH Nsf and Ndf have real faces
  set<int> active_sheets;
  for (auto & kv : sheets_Nsf)
    if (sheets_Ndf.count(kv.first) && !sheets_Ndf[kv.first].empty())
      active_sheets.insert(kv.first);

  // [sheets] diagnostic: first 10 collapses
  {
    static int sheet_log = 0;
    if (sheet_log < 10) {
      sheet_log++;
      fprintf(stderr, "[sheets] collapse #%zu vi=%d vj=%d active_sheets=%zu\n",
              decInfo.size(), vi, vj, active_sheets.size());
      for (int sid : active_sheets)
        fprintf(stderr, "  sheet=%d  Nsf=%zu  Ndf=%zu\n",
                sid, sheets_Nsf[sid].size(), sheets_Ndf[sid].size());
    }
  }

  // Walk face list builder: real faces of sheet si + adjacent infinity faces of vertex v
  auto get_walk_faces = [&](
      const vector<int> & all_faces,   // full Nsf or Ndf (real + infinity, from collect_onering)
      const vector<int> & real_si,     // real faces of sheet si for this vertex
      int center_v) -> vector<int>
  {
    vector<int> walk = real_si;
    set<int> vtx_set;
    for (int f : real_si)
      for (int c = 0; c < 3; c++) vtx_set.insert(F(f,c));
    // Attach adjacent infinity faces (boundary signal for fan_walk_local)
    for (int f : all_faces) {
      if (null_face(f) || f < numOrigFaces) continue;  // only infinity faces
      for (int c = 0; c < 3; c++) {
        int v = F(f,c);
        if (v == center_v || v == infVtx) continue;
        if (vtx_set.count(v)) { walk.push_back(f); break; }
      }
    }
    return walk;
  };

  // ---- Per-sheet UV loop ----
  bool    any_sheet_ok = false;
  set<int> FIdx_combined;
  data.sheets.clear();

  for (int sid : active_sheets) {
    const vector<int> & Nsf_si = sheets_Nsf[sid];
    const vector<int> & Ndf_si = sheets_Ndf[sid];

    vector<int> Nsf_walk = get_walk_faces(Nsf, Nsf_si, E(e,1));
    vector<int> Ndf_walk = get_walk_faces(Ndf, Ndf_si, E(e,0));

    // One-ring faces for this sheet (filters null + infinity faces)
    VectorXi FIdx_pre_si, FIdx_post_si;
    MatrixXi F_ring_pre_si, F_ring_post_si;
    bool validEdge = get_collapse_onering_faces(
        V, F, vi, vj, Nsf_si, Ndf_si,
        FIdx_pre_si, FIdx_post_si, F_ring_pre_si, F_ring_post_si);
    if (!validEdge) return false;

    // Compact local mesh for this sheet
    MatrixXd V_pre_si;
    MatrixXi FUV_pre_si;
    VectorXi subsetVIdx_si;
    {
      map<int,int> IM_tmp;
      remove_unreferenced_lessF(V, F_ring_pre_si, V_pre_si, FUV_pre_si, IM_tmp, subsetVIdx_si);
    }

    if (FUV_pre_si.rows() <= 2) continue;  // too few faces for this sheet

    // Find local indices of vi and vj (subsetVIdx is sorted ascending, so vi<vj globally → local vi<vj)
    VectorXi b_si(2);
    b_si.setConstant(-1);
    for (int ii = 0; ii < subsetVIdx_si.size(); ii++) {
      if      (subsetVIdx_si(ii) == vi) b_si(0) = ii;
      else if (subsetVIdx_si(ii) == vj) b_si(1) = ii;
    }

    // [b_si FAIL] diagnostic
    if (b_si(0) < 0 || b_si(1) < 0 || b_si(0) >= b_si(1)) {
      static int b_fail = 0;
      if (b_fail < 5) {
        b_fail++;
        int ndf_real=0, ndf_inf=0, nsf_real=0, nsf_inf=0;
        for (int f : Ndf_si) (f>=numOrigFaces ? ndf_inf : ndf_real)++;
        for (int f : Nsf_si) (f>=numOrigFaces ? nsf_inf : nsf_real)++;
        fprintf(stderr,
          "[b_si FAIL] sid=%d vi=%d vj=%d b=(%d,%d)  "
          "Ndf: real=%d inf=%d  Nsf: real=%d inf=%d\n",
          sid, vi, vj, b_si(0), b_si(1),
          ndf_real, ndf_inf, nsf_real, nsf_inf);
        if (ndf_real == 0)
          fprintf(stderr, "  HYPOTHESIS: sheets_Ndf[sid] contains only infinity faces\n");
        if (nsf_real == 0)
          fprintf(stderr, "  HYPOTHESIS: sheets_Nsf[sid] contains only infinity faces\n");
      }
      continue;
    }

    // Build post-collapse local mesh
    MatrixXd V_post_si = V_pre_si;
    V_post_si.row(b_si(0)) = p;
    MatrixXi FUV_post_si;
    VectorXi FUV_pre_keep_si;
    get_post_faces(FUV_pre_si, b_si(0), b_si(1), FUV_pre_keep_si, FUV_post_si);

    // Winding-order neighbour lists for joint_lscm
    // fan_walk_local(center, end_vtx, walk_faces, svIdx) → local indices, end_vtx last
    vector<int> Nsv_local = fan_walk_local(E(e,1), E(e,0), Nsf_walk, subsetVIdx_si);
    vector<int> Ndv_local = fan_walk_local(E(e,0), E(e,1), Ndf_walk, subsetVIdx_si);

    // [fan_walk EMPTY] diagnostic
    if (Nsv_local.empty() || Ndv_local.empty()) {
      static int fw_empty = 0;
      if (fw_empty < 5) {
        fw_empty++;
        fprintf(stderr,
          "[fan_walk EMPTY] sid=%d vi=%d vj=%d  "
          "Nsf_walk=%zu Ndf_walk=%zu  Nsv_local=%zu Ndv_local=%zu\n",
          sid, vi, vj,
          Nsf_walk.size(), Ndf_walk.size(),
          Nsv_local.size(), Ndv_local.size());
      }
      assert(false && "fan_walk_local returned empty — upstream topology bug");
      continue;
    }

    // [joint_lscm] diagnostic: first 10 calls
    {
      static int lscm_log = 0;
      if (lscm_log < 10) {
        lscm_log++;
        fprintf(stderr,
          "[joint_lscm #%d] sid=%d vi=%d(loc %d) vj=%d(loc %d)  pre=%d post=%d  Nsv=[",
          lscm_log, sid, vi, b_si(0), vj, b_si(1),
          (int)FUV_pre_si.rows(), (int)FUV_post_si.rows());
        for (int x : Nsv_local) fprintf(stderr, "%d,", x);
        fprintf(stderr, "]  Ndv=[");
        for (int x : Ndv_local) fprintf(stderr, "%d,", x);
        fprintf(stderr, "]\n");
      }
    }

    // joint_lscm (DO NOT MODIFY — uses infVtx=-1 sentinel and winding-order Nsv/Ndv)
    MatrixXd UV_pre_si, UV_post_si;
    bool isValid = joint_lscm(
        V_pre_si, FUV_pre_si, V_post_si, FUV_post_si,
        b_si(0), b_si(1), Nsv_local, Ndv_local,
        UV_pre_si, UV_post_si);
    if (!isValid) return false;

    // Store SheetData
    SheetData sd;
    sd.global_sheet_id = sid;
    sd.b          = b_si;
    sd.subsetVIdx = subsetVIdx_si;
    sd.UV_pre     = UV_pre_si;
    sd.UV_post    = UV_post_si;
    sd.FUV_pre    = FUV_pre_si;
    sd.FUV_post   = FUV_post_si;
    sd.FIdx_pre   = FIdx_pre_si;
    sd.FIdx_post  = FIdx_post_si;
    data.sheets.push_back(sd);

    // First successful sheet → set top-level 3D data (for display and Nsv/Ndv compat)
    if (!any_sheet_ok) {
      data.V_pre  = V_pre_si;
      data.V_post = V_post_si;
      data.Nsv    = Nsv_local;
      data.Ndv    = Ndv_local;
    }

    for (int ii = 0; ii < FIdx_pre_si.size(); ii++)
      FIdx_combined.insert(FIdx_pre_si(ii));

    any_sheet_ok = true;
  }

  if (!any_sheet_ok) return false;

  // Build combined FIdx_onering_pre (sorted union of all sheets)
  FIdx_onering_pre.resize((int)FIdx_combined.size());
  { int fi = 0; for (int f : FIdx_combined) FIdx_onering_pre(fi++) = f; }

  // ================================================================
  // VF-based two-pass topology update
  // (replaces EF/EI loop — EF/EI are left stale after this point)
  // nV2Fd = all faces of the absorbed vertex d (real + infinity)
  // ================================================================
  const vector<int> & nV2Fd = (!eflip ? Nsf : Ndf);

  V.row(s) = p;
  V.row(d) = p;

  const auto kill_edge = [&](const int ek) {
    E(ek,0) = IGL_COLLAPSE_EDGE_NULL; E(ek,1) = IGL_COLLAPSE_EDGE_NULL;
    EF(ek,0)= IGL_COLLAPSE_EDGE_NULL; EF(ek,1)= IGL_COLLAPSE_EDGE_NULL;
    EI(ek,0)= IGL_COLLAPSE_EDGE_NULL; EI(ek,1)= IGL_COLLAPSE_EDGE_NULL;
  };

  a_e1 = -1; a_e2 = -1; a_f1 = -1; a_f2 = -1;
  int flap_count = 0;

  // Pass 1: find flap faces (contain both s and d), kill them and their edges
  for (int f : nV2Fd) {
    if (null_face(f)) continue;
    int cs = -1, cd = -1;
    for (int c = 0; c < 3; c++) {
      if      (F(f,c) == s) cs = c;
      else if (F(f,c) == d) cd = c;
    }
    if (cs < 0 || cd < 0) continue;  // not a flap

    // edge opposite s-corner = edge(d, ca) → kill it
    // edge opposite d-corner = edge(s, ca) → keep it
    const int e_kill = EMAP(f + m*cs);
    const int e_keep = EMAP(f + m*cd);

    // Redirect EMAP in other faces of nV2Fd: e_kill → e_keep
    for (int fa : nV2Fd) {
      if (fa == f || null_face(fa)) continue;
      for (int c = 0; c < 3; c++)
        if (EMAP(fa + m*c) == e_kill) EMAP(fa + m*c) = e_keep;
    }

    kill_edge(e_kill);
    EQ(e_kill) = -1;  // mark dead in priority queue

    F(f,0) = IGL_COLLAPSE_EDGE_NULL;
    F(f,1) = IGL_COLLAPSE_EDGE_NULL;
    F(f,2) = IGL_COLLAPSE_EDGE_NULL;

    if (flap_count == 0) { a_e1 = e_kill; a_f1 = f; }
    else if (flap_count == 1) { a_e2 = e_kill; a_f2 = f; }
    flap_count++;
  }

  data.numFlapFaces = flap_count;  // 2 = manifold interior, >2 = seam edge

  // Pass 2: remap d → s in all surviving faces of nV2Fd
  for (int f : nV2Fd) {
    if (null_face(f)) continue;
    for (int c = 0; c < 3; c++) {
      if (F(f,c) == d) {
        const int e1 = EMAP(f + m*((c+1)%3));
        const int e2 = EMAP(f + m*((c+2)%3));
        if (E(e1,0) == d) E(e1,0) = s; else if (E(e1,1) == d) E(e1,1) = s;
        if (E(e2,0) == d) E(e2,0) = s; else if (E(e2,1) == d) E(e2,1) = s;
        F(f,c) = s;
        break;
      }
    }
  }

  kill_edge(e);  // kill the collapsed edge itself

  return true;
}


// ============================================================
// Outer overload
// Pops from queue, collects VF-based one-ring, calls inner overload,
// merges VF[d] → VF[s], re-enqueues surviving neighbor edges.
// ============================================================
bool SSP_collapse_edge(
  const decimate_cost_and_placement_func & cost_and_placement,
  const decimate_pre_collapse_func       & pre_collapse,
  const decimate_post_collapse_func      & post_collapse,
  Eigen::MatrixXd & V,
  Eigen::MatrixXi & F,
  Eigen::MatrixXi & E,
  Eigen::VectorXi & EMAP,
  Eigen::MatrixXi & EF,
  Eigen::MatrixXi & EI,
  min_heap< std::tuple<double,int,int> > & Q,
  Eigen::VectorXi & EQ,
  Eigen::MatrixXd & C,
  int & e,
  int & e1,
  int & e2,
  int & f1,
  int & f2,
  std::vector<single_collapse_data> & decInfo,
  std::vector<std::vector<int>> & decIM,
  std::vector<std::vector<int>> * VF,
  const Eigen::VectorXi & faceSheetID)
{
  using namespace Eigen;
  using namespace igl;
  using namespace std;

  const int numOrigFaces = (int)faceSheetID.size();
  const int m = F.rows();

  // VF-based one-ring collector (replaces igl::circulation)
  // Returns all non-null faces incident on v (real + infinity).
  // verts = sorted set of all neighbor vertices (includes infVtx if boundary).
  auto collect_onering = [&](const int v,
                              vector<int> & faces,
                              vector<int> & verts)
  {
    faces.clear(); verts.clear();
    set<int> vset;
    int inf_count = 0;
    for (const int f : (*VF)[v]) {
      if (F(f,0) == IGL_COLLAPSE_EDGE_NULL &&
          F(f,1) == IGL_COLLAPSE_EDGE_NULL &&
          F(f,2) == IGL_COLLAPSE_EDGE_NULL) continue;
      if (f >= numOrigFaces) inf_count++;
      faces.push_back(f);
      for (int c = 0; c < 3; c++)
        if (F(f,c) != v) vset.insert(F(f,c));
    }
    verts.assign(vset.begin(), vset.end());

    // [collect_onering] diagnostic
    static int co_log = 0;
    if (inf_count > 0 && co_log < 5) {
      co_log++;
      fprintf(stderr, "[collect_onering] v=%d  faces=%zu  inf=%d\n",
              v, faces.size(), inf_count);
    }
  };

  // Pop lowest-cost valid edge
  tuple<double,int,int> p;
  while (true) {
    if (Q.empty()) return false;
    p = Q.top();
    if (get<0>(p) == numeric_limits<double>::infinity()) return false;
    Q.pop();
    e = get<1>(p);
    if (get<2>(p) == EQ(e)) {
      // Safety: skip edges killed by VF topology update but not yet dequeued
      if (E(e,0) == IGL_COLLAPSE_EDGE_NULL) { EQ(e) = -1; continue; }
      break;
    }
    assert(get<2>(p) < EQ(e) || EQ(e) == -1);
  }

  // Capture s/d before inner overload calls kill_edge(e)
  const int sv = (E(e,0) < E(e,1)) ? E(e,0) : E(e,1);
  const int dv = (E(e,0) < E(e,1)) ? E(e,1) : E(e,0);

  // Collect one-rings via VF (real + infinity faces; order is unimportant here)
  vector<int> Nsf, Nsv_verts;   // faces and neighbour vertices of E(e,1)
  vector<int> Ndf, Ndv_verts;   // faces and neighbour vertices of E(e,0)
  collect_onering(E(e,1), Nsf, Nsv_verts);
  collect_onering(E(e,0), Ndf, Ndv_verts);

  // [BOGUS ONE-RING] diagnostic
  if ((int)Nsv_verts.size() < 2 || (int)Ndv_verts.size() < 2) {
    static int bogus = 0;
    if (bogus < 5) {
      bogus++;
      fprintf(stderr, "[BOGUS ONE-RING] e=%d E=(%d,%d)  Nsv=%zu  Ndv=%zu\n",
              e, E(e,0), E(e,1), Nsv_verts.size(), Ndv_verts.size());
    }
    EQ(e)++;
    Q.emplace(numeric_limits<double>::infinity(), e, EQ(e));
    return false;
  }

  bool collapsed = true;
  single_collapse_data data;
  VectorXi FIdx_onering_pre;

  if (pre_collapse(V, F, E, EMAP, EF, EI, Q, EQ, C, e)) {
    collapsed = SSP_collapse_edge(
        e, C.row(e),
        Nsv_verts, Nsf, Ndv_verts, Ndf,
        V, F, E, EMAP, EF, EI,
        e1, e2, f1, f2,
        decInfo, decIM, data, FIdx_onering_pre,
        faceSheetID, EQ);
  } else {
    collapsed = false;
  }

  post_collapse(V, F, E, EMAP, EF, EI, Q, EQ, C, e, e1, e2, f1, f2, collapsed);

  if (collapsed) {
    decInfo.push_back(data);

    // Update decIM for all pre-collapse real faces
    for (int ii = 0; ii < FIdx_onering_pre.size(); ii++)
      decIM[FIdx_onering_pre(ii)].push_back((int)decInfo.size() - 1);

    // Mark killed flap edges dead in queue (inner already set EQ(e_kill)=-1 directly;
    // guard for the first two for backward-compat with post_collapse callbacks)
    if (e1 >= 0) EQ(e1) = -1;
    if (e2 >= 0) EQ(e2) = -1;

    // Merge VF[dv] → VF[sv] (lazy: null faces filtered in collect_onering next call)
    (*VF)[sv].insert((*VF)[sv].end(), (*VF)[dv].begin(), (*VF)[dv].end());
    (*VF)[dv].clear();

    // Re-enqueue edges of surviving neighbor faces
    vector<int> Nf;
    Nf.reserve(Nsf.size() + Ndf.size());
    Nf.insert(Nf.end(), Nsf.begin(), Nsf.end());
    Nf.insert(Nf.end(), Ndf.begin(), Ndf.end());
    sort(Nf.begin(), Nf.end());
    Nf.erase(unique(Nf.begin(), Nf.end()), Nf.end());

    vector<int> Ne;
    Ne.reserve(3*Nf.size());
    for (auto & n : Nf) {
      if (F(n,0) == IGL_COLLAPSE_EDGE_NULL &&
          F(n,1) == IGL_COLLAPSE_EDGE_NULL &&
          F(n,2) == IGL_COLLAPSE_EDGE_NULL) continue;
      for (int v = 0; v < 3; v++)
        Ne.push_back(EMAP(v*m + n));
    }
    sort(Ne.begin(), Ne.end());
    Ne.erase(unique(Ne.begin(), Ne.end()), Ne.end());

    for (auto & ei : Ne) {
      if (E(ei,0) == IGL_COLLAPSE_EDGE_NULL) continue;  // skip killed edges
      double cost;
      RowVectorXd place;
      cost_and_placement(ei, V, F, E, EMAP, EF, EI, cost, place);
      EQ(ei)++;
      Q.emplace(cost, ei, EQ(ei));
      C.row(ei) = place;
    }
  } else {
    EQ(e)++;
    Q.emplace(numeric_limits<double>::infinity(), e, EQ(e));
  }

  return collapsed;
}
