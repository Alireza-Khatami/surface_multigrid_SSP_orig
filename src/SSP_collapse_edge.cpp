#include "SSP_collapse_edge.h"
#include <igl/edge_collapse_is_valid.h>
#include <always_try_never_care.h>
#include <vector>
#include <map>
#include <set>
#include <cstdio>
#include <cmath>

// ---- Seam-edge diagnostic log ----
// Call SSP_seam_log_open(path) once from main; all [SEAM-*] lines go there.
// Falls back to stderr if no file is open.
static FILE * s_seam_log = nullptr;
void SSP_seam_log_open(const char * path) {
  if (s_seam_log) fclose(s_seam_log);
  s_seam_log = path ? fopen(path, "w") : nullptr;
}
void SSP_seam_log_close() {
  if (s_seam_log) { fclose(s_seam_log); s_seam_log = nullptr; }
}
#define SEAM_LOG(fmt, ...) fprintf(s_seam_log ? s_seam_log : stderr, fmt, __VA_ARGS__)

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

  // Validity check.
  // Manifold edges (≤2 shared real faces): apply igl's link condition.
  // Non-manifold seam edges (>2 shared real faces): skip it — igl's check always
  // fails for seam edges because it expects exactly 2 shared vertices, but seam
  // edges have one per incident face.  The per-sheet LSCM loop below is the
  // correct validity gate for non-manifold topology.
  {
    const int _nOrig = (int)faceSheetID.size();
    auto _isnull = [&](int f) {
      return F(f,0)==IGL_COLLAPSE_EDGE_NULL && F(f,1)==IGL_COLLAPSE_EDGE_NULL
          && F(f,2)==IGL_COLLAPSE_EDGE_NULL; };
    set<int> _s;
    for (int f : Nsf) if (!_isnull(f) && f < _nOrig) _s.insert(f);
    int _sh = 0;
    for (int f : Ndf) if (!_isnull(f) && f < _nOrig && _s.count(f)) _sh++;

    if (_sh <= 2) {
      vector<int> Nsv_alec = Nsv;
      vector<int> Ndv_alec = Ndv;
      if (!igl::edge_collapse_is_valid(Nsv_alec, Ndv_alec))
        return false;
    }
    // _sh > 2: seam edge — skip link condition, proceed to per-sheet loop.
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

#ifdef SSP_LSCM_LOG
    // [fan_walk DEBUG] fire whenever infVtx appears in the face list (= open fan).
    if (nb_count.count(infVtx)) {
      const char * method = (start_vtx != -1) ? "count-1" : "closed-fallback";
      fprintf(stderr,
        "[fan_walk OPEN] center=%d end=%d  infVtx_count=%d  "
        "start_method=%s  start=%d%s\n",
        center, end_vtx, nb_count.at(infVtx), method, start_vtx,
        (start_vtx == infVtx) ? " *** INF-AS-START ***" : "");
      fprintf(stderr, "  nb count-1 (excl end=%d): [", end_vtx);
      for (auto & kv : nb_count)
        if (kv.second == 1 && kv.first != end_vtx)
          fprintf(stderr, "%d%s ", kv.first, kv.first == infVtx ? "(INF)" : "");
      fprintf(stderr, "]\n");
    }
#endif

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

#ifdef SSP_LSCM_LOG
    if (nb_count.count(infVtx)) {
      fprintf(stderr, "  walk result (local): [");
      for (int v : result) fprintf(stderr, "%d,", v);
      int inf_pos = -1;
      for (int i = 0; i < (int)result.size(); i++)
        if (result[i] == -1) { inf_pos = i; break; }
      fprintf(stderr, "]  -1 at pos %d / %d\n", inf_pos, (int)result.size());
    }
#endif

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

  const bool is_seam_collapse = (active_sheets.size() > 1);
  if (is_seam_collapse)
    SEAM_LOG("[SEAM-TRY]  e=(%d,%d) vi=%d vj=%d  active_sheets=%zu\n",
      E(e,0), E(e,1), vi, vj, active_sheets.size());

  // Helper: is vtx referenced by any face in flist?
  auto vtx_in_flist = [&](const vector<int> & flist, int vtx) -> bool {
    for (int f : flist)
      for (int c = 0; c < 3; c++)
        if (F(f,c) == vtx) return true;
    return false;
  };

#ifdef SSP_LSCM_LOG
  {
    static int sheet_log = 0;
    if (sheet_log < 40) {
      sheet_log++;
      fprintf(stderr, "[sheets] collapse #%zu  e=(%d,%d) vi=%d vj=%d  active_sheets=%zu\n",
              decInfo.size(), E(e,0), E(e,1), vi, vj, active_sheets.size());
      for (int sid : active_sheets) {
        bool nsf_has_e0 = vtx_in_flist(sheets_Nsf[sid], E(e,0));
        bool ndf_has_e1 = vtx_in_flist(sheets_Ndf[sid], E(e,1));
        fprintf(stderr, "  sheet=%d  Nsf=%zu(end_vtx_E0_present=%d)  Ndf=%zu(end_vtx_E1_present=%d)%s\n",
                sid, sheets_Nsf[sid].size(), (int)nsf_has_e0,
                sheets_Ndf[sid].size(), (int)ndf_has_e1,
                (!nsf_has_e0 || !ndf_has_e1) ? "  <<< MISSING END_VTX — BUG CONFIRMED" : "");
      }
    }
  }
#endif

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
    if (!validEdge) {
      if (is_seam_collapse)
        SEAM_LOG("[SEAM-FAIL-ONERING]  sid=%d  e=(%d,%d) vi=%d vj=%d\n",
          sid, E(e,0), E(e,1), vi, vj);
      return false;
    }

    // Compact local mesh for this sheet
    MatrixXd V_pre_si;
    MatrixXi FUV_pre_si;
    VectorXi subsetVIdx_si;
    {
      map<int,int> IM_tmp;
      remove_unreferenced_lessF(V, F_ring_pre_si, V_pre_si, FUV_pre_si, IM_tmp, subsetVIdx_si);
    }

    if (FUV_pre_si.rows() <= 2) {
      if (is_seam_collapse)
        SEAM_LOG("[SEAM-SKIP-FEW-FACES]  sid=%d  FUV_rows=%d  e=(%d,%d)\n",
          sid, FUV_pre_si.rows(), E(e,0), E(e,1));
      continue;
    }

    // Find local indices of vi and vj (subsetVIdx is sorted ascending, so vi<vj globally → local vi<vj)
    VectorXi b_si(2);
    b_si.setConstant(-1);
    for (int ii = 0; ii < subsetVIdx_si.size(); ii++) {
      if      (subsetVIdx_si(ii) == vi) b_si(0) = ii;
      else if (subsetVIdx_si(ii) == vj) b_si(1) = ii;
    }

    if (b_si(0) < 0 || b_si(1) < 0 || b_si(0) >= b_si(1)) {
      if (is_seam_collapse)
        SEAM_LOG("[SEAM-SKIP-BSI]  sid=%d  b=(%d,%d)  e=(%d,%d) vi=%d vj=%d\n",
          sid, b_si(0), b_si(1), E(e,0), E(e,1), vi, vj);
#ifdef SSP_LSCM_LOG
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
#endif
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

    if (Nsv_local.empty() || Ndv_local.empty()) {
      if (is_seam_collapse)
        SEAM_LOG("[SEAM-SKIP-FANWALK]  sid=%d  Nsv=%zu Ndv=%zu  e=(%d,%d)\n",
          sid, Nsv_local.size(), Ndv_local.size(), E(e,0), E(e,1));
#ifdef SSP_LSCM_LOG
      static int fw_empty = 0;
      if (fw_empty < 10) {
        fw_empty++;
        bool nsf_has_e0 = vtx_in_flist(Nsf_walk, E(e,0));
        bool ndf_has_e1 = vtx_in_flist(Ndf_walk, E(e,1));
        fprintf(stderr,
          "[fan_walk EMPTY] sid=%d  e=(%d,%d) vi=%d vj=%d\n"
          "  Nsf_walk=%zu faces  center=E(e,1)=%d  end_vtx=E(e,0)=%d  end_in_Nsf=%d  Nsv_local=%zu\n"
          "  Ndf_walk=%zu faces  center=E(e,0)=%d  end_vtx=E(e,1)=%d  end_in_Ndf=%d  Ndv_local=%zu\n",
          sid, E(e,0), E(e,1), vi, vj,
          Nsf_walk.size(), E(e,1), E(e,0), (int)nsf_has_e0, Nsv_local.size(),
          Ndf_walk.size(), E(e,0), E(e,1), (int)ndf_has_e1, Ndv_local.size());
        if (!nsf_has_e0 || !ndf_has_e1) {
          fprintf(stderr, "  <<< END_VTX MISSING — hypothesis confirmed\n");
          fprintf(stderr, "  Nsf_walk faces:");
          for (int f : Nsf_walk)
            fprintf(stderr, " [%d,%d,%d]", F(f,0), F(f,1), F(f,2));
          fprintf(stderr, "\n  Ndf_walk faces:");
          for (int f : Ndf_walk)
            fprintf(stderr, " [%d,%d,%d]", F(f,0), F(f,1), F(f,2));
          fprintf(stderr, "\n");
        }
      }
#endif
      continue;
    }

    // ---- pre-joint_lscm invariant check ----
    // joint_lscm reads Nsv/Ndv at lines 27-34 (e0/e1 init), 40-53 (onBd),
    // 64-68 (isFlap), 126-175 (bdLoop), 924 (vk assert in case2).
    {
      const int loc_vi = b_si(0);
      const int loc_vj = b_si(1);
      const int nV_loc  = (int)V_pre_si.rows();

      // Local indices of the two global edge endpoints inside subsetVIdx_si.
      // fan_walk_local(center=E(e,1), end=E(e,0)) → result must end at loc_e0.
      // fan_walk_local(center=E(e,0), end=E(e,1)) → result must end at loc_e1.
      int loc_e0 = -99, loc_e1 = -99;
      for (int ii = 0; ii < subsetVIdx_si.size(); ii++) {
        if (subsetVIdx_si(ii) == E(e,0)) loc_e0 = ii;
        if (subsetVIdx_si(ii) == E(e,1)) loc_e1 = ii;
      }

      // INV-A: walks must be non-empty
      bool inv_A = !Nsv_local.empty() && !Ndv_local.empty();

      int nsv_last = inv_A ? Nsv_local.back() : -999;
      int ndv_last = inv_A ? Ndv_local.back() : -999;

      // INV-B: last element of Nsv OR Ndv must equal loc_vi.
      //   joint_lscm lines 27-34: e0/e1 only assigned if Nsv[end]==vi or Ndv[end]==vi;
      //   otherwise e0/e1 are UNINITIALIZED → UB in bdLoop construction at line 131.
      bool nsv_ends_vi = (nsv_last == loc_vi);
      bool ndv_ends_vi = (ndv_last == loc_vi);
      bool inv_B = nsv_ends_vi || ndv_ends_vi;

      // INV-C: all local indices must be in [-1, nV_loc-1]; -1 is infVtx sentinel.
      auto all_in_range = [&](const vector<int> & v) -> bool {
        for (int x : v) if (x < -1 || x >= nV_loc) return false;
        return true;
      };
      bool inv_C = all_in_range(Nsv_local) && all_in_range(Ndv_local);

      // INV-D: loc_vi and loc_vj must each appear somewhere in Nsv or Ndv.
      auto contains_val = [&](const vector<int> & v, int x) -> bool {
        for (int u : v) if (u == x) return true;
        return false;
      };
      bool vi_in_nsv = contains_val(Nsv_local, loc_vi);
      bool vi_in_ndv = contains_val(Ndv_local, loc_vi);
      bool vj_in_nsv = contains_val(Nsv_local, loc_vj);
      bool vj_in_ndv = contains_val(Ndv_local, loc_vj);
      bool inv_D = (vi_in_nsv || vi_in_ndv) && (vj_in_nsv || vj_in_ndv);

      // INV-F: each walk must reach its declared end_vtx.
      //   If the fan walk died at a dead-end, the last element won't be loc_e0/loc_e1.
      //   A mis-terminated walk produces a bdLoop where vi and vj are not adjacent,
      //   causing vk==-1 assertion at joint_lscm.cpp:924.
      bool inv_F = inv_A && (nsv_last == loc_e0) && (ndv_last == loc_e1);

      // INV-G: neither walk may end on the infVtx sentinel (-1).
      //   A trailing -1 means the walk started from infVtx (INF-AS-START) and never
      //   reached the real end_vtx; joint_lscm line 27 would then leave e0/e1 uninit.
      bool inv_G = inv_A && (nsv_last != -1) && (ndv_last != -1);

      // INV-E (informational only): position of -1 in each walk.
      int nsv_inf_pos = -1, ndv_inf_pos = -1;
      for (int k = 0; k < (int)Nsv_local.size(); k++)
        if (Nsv_local[k] == -1) { nsv_inf_pos = k; break; }
      for (int k = 0; k < (int)Ndv_local.size(); k++)
        if (Ndv_local[k] == -1) { ndv_inf_pos = k; break; }

      // INV-H: -1 must not be at position 0 (walk must not start from infVtx).
      //   INF-AS-START: fan_walk picked infVtx as start_vtx because it was the only
      //   count-1 boundary vertex other than end_vtx.  Result starts with [-1,...].
      //   Even if the walk then reaches end_vtx (passes INV-F/G), the bdLoop built
      //   inside joint_lscm has e0/e1 at the wrong position, so vi and vj are not
      //   adjacent → vk==-1 assertion at joint_lscm.cpp:924.
      bool inv_H = (nsv_inf_pos != 0) && (ndv_inf_pos != 0);

      bool all_ok = inv_A && inv_B && inv_C && inv_D && inv_F && inv_G && inv_H;

#ifdef SSP_LSCM_LOG
      static int pre_log = 0;
      if (!all_ok || pre_log < 5) {
        pre_log++;
        fprintf(stderr,
          "[pre-jlscm] sid=%d  e=(%d,%d)  loc_vi=%d loc_vj=%d  loc_e0=%d loc_e1=%d  nV_loc=%d\n"
          "  INV-A=%d  INV-B(nsv_ends_vi=%d ndv_ends_vi=%d)=%d  INV-C=%d  INV-D=%d\n"
          "  INV-F(nsv_last=%d==loc_e0=%d  ndv_last=%d==loc_e1=%d)=%d"
          "  INV-G(no trailing -1)=%d  INV-H(no leading -1: nsv_pos=%d ndv_pos=%d)=%d\n"
          "  Nsv[%zu] inf_at=%d  vi_in=%d vj_in=%d : [",
          sid, E(e,0), E(e,1), loc_vi, loc_vj, loc_e0, loc_e1, nV_loc,
          (int)inv_A, (int)nsv_ends_vi, (int)ndv_ends_vi, (int)inv_B, (int)inv_C, (int)inv_D,
          nsv_last, loc_e0, ndv_last, loc_e1, (int)inv_F,
          (int)inv_G, nsv_inf_pos, ndv_inf_pos, (int)inv_H,
          Nsv_local.size(), nsv_inf_pos, (int)vi_in_nsv, (int)vj_in_nsv);
        for (int x : Nsv_local) fprintf(stderr, "%d,", x);
        fprintf(stderr, "]\n  Ndv[%zu] inf_at=%d  vi_in=%d vj_in=%d : [",
          Ndv_local.size(), ndv_inf_pos, (int)vi_in_ndv, (int)vj_in_ndv);
        for (int x : Ndv_local) fprintf(stderr, "%d,", x);
        fprintf(stderr, "]\n  %s\n",
          all_ok ? "ALL OK" :
          !inv_A ? "FAIL INV-A: empty walk" :
          !inv_G ? "FAIL INV-G: walk ended at infVtx sentinel (-1)" :
          !inv_H ? "FAIL INV-H: walk started at infVtx (-1 at pos 0) — INF-AS-START" :
          !inv_F ? "FAIL INV-F: walk did not reach declared end_vtx — dead-end in fan" :
          !inv_B ? "FAIL INV-B: neither Nsv nor Ndv ends with loc_vi — e0/e1 uninit" :
          !inv_C ? "FAIL INV-C: local index out of range" :
                   "FAIL INV-D: loc_vi or loc_vj absent from Nsv+Ndv");
      }
#endif

      if (!all_ok) {
        if (is_seam_collapse)
          SEAM_LOG("[SEAM-SKIP-INV]  sid=%d  e=(%d,%d)  fail=%s\n",
            sid, E(e,0), E(e,1),
            !inv_A ? "INV-A(empty-walk)" :
            !inv_G ? "INV-G(walk-ends-at-inf)" :
            !inv_H ? "INV-H(inf-as-start)" :
            !inv_F ? "INV-F(walk-dead-end)" :
            !inv_B ? "INV-B(e0/e1-uninit)" :
            !inv_C ? "INV-C(idx-out-of-range)" :
                     "INV-D(vi/vj-absent)");
        continue;
      }
    }
    // ---- end pre-joint_lscm invariant check ----

    // Seam edge: the collapsed edge borders multiple sheets.  The faces from other
    // sheets are excluded from this sheet's local patch, so each endpoint's fan is
    // naturally open at the seam.  Inject -1 (infVtx sentinel) at position 0 of
    // each walk so joint_lscm treats both vi and vj as boundary vertices
    // (onBd.sum()==2 → Case 2).  Injection is after the invariant check so INV-H
    // (which rejects -1 at position 0 as an INF-AS-START bug) does not fire.
    if (active_sheets.size() > 1) {
      if (std::find(Nsv_local.begin(), Nsv_local.end(), -1) == Nsv_local.end())
        Nsv_local.insert(Nsv_local.begin(), -1);
      if (std::find(Ndv_local.begin(), Ndv_local.end(), -1) == Ndv_local.end())
        Ndv_local.insert(Ndv_local.begin(), -1);
    }

    // joint_lscm (DO NOT MODIFY — uses infVtx=-1 sentinel and winding-order Nsv/Ndv)
    MatrixXd UV_pre_si, UV_post_si;
    bool isValid = joint_lscm(
        V_pre_si, FUV_pre_si, V_post_si, FUV_post_si,
        b_si(0), b_si(1), Nsv_local, Ndv_local,
        UV_pre_si, UV_post_si,
        any_sheet_ok ? nullptr : &data.lscm_case);
    if (!isValid) {
      if (is_seam_collapse)
        SEAM_LOG("[SEAM-FAIL-LSCM]  sid=%d  e=(%d,%d) vi=%d vj=%d\n",
          sid, E(e,0), E(e,1), vi, vj);
#ifdef SSP_LSCM_LOG
      fprintf(stderr,
        "[LSCM-FAIL] after_collapse=%zu  e=(%d,%d)  sid=%d  vi=%d vj=%d\n",
        decInfo.size(), E(e,0), E(e,1), sid, vi, vj);
#endif
      return false;
    }

    // ── Paper §4: UV Face Flips (Neural Subdivision, Hsueh-Ti et al. 2020) ──
    // Reject if any post-collapse UV face has non-positive signed area.
    {
      bool uv_flip = false;
      for (int fi = 0; fi < FUV_post_si.rows() && !uv_flip; ++fi) {
        const int ua = FUV_post_si(fi,0), ub = FUV_post_si(fi,1), uc = FUV_post_si(fi,2);
        const double ax = UV_post_si(ua,0), ay = UV_post_si(ua,1);
        const double bx = UV_post_si(ub,0), by = UV_post_si(ub,1);
        const double cx = UV_post_si(uc,0), cy = UV_post_si(uc,1);
        if ((bx-ax)*(cy-ay) - (cx-ax)*(by-ay) <= 0.0) uv_flip = true;
      }
      if (uv_flip) {
        fprintf(stderr, "[UV-REJECT] uv_face_flip  sid=%d  e=(%d,%d)\n",
                sid, E(e,0), E(e,1));
        if (FILE* lf = fopen("collapse_rejections_log.txt", "a"))
        { fprintf(lf, "[UV-REJECT] uv_face_flip  sid=%d  e=(%d,%d)\n", sid, E(e,0), E(e,1)); fclose(lf); }
        return false;
      }
    }

    // ── Paper §4: Overlapped UV Faces (Neural Subdivision, Hsueh-Ti et al. 2020) ──
    // For each interior vertex of the post-collapse UV patch, the total angle
    // sum around it must equal 2π (discrete flatness — no UV overlaps).
    {
      const int nUV  = (int)UV_post_si.rows();
      const int nFuv = (int)FUV_post_si.rows();

      // Boundary vertices: on edges that appear exactly once
      map<pair<int,int>, int> uv_edge_cnt;
      for (int fi = 0; fi < nFuv; ++fi)
        for (int c = 0; c < 3; ++c) {
          int a = FUV_post_si(fi,c), b = FUV_post_si(fi,(c+1)%3);
          if (a > b) std::swap(a,b);
          uv_edge_cnt[{a,b}]++;
        }
      set<int> uv_bd;
      for (auto & kv : uv_edge_cnt)
        if (kv.second == 1) { uv_bd.insert(kv.first.first); uv_bd.insert(kv.first.second); }

      bool bad_angle = false;
      int  bad_angle_v = -1;
      double bad_angle_sum = 0.0;
      static constexpr double kTwoPi = 6.283185307179586;
      int n_interior = 0;
      for (int v = 0; v < nUV && !bad_angle; ++v) {
        if (uv_bd.count(v)) continue;  // skip boundary vertices
        ++n_interior;
        double angle_sum = 0.0;
        bool degenerate = false;
        for (int fi = 0; fi < nFuv; ++fi) {
          int vb_idx = -1, vc_idx = -1;
          for (int c = 0; c < 3; ++c) {
            if (FUV_post_si(fi,c) == v) {
              vb_idx = FUV_post_si(fi,(c+1)%3);
              vc_idx = FUV_post_si(fi,(c+2)%3);
              break;
            }
          }
          if (vb_idx < 0) continue;
          Eigen::Vector2d pa(UV_post_si(v,0),      UV_post_si(v,1));
          Eigen::Vector2d pb(UV_post_si(vb_idx,0), UV_post_si(vb_idx,1));
          Eigen::Vector2d pc(UV_post_si(vc_idx,0), UV_post_si(vc_idx,1));
          Eigen::Vector2d e1 = pb - pa, e2 = pc - pa;
          double l1 = e1.norm(), l2 = e2.norm();
          if (l1 < 1e-15 || l2 < 1e-15) { degenerate = true; bad_angle = true; bad_angle_v = v; break; }
          double cos_a = std::max(-1.0, std::min(1.0, e1.dot(e2) / (l1*l2)));
          angle_sum += std::acos(cos_a);
        }
        if (!degenerate && std::abs(angle_sum - kTwoPi) > 1e-15)
        { bad_angle = true; bad_angle_v = v; bad_angle_sum = angle_sum; }
      }
      if (bad_angle) {
        static int s_angle_rej = 0;
        ++s_angle_rej;
        if (s_angle_rej <= 50) {
          fprintf(stderr,
            "[UV-REJECT] uv_angle_sum  sid=%d  e=(%d,%d)"
            "  bad_v=%d  angle_sum=%.6f  diff_from_2pi=%.6f"
            "  n_interior=%d  nUV=%d  nFuv=%d  (reject#%d)\n",
            sid, E(e,0), E(e,1),
            bad_angle_v, bad_angle_sum, bad_angle_sum - kTwoPi,
            n_interior, nUV, nFuv, s_angle_rej);
          if (FILE* lf = fopen("collapse_rejections_log.txt", "a")) {
            fprintf(lf,
              "[UV-REJECT] uv_angle_sum  sid=%d  e=(%d,%d)"
              "  bad_v=%d  angle_sum=%.6f  diff_from_2pi=%.6f"
              "  n_interior=%d  nUV=%d  nFuv=%d\n",
              sid, E(e,0), E(e,1),
              bad_angle_v, bad_angle_sum, bad_angle_sum - kTwoPi,
              n_interior, nUV, nFuv);
            fclose(lf);
          }
        }
        return false;
      }
    }

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

  if (!any_sheet_ok) {
    if (is_seam_collapse)
      SEAM_LOG("[SEAM-FAIL-ALL-SHEETS]  e=(%d,%d) vi=%d vj=%d  active_sheets=%zu\n",
        E(e,0), E(e,1), vi, vj, active_sheets.size());
#ifdef SSP_LSCM_LOG
    static int saf_count = 0;
    static int saf_total = 0;
    saf_total++;
    if (saf_count < 40) {
      saf_count++;
      const vector<int> & dfaces = (!eflip ? Nsf : Ndf);
      fprintf(stderr,
        "[SHEET-ALL-FAILED] reject#%d(total=%d)  seq=%zu  e=(%d,%d) s=%d d=%d  "
        "active_sheets=%zu  real_faces_of_d=[",
        saf_count, saf_total, decInfo.size(),
        E(e,0), E(e,1), s, d, active_sheets.size());
      for (int f : dfaces)
        if (!null_face(f) && f < numOrigFaces) fprintf(stderr, "%d ", f);
      fprintf(stderr, "]\n");
    }
#endif
    return false;
  }

  // Build combined FIdx_onering_pre (sorted union of all sheets)
  FIdx_onering_pre.resize((int)FIdx_combined.size());
  { int fi = 0; for (int f : FIdx_combined) FIdx_onering_pre(fi++) = f; }

  // [NON-ACTIVE-SHEET] Log real faces of d that Pass 2 will remap (d→s in gF)
  // but that are NOT in any active sheet's one-ring → no decIM entry will be written.
  // These faces will have stale gF entries when the backward walk queries them.
  {
    const vector<int> & nV2Fd_check = (!eflip ? Nsf : Ndf);
    int nas_count = 0;
#ifdef SSP_LSCM_LOG
    static int nas_log = 0;
#endif
    for (int f : nV2Fd_check) {
      if (null_face(f) || f >= numOrigFaces) continue;
      if (FIdx_combined.find(f) == FIdx_combined.end()) {
        nas_count++;

        NonActiveSheetFace naf;
        naf.face_idx  = f;
        naf.sheet_id  = (int)faceSheetID(f);
        for (int c = 0; c < 3; c++)
          if (std::isinf(V(F(f,c), 0))) { naf.is_infinity_face = true; break; }
        naf.p0 = V.row(F(f,0)).head<3>().transpose();
        naf.p1 = V.row(F(f,1)).head<3>().transpose();
        naf.p2 = V.row(F(f,2)).head<3>().transpose();
        data.non_active_faces.push_back(naf);

#ifdef SSP_LSCM_LOG
        if (nas_log < 30) {
          nas_log++;
          if (naf.is_infinity_face) {
            fprintf(stderr,
              "[NON-ACTIVE-SHEET-FACE] collapse=%zu  face=%d  sheet=%d"
              "  → infinity face, not a geometry sheet\n",
              decInfo.size(), f, naf.sheet_id);
          } else {
            fprintf(stderr,
              "[NON-ACTIVE-SHEET-FACE] collapse=%zu  face=%d  sheet=%d  "
              "d=%d s=%d  active_sheets=[",
              decInfo.size(), f, naf.sheet_id, d, s);
            for (int sid : active_sheets) fprintf(stderr, "%d ", sid);
            fprintf(stderr, "]  → gF will get d→s remap but NO decIM entry\n");
          }
        }
#endif
      }
    }
#ifdef SSP_LSCM_LOG
    if (nas_count > 0) {
      static int nas_summary = 0;
      if (nas_summary < 10) {
        nas_summary++;
        fprintf(stderr,
          "[NON-ACTIVE-SHEET-SUMMARY] collapse=%zu  %d face(s) remapped without decIM entry\n",
          decInfo.size(), nas_count);
      }
    }
#endif
  }

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

#ifdef SSP_LSCM_LOG
    static int co_log = 0;
    if (inf_count > 0 && co_log < 5) {
      co_log++;
      fprintf(stderr, "[collect_onering] v=%d  faces=%zu  inf=%d\n",
              v, faces.size(), inf_count);
    }
#endif
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

  if ((int)Nsv_verts.size() < 2 || (int)Ndv_verts.size() < 2) {
#ifdef SSP_LSCM_LOG
    static int bogus = 0;
    if (bogus < 5) {
      bogus++;
      fprintf(stderr, "[BOGUS ONE-RING] e=%d E=(%d,%d)  Nsv=%zu  Ndv=%zu\n",
              e, E(e,0), E(e,1), Nsv_verts.size(), Ndv_verts.size());
    }
#endif
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
