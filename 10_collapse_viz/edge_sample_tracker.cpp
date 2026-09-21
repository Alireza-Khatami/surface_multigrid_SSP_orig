#include "edge_sample_tracker.h"
#include "load_matstruct_edges.h"

#include <single_collapse_data.h>
#include <compute_barycentric.h>

#ifdef C2F_VIZ_DIAGNOSTIC
#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>
#endif

#include <Eigen/Dense>
#include <fstream>
#include <cstdio>
#include <random>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <vector>
#include <utility>
#include <cstdint>

using namespace Eigen;

// ---- globals from main.cpp ----
extern MatrixXd gV;
extern MatrixXd gVO;
extern MatrixXi gF;
extern MatrixXi gFO;
extern std::vector<single_collapse_data> gDecInfo;

// ---- sample state (fully separate from face_sample_tracker's gSamples) ----

struct EdgeSample {
    int         id;
    int         type_id;      // 1 = seam, 2 = boundary (matches .ma_struct convention)
    int         struct_id;
    int         src_v0, src_v1; // original fine-mesh (gVO) edge endpoints
    double      t;             // static: param in [0,1] along (src_v0 -> src_v1), never updated
    int         seed_face_id;  // gFO face this sample was seeded on (disambiguates multi-sheet seams)
    // Frozen at seeding time (fine mesh):
    int         fine_face_id;
    RowVector3d fine_BC;
    // Live tracking (evolves with each collapse):
    int         cur_FIdx;
    RowVector3d cur_BC;
    RowVector3i cur_BF;
};

static std::vector<EdgeSample>                  gEdgeSamples;
// Reverse map: cur_FIdx -> list of indices into gEdgeSamples currently on that face.
static std::unordered_map<int,std::vector<int>> gEdgeSamplesByFace;

static void es_insert(int fi, int si) { gEdgeSamplesByFace[fi].push_back(si); }

static void es_remove(int fi, int si)
{
    auto it = gEdgeSamplesByFace.find(fi);
    if (it == gEdgeSamplesByFace.end()) return;
    auto& v = it->second;
    v.erase(std::remove(v.begin(), v.end(), si), v.end());
    if (v.empty()) gEdgeSamplesByFace.erase(it);
}

// Unordered vertex-pair key for an edge -> incident face lookup.
static inline int64_t edge_key(int a, int b)
{
    if (a > b) std::swap(a, b);
    return (int64_t)a << 32 | (uint32_t)b;
}

// ---- seeding helpers ----

// Find the column c in gFO.row(face) such that gFO(face,c) == vid. -1 if not found.
static int face_col_of(int face, int vid)
{
    for (int c = 0; c < 3; ++c)
        if (gFO(face, c) == vid) return c;
    return -1;
}

void edge_sample_tracker_init(const std::string& matstruct_path, int samples_per_struct)
{
    gEdgeSamples.clear();
    gEdgeSamplesByFace.clear();

    std::vector<MatStructEdge> boundary_edges, seam_edges;
    if (!load_matstruct_edges(matstruct_path, boundary_edges, seam_edges)) {
        fprintf(stderr, "[edge_sample_tracker] load_matstruct_edges failed for '%s' — no edge samples seeded\n",
                matstruct_path.c_str());
        return;
    }

    // Build edge (v0,v1) -> list of incident gFO faces. Naturally captures
    // non-manifold multiplicity (3+ faces sharing an edge at a junction).
    std::unordered_map<int64_t, std::vector<int>> edgeFaces;
    const int nFO = gFO.rows();
    for (int f = 0; f < nFO; ++f) {
        for (int c = 0; c < 3; ++c) {
            int a = gFO(f, c), b = gFO(f, (c + 1) % 3);
            edgeFaces[edge_key(a, b)].push_back(f);
        }
    }

    // Group edges by (type_id, struct_id) so each seam/boundary curve gets
    // its own arc-length CDF and its own samples_per_struct draws.
    struct StructKey { int type_id; int struct_id; bool operator<(const StructKey& o) const {
        return type_id != o.type_id ? type_id < o.type_id : struct_id < o.struct_id; } };
    std::map<StructKey, std::vector<MatStructEdge>> byStruct;
    for (const auto& e : boundary_edges) byStruct[{2, e.struct_id}].push_back(e);
    for (const auto& e : seam_edges)     byStruct[{1, e.struct_id}].push_back(e);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    int id = 0;
    int n_edges_missing_face = 0;

    for (const auto& kv : byStruct) {
        const int type_id   = kv.first.type_id;
        const int struct_id = kv.first.struct_id;
        const std::vector<MatStructEdge>& edges = kv.second;
        if (edges.empty()) continue;

        // Arc-length CDF over this struct's edges.
        std::vector<double> len_cdf(edges.size());
        for (size_t i = 0; i < edges.size(); ++i) {
            Vector3d p0 = gVO.row(edges[i].v0);
            Vector3d p1 = gVO.row(edges[i].v1);
            len_cdf[i] = (p1 - p0).norm();
            if (i > 0) len_cdf[i] += len_cdf[i - 1];
        }
        const double total_len = len_cdf.back();
        if (total_len <= 0.0) continue;

        for (int k = 0; k < samples_per_struct; ++k) {
            double q = U(rng) * total_len;
            int ei = (int)(std::lower_bound(len_cdf.begin(), len_cdf.end(), q) - len_cdf.begin());
            if (ei >= (int)edges.size()) ei = (int)edges.size() - 1;

            const MatStructEdge& me = edges[ei];
            const double t = U(rng); // uniform position along this edge, param src_v0 -> src_v1

            auto it = edgeFaces.find(edge_key(me.v0, me.v1));
            if (it == edgeFaces.end() || it->second.empty()) {
                ++n_edges_missing_face;
                continue;
            }

            // NOTE: previously seeded one sample per incident face (one per
            // sheet touching this seam edge, so 3+ at non-manifold junction
            // edges) to independently verify every sheet's UV tracking.
            // Disabled per request — joint_lscm_seam_pinned already forces
            // vi/vj/vk to a shared UV target across sheets for each seam
            // collapse (joint_lscm_pinned.cpp:274-298), so we now trust that
            // and only seed ONE sample per point (first incident face found),
            // to keep sample counts/collapse-time cost manageable.
            {
                int face = it->second[0];
                int c0 = face_col_of(face, me.v0);
                int c1 = face_col_of(face, me.v1);
                if (c0 >= 0 && c1 >= 0) {
                    RowVector3d bc = RowVector3d::Zero();
                    bc(c0) = 1.0 - t;
                    bc(c1) = t;
                    RowVector3i bf;
                    bf << gFO(face,0), gFO(face,1), gFO(face,2);

                    EdgeSample s;
                    s.id            = id++;
                    s.type_id       = type_id;
                    s.struct_id     = struct_id;
                    s.src_v0        = me.v0;
                    s.src_v1        = me.v1;
                    s.t             = t;
                    s.seed_face_id  = face;
                    s.fine_face_id  = face;
                    s.fine_BC       = bc;
                    s.cur_FIdx      = face;
                    s.cur_BC        = bc;
                    s.cur_BF        = bf;

                    es_insert(face, (int)gEdgeSamples.size());
                    gEdgeSamples.push_back(s);
                }
            }
        }
    }

    if (n_edges_missing_face > 0)
        fprintf(stderr, "[edge_sample_tracker] WARNING: %d edge draws had no incident gFO face (skipped)\n",
                n_edges_missing_face);

    fprintf(stderr,
        "[edge_sample_tracker] init: %d samples  (%zu boundary edges, %zu seam edges, %zu structs)  %s\n",
        id, boundary_edges.size(), seam_edges.size(), byStruct.size(), matstruct_path.c_str());
}

void edge_sample_tracker_update()
{
    if (gDecInfo.empty() || gEdgeSamples.empty()) return;
    const single_collapse_data& d = gDecInfo.back();

    for (int shi = 0; shi < (int)d.sheets.size(); shi++) {
        const SheetData& sd = d.sheets[shi];
        if (sd.FIdx_pre.size() == 0 || sd.FIdx_post.size() == 0) continue;

        std::unordered_map<int,int> preFaceRow;
        preFaceRow.reserve(sd.FIdx_pre.size());
        for (int r = 0; r < (int)sd.FIdx_pre.size(); r++)
            preFaceRow[sd.FIdx_pre(r)] = r;

        std::vector<int> to_remap;
        for (auto& [face, row] : preFaceRow) {
            auto it = gEdgeSamplesByFace.find(face);
            if (it == gEdgeSamplesByFace.end()) continue;
            for (int si : it->second) to_remap.push_back(si);
        }

        for (int si : to_remap) {
            EdgeSample& s = gEdgeSamples[si];
            auto rit = preFaceRow.find(s.cur_FIdx);
            if (rit == preFaceRow.end()) continue;
            int pre_row = rit->second;

            // ---- UV CAST (mirrors sample_tracker_update() in face_sample_tracker.cpp) ----
            int lv0 = sd.FUV_pre(pre_row, 0);
            int lv1 = sd.FUV_pre(pre_row, 1);
            int lv2 = sd.FUV_pre(pre_row, 2);
            Vector2d queryUV =
                s.cur_BC(0) * sd.UV_pre.row(lv0).transpose()
              + s.cur_BC(1) * sd.UV_pre.row(lv1).transpose()
              + s.cur_BC(2) * sd.UV_pre.row(lv2).transpose();

            MatrixXd B;
            compute_barycentric(queryUV, sd.UV_post, sd.FUV_post, B);

            VectorXd d2v = -B.rowwise().minCoeff();
            int best = 0; double minD = 1.0;
            for (int bb = 0; bb < (int)d2v.size(); bb++)
                if (d2v(bb) < minD) { minD = d2v(bb); best = bb; }

            for (int c = 0; c < 3; c++) B(best,c) = std::max(0.0, B(best,c));
            double bsum = B.row(best).sum();
            if (bsum > 1e-12) B.row(best) /= bsum;

            int new_FIdx = sd.FIdx_post(best);
            RowVector3d new_BC = B.row(best);
            RowVector3i new_BF;
            new_BF << sd.subsetVIdx(sd.FUV_post(best,0)),
                      sd.subsetVIdx(sd.FUV_post(best,1)),
                      sd.subsetVIdx(sd.FUV_post(best,2));

            es_remove(s.cur_FIdx, si);
            s.cur_FIdx = new_FIdx;
            s.cur_BC   = new_BC;
            s.cur_BF   = new_BF;
            es_insert(new_FIdx, si);
        }
    }

    // Vertex fixup: patch d->s in cur_BF, same as sample_tracker_update().
    for (const SheetData& sd : d.sheets) {
        if (sd.b.size() < 2) continue;
        int global_d = sd.subsetVIdx(sd.b(1));
        int global_s = sd.subsetVIdx(sd.b(0));
        for (EdgeSample& s : gEdgeSamples)
            for (int c = 0; c < 3; c++)
                if (s.cur_BF(c) == global_d) s.cur_BF(c) = global_s;
    }
}

void edge_sample_tracker_save(const std::string& path)
{
    if (gEdgeSamples.empty()) {
        fprintf(stderr, "[edge_sample_tracker] no samples to save\n");
        return;
    }

    std::ofstream f(path);
    if (!f) {
        fprintf(stderr, "[edge_sample_tracker] cannot write %s\n", path.c_str());
        return;
    }

    f << "# id type_id struct_id src_v0 src_v1 t seed_face_id cur_FIdx b0 b1 b2 bv0 bv1 bv2\n";
    f << gEdgeSamples.size() << "\n";
    for (const EdgeSample& s : gEdgeSamples) {
        f << s.id << " " << s.type_id << " " << s.struct_id
          << " " << s.src_v0 << " " << s.src_v1 << " " << s.t
          << " " << s.seed_face_id << " " << s.cur_FIdx
          << " " << s.cur_BC(0) << " " << s.cur_BC(1) << " " << s.cur_BC(2)
          << " " << s.cur_BF(0) << " " << s.cur_BF(1) << " " << s.cur_BF(2) << "\n";
    }
    fprintf(stderr, "[edge_sample_tracker] wrote %zu edge samples -> %s\n",
            gEdgeSamples.size(), path.c_str());
}

void edge_sample_tracker_show()
{
#ifdef C2F_VIZ_DIAGNOSTIC
    if (gEdgeSamples.empty()) return;

    std::vector<const EdgeSample*> boundary, seam;
    for (const EdgeSample& s : gEdgeSamples)
        (s.type_id == 2 ? boundary : seam).push_back(&s);

    auto build_pts = [](const std::vector<const EdgeSample*>& v) {
        MatrixXd pts((int)v.size(), 3);
        for (int i = 0; i < (int)v.size(); ++i) {
            const EdgeSample& s = *v[i];
            pts.row(i) =
                s.cur_BC(0) * gV.row(s.cur_BF(0)).leftCols(3)
              + s.cur_BC(1) * gV.row(s.cur_BF(1)).leftCols(3)
              + s.cur_BC(2) * gV.row(s.cur_BF(2)).leftCols(3);
        }
        return pts;
    };

    if (!boundary.empty()) {
        MatrixXd pts = build_pts(boundary);
        polyscope::registerPointCloud("edge_sample_boundary_pts", pts)
            ->setPointColor({0.86f, 0.08f, 0.24f})  // crimson, matches simp_viz MS_Boundary
            ->setPointRadius(0.0015, true)
            ->setEnabled(true);
    }
    if (!seam.empty()) {
        MatrixXd pts = build_pts(seam);
        polyscope::registerPointCloud("edge_sample_seam_pts", pts)
            ->setPointColor({1.0f, 0.65f, 0.0f})    // orange, matches simp_viz MS_Seam
            ->setPointRadius(0.0015, true)
            ->setEnabled(true);
    }
#endif
}
