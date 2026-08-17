#include "face_sample_tracker.h"

#include <single_collapse_data.h>
#include <compute_barycentric.h>
#include "face_dead.h"

#ifdef C2F_VIZ_DIAGNOSTIC
#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>
#endif

#include <igl/writeOBJ.h>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>

using namespace Eigen;

// ---- globals from main.cpp ----
extern MatrixXd gV;
extern MatrixXd gVO;
extern MatrixXi gF;
extern MatrixXi gFO;
extern std::vector<single_collapse_data> gDecInfo;

// ---- trace configuration ----
static std::unordered_set<int> gTraceVIDSet;   // vertex IDs to trace; empty = trace all
static std::ofstream           gTraceFile;
static int                     gCollapseIdx = 0;

// ---- sample state ----

struct Sample {
    int         id;
    bool        is_vertex;
    int         fine_vertex_id; // gVO vertex index for vertex samples; -1 for interior
    // Frozen at seeding time (fine mesh):
    int         fine_face_id;  // gFO face index
    RowVector3d fine_BC;
    // Live tracking (evolves with each collapse):
    int         cur_FIdx;      // current face index in gF / gDecIM
    RowVector3d cur_BC;        // barycentric coords in cur_FIdx
    RowVector3i cur_BF;        // global vertex indices of cur_FIdx's corners
    // Current live global vertex index this sample tracks (starts == fine_vertex_id,
    // updated to s whenever fine_vertex_id's vertex is absorbed as d in a collapse).
    // Used to snap cur_BC back to a one-hot after UV remap, preventing drift.
    int         cur_vertex_id;
};

static std::vector<Sample>                      gSamples;
// Reverse map: cur_FIdx → list of sample indices into gSamples
static std::unordered_map<int,std::vector<int>> gFaceSamples;

// Per-remap record captured during sample_tracker_update().
// Stores the UV face row + BC both before and after each collapse so that
// sample_tracker_show_canonical() can interpolate positions directly on the
// canonical-frame UV meshes without re-deriving geometry.
struct RingRemapEntry {
    int         pre_face_row;   // row in this sheet's FUV_pre
    RowVector3d pre_bc;
    int         post_face_row;  // row in this sheet's FUV_post (= best)
    RowVector3d post_bc;
    bool        on_snap_sheet;  // true when from sheets[0] (= the sheet shown in gSnap/uv_pre_3d)
};
static std::vector<RingRemapEntry> gLastRingRemap;

static void fs_insert(int fi, int si) { gFaceSamples[fi].push_back(si); }

static void fs_remove(int fi, int si)
{
    auto it = gFaceSamples.find(fi);
    if (it == gFaceSamples.end()) return;
    auto& v = it->second;
    v.erase(std::remove(v.begin(), v.end(), si), v.end());
    if (v.empty()) gFaceSamples.erase(it);
}

// ---- public API ----

void sample_tracker_set_trace(const std::string& vertex_list_path,
                               const std::string& trace_output_path)
{
    gTraceVIDSet.clear();
    gCollapseIdx = 0;

    std::ifstream vf(vertex_list_path);
    if (!vf) {
        fprintf(stderr, "[sample_tracker] WARNING: cannot open vertex list '%s' — tracing all vertices\n",
                vertex_list_path.c_str());
    } else {
        int v;
        while (vf >> v) gTraceVIDSet.insert(v);
        fprintf(stderr, "[sample_tracker] trace: %zu vertices from '%s'\n",
                gTraceVIDSet.size(), vertex_list_path.c_str());
    }

    if (gTraceFile.is_open()) gTraceFile.close();
    gTraceFile.open(trace_output_path);
    if (!gTraceFile)
        fprintf(stderr, "[sample_tracker] WARNING: cannot open trace file '%s'\n",
                trace_output_path.c_str());
    else {
        gTraceFile << "# ci  vi  sheet  face_pre  bc_pre(0,1,2)  vid_pre  face_post  bc_post(0,1,2)  vid_post  method\n";
        fprintf(stderr, "[sample_tracker] trace output → '%s'\n", trace_output_path.c_str());
    }
}

void sample_tracker_init(int n_total)
{
    gSamples.clear();
    gFaceSamples.clear();

    const int nFO = gFO.rows();
    const int nVO = (int)gVO.rows();

    // Build vertex → first incident gFO face
    std::vector<int> vtxFace(nVO, -1);
    for (int f = 0; f < nFO; f++)
        for (int c = 0; c < 3; c++) {
            int v = gFO(f, c);
            if (v >= 0 && v < nVO && vtxFace[v] < 0) vtxFace[v] = f;
        }

    int id = 0;

    // --- vertex samples ---
    // Seed ALL fine-mesh vertices. gTraceVIDSet controls trace-file output only, not seeding.
    for (int vi = 0; vi < nVO; vi++) {
        int fi = vtxFace[vi];
        if (fi < 0) continue;
        int col = -1;
        for (int c = 0; c < 3; c++) if (gFO(fi, c) == vi) { col = c; break; }
        if (col < 0) continue;

        RowVector3d bc = RowVector3d::Zero();
        bc(col) = 1.0;
        RowVector3i bf;
        bf << gFO(fi,0), gFO(fi,1), gFO(fi,2);

        Sample s;
        s.id             = id++;
        s.is_vertex      = true;
        s.fine_vertex_id = vi;
        s.fine_face_id   = fi;
        s.fine_BC        = bc;
        s.cur_FIdx       = fi;
        s.cur_BC         = bc;
        s.cur_BF         = bf;
        s.cur_vertex_id  = vi;
        fs_insert(fi, (int)gSamples.size());
        gSamples.push_back(s);
    }

    // INTERIOR_SAMPLES: disabled while debugging vertex tracking.
    // TODO: uncomment the block below once vertex correspondence is validated,
    //       to restore area-weighted interior sample tracking.
    /*
    // --- interior barycentric samples (area-weighted) ---
    // Build CDF over face areas so larger faces receive proportionally more samples.
    std::vector<double> area_cdf(nFO);
    for (int fi = 0; fi < nFO; fi++) {
        Vector3d v0 = gVO.row(gFO(fi, 0));
        Vector3d v1 = gVO.row(gFO(fi, 1));
        Vector3d v2 = gVO.row(gFO(fi, 2));
        area_cdf[fi] = 0.5 * (v1 - v0).cross(v2 - v0).norm();
        if (fi > 0) area_cdf[fi] += area_cdf[fi - 1];
    }
    const double total_area = area_cdf.back();

    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    for (int k = 0; k < n_total; k++) {
        // Pick a face proportional to area via binary search on the CDF.
        double t = U(rng) * total_area;
        int fi = (int)(std::lower_bound(area_cdf.begin(), area_cdf.end(), t) - area_cdf.begin());
        if (fi >= nFO) fi = nFO - 1;

        RowVector3i bf;
        bf << gFO(fi,0), gFO(fi,1), gFO(fi,2);

        // Uniform sampling inside the triangle (Osada et al.).
        double r1 = U(rng), r2 = U(rng);
        if (r1 + r2 > 1.0) { r1 = 1.0 - r1; r2 = 1.0 - r2; }
        RowVector3d bc;
        bc << r1, r2, 1.0 - r1 - r2;

        Sample s;
        s.id             = id++;
        s.is_vertex      = false;
        s.fine_vertex_id = -1;
        s.fine_face_id   = fi;
        s.fine_BC        = bc;
        s.cur_FIdx       = fi;
        s.cur_BC         = bc;
        s.cur_BF         = bf;
        s.cur_vertex_id  = -1;
        fs_insert(fi, (int)gSamples.size());
        gSamples.push_back(s);
    }
    */
    const int n_interior = 0; // interior samples disabled; was n_total

    fprintf(stderr,
        "[sample_tracker] init: %d samples (%d vertex + %d interior)  nFO=%d  nVO=%d  trace_set=%zu\n",
        id, id, n_interior, nFO, nVO, gTraceVIDSet.size());
}

void sample_tracker_update()
{
    if (gDecInfo.empty() || gSamples.empty()) return;
    const single_collapse_data& d = gDecInfo.back();
    const int ci = gCollapseIdx++;

    // Rebuild per-collapse remap record for canonical-view display.
    gLastRingRemap.clear();

    for (int shi = 0; shi < (int)d.sheets.size(); shi++) {
        const SheetData& sd = d.sheets[shi];
        if (sd.FIdx_pre.size() == 0 || sd.FIdx_post.size() == 0) continue;
        const bool on_snap = (shi == 0); // sheets[0] is the sheet gSnap/uv_pre_3d represents

        // Absorbed / survivor global vertex indices for this sheet (requires sd.b from v4 bundle).
        const int global_d = (sd.b.size() >= 2) ? (int)sd.subsetVIdx(sd.b(1)) : -1;
        const int global_s = (sd.b.size() >= 2) ? (int)sd.subsetVIdx(sd.b(0)) : -1;

        // Build pre-face → row index for O(1) lookup
        std::unordered_map<int,int> preFaceRow;
        preFaceRow.reserve(sd.FIdx_pre.size());
        for (int r = 0; r < (int)sd.FIdx_pre.size(); r++)
            preFaceRow[sd.FIdx_pre(r)] = r;

        // Gather samples on pre-collapse faces of this sheet
        std::vector<int> to_remap;
        for (auto& [face, row] : preFaceRow) {
            auto it = gFaceSamples.find(face);
            if (it == gFaceSamples.end()) continue;
            for (int si : it->second) to_remap.push_back(si);
        }

        for (int si : to_remap) {
            Sample& s = gSamples[si];
            auto rit = preFaceRow.find(s.cur_FIdx);
            if (rit == preFaceRow.end()) continue;
            int pre_row = rit->second;

            const bool should_trace = gTraceFile.is_open() && s.is_vertex &&
                (gTraceVIDSet.empty() || gTraceVIDSet.count(s.fine_vertex_id));

            // ---- UV CAST (always; no SNAP special-case — see snap_vs_uv_cast.md) ----
            // argmax(-B.rowwise().minCoeff()) picks the best face even when the query
            // lands in the deleted-triangle region of UV_post (other faces expand to cover it).
            // Express sample position in UV_pre (uses this sheet's UV, not a global assumption)
            int lv0 = sd.FUV_pre(pre_row, 0);
            int lv1 = sd.FUV_pre(pre_row, 1);
            int lv2 = sd.FUV_pre(pre_row, 2);
            Vector2d queryUV =
                s.cur_BC(0) * sd.UV_pre.row(lv0).transpose()
              + s.cur_BC(1) * sd.UV_pre.row(lv1).transpose()
              + s.cur_BC(2) * sd.UV_pre.row(lv2).transpose();

            // Find corresponding position in UV_post (same sheet)
            MatrixXd B;
            compute_barycentric(queryUV, sd.UV_post, sd.FUV_post, B);

            // Pick face that minimises the largest negative BC component (first-wins).
            VectorXd d2v = -B.rowwise().minCoeff();
            int best = 0; double minD = 1.0;
            for (int bb = 0; bb < (int)d2v.size(); bb++)
                if (d2v(bb) < minD) { minD = d2v(bb); best = bb; }

            // Clamp negative coords and renormalize
            for (int c = 0; c < 3; c++) B(best,c) = std::max(0.0, B(best,c));
            double bsum = B.row(best).sum();
            if (bsum > 1e-12) B.row(best) /= bsum;

            int new_FIdx = sd.FIdx_post(best);
            RowVector3d new_BC = B.row(best);
            RowVector3i new_BF;
            new_BF << sd.subsetVIdx(sd.FUV_post(best,0)),
                      sd.subsetVIdx(sd.FUV_post(best,1)),
                      sd.subsetVIdx(sd.FUV_post(best,2));

            if (should_trace) {
                gTraceFile
                    << ci << "  " << s.fine_vertex_id << "  " << shi
                    << "  " << s.cur_FIdx
                    << "  " << s.cur_BC(0) << " " << s.cur_BC(1) << " " << s.cur_BC(2)
                    << "  " << s.cur_BF(0) << "/" << s.cur_BF(1) << "/" << s.cur_BF(2)
                    << "  " << new_FIdx
                    << "  " << new_BC(0) << " " << new_BC(1) << " " << new_BC(2)
                    << "  CAST\n";
            }

            // Record UV face row + BC for canonical-view placement on UV surfaces.
            gLastRingRemap.push_back({pre_row, s.cur_BC, best, new_BC, on_snap});

            fs_remove(s.cur_FIdx, si);
            s.cur_FIdx = new_FIdx;
            s.cur_BC   = new_BC;
            s.cur_BF   = new_BF;
            fs_insert(new_FIdx, si);
        }
    }

    // Vertex fixup: after UV remap, cur_BF may still reference the absorbed
    // vertex d (subsetVIdx was captured pre-collapse).  Patch d→s across all
    // samples so gV lookups stay correct.
    for (const SheetData& sd : d.sheets) {
        if (sd.b.size() < 2) continue;
        int global_d = sd.subsetVIdx(sd.b(1)); // absorbed
        int global_s = sd.subsetVIdx(sd.b(0)); // survivor
        for (Sample& s : gSamples)
            for (int c = 0; c < 3; c++)
                if (s.cur_BF(c) == global_d) s.cur_BF(c) = global_s;
    }
}

void sample_tracker_save(const std::string& fine_path,
                         const std::string& coarse_path,
                         const std::string& vertices_path)
{
    if (gSamples.empty()) {
        fprintf(stderr, "[sample_tracker] no samples to save\n");
        return;
    }

    {
        std::ofstream f(fine_path);
        if (!f) {
            fprintf(stderr, "[sample_tracker] cannot write %s\n", fine_path.c_str());
        } else {
            f << "# sample_id is_vertex fine_face_id b0 b1 b2\n";
            f << gSamples.size() << "\n";
            for (const Sample& s : gSamples)
                f << s.id << " " << (s.is_vertex?1:0) << " " << s.fine_face_id
                  << " " << s.fine_BC(0) << " " << s.fine_BC(1) << " " << s.fine_BC(2) << "\n";
            fprintf(stderr, "[sample_tracker] wrote %zu fine samples → %s\n",
                    gSamples.size(), fine_path.c_str());
        }
    }
    {
        std::ofstream f(coarse_path);
        if (!f) {
            fprintf(stderr, "[sample_tracker] cannot write %s\n", coarse_path.c_str());
        } else {
            f << "# sample_id is_vertex coarse_face_id b0 b1 b2 bv0 bv1 bv2\n";
            f << gSamples.size() << "\n";
            for (const Sample& s : gSamples)
                f << s.id << " " << (s.is_vertex?1:0) << " " << s.cur_FIdx
                  << " " << s.cur_BC(0) << " " << s.cur_BC(1) << " " << s.cur_BC(2)
                  << " " << s.cur_BF(0) << " " << s.cur_BF(1) << " " << s.cur_BF(2) << "\n";
            fprintf(stderr, "[sample_tracker] wrote %zu coarse correspondences → %s\n",
                    gSamples.size(), coarse_path.c_str());
        }
    }
    {
        // One row per original fine-mesh vertex: fine_vertex_id + coarse correspondence.
        std::vector<const Sample*> vtx_samples;
        for (const Sample& s : gSamples)
            if (s.is_vertex) vtx_samples.push_back(&s);

        std::ofstream f(vertices_path);
        if (!f) {
            fprintf(stderr, "[sample_tracker] cannot write %s\n", vertices_path.c_str());
        } else {
            f << "# fine_vertex_id coarse_face_id b0 b1 b2 bv0 bv1 bv2\n";
            f << vtx_samples.size() << "\n";
            for (const Sample* s : vtx_samples)
                f << s->fine_vertex_id << " " << s->cur_FIdx
                  << " " << s->cur_BC(0) << " " << s->cur_BC(1) << " " << s->cur_BC(2)
                  << " " << s->cur_BF(0) << " " << s->cur_BF(1) << " " << s->cur_BF(2) << "\n";
            fprintf(stderr, "[sample_tracker] wrote %zu vertex tracks → %s\n",
                    vtx_samples.size(), vertices_path.c_str());
        }
    }
}

void sample_tracker_export_deformed_mesh(const std::string& path)
{
    if (gSamples.empty() || gVO.rows() == 0) {
        fprintf(stderr, "[sample_tracker] no data to export deformed mesh\n");
        return;
    }

    Eigen::MatrixXd deformed = gVO;
    int count = 0;
    for (const Sample& s : gSamples) {
        if (!s.is_vertex) continue;
        Eigen::RowVector3d cp =
            s.cur_BC(0) * gV.row(s.cur_BF(0)).leftCols(3)
          + s.cur_BC(1) * gV.row(s.cur_BF(1)).leftCols(3)
          + s.cur_BC(2) * gV.row(s.cur_BF(2)).leftCols(3);
        deformed.row(s.fine_vertex_id) = cp;
        count++;
    }

    if (!igl::writeOBJ(path, deformed, gFO))
        fprintf(stderr, "[sample_tracker] writeOBJ failed: %s\n", path.c_str());
    else
        fprintf(stderr, "[sample_tracker] deformed fine mesh -> %s  (%d verts deformed, %d total, %d faces)\n",
            path.c_str(), count, (int)deformed.rows(), (int)gFO.rows());
}

// Shared helper: compute fine and coarse 3D positions for all samples.
static void build_sample_pts(MatrixXd& finePts, MatrixXd& coarsePts)
{
    const int nS = (int)gSamples.size();
    finePts.resize(nS, 3);
    coarsePts.resize(nS, 3);
    for (int i = 0; i < nS; i++) {
        const Sample& s = gSamples[i];
        finePts.row(i) =
            s.fine_BC(0) * gVO.row(gFO(s.fine_face_id, 0))
          + s.fine_BC(1) * gVO.row(gFO(s.fine_face_id, 1))
          + s.fine_BC(2) * gVO.row(gFO(s.fine_face_id, 2));
        coarsePts.row(i) =
            s.cur_BC(0) * gV.row(s.cur_BF(0)).leftCols(3)
          + s.cur_BC(1) * gV.row(s.cur_BF(1)).leftCols(3)
          + s.cur_BC(2) * gV.row(s.cur_BF(2)).leftCols(3);
    }
}

void sample_tracker_show()
{
#ifdef C2F_VIZ_DIAGNOSTIC
    if (gSamples.empty()) return;

    MatrixXd finePts, coarsePts;
    build_sample_pts(finePts, coarsePts);

    polyscope::registerPointCloud("sample_fine_pts", finePts)
        ->setPointColor({0.3f, 0.9f, 0.4f})
        ->setPointRadius(0.0005, true)
        ->setEnabled(true);

    polyscope::registerPointCloud("sample_coarse_pts", coarsePts)
        ->setPointColor({0.9f, 0.4f, 0.2f})
        ->setPointRadius(0.0005, true)
        ->setEnabled(true);
#endif
}

void sample_tracker_show_vertices()
{
#ifdef C2F_VIZ_DIAGNOSTIC
    // Collect vertex-only samples
    std::vector<const Sample*> vsamps;
    for (const Sample& s : gSamples)
        if (s.is_vertex) vsamps.push_back(&s);
    if (vsamps.empty()) return;

    const int nV = (int)vsamps.size();
    MatrixXd finePts(nV, 3), coarsePts(nV, 3), arrows(nV, 3);
    for (int i = 0; i < nV; i++) {
        const Sample& s = *vsamps[i];
        // Fine position: directly from original vertex (no BC interpolation needed)
        RowVector3d fp = gVO.row(s.fine_vertex_id);
        // Coarse position: interpolate with live BC + vertex indices
        RowVector3d cp =
            s.cur_BC(0) * gV.row(s.cur_BF(0)).leftCols(3)
          + s.cur_BC(1) * gV.row(s.cur_BF(1)).leftCols(3)
          + s.cur_BC(2) * gV.row(s.cur_BF(2)).leftCols(3);
        finePts.row(i)  = fp;
        coarsePts.row(i) = cp;
        arrows.row(i)   = cp - fp;
    }

    auto* pc_fine = polyscope::registerPointCloud("vtx_fine_pts", finePts);
    pc_fine->setPointColor({0.2f, 0.7f, 1.0f})->setPointRadius(0.0008, true)->setEnabled(true);
    pc_fine->addVectorQuantity("to_coarse", arrows, polyscope::VectorType::AMBIENT)
           ->setVectorColor({1.0f, 0.8f, 0.1f})->setEnabled(true);

    polyscope::registerPointCloud("vtx_coarse_pts", coarsePts)
        ->setPointColor({1.0f, 0.35f, 0.1f})->setPointRadius(0.0008, true)->setEnabled(true);
#endif
}

// ---- face flip tracker ----

static int    gFFT_faceIdx       = -1;
static int    gFFT_sampleIdx[3]  = {-1, -1, -1};
static Eigen::Vector3d gFFT_origNormal;
static Eigen::Vector3d gFFT_posBefore[3];
static std::vector<Eigen::Vector3d> gFFT_traj[3];
static bool   gFFT_flipDetected  = false;
static int    gFFT_flipAtCollapse = -1;

extern int gCollapseCount;

static Eigen::Vector3d fft_sample_pos(const Sample& s)
{
    return s.cur_BC(0) * gV.row(s.cur_BF(0)).transpose()
         + s.cur_BC(1) * gV.row(s.cur_BF(1)).transpose()
         + s.cur_BC(2) * gV.row(s.cur_BF(2)).transpose();
}

void face_flip_tracker_init(int face_idx)
{
    gFFT_faceIdx        = -1;
    gFFT_flipDetected   = false;
    gFFT_flipAtCollapse = -1;
    for (int i = 0; i < 3; i++) { gFFT_sampleIdx[i] = -1; gFFT_traj[i].clear(); }

    if (face_idx < 0 || face_idx >= gFO.rows()) {
        fprintf(stderr, "[face_flip_tracker] face %d out of range\n", face_idx);
        return;
    }
    gFFT_faceIdx = face_idx;

    int verts[3] = { gFO(face_idx,0), gFO(face_idx,1), gFO(face_idx,2) };

    for (int si = 0; si < (int)gSamples.size(); si++) {
        const Sample& s = gSamples[si];
        if (!s.is_vertex) continue;
        for (int i = 0; i < 3; i++)
            if (s.fine_vertex_id == verts[i] && gFFT_sampleIdx[i] < 0)
                gFFT_sampleIdx[i] = si;
    }

    Eigen::Vector3d p0 = gVO.row(verts[0]).transpose();
    Eigen::Vector3d p1 = gVO.row(verts[1]).transpose();
    Eigen::Vector3d p2 = gVO.row(verts[2]).transpose();
    gFFT_origNormal = (p1 - p0).cross(p2 - p0).normalized();

    for (int i = 0; i < 3; i++)
        gFFT_traj[i].push_back(gVO.row(verts[i]).transpose());

    fprintf(stderr,
        "[face_flip_tracker] tracking face %d  verts=(%d,%d,%d)  samples=(%d,%d,%d)\n",
        face_idx, verts[0], verts[1], verts[2],
        gFFT_sampleIdx[0], gFFT_sampleIdx[1], gFFT_sampleIdx[2]);
}

void face_flip_tracker_pre_update()
{
    if (gFFT_faceIdx < 0) return;
    for (int i = 0; i < 3; i++) {
        int si = gFFT_sampleIdx[i];
        if (si < 0) continue;
        gFFT_posBefore[i] = fft_sample_pos(gSamples[si]);
    }
}

void face_flip_tracker_post_update()
{
    if (gFFT_faceIdx < 0 || gFFT_flipDetected) return;

    Eigen::Vector3d posAfter[3];
    bool any_moved = false;
    for (int i = 0; i < 3; i++) {
        int si = gFFT_sampleIdx[i];
        if (si < 0) { posAfter[i] = gFFT_posBefore[i]; continue; }
        posAfter[i] = fft_sample_pos(gSamples[si]);
        if ((posAfter[i] - gFFT_posBefore[i]).norm() > 1e-15) {
            any_moved = true;
            gFFT_traj[i].push_back(posAfter[i]);
        }
    }

    if (!any_moved) return;

    Eigen::Vector3d curNormal =
        (posAfter[1] - posAfter[0]).cross(posAfter[2] - posAfter[0]);
    if (curNormal.dot(gFFT_origNormal) < 0.0) {
        gFFT_flipDetected   = true;
        gFFT_flipAtCollapse = gCollapseCount;
        fprintf(stderr,
            "[face_flip_tracker] *** FLIP DETECTED at collapse #%d — original face %d ***\n",
            gCollapseCount, gFFT_faceIdx);
    }
}

bool face_flip_tracker_enabled()         { return gFFT_faceIdx >= 0; }
bool face_flip_tracker_flip_detected()   { return gFFT_flipDetected; }
int  face_flip_tracker_flip_at_collapse(){ return gFFT_flipAtCollapse; }
int  face_flip_tracker_face_idx()        { return gFFT_faceIdx; }

Eigen::Vector3d face_flip_tracker_cur_pos(int i)
{
    if (i < 0 || i > 2 || gFFT_sampleIdx[i] < 0) return Eigen::Vector3d::Zero();
    return fft_sample_pos(gSamples[gFFT_sampleIdx[i]]);
}

const std::vector<Eigen::Vector3d>& face_flip_tracker_traj(int i)
{
    static const std::vector<Eigen::Vector3d> empty;
    if (i < 0 || i > 2) return empty;
    return gFFT_traj[i];
}

// ---- end face flip tracker ----

// ---- vertex watch tracker ----

static int           gVWT_fineVtxId  = -1;
static int           gVWT_sampleIdx  = -1;
static Eigen::RowVector3i gVWT_snapBF;          // cur_BF snapshot before tries loop
static bool          gVWT_triggered  = false;
static int           gVWT_triggerAt  = -1;

void vertex_watch_set(int fine_vtx_id)
{
    gVWT_fineVtxId = -1;
    gVWT_sampleIdx = -1;
    gVWT_triggered = false;
    gVWT_triggerAt = -1;
    gVWT_snapBF    = Eigen::RowVector3i(-1, -1, -1);

    if (fine_vtx_id < 0 || fine_vtx_id >= (int)gVO.rows()) {
        fprintf(stderr, "[vertex_watch] vertex %d out of range (gVO has %d rows)\n",
                fine_vtx_id, (int)gVO.rows());
        return;
    }
    for (int si = 0; si < (int)gSamples.size(); si++) {
        const Sample& s = gSamples[si];
        if (s.is_vertex && s.fine_vertex_id == fine_vtx_id) {
            gVWT_sampleIdx = si;
            break;
        }
    }
    if (gVWT_sampleIdx < 0) {
        fprintf(stderr, "[vertex_watch] no sample found for fine vertex %d\n", fine_vtx_id);
        return;
    }
    gVWT_fineVtxId = fine_vtx_id;
    fprintf(stderr, "[vertex_watch] watching fine vertex %d  sample_idx=%d\n",
            fine_vtx_id, gVWT_sampleIdx);
}

void vertex_watch_clear()
{
    gVWT_fineVtxId = -1;
    gVWT_sampleIdx = -1;
    gVWT_triggered = false;
    gVWT_triggerAt = -1;
}

void vertex_watch_pre_step()
{
    if (gVWT_fineVtxId < 0 || gVWT_sampleIdx < 0) return;
    gVWT_snapBF = gSamples[gVWT_sampleIdx].cur_BF;
}

void vertex_watch_check_collapse(int s_vtx, int d_vtx)
{
    if (gVWT_fineVtxId < 0 || gVWT_sampleIdx < 0 || gVWT_triggered) return;
    for (int k = 0; k < 3; k++) {
        int bfv = gVWT_snapBF(k);
        if (bfv == s_vtx || bfv == d_vtx) {
            gVWT_triggered = true;
            gVWT_triggerAt = gCollapseCount;
            fprintf(stderr,
                "[vertex_watch] *** TRIGGERED at collapse #%d ***"
                "  fine_vtx=%d  cur_BF[%d]=%d  matched_%s\n",
                gCollapseCount, gVWT_fineVtxId, k, bfv,
                (bfv == s_vtx) ? "survivor" : "absorbed");
            return;
        }
    }
}

bool vertex_watch_active()             { return gVWT_fineVtxId >= 0; }
bool vertex_watch_triggered()          { return gVWT_triggered; }
int  vertex_watch_trigger_at_collapse(){ return gVWT_triggerAt; }
int  vertex_watch_fine_vtx()           { return gVWT_fineVtxId; }

Eigen::Vector3i vertex_watch_cur_BF()
{
    if (gVWT_sampleIdx < 0) return Eigen::Vector3i(-1, -1, -1);
    return gSamples[gVWT_sampleIdx].cur_BF.transpose();
}

// ---- end vertex watch tracker ----

void sample_tracker_show_canonical(const Eigen::MatrixXd& uv_pre_3d,
                                   const Eigen::MatrixXd& uv_post_3d,
                                   const Eigen::MatrixXi& FUV_pre,
                                   const Eigen::MatrixXi& FUV_post)
{
#ifdef C2F_VIZ_DIAGNOSTIC
    // Collect entries from the displayed sheet (on_snap_sheet == true).
    // These are the only samples whose UV face rows index into FUV_pre/FUV_post.
    std::vector<int> idx;
    for (int i = 0; i < (int)gLastRingRemap.size(); i++)
        if (gLastRingRemap[i].on_snap_sheet) idx.push_back(i);
    if (idx.empty()) return;

    const int nR = (int)idx.size();
    MatrixXd prePts(nR, 3), postPts(nR, 3), arrows(nR, 3);
    for (int i = 0; i < nR; i++) {
        const RingRemapEntry& e = gLastRingRemap[idx[i]];

        // Interpolate the canonical-frame UV_pre surface at the pre-collapse BC.
        RowVector3d pre_pos =
            e.pre_bc(0) * uv_pre_3d.row(FUV_pre(e.pre_face_row, 0))
          + e.pre_bc(1) * uv_pre_3d.row(FUV_pre(e.pre_face_row, 1))
          + e.pre_bc(2) * uv_pre_3d.row(FUV_pre(e.pre_face_row, 2));

        // Interpolate the canonical-frame UV_post surface at the post-collapse BC.
        RowVector3d post_pos =
            e.post_bc(0) * uv_post_3d.row(FUV_post(e.post_face_row, 0))
          + e.post_bc(1) * uv_post_3d.row(FUV_post(e.post_face_row, 1))
          + e.post_bc(2) * uv_post_3d.row(FUV_post(e.post_face_row, 2));

        prePts.row(i)  = pre_pos;
        postPts.row(i) = post_pos;
        arrows.row(i)  = post_pos - pre_pos;  // UV_pre position → UV_post position
    }

    // Pre samples sit on the UV_pre surface (green), with arrows to their UV_post destinations.
    auto * pc_pre = polyscope::registerPointCloud("ring_sample_pre", prePts);
    pc_pre->setPointColor({0.3f, 0.9f, 0.4f})->setPointRadius(0.003, true)->setEnabled(true);
    pc_pre->addVectorQuantity("to_uv_post", arrows, polyscope::VectorType::AMBIENT)
         ->setVectorColor({1.0f, 0.95f, 0.1f})
         ->setEnabled(true);

    // Post samples sit on the UV_post surface (orange).
    polyscope::registerPointCloud("ring_sample_post", postPts)
        ->setPointColor({0.9f, 0.4f, 0.2f})->setPointRadius(0.003, true)->setEnabled(true);
#endif
}
