#include "coarse_fine_viz.h"

#include <limits>

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>
#include <polyscope/point_cloud.h>
#include <polyscope/structure.h>
#include <polyscope/pick.h>

#include <igl/collapse_edge.h>       // IGL_COLLAPSE_EDGE_NULL
#include <igl/writeOBJ.h>
#include "face_dead.h"

#include <query_coarse_to_fine.h>
#include <single_collapse_data.h>

#ifdef C2F_VIZ_DIAGNOSTIC
#include <compute_barycentric.h>
#endif

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

using namespace Eigen;

// ---- globals from main.cpp ----
extern MatrixXd gV;
extern MatrixXd gVO;
extern MatrixXi gF;
extern MatrixXi gFO;
extern VectorXi gFaceSheetID;
extern std::vector<std::vector<int>> gVF;
extern std::vector<single_collapse_data> gDecInfo;
extern std::vector<std::vector<int>> gDecIM;
extern bool gFinished;

// ---- internal state ----
static bool  gC2FLoaded  = false;
static char  gC2FPath[512] = "coarse_to_fine.txt";
static float gC2FzOffset = 0.0f;

// Stored after load so the Z-offset slider can re-register without re-reading the file.
static MatrixXd gCoarseMeshV;  // all gV vertices (no Z offset, infinity zeroed out)
static MatrixXi gCoarseMeshF;  // live coarse faces (null and infinity faces removed)
static MatrixXd gCorrVectors;  // per-vertex (fine_pos - coarse_pos), zero if no correspondence

// Names of structures that were enabled before c2f load, so Hide can restore them.
static std::vector<std::pair<std::string,std::string>> gPrevEnabled; // (type, name)

// ---- vertex-pick c2f state ----
static bool    gPickModeEnabled  = false;
static bool    gPickResultValid  = false;
static int     gPickedGlobalVtx  = -1;
static float   gPickFineZOff     = 0.5f;
static Vector3d gPickFinePos     = Vector3d::Zero(); // cached fine-mesh world position

// Compact list of live coarse vertices registered as the pick point cloud
static std::vector<int> gPickCompactToGlobal;

// Track last polyscope selection so we only react on changes
static std::pair<polyscope::Structure*, size_t> gPickLastSel = {nullptr, (size_t)-1};

// ---- vertex-pick helpers ----

static void build_pick_pts()
{
    const int nFO = (int)gFaceSheetID.size();
    gPickCompactToGlobal.clear();
    for (int v = 0; v < (int)gV.rows(); v++) {
        if (std::isinf(gV(v, 0))) continue;
        for (int f : gVF[v]) {
            if (f >= nFO) continue;
            if (is_face_dead(gF, f)) continue;
            if (std::isinf(gV(gF(f, 0), 0))) continue;
            gPickCompactToGlobal.push_back(v);
            break;
        }
    }
    if (gPickCompactToGlobal.empty()) return;

    MatrixXd pts((int)gPickCompactToGlobal.size(), 3);
    for (int i = 0; i < (int)gPickCompactToGlobal.size(); i++)
        pts.row(i) = gV.row(gPickCompactToGlobal[i]).leftCols(3);

    polyscope::registerPointCloud("c2f_pick_verts", pts)
        ->setPointColor({1.0f, 0.9f, 0.1f})
        ->setPointRadius(0.005, true)   // relative to bounding box
        ->setEnabled(true);
}

// Rebuild only the polyscope structures from the cached pick result (for Z-slider updates).
static void rebuild_pick_viz()
{
    if (!gPickResultValid || gPickedGlobalVtx < 0) return;

    Vector3d coarse_pos = gV.row(gPickedGlobalVtx).head(3);
    Vector3d fine_off   = gPickFinePos; fine_off(2) += (double)gPickFineZOff;

    // Offset fine mesh
    MatrixXd fineV = gVO;
    fineV.col(2).array() += (double)gPickFineZOff;
    polyscope::registerSurfaceMesh("c2f_pick_fine_mesh", fineV, gFO)
        ->setSurfaceColor({0.5f, 0.85f, 0.5f})
        ->setEdgeWidth(0.3f)
        ->setSmoothShade(false)
        ->setTransparency(0.4f);

    // Highlight picked coarse vertex
    MatrixXd coarsePt(1, 3); coarsePt.row(0) = coarse_pos.transpose();
    polyscope::registerPointCloud("c2f_pick_coarse_pt", coarsePt)
        ->setPointColor({1.0f, 0.3f, 0.1f})
        ->setPointRadius(0.0008, true);

    // Correspondence point on offset fine mesh
    MatrixXd finePt(1, 3); finePt.row(0) = fine_off.transpose();
    polyscope::registerPointCloud("c2f_pick_fine_pt", finePt)
        ->setPointColor({0.1f, 1.0f, 0.3f})
        ->setPointRadius(0.0008, true);

    // Arrow: coarse vertex → correspondence point on offset fine mesh
    MatrixXd arrowV(2, 3);
    arrowV.row(0) = coarse_pos.transpose();
    arrowV.row(1) = fine_off.transpose();
    MatrixXi arrowE(1, 2); arrowE << 0, 1;
    polyscope::registerCurveNetwork("c2f_pick_arrow", arrowV, arrowE)
        ->setColor({1.0f, 0.85f, 0.05f})
        ->setRadius(0.0002, true);
}

static void clear_pick_viz()
{
    polyscope::removeStructure("c2f_pick_fine_mesh", false);
    polyscope::removeStructure("c2f_pick_fine_pt",   false);
    polyscope::removeStructure("c2f_pick_coarse_pt", false);
    polyscope::removeStructure("c2f_pick_arrow",     false);
    gPickResultValid  = false;
    gPickedGlobalVtx  = -1;
    gPickLastSel      = {nullptr, (size_t)-1};
    polyscope::pick::resetSelection();
}

// ---- ring_post diagnostic structs ----
#ifdef C2F_VIZ_DIAGNOSTIC

struct WalkStep {
    int dIdx;
    int FIdx_in;
    int sheet_id;
    Eigen::RowVector3d BC_in;
    Eigen::RowVector3i BF_in;
    Eigen::RowVector3d BC_out;
    Eigen::RowVector3i BF_out;
    int FIdx_out;
};

struct VertexWalkResult {
    int global_vtx;
    Eigen::RowVector3i final_BF;
    Eigen::RowVector3d final_BC;
    std::vector<WalkStep> chain;
    bool init_ok = false;
};

// Same initialization logic as coarse_fine_compute_and_save, but runs the walk
// step-by-step and records the full chain of (dIdx, FIdx, BC, BF) at each step.
static VertexWalkResult traced_walk(int vi)
{
    VertexWalkResult res;
    res.global_vtx = vi;

    const int nFO  = (int)gFaceSheetID.size();
    const int nDec = (int)gDecInfo.size();

    // Find a live face touching vi
    int fi = -1;
    for (int f : gVF[vi]) {
        if (f >= nFO) continue;
        if (is_face_dead(gF, f)) continue;
        if (std::isinf(gV(gF(f, 0), 0))) continue;
        fi = f; break;
    }
    if (fi < 0) {
        fprintf(stderr, "[TRACED-WALK] vtx=%d: no live face found\n", vi);
        return res;
    }

    // Initialize BF/BC/FIdx (mirrors coarse_fine_compute_and_save)
    Eigen::RowVector3i BF_cur;
    Eigen::RowVector3d BC_cur;
    int FIdx_cur;

    int dIdx_init = -1;
    if (fi < (int)gDecIM.size()) {
        const auto & dList = gDecIM[fi];
        for (int ii = (int)dList.size()-1; ii >= 0; ii--)
            if (dList[ii] < nDec) { dIdx_init = dList[ii]; break; }
    }

    bool initOk = false;
    if (dIdx_init >= 0) {
        const int sid = (fi < nFO) ? (int)gFaceSheetID(fi) : 0;
        const SheetData * sd_ptr = nullptr;
        for (auto & sd : gDecInfo[dIdx_init].sheets)
            if (sd.global_sheet_id == sid) { sd_ptr = &sd; break; }
        if (!sd_ptr && !gDecInfo[dIdx_init].sheets.empty())
            sd_ptr = &gDecInfo[dIdx_init].sheets[0];
        if (sd_ptr) {
            const SheetData & sd = *sd_ptr;
            int pre_row = -1;
            for (int r = 0; r < (int)sd.FIdx_pre.size(); r++)
                if (sd.FIdx_pre(r) == fi) { pre_row = r; break; }
            if (pre_row >= 0) {
                int gv0 = sd.subsetVIdx(sd.FUV_pre(pre_row, 0));
                int gv1 = sd.subsetVIdx(sd.FUV_pre(pre_row, 1));
                int gv2 = sd.subsetVIdx(sd.FUV_pre(pre_row, 2));
                int col = (gv0==vi)?0 : (gv1==vi)?1 : (gv2==vi)?2 : -1;
                if (col >= 0) {
                    BF_cur = {gv0, gv1, gv2};
                    BC_cur = {(col==0)?1.0:0.0, (col==1)?1.0:0.0, (col==2)?1.0:0.0};
                    FIdx_cur = fi;
                    initOk = true;
                }
            }
        }
    }
    if (!initOk) {
        int col = 0;
        for (int c = 0; c < 3; c++) if (gF(fi, c) == vi) { col = c; break; }
        BF_cur   = {gF(fi, col), gF(fi, (col+1)%3), gF(fi, (col+2)%3)};
        BC_cur   = {1.0, 0.0, 0.0};
        FIdx_cur = fi;
    }
    res.init_ok = true;

    // Backward walk — record every step
    int dIdx = nDec;
    while (true) {
        if (FIdx_cur < 0 || FIdx_cur >= (int)gDecIM.size()) break;
        const std::vector<int> & dList = gDecIM[FIdx_cur];
        int next_dIdx = -1;
        for (int ii = (int)dList.size()-1; ii >= 0; ii--) {
            if (dIdx > dList[ii]) { next_dIdx = dList[ii]; break; }
        }
        if (next_dIdx < 0) break;
        dIdx = next_dIdx;

        const int sid = (FIdx_cur < nFO) ? (int)gFaceSheetID(FIdx_cur) : 0;
        const SheetData * sd_ptr = nullptr;
        for (auto & sd : gDecInfo[dIdx].sheets)
            if (sd.global_sheet_id == sid) { sd_ptr = &sd; break; }
        if (!sd_ptr && !gDecInfo[dIdx].sheets.empty())
            sd_ptr = &gDecInfo[dIdx].sheets[0];
        if (!sd_ptr) continue;
        const SheetData & sd = *sd_ptr;

        int pre_row = -1;
        for (int r = 0; r < (int)sd.FIdx_pre.size(); r++)
            if (sd.FIdx_pre(r) == FIdx_cur) { pre_row = r; break; }
        if (pre_row < 0) continue;

        int v0 = sd.FUV_pre(pre_row, 0);
        int v1 = sd.FUV_pre(pre_row, 1);
        int v2 = sd.FUV_pre(pre_row, 2);

        Eigen::Vector2d queryUV =
            BC_cur(0) * sd.UV_post.row(v0).transpose()
          + BC_cur(1) * sd.UV_post.row(v1).transpose()
          + BC_cur(2) * sd.UV_post.row(v2).transpose();

        Eigen::MatrixXd B;
        compute_barycentric(queryUV, sd.UV_pre, sd.FUV_pre, B);

        Eigen::VectorXd distToValid = -B.rowwise().minCoeff();
        double minD = 1.0; int best = 0;
        for (int bb = 0; bb < (int)distToValid.size(); bb++)
            if (distToValid(bb) < minD) { minD = distToValid(bb); best = bb; }

        B(best,0) = std::max(0.0, B(best,0));
        B(best,1) = std::max(0.0, B(best,1));
        B(best,2) = std::max(0.0, B(best,2));
        double s = B.row(best).sum(); if (s > 0) B.row(best) /= s;

        int nv0 = sd.subsetVIdx(sd.FUV_pre(best, 0));
        int nv1 = sd.subsetVIdx(sd.FUV_pre(best, 1));
        int nv2 = sd.subsetVIdx(sd.FUV_pre(best, 2));
        int new_FIdx = sd.FIdx_pre(best);

        WalkStep step;
        step.dIdx     = dIdx;
        step.FIdx_in  = FIdx_cur;
        step.sheet_id = sid;
        step.BC_in    = BC_cur;
        step.BF_in    = BF_cur;
        step.BC_out   = B.row(best);
        step.BF_out   = {nv0, nv1, nv2};
        step.FIdx_out = new_FIdx;
        res.chain.push_back(step);

        BC_cur   = B.row(best);
        BF_cur   = {nv0, nv1, nv2};
        FIdx_cur = new_FIdx;
    }

    res.final_BF = BF_cur;
    res.final_BC = BC_cur;
    return res;
}

#endif // C2F_VIZ_DIAGNOSTIC

// Run the full backward walk for one coarse vertex, then display results.
static void run_single_vertex_c2f(int vi)
{
    const int nFO  = (int)gFaceSheetID.size();
    const int nDec = (int)gDecInfo.size();

    // Pick any live face touching vi
    int fi = -1;
    for (int f : gVF[vi]) {
        if (f >= nFO) continue;
        if (is_face_dead(gF, f)) continue;
        if (std::isinf(gV(gF(f, 0), 0))) continue;
        fi = f; break;
    }
    if (fi < 0) {
        fprintf(stderr, "[c2f-pick] vtx=%d: no live face found\n", vi);
        return;
    }

    MatrixXd BC(1, 3);
    MatrixXi BF(1, 3);
    VectorXi FIdx(1);

    // Identical initialization to coarse_fine_compute_and_save
    int dIdx = -1;
    if (fi < (int)gDecIM.size()) {
        const auto & dList = gDecIM[fi];
        for (int ii = (int)dList.size()-1; ii >= 0; ii--)
            if (dList[ii] < nDec) { dIdx = dList[ii]; break; }
    }

    auto useCurrentGF = [&]() {
        int col = 0;
        for (int c = 0; c < 3; c++) if (gF(fi, c) == vi) { col = c; break; }
        BF(0, 0) = gF(fi,  col);
        BF(0, 1) = gF(fi, (col + 1) % 3);
        BF(0, 2) = gF(fi, (col + 2) % 3);
        BC(0, 0) = 1.0; BC(0, 1) = 0.0; BC(0, 2) = 0.0;
        FIdx(0)  = fi;
    };

    bool initOk = false;
    if (dIdx >= 0) {
        const int sid = (fi < nFO) ? (int)gFaceSheetID(fi) : 0;
        const SheetData * sd_ptr = nullptr;
        for (auto & sd : gDecInfo[dIdx].sheets)
            if (sd.global_sheet_id == sid) { sd_ptr = &sd; break; }
        if (!sd_ptr && !gDecInfo[dIdx].sheets.empty())
            sd_ptr = &gDecInfo[dIdx].sheets[0];
        if (sd_ptr) {
            const SheetData & sd = *sd_ptr;
            int pre_row = -1;
            for (int r = 0; r < (int)sd.FIdx_pre.size(); r++)
                if (sd.FIdx_pre(r) == fi) { pre_row = r; break; }
            if (pre_row >= 0) {
                int gv0 = sd.subsetVIdx(sd.FUV_pre(pre_row, 0));
                int gv1 = sd.subsetVIdx(sd.FUV_pre(pre_row, 1));
                int gv2 = sd.subsetVIdx(sd.FUV_pre(pre_row, 2));
                int col = (gv0==vi)?0 : (gv1==vi)?1 : (gv2==vi)?2 : -1;
                if (col >= 0) {
                    BF(0, 0) = gv0;
                    BF(0, 1) = gv1;
                    BF(0, 2) = gv2;
                    BC(0, 0) = (col==0) ? 1.0 : 0.0;
                    BC(0, 1) = (col==1) ? 1.0 : 0.0;
                    BC(0, 2) = (col==2) ? 1.0 : 0.0;
                    FIdx(0)  = fi;
                    initOk = true;
                }
            }
        }
    }
    if (!initOk) useCurrentGF();

    VectorXi IM  = VectorXi::LinSpaced((int)gV.rows(), 0, (int)gV.rows() - 1);
    VectorXi IMF = VectorXi::LinSpaced((int)gF.rows(), 0, (int)gF.rows() - 1);
    query_coarse_to_fine(gDecInfo, IM, gDecIM, IMF, gFaceSheetID, BC, BF, FIdx);

    // Cache the result
    gPickFinePos = BC(0,0)*gVO.row(BF(0,0)).transpose()
                 + BC(0,1)*gVO.row(BF(0,1)).transpose()
                 + BC(0,2)*gVO.row(BF(0,2)).transpose();
    gPickedGlobalVtx = vi;
    gPickResultValid = true;

    fprintf(stderr, "[c2f-pick] vtx=%d → fine_pos=(%.4f,%.4f,%.4f)  BC=(%.3f,%.3f,%.3f)  BF=(%d,%d,%d)\n",
        vi, gPickFinePos(0), gPickFinePos(1), gPickFinePos(2),
        BC(0,0), BC(0,1), BC(0,2), BF(0,0), BF(0,1), BF(0,2));

    rebuild_pick_viz();
}

// ---- end vertex-pick helpers ----

static void rebuild_coarse_mesh_viz()
{
    MatrixXd V = gCoarseMeshV;
    V.col(2).array() += (double)gC2FzOffset;

    auto * ps = polyscope::registerSurfaceMesh("coarse_mesh_c2f", V, gCoarseMeshF);
    ps->setSurfaceColor({0.25f, 0.52f, 0.95f});
    ps->setEdgeWidth(0.5f);
    ps->setSmoothShade(false);

    // Adjust Z component: coarse mesh moved by zOffset, so the vector to the (stationary)
    // fine mesh shrinks by that same amount in Z.
    MatrixXd corrAdj = gCorrVectors;
    corrAdj.col(2).array() -= (double)gC2FzOffset;

    auto * vq = ps->addVertexVectorQuantity("correspondence", corrAdj, polyscope::VectorType::AMBIENT);
    vq->setEnabled(true);
    vq->setVectorColor({1.0f, 0.35f, 0.05f});
    // Draw vectors at true world-space length so tips land exactly on the fine mesh.
    // isRelative=false means a vector of magnitude 1 is drawn as 1 world unit.
    vq->setVectorLengthScale(1.0, false);
}

// ---- implementation ----

void coarse_fine_compute_and_save(const std::string & path)
{
    const int nFO = (int)gFaceSheetID.size();   // original face count (no infinity faces)

    // Collect live coarse vertices: non-infinity, referenced by at least one live original face.
    std::vector<int> coarseVerts;
    coarseVerts.reserve(256);
    for (int v = 0; v < (int)gV.rows(); v++) {
        if (std::isinf(gV(v, 0))) continue;
        for (int f : gVF[v]) {
            if (f >= nFO) continue;
            if (is_face_dead(gF, f)) continue;
            if (std::isinf(gV(gF(f, 0), 0))) continue;
            coarseVerts.push_back(v);
            break;
        }
    }

    const int N = (int)coarseVerts.size();
    if (N == 0) { std::cerr << "[c2f] No live coarse vertices found.\n"; return; }

    MatrixXd BC(N, 3);
    MatrixXi BF(N, 3);
    VectorXi FIdx(N);

    const int nDec = (int)gDecInfo.size();

    for (int i = 0; i < N; i++) {
        const int vi = coarseVerts[i];
        int fi = -1;
        for (int f : gVF[vi]) {
            if (f >= nFO) continue;
            if (is_face_dead(gF, f)) continue;
            if (std::isinf(gV(gF(f, 0), 0))) continue;
            fi = f; break;
        }
        if (fi < 0) {
            BC.row(i).setZero(); BF.row(i).setZero(); FIdx(i) = 0;
            continue;
        }

        // Find the last active collapse that touched fi.
        int dIdx = -1;
        if (fi < (int)gDecIM.size()) {
            const auto & dList = gDecIM[fi];
            for (int ii = (int)dList.size()-1; ii >= 0; ii--)
                if (dList[ii] < nDec) { dIdx = dList[ii]; break; }
        }

        // Helper: initialize BF/BC/FIdx from current gF (fallback).
        auto useCurrentGF = [&]() {
            int col = 0;
            for (int c = 0; c < 3; c++) if (gF(fi, c) == vi) { col = c; break; }
            BF(i, 0) = gF(fi,  col);
            BF(i, 1) = gF(fi, (col + 1) % 3);
            BF(i, 2) = gF(fi, (col + 2) % 3);
            BC(i, 0) = 1.0; BC(i, 1) = 0.0; BC(i, 2) = 0.0;
            FIdx(i)  = fi;
        };

        if (dIdx < 0) {
            // Face never in any active collapse → gF is correct (no remapping).
            useCurrentGF();
            continue;
        }

        // Route to the SheetData for fi's sheet at collapse dIdx.
        const int sid = (fi < nFO) ? (int)gFaceSheetID(fi) : 0;
        const SheetData * sd_ptr = nullptr;
        for (auto & sd : gDecInfo[dIdx].sheets)
            if (sd.global_sheet_id == sid) { sd_ptr = &sd; break; }
        if (!sd_ptr && !gDecInfo[dIdx].sheets.empty())
            sd_ptr = &gDecInfo[dIdx].sheets[0];

        if (!sd_ptr) { useCurrentGF(); continue; }

        // Find fi in FIdx_pre and align BC with FUV_pre column ordering.
        // BC[j] must correspond to FUV_pre column j because query_coarse_to_fine's
        // first step computes queryUV = BC(0)*UV_post[FUV_pre(row,0)] + ...
        // Using FUV_post ordering for BC was wrong: FUV_pre and FUV_post can list
        // the same face's vertices in different column orders.
        const SheetData & sd = *sd_ptr;
        int pre_row = -1;
        for (int r = 0; r < (int)sd.FIdx_pre.size(); r++)
            if (sd.FIdx_pre(r) == fi) { pre_row = r; break; }

        if (pre_row < 0) { useCurrentGF(); continue; }

        const int gv0 = sd.subsetVIdx(sd.FUV_pre(pre_row, 0));
        const int gv1 = sd.subsetVIdx(sd.FUV_pre(pre_row, 1));
        const int gv2 = sd.subsetVIdx(sd.FUV_pre(pre_row, 2));

        int col = -1;
        if (gv0 == vi) col = 0;
        else if (gv1 == vi) col = 1;
        else if (gv2 == vi) col = 2;

        if (col < 0) {
            // vi not present in FUV_pre (introduced by a later non-active-sheet collapse).
            useCurrentGF(); continue;
        }

        BF(i, 0) = gv0;
        BF(i, 1) = gv1;
        BF(i, 2) = gv2;
        BC(i, 0) = (col == 0) ? 1.0 : 0.0;
        BC(i, 1) = (col == 1) ? 1.0 : 0.0;
        BC(i, 2) = (col == 2) ? 1.0 : 0.0;
        FIdx(i)  = fi;
    }

    // SSP never renumbers vertices or faces — identity maps are correct.
    VectorXi IM  = VectorXi::LinSpaced((int)gV.rows(), 0, (int)gV.rows()  - 1);
    VectorXi IMF = VectorXi::LinSpaced((int)gF.rows(), 0, (int)gF.rows()  - 1);

    query_coarse_to_fine(gDecInfo, IM, gDecIM, IMF, gFaceSheetID, BC, BF, FIdx);

    // Write: first line = count, then one line per query.
    std::ofstream out(path);
    if (!out) { std::cerr << "[c2f] Cannot write " << path << "\n"; return; }
    out << N << "\n";
    for (int i = 0; i < N; i++) {
        const int vi = coarseVerts[i];
        // Skip entries where the walk never found a live face (BC zeroed above).
        if (BC.row(i).isZero()) continue;
        out << vi
            << " " << BC(i,0) << " " << BC(i,1) << " " << BC(i,2)
            << " " << BF(i,0) << " " << BF(i,1) << " " << BF(i,2)
            << "\n";
    }
    std::cout << "[c2f] Saved " << N << " correspondences → " << path << "\n";
}

// -----------------------------------------------------------------------
// Bundle format (little-endian binary, extension .c2f):
//   magic   uint32  0xC2F50001
//   NC      uint32  compact coarse vertex count
//   FC      uint32  coarse face count
//   NF      uint32  fine vertex count
//   FF      uint32  fine face count
//   coarse vertices  NC×3 double
//   coarse faces     FC×3 uint32  (compact indices 0..NC-1)
//   fine vertices    NF×3 double
//   fine faces       FF×3 uint32
//   correspondence   NC × (bc0 bc1 bc2 double  +  fv0 fv1 fv2 uint32)
//                    one entry per compact coarse vertex, in order
// -----------------------------------------------------------------------
void coarse_fine_save_bundle(const std::string & corrPath, const std::string & bundlePath)
{
    std::ifstream in(corrPath);
    if (!in) { std::cerr << "[bundle] Cannot read " << corrPath << "\n"; return; }
    int N; in >> N;
    if (N <= 0) { std::cerr << "[bundle] Empty correspondence file.\n"; return; }

    const int nV  = (int)gV.rows();
    const int nFO = (int)gFaceSheetID.size();

    struct Corr { double bc0, bc1, bc2; uint32_t fv0, fv1, fv2; };
    std::vector<Corr> corrGlobal(nV, {1.0/3,1.0/3,1.0/3, 0,0,0});
    std::vector<bool> hasCorr(nV, false);

    for (int i = 0; i < N; i++) {
        int vi, fv0, fv1, fv2; double bc0, bc1, bc2;
        if (!(in >> vi >> bc0 >> bc1 >> bc2 >> fv0 >> fv1 >> fv2)) break;
        if (vi < 0 || vi >= nV) continue;
        corrGlobal[vi] = {bc0, bc1, bc2, (uint32_t)fv0, (uint32_t)fv1, (uint32_t)fv2};
        hasCorr[vi] = true;
    }

    // Build compact coarse mesh (same logic as load_and_show).
    // Also track original (global) face indices for the SSP query data.
    MatrixXi tmpF(nFO, 3);
    std::vector<int> fullFOrigIdx;   // compact face idx → global face idx
    fullFOrigIdx.reserve(512);
    int nLive = 0;
    for (int f = 0; f < nFO; f++) {
        if (is_face_dead(gF, f)) continue;
        bool has_inf = false;
        for (int c = 0; c < 3; c++)
            if (std::isinf(gV(gF(f, c), 0))) { has_inf = true; break; }
        if (has_inf) continue;
        tmpF.row(nLive++) = gF.row(f);
        fullFOrigIdx.push_back(f);
    }
    MatrixXi fullF = tmpF.topRows(nLive);

    std::vector<int> oldToNew(nV, -1);
    std::vector<int> newToOld;
    newToOld.reserve(512);
    for (int f = 0; f < fullF.rows(); f++)
        for (int c = 0; c < 3; c++) {
            int v = fullF(f, c);
            if (oldToNew[v] < 0) { oldToNew[v] = (int)newToOld.size(); newToOld.push_back(v); }
        }

    const uint32_t NC = (uint32_t)newToOld.size();
    const uint32_t FC = (uint32_t)fullF.rows();
    const uint32_t NF = (uint32_t)gVO.rows();
    const uint32_t FF = (uint32_t)gFO.rows();

    std::ofstream out(bundlePath, std::ios::binary);
    if (!out) { std::cerr << "[bundle] Cannot write " << bundlePath << "\n"; return; }

    const uint32_t magic = 0xC2F50006;  // v6: adds DC UV data (UV_dc_pre/post, FUV_dc_pre/post) per sheet
    out.write((const char*)&magic, 4);
    out.write((const char*)&NC, 4);
    out.write((const char*)&FC, 4);
    out.write((const char*)&NF, 4);
    out.write((const char*)&FF, 4);

    for (uint32_t i = 0; i < NC; i++) {
        double xyz[3] = { gV(newToOld[i],0), gV(newToOld[i],1), gV(newToOld[i],2) };
        out.write((const char*)xyz, 24);
    }
    for (int f = 0; f < (int)FC; f++) {
        uint32_t tri[3] = { (uint32_t)oldToNew[fullF(f,0)],
                            (uint32_t)oldToNew[fullF(f,1)],
                            (uint32_t)oldToNew[fullF(f,2)] };
        out.write((const char*)tri, 12);
    }
    for (uint32_t i = 0; i < NF; i++) {
        double xyz[3] = { gVO(i,0), gVO(i,1), gVO(i,2) };
        out.write((const char*)xyz, 24);
    }
    for (uint32_t f = 0; f < FF; f++) {
        uint32_t tri[3] = { (uint32_t)gFO(f,0), (uint32_t)gFO(f,1), (uint32_t)gFO(f,2) };
        out.write((const char*)tri, 12);
    }
    for (uint32_t i = 0; i < NC; i++) {
        const Corr & c = corrGlobal[newToOld[i]];
        out.write((const char*)&c.bc0, 8);
        out.write((const char*)&c.bc1, 8);
        out.write((const char*)&c.bc2, 8);
        out.write((const char*)&c.fv0, 4);
        out.write((const char*)&c.fv1, 4);
        out.write((const char*)&c.fv2, 4);
    }

    // ---- SSP query data (v2) ----
    // Sizes for reconstructing identity IM and IMF inside query_coarse_to_fine.
    const uint32_t nV_total  = (uint32_t)gV.rows();
    const uint32_t nF_decIM  = (uint32_t)gDecIM.size();
    const uint32_t nFO_u     = (uint32_t)nFO;
    out.write((const char*)&nV_total, 4);
    out.write((const char*)&nF_decIM, 4);
    out.write((const char*)&nFO_u,    4);

    // compact→global vertex map (NC int32s)
    for (uint32_t i = 0; i < NC; i++) {
        int32_t gv = (int32_t)newToOld[i];
        out.write((const char*)&gv, 4);
    }

    // compact→global face map (FC int32s)
    for (uint32_t f = 0; f < FC; f++) {
        int32_t gf = (int32_t)fullFOrigIdx[f];
        out.write((const char*)&gf, 4);
    }

    // gFaceSheetID (nFO int32s)
    for (int f = 0; f < nFO; f++) {
        int32_t sid = (int32_t)gFaceSheetID(f);
        out.write((const char*)&sid, 4);
    }

    // gDecIM: for each face, variable-length list of collapse indices
    for (uint32_t f = 0; f < nF_decIM; f++) {
        uint32_t cnt = (uint32_t)gDecIM[f].size();
        out.write((const char*)&cnt, 4);
        for (int idx : gDecIM[f]) {
            int32_t v = (int32_t)idx;
            out.write((const char*)&v, 4);
        }
    }

    // gDecInfo: collapse history (only fields used by query_coarse_to_fine)
    uint32_t nDec = (uint32_t)gDecInfo.size();
    out.write((const char*)&nDec, 4);
    for (const auto & cd : gDecInfo) {
        uint32_t nSheets = (uint32_t)cd.sheets.size();
        out.write((const char*)&nSheets, 4);
        for (const auto & sd : cd.sheets) {
            int32_t sheet_id = (int32_t)sd.global_sheet_id;
            out.write((const char*)&sheet_id, 4);

            uint32_t svSz = (uint32_t)sd.subsetVIdx.size();
            out.write((const char*)&svSz, 4);
            for (int j = 0; j < (int)svSz; j++) {
                int32_t v = (int32_t)sd.subsetVIdx(j);
                out.write((const char*)&v, 4);
            }

            uint32_t uvRows = (uint32_t)sd.UV_pre.rows();
            out.write((const char*)&uvRows, 4);
            for (uint32_t r = 0; r < uvRows; r++) {
                double u = sd.UV_pre(r, 0), v = sd.UV_pre(r, 1);
                out.write((const char*)&u, 8); out.write((const char*)&v, 8);
            }
            for (uint32_t r = 0; r < uvRows; r++) {
                double u = sd.UV_post(r, 0), v = sd.UV_post(r, 1);
                out.write((const char*)&u, 8); out.write((const char*)&v, 8);
            }

            uint32_t fuvRows = (uint32_t)sd.FUV_pre.rows();
            out.write((const char*)&fuvRows, 4);
            for (uint32_t r = 0; r < fuvRows; r++) {
                int32_t a = sd.FUV_pre(r,0), b = sd.FUV_pre(r,1), c2 = sd.FUV_pre(r,2);
                out.write((const char*)&a, 4);
                out.write((const char*)&b, 4);
                out.write((const char*)&c2, 4);
            }
            for (uint32_t r = 0; r < fuvRows; r++) {
                int32_t fi = (int32_t)sd.FIdx_pre(r);
                out.write((const char*)&fi, 4);
            }

            uint32_t fuvPostRows = (uint32_t)sd.FUV_post.rows();
            out.write((const char*)&fuvPostRows, 4);
            for (uint32_t r = 0; r < fuvPostRows; r++) {
                int32_t a = sd.FUV_post(r,0), b = sd.FUV_post(r,1), c2 = sd.FUV_post(r,2);
                out.write((const char*)&a, 4);
                out.write((const char*)&b, 4);
                out.write((const char*)&c2, 4);
            }
            for (uint32_t r = 0; r < fuvPostRows; r++) {
                int32_t fi = (int32_t)sd.FIdx_post(r);
                out.write((const char*)&fi, 4);
            }

            // v4: sd.b — (survivor_local, absorbed_local) for vertex fixup
            int32_t b0 = (sd.b.size() >= 2) ? (int32_t)sd.b(0) : -1;
            int32_t b1 = (sd.b.size() >= 2) ? (int32_t)sd.b(1) : -1;
            out.write((const char*)&b0, 4);
            out.write((const char*)&b1, 4);

            // v5: V_pre/V_post — 3D ring geometry in local index space (uvRows × 3 float64 each)
            if ((uint32_t)sd.V_pre.rows() != uvRows || (uint32_t)sd.V_post.rows() != uvRows) {
                std::cerr << "[bundle] WARNING: V_pre/V_post unpopulated for sheet "
                          << sd.global_sheet_id << " (uvRows=" << uvRows
                          << " V_pre.rows()=" << sd.V_pre.rows()
                          << " V_post.rows()=" << sd.V_post.rows() << ") — writing zeros\n";
                const double z3[3] = {0,0,0};
                for (uint32_t r = 0; r < uvRows; r++) out.write((const char*)z3, 24);
                for (uint32_t r = 0; r < uvRows; r++) out.write((const char*)z3, 24);
            } else {
                for (uint32_t r = 0; r < uvRows; r++) {
                    double xyz[3] = { sd.V_pre(r,0), sd.V_pre(r,1), sd.V_pre(r,2) };
                    out.write((const char*)xyz, 24);
                }
                for (uint32_t r = 0; r < uvRows; r++) {
                    double xyz[3] = { sd.V_post(r,0), sd.V_post(r,1), sd.V_post(r,2) };
                    out.write((const char*)xyz, 24);
                }
            }

            // v6: DC UV data — raw LSCM result for the double cover (boundary cases)
            {
                uint32_t has_dc = sd.has_double_cover ? 1u : 0u;
                out.write((const char*)&has_dc, 4);
                if (sd.has_double_cover) {
                    uint32_t dc_uvRows = (uint32_t)sd.UV_dc_pre.rows();
                    out.write((const char*)&dc_uvRows, 4);
                    for (uint32_t r = 0; r < dc_uvRows; r++) {
                        double u = sd.UV_dc_pre(r,0), v = sd.UV_dc_pre(r,1);
                        out.write((const char*)&u, 8); out.write((const char*)&v, 8);
                    }
                    for (uint32_t r = 0; r < dc_uvRows; r++) {
                        double u = sd.UV_dc_post(r,0), v = sd.UV_dc_post(r,1);
                        out.write((const char*)&u, 8); out.write((const char*)&v, 8);
                    }
                    uint32_t dc_fPre = (uint32_t)sd.FUV_dc_pre.rows();
                    out.write((const char*)&dc_fPre, 4);
                    for (uint32_t r = 0; r < dc_fPre; r++) {
                        int32_t a = sd.FUV_dc_pre(r,0), b = sd.FUV_dc_pre(r,1), c2 = sd.FUV_dc_pre(r,2);
                        out.write((const char*)&a, 4); out.write((const char*)&b, 4); out.write((const char*)&c2, 4);
                    }
                    uint32_t dc_fPost = (uint32_t)sd.FUV_dc_post.rows();
                    out.write((const char*)&dc_fPost, 4);
                    for (uint32_t r = 0; r < dc_fPost; r++) {
                        int32_t a = sd.FUV_dc_post(r,0), b = sd.FUV_dc_post(r,1), c2 = sd.FUV_dc_post(r,2);
                        out.write((const char*)&a, 4); out.write((const char*)&b, 4); out.write((const char*)&c2, 4);
                    }
                }
            }
        }
    }

    std::cerr << "[bundle] Saved NC=" << NC << " FC=" << FC
              << " NF=" << NF << " FF=" << FF
              << " nDec=" << nDec << " → " << bundlePath << "\n";

    // Export the coarse mesh as OBJ alongside the bundle.
    {
        std::string objPath = bundlePath;
        size_t dot = objPath.rfind('.');
        if (dot != std::string::npos) objPath = objPath.substr(0, dot);
        objPath += ".obj";

        MatrixXd Vc(NC, 3);
        for (uint32_t i = 0; i < NC; i++)
            Vc.row(i) << gV(newToOld[i],0), gV(newToOld[i],1), gV(newToOld[i],2);
        MatrixXi Fc(FC, 3);
        for (int f = 0; f < (int)FC; f++)
            Fc.row(f) << oldToNew[fullF(f,0)], oldToNew[fullF(f,1)], oldToNew[fullF(f,2)];

        if (!igl::writeOBJ(objPath, Vc, Fc))
            std::cerr << "[bundle] writeOBJ failed: " << objPath << "\n";
        else
            std::cerr << "[bundle] coarse mesh OBJ → " << objPath << "\n";
    }
}

void coarse_fine_clear()
{
    // Remove c2f-specific structures.
    polyscope::removeStructure("fine_mesh",       /*errorIfAbsent=*/false);
    polyscope::removeStructure("coarse_mesh_c2f", /*errorIfAbsent=*/false);

    // Restore whatever was enabled before c2f was loaded.
    for (auto & p : gPrevEnabled) {
        const std::string & typeName = p.first;
        const std::string & name     = p.second;
        auto typeIt = polyscope::state::structures.find(typeName);
        if (typeIt == polyscope::state::structures.end()) continue;
        auto nameIt = typeIt->second.find(name);
        if (nameIt == typeIt->second.end()) continue;
        nameIt->second->setEnabled(true);
    }
    gPrevEnabled.clear();
    gC2FLoaded = false;
}

void coarse_fine_load_and_show(const std::string & path)
{
    std::ifstream in(path);
    if (!in) { std::cerr << "[c2f] Cannot read " << path << "\n"; return; }

    int N; in >> N;
    if (N <= 0) { std::cerr << "[c2f] Empty correspondence file.\n"; return; }

    const int nV  = (int)gV.rows();
    const int nFO = (int)gFaceSheetID.size();

    // Per-vertex correspondence vectors indexed by global vertex index.
    // Exclude the infinity vertex (last row of gV) — it has no correspondence and
    // its (inf,inf,inf) position would corrupt polyscope's bounding box.
    gCorrVectors = MatrixXd::Zero(nV - 1, 3);

    int loaded = 0;
    for (int i = 0; i < N; i++) {
        int vi, fv0, fv1, fv2;
        double bc0, bc1, bc2;
        if (!(in >> vi >> bc0 >> bc1 >> bc2 >> fv0 >> fv1 >> fv2)) break;
        if (vi >= nV || fv0 >= gVO.rows() || fv1 >= gVO.rows() || fv2 >= gVO.rows()) continue;

        RowVector3d fine_pos   = bc0 * gVO.row(fv0) + bc1 * gVO.row(fv1) + bc2 * gVO.row(fv2);
        RowVector3d coarse_pos = gV.row(vi).leftCols(3);
        gCorrVectors.row(vi)   = fine_pos - coarse_pos;
        loaded++;
    }

    if (loaded == 0) { std::cerr << "[c2f] No valid entries.\n"; return; }

    // Build live coarse mesh faces (skip nulled and infinity-vertex faces).
    MatrixXi tmpF(nFO, 3);
    int nLive = 0;
    for (int f = 0; f < nFO; f++) {
        if (is_face_dead(gF, f)) continue;
        bool has_inf = false;
        for (int c = 0; c < 3; c++)
            if (std::isinf(gV(gF(f, c), 0))) { has_inf = true; break; }
        if (has_inf) continue;
        tmpF.row(nLive++) = gF.row(f);
    }
    MatrixXi fullF = tmpF.topRows(nLive);

    // Compact: keep only vertices actually referenced by live faces.
    // gV has ~N_fine rows (absorbed vertices still exist there with empty VF).
    // Passing the full gV to polyscope would render one arrow per original fine vertex —
    // most zero-length — making it look like a fine-to-coarse correspondence.
    std::vector<int> oldToNew(nV, -1);
    std::vector<int> newToOld;
    newToOld.reserve(512);
    for (int f = 0; f < fullF.rows(); f++)
        for (int c = 0; c < 3; c++) {
            int v = fullF(f, c);
            if (oldToNew[v] < 0) {
                oldToNew[v] = (int)newToOld.size();
                newToOld.push_back(v);
            }
        }

    const int nCoarse = (int)newToOld.size();
    gCoarseMeshV.resize(nCoarse, 3);
    for (int i = 0; i < nCoarse; i++)
        gCoarseMeshV.row(i) = gV.row(newToOld[i]).leftCols(3);

    gCoarseMeshF.resize(fullF.rows(), 3);
    for (int f = 0; f < fullF.rows(); f++)
        for (int c = 0; c < 3; c++)
            gCoarseMeshF(f, c) = oldToNew[fullF(f, c)];

    // Remap gCorrVectors to compact indexing.
    MatrixXd compactCorr = MatrixXd::Zero(nCoarse, 3);
    for (int i = 0; i < nCoarse; i++)
        compactCorr.row(i) = gCorrVectors.row(newToOld[i]);
    gCorrVectors = compactCorr;

    // Disable all currently enabled structures so only c2f view is shown.
    // Save their names so Hide can restore them.
    gPrevEnabled.clear();
    for (auto & typeEntry : polyscope::state::structures) {
        const std::string & typeName = typeEntry.first;
        for (auto & nameEntry : typeEntry.second) {
            const std::string & name = nameEntry.first;
            polyscope::Structure * ptr = nameEntry.second.get();
            if (name == "fine_mesh" || name == "coarse_mesh_c2f") continue;
            if (ptr->isEnabled()) {
                gPrevEnabled.push_back(std::make_pair(typeName, name));
                ptr->setEnabled(false);
            }
        }
    }

    // Fine (original) mesh — semi-transparent, in its natural position.
    polyscope::registerSurfaceMesh("fine_mesh", gVO, gFO)
        ->setSurfaceColor({0.55f, 0.55f, 0.55f})
        ->setEdgeWidth(0.3f)
        ->setSmoothShade(false)
        ->setTransparency(0.6f);

    // Coarse mesh with correspondence vector arrows.
    rebuild_coarse_mesh_viz();

    gC2FLoaded = true;
    std::cerr << "[c2f] Loaded " << loaded << " correspondences from " << path << "\n";
}

void coarse_fine_pick_check()
{
    if (!gPickModeEnabled || gPickCompactToGlobal.empty()) return;
    if (!polyscope::pick::haveSelection()) return;

    auto sel = polyscope::pick::getSelection();
    if (sel == gPickLastSel) return;
    gPickLastSel = sel;

    polyscope::Structure * structure = sel.first;
    size_t localIdx = sel.second;
    if (!structure || structure->name != "c2f_pick_verts") return;
    if ((int)localIdx >= (int)gPickCompactToGlobal.size()) return;

    run_single_vertex_c2f(gPickCompactToGlobal[(int)localIdx]);
}

void coarse_fine_imgui_section()
{
    if (!ImGui::CollapsingHeader("Coarse → Fine Correspondence")) return;

    // ---- vertex picker — always available ----
    ImGui::TextUnformatted("Vertex Picker (single coarse→fine query):");

    bool pickToggle = ImGui::Checkbox("Pick mode##c2fpick", &gPickModeEnabled);
    if (pickToggle) {
        if (gPickModeEnabled) {
            build_pick_pts();
        } else {
            polyscope::removeStructure("c2f_pick_verts", false);
            clear_pick_viz();
        }
    }

    if (gPickModeEnabled) {
        ImGui::SameLine();
        ImGui::TextDisabled("(click a yellow vertex)");

        if (gPickResultValid && gPickedGlobalVtx >= 0) {
            ImGui::Text("Picked vtx: %d", gPickedGlobalVtx);
            ImGui::Text("Fine pos: (%.3f, %.3f, %.3f)",
                gPickFinePos(0), gPickFinePos(1), gPickFinePos(2));

            ImGui::SetNextItemWidth(220);
            if (ImGui::SliderFloat("Fine mesh Z##pick", &gPickFineZOff, -3.0f, 3.0f))
                rebuild_pick_viz();

            if (ImGui::Button("Clear##c2fpick"))
                clear_pick_viz();
        } else {
            ImGui::TextDisabled("No vertex picked yet.");
        }
    }

    // ---- batch save / load (requires gFinished) ----
    ImGui::Separator();
    if (!gFinished) {
        ImGui::TextDisabled("Decimate to target to enable Save/Load.");
        return;
    }

    static char gBundlePath[512] = "c2f_bundle.c2f";

    ImGui::SetNextItemWidth(230);
    ImGui::InputText("##c2fpath", gC2FPath, sizeof(gC2FPath));
    ImGui::SameLine();
    if (ImGui::Button("Save##c2f"))
        coarse_fine_compute_and_save(std::string(gC2FPath));

    ImGui::SetNextItemWidth(230);
    ImGui::InputText("##bundlepath", gBundlePath, sizeof(gBundlePath));
    ImGui::SameLine();
    if (ImGui::Button("Save Bundle##c2f"))
        coarse_fine_save_bundle(std::string(gC2FPath), std::string(gBundlePath));

    if (ImGui::Button("Load & Show##c2f")) {
        coarse_fine_clear();
        coarse_fine_load_and_show(std::string(gC2FPath));
    }
    if (gC2FLoaded) {
        ImGui::SameLine();
        if (ImGui::Button("Hide##c2f"))
            coarse_fine_clear();

        ImGui::SetNextItemWidth(220);
        if (ImGui::SliderFloat("Z offset##c2f", &gC2FzOffset, -2.0f, 2.0f))
            rebuild_coarse_mesh_viz();
    }
}

// ---------------------------------------------------------------------------
// ring_post_c2f_diagnostic
//
// After each collapse, collect all ring_post vertices (those in FUV_post of
// the latest SheetData), run the full traced backward walk for each, then:
//  - log any group that lands on the same fine-mesh position (within 1e-8)
//  - show per-step chain info for every member of each duplicate group
//  - register polyscope structures for visual inspection
// ---------------------------------------------------------------------------
void ring_post_c2f_diagnostic()
{
#ifdef C2F_VIZ_DIAGNOSTIC
    if (gDecInfo.empty()) return;
    const int collapse_idx = (int)gDecInfo.size() - 1;
    const single_collapse_data & d = gDecInfo.back();

    // Collect unique ring_post global vertex indices across all sheets
    std::vector<int> ring_post_verts;
    for (auto & sd : d.sheets) {
        for (int r = 0; r < sd.FUV_post.rows(); r++) {
            for (int c = 0; c < 3; c++) {
                int gv = sd.subsetVIdx(sd.FUV_post(r, c));
                if (std::find(ring_post_verts.begin(), ring_post_verts.end(), gv)
                        == ring_post_verts.end())
                    ring_post_verts.push_back(gv);
            }
        }
    }
    if (ring_post_verts.empty()) return;

    // Run traced walk for each ring_post vertex
    std::vector<VertexWalkResult> results;
    results.reserve(ring_post_verts.size());
    for (int gv : ring_post_verts)
        results.push_back(traced_walk(gv));

    // Compute final 3D fine-mesh positions
    // Skip entries where traced_walk failed (init_ok=false) — final_BF is
    // uninitialized and may contain the infinity-cap vertex index (>= gVO.rows()).
    std::vector<Eigen::Vector3d> fine_pos(results.size(),
        Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()));
    for (int i = 0; i < (int)results.size(); i++) {
        const auto & r = results[i];
        if (!r.init_ok) continue;
        if (r.final_BF(0) >= (int)gVO.rows() ||
            r.final_BF(1) >= (int)gVO.rows() ||
            r.final_BF(2) >= (int)gVO.rows()) {
            fprintf(stderr,
                "[RING-POST-DIAG] collapse=%d vtx=%d: final_BF(%d,%d,%d) out of gVO range %d — skipped\n",
                collapse_idx, r.global_vtx,
                r.final_BF(0), r.final_BF(1), r.final_BF(2), (int)gVO.rows());
            continue;
        }
        fine_pos[i] = r.final_BC(0) * gVO.row(r.final_BF(0)).transpose()
                    + r.final_BC(1) * gVO.row(r.final_BF(1)).transpose()
                    + r.final_BC(2) * gVO.row(r.final_BF(2)).transpose();
    }

    // Find duplicate groups (same fine-mesh position within tolerance)
    const double tol = 1e-8;
    std::vector<bool> grouped(results.size(), false);
    bool any_dup = false;

    for (int i = 0; i < (int)results.size(); i++) {
        if (grouped[i] || !results[i].init_ok) continue;
        std::vector<int> grp = {i};
        for (int j = i+1; j < (int)results.size(); j++) {
            if (!grouped[j] && (fine_pos[i] - fine_pos[j]).norm() < tol)
                grp.push_back(j);
        }
        if ((int)grp.size() < 2) continue;

        any_dup = true;
        for (int idx : grp) grouped[idx] = true;

#ifdef SSP_LSCM_LOG
        fprintf(stderr,
            "\n[RING-POST-DUP] collapse=%d  %zu coarse verts → same fine pos"
            " (%.8f, %.8f, %.8f)\n",
            collapse_idx, grp.size(),
            fine_pos[i](0), fine_pos[i](1), fine_pos[i](2));

        for (int idx : grp) {
            const VertexWalkResult & r = results[idx];
            fprintf(stderr,
                "  coarse_vtx=%d  init_ok=%d  chain_len=%zu\n"
                "  final_BF=(%d,%d,%d)  final_BC=(%.8f,%.8f,%.8f)\n",
                r.global_vtx, (int)r.init_ok, r.chain.size(),
                r.final_BF(0), r.final_BF(1), r.final_BF(2),
                r.final_BC(0), r.final_BC(1), r.final_BC(2));
            for (int s = 0; s < (int)r.chain.size(); s++) {
                const WalkStep & st = r.chain[s];
                fprintf(stderr,
                    "    step[%d] dIdx=%d sheet=%d\n"
                    "      FIdx_in=%d  BC_in=(%.6f,%.6f,%.6f)"
                    "  BF_in=(%d,%d,%d)\n"
                    "      FIdx_out=%d  BC_out=(%.6f,%.6f,%.6f)"
                    "  BF_out=(%d,%d,%d)\n",
                    s, st.dIdx, st.sheet_id,
                    st.FIdx_in,
                    st.BC_in(0), st.BC_in(1), st.BC_in(2),
                    st.BF_in(0), st.BF_in(1), st.BF_in(2),
                    st.FIdx_out,
                    st.BC_out(0), st.BC_out(1), st.BC_out(2),
                    st.BF_out(0), st.BF_out(1), st.BF_out(2));
            }
        }
#endif
    }

#ifdef SSP_LSCM_LOG
    if (!any_dup) {
        fprintf(stderr,
            "[RING-POST-DUP] collapse=%d  no duplicates among %zu ring_post verts\n",
            collapse_idx, ring_post_verts.size());
    }
#endif

    // ---- Polyscope visualization ----

    // Ring_post coarse positions (current gV)
    MatrixXd rp_pts((int)ring_post_verts.size(), 3);
    for (int i = 0; i < (int)ring_post_verts.size(); i++)
        rp_pts.row(i) = gV.row(ring_post_verts[i]).leftCols(3);

    // Fine correspondence positions
    MatrixXd fp_pts((int)results.size(), 3);
    for (int i = 0; i < (int)results.size(); i++)
        fp_pts.row(i) = fine_pos[i].transpose();

    // Per-point duplicate flag (1 = duplicate, 0 = unique)
    MatrixXd dup_flag((int)results.size(), 1);
    for (int i = 0; i < (int)results.size(); i++)
        dup_flag(i, 0) = grouped[i] ? 1.0 : 0.0;

    auto * rp_cloud = polyscope::registerPointCloud("diag_ring_post", rp_pts);
    rp_cloud->setPointColor({1.0f, 0.85f, 0.1f});
    rp_cloud->setPointRadius(0.0006, true);
    rp_cloud->setEnabled(true);

    auto * fp_cloud = polyscope::registerPointCloud("diag_fine_corr", fp_pts);
    fp_cloud->setPointRadius(0.0006, true);
    fp_cloud->addScalarQuantity("is_duplicate", dup_flag.col(0))
        ->setEnabled(true);
    fp_cloud->setEnabled(true);

    // Arrows: each ring_post coarse vertex → its fine correspondence point
    MatrixXd arrowV(2 * (int)results.size(), 3);
    MatrixXi arrowE((int)results.size(), 2);
    for (int i = 0; i < (int)results.size(); i++) {
        arrowV.row(2*i)   = rp_pts.row(i);
        arrowV.row(2*i+1) = fp_pts.row(i);
        arrowE.row(i) << 2*i, 2*i+1;
    }
    polyscope::registerCurveNetwork("diag_c2f_arrows", arrowV, arrowE)
        ->setColor({0.5f, 0.5f, 1.0f})
        ->setRadius(0.00015, true)
        ->setEnabled(true);
#endif // C2F_VIZ_DIAGNOSTIC
}
