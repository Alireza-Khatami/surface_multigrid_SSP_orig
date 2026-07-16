#include "orient_faces_consistently.h"

#include <igl/read_triangle_mesh.h>
#include <igl/connect_boundary_to_infinity.h>
#include <igl/edge_flaps.h>
#include <igl/shortest_edge_and_midpoint.h>
#include <igl/per_vertex_point_to_plane_quadrics.h>
#include <igl/parallel_for.h>
#include <igl/collapse_edge.h>  // IGL_COLLAPSE_EDGE_NULL
#include "face_dead.h"
#include <igl/vertex_triangle_adjacency.h>

#include <SSP_qslim_optimal_collapse_edge_callbacks.h>
#include <always_try_never_care.h>
#include <decimate_func_types.h>
#include "meshlab_qslim_callbacks.h"

#ifdef C2F_VIZ_DIAGNOSTIC
#include <polyscope/polyscope.h>
#endif

#include <SSP_collapse_edge.h>
#include <single_collapse_data.h>
#include <partition_into_sheets.h>
#include <min_heap.h>

#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <tuple>
#include <limits>
#include <string>
#include <cmath>

#ifdef C2F_VIZ_DIAGNOSTIC
#include "visualizer.h"
#include "coarse_fine_viz.h"
#else
void coarse_fine_compute_and_save(const std::string & path);
void coarse_fine_save_bundle(const std::string & corrPath, const std::string & bundlePath);
#endif

using namespace Eigen;

// ---- SSP loop state (extern'd by visualizer.cpp) ----
MatrixXd gV;
MatrixXd gVO;           // original mesh vertices (before connect_boundary_to_infinity)
MatrixXi gF, gE;
MatrixXi gFO;           // original mesh faces
VectorXi gFaceFlipped;  // 1 = face was CW and got re-wound, 0 = already CCW
VectorXi gEMAP;
MatrixXi gEF, gEI;
igl::min_heap<std::tuple<double,int,int>> gQ;
VectorXi gEQ;
MatrixXd gC;
std::vector<single_collapse_data> gDecInfo;
std::vector<std::vector<int>> gDecIM;
VectorXi gFaceSheetID;
int gNumSheets    = 1;
std::vector<std::vector<int>> gVF;
std::vector<std::pair<int,int>> gSeamEdgeList;  // vertex pairs of seam (non-manifold) edges
int gTargetFaces  = 100;
int gCollapseCount = 0;
bool gFinished    = false;

// Decimation strategy: 0 = midpoint, 1 = qslim, 2 = meshlab
int gDecType = 1;

// Unified callbacks — set by init_ssp based on gDecType
typedef std::tuple<MatrixXd, RowVectorXd, double> Quadric;
static std::vector<Quadric> gQuadrics;
static int gQSlimV1 = -1, gQSlimV2 = -1;
static int gMlV1    = -1, gMlV2    = -1;
static MeshlabQEMConfig gMlCfg;
static decimate_cost_and_placement_func gCostFn;
static decimate_pre_collapse_func       gPreFn;
static decimate_post_collapse_func      gPostFn;

// ---- init ----
static void init_ssp(const std::string & mesh_path, int tarF)
{
    MatrixXd VO; MatrixXi FO;
    igl::read_triangle_mesh(mesh_path, VO, FO);
    std::cout << "Loaded: |V|=" << VO.rows() << "  |F|=" << FO.rows() << "\n";

    gTargetFaces = tarF;

    {
        MatrixXi FF; VectorXi C;
        int n_flipped = orient_faces_consistently(VO, FO, FF, C);
        // Record which faces were re-wound (CW → CCW) before overwriting FO
        gFaceFlipped.resize(FO.rows());
        for (int f = 0; f < FO.rows(); f++)
            gFaceFlipped(f) = (FF(f,0) != FO(f,0)) ? 1 : 0;
        FO = FF;
        std::cout << "orient_faces_consistently: " << n_flipped
                  << " / " << FO.rows() << " faces re-wound\n";
    }

    partition_into_sheets(FO, gFaceSheetID, gNumSheets);
    std::cout << "Sheets: " << gNumSheets << "\n";

    gVO = VO;   // save original mesh before boundary extension
    gFO = FO;

    igl::connect_boundary_to_infinity(VO, FO, gV, gF);

    // Face-count comparison immediately after connect_boundary_to_infinity,
    // before parallel_for, so it always runs regardless of later crashes.
    {
        int live = 0, null_faces = 0, inf_faces = 0;
        for (int f = 0; f < gF.rows(); f++) {
            int v0 = gF(f,0), v1 = gF(f,1), v2 = gF(f,2);
            if (is_face_dead(gF, f)) {
                null_faces++;
                fprintf(stderr, "[INIT null-face] f=%d  v=(%d,%d,%d)  in_orig_range=%s\n",
                    f, v0, v1, v2, (f < gFO.rows()) ? "YES" : "no(cap)");
                continue;
            }
            if (std::isinf(gV(v0,0)) || std::isinf(gV(v1,0)) || std::isinf(gV(v2,0))) { inf_faces++; continue; }
            live++;
        }
        fprintf(stderr, "[INIT face-count]"
                "  IGL_COLLAPSE_EDGE_NULL=%d"
                "  gFO.rows()=%d  gF.rows()=%d"
                "  live=%d  inf_cap=%d  null=%d"
                "  diff(gFO-live)=%d\n",
                (int)IGL_COLLAPSE_EDGE_NULL,
                (int)gFO.rows(), (int)gF.rows(),
                live, inf_faces, null_faces,
                (int)gFO.rows() - live);
    }

    igl::edge_flaps(gF, gE, gEMAP, gEF, gEI);

    {
        std::vector<std::vector<int>> VFi_unused;
        igl::vertex_triangle_adjacency((int)gV.rows(), gF, gVF, VFi_unused);
    }

    // Seam edges: edges where incident real faces span >1 sheet, or 3+ faces share the edge.
    // Only considers original faces (f < gFaceSheetID.size()); infinity faces are excluded.
    {
        const int nFO = (int)gFaceSheetID.size();
        std::map<std::pair<int,int>, std::vector<int>> edgeFaces;
        for (int f = 0; f < nFO; f++)
            for (int c = 0; c < 3; c++) {
                int u = gF(f,c), v = gF(f,(c+1)%3);
                edgeFaces[{std::min(u,v), std::max(u,v)}].push_back(f);
            }
        gSeamEdgeList.clear();
        for (auto & kv : edgeFaces) {
            const auto & fv = kv.second;
            if (fv.size() > 2)  // 3+ faces → non-manifold seam edge
                gSeamEdgeList.push_back(kv.first);
        }
        std::cout << "Seam edges: " << gSeamEdgeList.size() << "\n";
    }

    // Build cost/placement callbacks based on chosen strategy
    if (gDecType == 0) {
        // Midpoint: cost = edge length, placement = midpoint
        gCostFn = [](int e, const MatrixXd & V, const MatrixXi & F,
                     const MatrixXi & E, const VectorXi & EMAP,
                     const MatrixXi & EF, const MatrixXi & EI,
                     double & cost, RowVectorXd & p) {
            igl::shortest_edge_and_midpoint(e, V, F, E, EMAP, EF, EI, cost, p);
        };
        always_try_never_care(gPreFn, gPostFn);
        std::cout << "Decimation: midpoint\n";
    } else if (gDecType == 1) {
        // QSlim: quadric error metric, optimal placement
        igl::per_vertex_point_to_plane_quadrics(gV, gF, gEMAP, gEF, gEI, gQuadrics);
        SSP_qslim_optimal_collapse_edge_callbacks(
            gE, gQuadrics, gQSlimV1, gQSlimV2, gCostFn, gPreFn, gPostFn);
        std::cout << "Decimation: qslim\n";
    } else {
        // MeshLab QEM: area-weighted quadrics + boundary reinforcement
        meshlab_setup_callbacks(gV, gF, gE, gEF, gVF, gMlV1, gMlV2, gMlCfg, gCostFn, gPreFn, gPostFn);
        std::cout << "Decimation: meshlab\n";
    }

    gEQ = VectorXi::Zero(gE.rows());
    gC.resize(gE.rows(), gV.cols());
    {
        VectorXd costs(gE.rows());
        igl::parallel_for(gE.rows(), [&](const int e) {
            double cost; RowVectorXd p;
            gCostFn(e, gV, gF, gE, gEMAP, gEF, gEI, cost, p);
            gC.row(e) = p;
            costs(e) = cost;
        }, 10000);
        for (int e = 0; e < gE.rows(); e++)
            gQ.emplace(costs(e), e, 0);
    }

    gDecIM.resize(gF.rows());
    gDecInfo.reserve(FO.rows() - tarF + 1);
}

// ---- one step ----
bool do_next_step()
{
    if (gFinished) return false;

    for (int tries = 0; tries < 1'000'000; tries++) {
        if (gQ.empty() || std::get<0>(gQ.top()) == std::numeric_limits<double>::infinity()) {
            gFinished = true;
            int live = 0;
            for (int f = 0; f < gF.rows(); f++)
                if (!is_face_dead(gF, f) && !std::isinf(gV(gF(f,0), 0)))
                    live++;
            std::cerr << "[FINISHED] queue_exhausted  live_faces=" << live
                      << "  target=" << gTargetFaces
                      << "  collapses=" << gCollapseCount << "\n";
            return false;
        }
        int e, e1, e2, f1, f2;
        bool ok = SSP_collapse_edge(
            gCostFn, gPreFn, gPostFn,
            gV, gF, gE, gEMAP, gEF, gEI,
            gQ, gEQ, gC, e, e1, e2, f1, f2,
            gDecInfo, gDecIM,
            &gVF, gFaceSheetID);

        if (ok) {
            gCollapseCount++;
            std::cerr << "############## collapse ############ " << gCollapseCount << "\n";

            int live = 0;
            for (int f = 0; f < gF.rows(); f++)
                if (!is_face_dead(gF, f) && !std::isinf(gV(gF(f,0), 0)))
                    live++;
            if (live <= gTargetFaces) {
                gFinished = true;
                std::cerr << "[FINISHED] target_reached  live_faces=" << live
                          << "  target=" << gTargetFaces
                          << "  collapses=" << gCollapseCount << "\n";
            }
            return true;
        }
    }
    gFinished = true;
    std::cerr << "[FINISHED] max_tries_exceeded"
              << "  target=" << gTargetFaces
              << "  collapses=" << gCollapseCount << "\n";
    return false;
}

// ---- main ----
int main(int argc, char * argv[])
{
    if (argc < 3) {
        std::cerr << "usage: collapse_viz_bin  <mesh_path>  <target_faces>  [midpoint|qslim]\n";
        return 1;
    }
    if (argc >= 4) {
        std::string mode = argv[3];
        if (mode == "midpoint")     gDecType = 0;
        else if (mode == "qslim")    gDecType = 1;
        else if (mode == "meshlab")  gDecType = 2;
        else {
            std::cerr << "unknown decimation mode '" << mode << "' — use midpoint, qslim or meshlab\n";
            return 1;
        }
    }

    if (gDecType == 2) {
        if (gMlCfg.read("meshlab_qem.ini"))
            std::cout << "Loaded meshlab_qem.ini\n";
        else
            std::cout << "meshlab_qem.ini not found — using defaults\n";
    }

    init_ssp(argv[1], std::stoi(argv[2]));
    SSP_qslim_enable_log(true);   // activate ML_QEM_LOG output now that init cost pass is done
    if (gDecType == 2)
        meshlab_enable_cost_logging();

    // Build output file names from mesh stem: c2f_<stem>.txt, correspondence_<stem>.c2f
    std::string stem = argv[1];
    {
        size_t slash = stem.find_last_of("/\\");
        if (slash != std::string::npos) stem = stem.substr(slash + 1);
        size_t dot = stem.rfind('.');
        if (dot != std::string::npos) stem = stem.substr(0, dot);
    }
    const std::string c2f_path    = "c2f_" + stem + ".txt";
    const std::string bundle_path = "correspondence_" + stem + ".c2f";

#ifdef C2F_VIZ_DIAGNOSTIC
    polyscope::init();
    polyscope::state::userCallback = ui_callback;
    update_display();
    polyscope::show();
#else
    // Headless: run all collapses to target without any GUI.
    while (!gFinished)
        do_next_step();
#endif

    // Auto-save on exit regardless of C2F_VIZ_DIAGNOSTIC and regardless of
    // whether decimation reached the target face count.
    if (!gDecInfo.empty()) {
        coarse_fine_compute_and_save(c2f_path);
        coarse_fine_save_bundle(c2f_path, bundle_path);
    }

    return 0;
}
