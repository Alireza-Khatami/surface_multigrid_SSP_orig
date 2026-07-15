#include "orient_faces_consistently.h"

#include <igl/read_triangle_mesh.h>
#include <igl/connect_boundary_to_infinity.h>
#include <igl/edge_flaps.h>
#include <igl/shortest_edge_and_midpoint.h>
#include <igl/parallel_for.h>
#include <igl/collapse_edge.h>  // IGL_COLLAPSE_EDGE_NULL
#include <igl/vertex_triangle_adjacency.h>

#ifdef C2F_VIZ_DIAGNOSTIC
#include <polyscope/polyscope.h>
#endif

#include <SSP_collapse_edge.h>
#include <single_collapse_data.h>
#include <partition_into_sheets.h>
#include <always_try_never_care.h>
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
        FO = FF;
        std::cout << "orient_faces_consistently: " << n_flipped
                  << " / " << FO.rows() << " faces re-wound\n";
    }

    partition_into_sheets(FO, gFaceSheetID, gNumSheets);
    std::cout << "Sheets: " << gNumSheets << "\n";

    gVO = VO;   // save original mesh before boundary extension
    gFO = FO;

    igl::connect_boundary_to_infinity(VO, FO, gV, gF);
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

    gEQ = VectorXi::Zero(gE.rows());
    gC.resize(gE.rows(), gV.cols());
    {
        VectorXd costs(gE.rows());
        igl::parallel_for(gE.rows(), [&](const int e) {
            double cost = e;
            RowVectorXd p(1, 3);
            igl::shortest_edge_and_midpoint(e, gV, gF, gE, gEMAP, gEF, gEI, cost, p);
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

    decimate_pre_collapse_func  always_try;
    decimate_post_collapse_func never_care;
    always_try_never_care(always_try, never_care);

    for (int tries = 0; tries < 1'000'000; tries++) {
        if (gQ.empty() || std::get<0>(gQ.top()) == std::numeric_limits<double>::infinity()) {
            gFinished = true;
            return false;
        }
        int e, e1, e2, f1, f2;
        bool ok = SSP_collapse_edge(
            igl::shortest_edge_and_midpoint, always_try, never_care,
            gV, gF, gE, gEMAP, gEF, gEI,
            gQ, gEQ, gC, e, e1, e2, f1, f2,
            gDecInfo, gDecIM,
            &gVF, gFaceSheetID);

        if (ok) {
            gCollapseCount++;

            // Count live (non-inf) faces for stopping condition
            int live = 0;
            for (int f = 0; f < gF.rows(); f++)
                if (gF(f,0) != IGL_COLLAPSE_EDGE_NULL && !std::isinf(gV(gF(f,0), 0)))
                    live++;
            if (live <= gTargetFaces) gFinished = true;

            return true;
        }
    }
    gFinished = true;
    return false;
}

// ---- main ----
int main(int argc, char * argv[])
{
    if (argc < 3) {
        std::cerr << "usage: collapse_viz_bin  <mesh_path>  <target_faces>\n";
        return 1;
    }

    init_ssp(argv[1], std::stoi(argv[2]));

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
