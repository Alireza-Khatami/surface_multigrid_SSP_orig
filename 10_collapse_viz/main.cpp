#include "visualizer.h"

#include <igl/read_triangle_mesh.h>
#include <igl/connect_boundary_to_infinity.h>
#include <igl/edge_flaps.h>
#include <igl/shortest_edge_and_midpoint.h>
#include <igl/parallel_for.h>
#include <igl/collapse_edge.h>  // IGL_COLLAPSE_EDGE_NULL
#include <igl/vertex_triangle_adjacency.h>

#include <polyscope/polyscope.h>

#include <SSP_collapse_edge.h>
#include <single_collapse_data.h>
#include <partition_into_sheets.h>
#include <always_try_never_care.h>
#include <min_heap.h>

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <tuple>
#include <limits>
#include <string>
#include <cmath>

using namespace Eigen;

// ---- SSP loop state (extern'd by visualizer.cpp) ----
MatrixXd gV;
MatrixXi gF, gE;
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

    partition_into_sheets(FO, gFaceSheetID, gNumSheets);
    std::cout << "Sheets: " << gNumSheets << "\n";

    igl::connect_boundary_to_infinity(VO, FO, gV, gF);
    igl::edge_flaps(gF, gE, gEMAP, gEF, gEI);

    {
        std::vector<std::vector<int>> VFi_unused;
        igl::vertex_triangle_adjacency((int)gV.rows(), gF, gVF, VFi_unused);
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

    polyscope::init();
    polyscope::state::userCallback = ui_callback;
    update_display();
    polyscope::show();
    return 0;
}
