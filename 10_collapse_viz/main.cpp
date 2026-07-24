#include "orient_faces_consistently.h"

#include <igl/read_triangle_mesh.h>
#include <igl/remove_unreferenced.h>
#include <igl/writeOBJ.h>
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
std::vector<double> gInitCosts;               // initial cost per edge (index = gE row)
int gTargetFaces  = 100;
int gCollapseCount = 0;
int gSeamCollapseCount = 0;   // collapses where the edge was still a seam at collapse time
int gSeamAttemptCount = 0;    // pre_collapse calls where edge was still a seam (success or not)
bool gLastCollapseWasSeam = false;
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
        gInitCosts.assign(costs.data(), costs.data() + costs.size());
        for (int e = 0; e < gE.rows(); e++)
            gQ.emplace(costs(e), e, 0);
    }

    gDecIM.resize(gF.rows());
    gDecInfo.reserve(FO.rows() - tarF + 1);
}

// ---- seam edge cost diagnostic ----
// Called once after init_ssp. For each seam edge (3+ incident faces) prints
// the edge index, vertex pair, endpoint positions and collapse cost.
static void print_seam_edge_costs(const std::string & out_path = "seam_edge_costs.txt")
{
    if (gSeamEdgeList.empty()) return;

    FILE * fout = fopen(out_path.c_str(), "w");
    if (!fout) {
        fprintf(stderr, "[SEAM-COSTS] could not open %s for writing\n", out_path.c_str());
        return;
    }

    // Build vertex-pair → edge index lookup from gE
    std::map<std::pair<int,int>, int> edgeIdx;
    for (int e = 0; e < gE.rows(); e++) {
        int u = gE(e,0), v = gE(e,1);
        edgeIdx[{std::min(u,v), std::max(u,v)}] = e;
    }

    // Sorted copy of all initial costs for rank lookup (1-based: rank 1 = lowest cost)
    std::vector<double> sorted_costs = gInitCosts;
    std::sort(sorted_costs.begin(), sorted_costs.end());
    const int n_edges = (int)sorted_costs.size();

    fprintf(fout, "seam_edges=%d  total_edges=%d\n",
            (int)gSeamEdgeList.size(), n_edges);
    for (int i = 0; i < (int)gSeamEdgeList.size(); i++) {
        int u = gSeamEdgeList[i].first, v = gSeamEdgeList[i].second;
        auto it = edgeIdx.find({u, v});
        if (it == edgeIdx.end()) {
            fprintf(fout, "[%d] v=(%d,%d)  edge=NOT_FOUND\n", i, u, v);
            continue;
        }
        int e = it->second;
        double cost; RowVectorXd p;
        gCostFn(e, gV, gF, gE, gEMAP, gEF, gEI, cost, p);

        // rank = 1-based position in the sorted queue (number of edges with lower cost + 1)
        int rank = (int)(std::lower_bound(sorted_costs.begin(), sorted_costs.end(), cost)
                         - sorted_costs.begin()) + 1;

        Eigen::RowVectorXd va = gV.row(u), vb = gV.row(v);
        fprintf(fout,
            "[%d] e=%d  v=(%d,%d)"
            "  va=(%.6g,%.6g,%.6g)  vb=(%.6g,%.6g,%.6g)"
            "  cost=%.6g  rank=%d/%d\n",
            i, e, u, v,
            va(0), va(1), va(2),
            vb(0), vb(1), vb(2),
            cost, rank, n_edges);
    }

    fclose(fout);
    fprintf(stderr, "[SEAM-COSTS] wrote %d seam edges to %s\n",
            (int)gSeamEdgeList.size(), out_path.c_str());
}

// ---- export simplified mesh ----
static void save_simplified_mesh(const std::string & path)
{
    // Collect live faces: not dead, not incident to the infinity cap vertex
    std::vector<std::array<int,3>> rows;
    for (int f = 0; f < gF.rows(); f++) {
        if (is_face_dead(gF, f)) continue;
        int v0 = gF(f,0), v1 = gF(f,1), v2 = gF(f,2);
        if (std::isinf(gV(v0,0)) || std::isinf(gV(v1,0)) || std::isinf(gV(v2,0))) continue;
        rows.push_back({v0, v1, v2});
    }
    MatrixXi Flive((int)rows.size(), 3);
    for (int i = 0; i < (int)rows.size(); i++)
        Flive.row(i) << rows[i][0], rows[i][1], rows[i][2];

    // Strip the infinity cap vertex and compact vertex indices
    MatrixXd Vout; MatrixXi Fout; VectorXi I, J;
    igl::remove_unreferenced(gV.leftCols(3), Flive, Vout, Fout, I, J);

    if (!igl::writeOBJ(path, Vout, Fout))
        fprintf(stderr, "[SAVE-MESH] writeOBJ failed: %s\n", path.c_str());
    else
        fprintf(stderr, "[SAVE-MESH] wrote %d faces  %d verts  → %s\n",
            (int)Fout.rows(), (int)Vout.rows(), path.c_str());
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
                      << "  collapses=" << gCollapseCount
                      << "  seam_attempts=" << gSeamAttemptCount
                      << "  seam_collapses=" << gSeamCollapseCount << "\n";
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
            if (gLastCollapseWasSeam) gSeamCollapseCount++;
            std::cerr << "############## collapse ############ " << gCollapseCount
                      << "  seam=" << gSeamCollapseCount << "/" << gCollapseCount << "\n";

            int live = 0;
            for (int f = 0; f < gF.rows(); f++)
                if (!is_face_dead(gF, f) && !std::isinf(gV(gF(f,0), 0)))
                    live++;
            if (live <= gTargetFaces) {
                gFinished = true;
                std::cerr << "[FINISHED] target_reached  live_faces=" << live
                          << "  target=" << gTargetFaces
                          << "  collapses=" << gCollapseCount
                          << "  seam_attempts=" << gSeamAttemptCount
                          << "  seam_collapses=" << gSeamCollapseCount << "\n";
            }
            return true;
        }
    }
    gFinished = true;
    std::cerr << "[FINISHED] max_tries_exceeded"
              << "  target=" << gTargetFaces
              << "  collapses=" << gCollapseCount
              << "  seam_attempts=" << gSeamAttemptCount
              << "  seam_collapses=" << gSeamCollapseCount << "\n";
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

    // Wrap gPreFn to detect seam collapses at collapse time.
    // A seam edge is one where 3+ live real faces share the same vertex pair in the
    // CURRENT topology (rechecked every collapse, not just from the initial seam list).
    {
        auto orig_pre = gPreFn;
        gPreFn = [orig_pre](
            const MatrixXd & V, const MatrixXi & F, const MatrixXi & E,
            const VectorXi & EMAP, const MatrixXi & EF, const MatrixXi & EI,
            const igl::min_heap<std::tuple<double,int,int>> & Q,
            const VectorXi & EQ, const MatrixXd & C, const int e) -> bool
        {
            bool result = orig_pre(V, F, E, EMAP, EF, EI, Q, EQ, C, e);
            if (result) {
                int u = E(e,0), v = E(e,1);
                const int nFSheet = (int)gFaceSheetID.size();
                // Build set of live real faces incident to u, then count those also in v's list
                std::set<int> u_faces;
                for (int f : gVF[u])
                    if (f < nFSheet && !is_face_dead(F, f))
                        u_faces.insert(f);
                int shared = 0;
                for (int f : gVF[v])
                    if (u_faces.count(f)) shared++;
                gLastCollapseWasSeam = (shared > 2);
                if (gLastCollapseWasSeam) gSeamAttemptCount++;
            }
            return result;
        };
    }

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

    print_seam_edge_costs("seam_edge_costs_" + stem + ".txt");
    SSP_seam_log_open(("seam_diag_" + stem + ".txt").c_str());

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

    SSP_seam_log_close();

    // Export the simplified mesh regardless of how many collapses happened.
    save_simplified_mesh("simplified_" + stem + ".obj");

    // Auto-save on exit regardless of C2F_VIZ_DIAGNOSTIC and regardless of
    // whether decimation reached the target face count.
    if (!gDecInfo.empty()) {
        coarse_fine_compute_and_save(c2f_path);
        coarse_fine_save_bundle(c2f_path, bundle_path);
    }

    return 0;
}
