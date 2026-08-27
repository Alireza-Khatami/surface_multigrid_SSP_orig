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
#include <polyscope/point_cloud.h>
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
#include <cstdint>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <filesystem>

#ifdef C2F_VIZ_DIAGNOSTIC
#include "visualizer.h"
#include "coarse_fine_viz.h"
#else
void coarse_fine_compute_and_save(const std::string & path);
void coarse_fine_save_bundle(const std::string & corrPath, const std::string & bundlePath);
#endif

#include "face_sample_tracker.h"
#include "load_matstruct.h"

using namespace Eigen;

// ---- output path globals (extern'd by visualizer.cpp) ----
std::string gStem;    // mesh filename without extension, e.g. "bunny"
std::string gOutDir;  // output directory with trailing separator, e.g. "output/"

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
std::vector<std::set<int>> gVertexStructIDs;  // per-vertex struct ID sets from .ma_struct file
FILE* gStructGateLog = nullptr;               // dedicated log file for struct-ID gate decisions
std::vector<std::pair<int,int>> gSeamEdgeList;  // vertex pairs of seam (non-manifold) edges
std::vector<double> gInitCosts;               // initial cost per edge (index = gE row)

// Stale chains: naked l-element edges in the fine mesh that SSP must not touch.
// Populated by detect_stale_chains() inside init_ssp, before any collapse.
std::vector<std::vector<int>> gStaleChains;    // each chain = ordered vertex ID sequence
std::unordered_set<int>       gStaleVertexSet; // fast lookup used by the pre-collapse lock

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

// ---- stale chain detection ----
// Loads an OBJ file in a single pass, extracting vertex positions (V), triangle
// faces (F), and raw line-element edges (l_edges, 0-based).  This replaces the
// igl::read_triangle_mesh call so that 'l' elements are never seen by libigl's
// parser (which would print a warning for every such line).
static bool load_obj_vfl(
    const std::string & path,
    MatrixXd & V,
    MatrixXi & F,
    std::vector<std::pair<int,int>> & l_edges)
{
    std::ifstream fh(path);
    if (!fh.is_open()) {
        fprintf(stderr, "[OBJ] cannot open '%s'\n", path.c_str());
        return false;
    }
    std::vector<std::array<double,3>> verts;
    std::vector<std::array<int,3>>   faces;
    std::string line;
    while (std::getline(fh, line)) {
        if (line.empty()) continue;
        // strip leading whitespace
        size_t s = line.find_first_not_of(" \t\r");
        if (s == std::string::npos || line[s] == '#') continue;
        char token = line[s];
        if (token == 'v' && (s+1 < line.size()) && std::isspace((unsigned char)line[s+1])) {
            std::istringstream ss(line.substr(s+1));
            double x, y, z; ss >> x >> y >> z;
            verts.push_back({x, y, z});
        } else if (token == 'f' && (s+1 < line.size()) && std::isspace((unsigned char)line[s+1])) {
            std::istringstream ss(line.substr(s+1));
            std::array<int,3> tri;
            for (int k = 0; k < 3; k++) {
                std::string tok; ss >> tok;
                // handle v, v/vt, v/vt/vn, v//vn
                tri[k] = std::stoi(tok) - 1;  // 1-based → 0-based
            }
            faces.push_back(tri);
        } else if (token == 'l' && (s+1 < line.size()) && std::isspace((unsigned char)line[s+1])) {
            std::istringstream ss(line.substr(s+1));
            std::vector<int> vs;
            int vi;
            while (ss >> vi) vs.push_back(vi - 1);
            for (int i = 0; i + 1 < (int)vs.size(); i++)
                l_edges.push_back({vs[i], vs[i+1]});
        }
        // 'vn', 'vt', 'usemtl', etc. are silently skipped — we don't need them
    }
    V.resize((int)verts.size(), 3);
    for (int i = 0; i < (int)verts.size(); i++)
        V.row(i) << verts[i][0], verts[i][1], verts[i][2];
    F.resize((int)faces.size(), 3);
    for (int i = 0; i < (int)faces.size(); i++)
        F.row(i) << faces[i][0], faces[i][1], faces[i][2];
    return true;
}

// Filters raw l-element edges to keep only those that appear in no triangle face,
// then builds a maximal chain decomposition of those naked edges.
// Populates gStaleChains and gStaleVertexSet.
static void detect_stale_chains(
    const std::vector<std::pair<int,int>> & l_edges_raw,
    const MatrixXi & FO)
{
    // 1. Already have l-edges from the OBJ parse; nothing to re-read.
    std::vector<std::pair<int,int>> raw_edges = l_edges_raw;
    if (raw_edges.empty()) {
        std::cout << "[STALE] no l-elements found — no stale chains\n";
        return;
    }
    std::cout << "[STALE] " << raw_edges.size() << " l-edges parsed\n";

    // 2. Build a set of all face edges so we can exclude them.
    //    A true naked/stale edge must NOT appear in any triangle face.
    std::set<std::pair<int,int>> face_edges;
    for (int f = 0; f < FO.rows(); f++)
        for (int c = 0; c < 3; c++) {
            int u = FO(f, c), v = FO(f, (c + 1) % 3);
            face_edges.insert({std::min(u, v), std::max(u, v)});
        }

    {
        std::vector<std::pair<int,int>> naked;
        naked.reserve(raw_edges.size());
        int n_face = 0;
        for (auto & e : raw_edges) {
            auto key = std::make_pair(std::min(e.first, e.second),
                                      std::max(e.first, e.second));
            if (face_edges.count(key)) { n_face++; continue; }
            naked.push_back(e);
        }
        if (n_face)
            std::cout << "[STALE] " << n_face << " l-edges skipped (also a face edge)\n";
        raw_edges = std::move(naked);
    }

    if (raw_edges.empty()) {
        std::cout << "[STALE] all l-edges were face edges — no stale chains\n";
        return;
    }
    std::cout << "[STALE] " << raw_edges.size() << " naked l-edges remain\n";

    // 3. Build adjacency from naked edges (deduplicated)
    std::map<int, std::vector<int>> adj;
    for (auto & e : raw_edges) {
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
    }
    for (auto & kv : adj) {
        auto & nb = kv.second;
        std::sort(nb.begin(), nb.end());
        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
    }

    // 4. Build maximal chains (same algorithm as visualize_naked_edge_chains.py)
    std::set<std::pair<int,int>> visited;
    auto mark_edge = [&](int u, int v) {
        visited.insert({std::min(u,v), std::max(u,v)});
    };
    auto edge_visited = [&](int u, int v) -> bool {
        return visited.count({std::min(u,v), std::max(u,v)}) > 0;
    };

    // Trace one chain starting from 'start' stepping first to 'nxt'
    auto trace = [&](int start, int nxt) -> std::vector<int> {
        std::vector<int> chain = {start};
        int prev = start, cur = nxt;
        while (true) {
            chain.push_back(cur);
            mark_edge(prev, cur);
            const auto & nbrs = adj[cur];
            if ((int)nbrs.size() != 2) break;   // endpoint or junction — stop
            int next_v = (nbrs[0] != prev) ? nbrs[0] : nbrs[1];
            if (edge_visited(cur, next_v)) break; // loop closed — stop
            prev = cur; cur = next_v;
        }
        return chain;
    };

    // Pass 0: endpoints (degree 1), Pass 1: junctions (degree > 2)
    for (int pass = 0; pass < 2; pass++) {
        for (auto & kv : adj) {
            int v   = kv.first;
            int deg = (int)kv.second.size();
            if (pass == 0 ? deg != 1 : deg <= 2) continue;
            for (int nb : kv.second)
                if (!edge_visited(v, nb))
                    gStaleChains.push_back(trace(v, nb));
        }
    }

    // Pass 2: closed loops — unvisited edges among degree-2 vertices
    for (auto & kv : adj) {
        if ((int)kv.second.size() != 2) continue;
        int v = kv.first;
        for (int nb : kv.second) {
            if (!edge_visited(v, nb)) {
                auto chain = trace(v, nb);
                if (chain.front() != chain.back())
                    chain.push_back(chain.front()); // close the loop
                gStaleChains.push_back(chain);
                break;
            }
        }
    }

    // 5. Populate vertex set
    for (const auto & chain : gStaleChains)
        for (int vid : chain)
            gStaleVertexSet.insert(vid);

    std::cout << "[STALE] " << gStaleChains.size() << " chains  ("
              << gStaleVertexSet.size() << " unique vertices protected)\n";
    for (int i = 0; i < (int)gStaleChains.size(); i++)
        printf("[STALE]   chain[%d]: %d vertices\n", i, (int)gStaleChains[i].size());
}

// ---- init ----
static void init_ssp(const std::string & mesh_path, int tarF, const std::string & out_dir)
{
    MatrixXd VO; MatrixXi FO;
    std::vector<std::pair<int,int>> l_edges;
    {
        // Use our own parser for .obj so that 'l' elements are captured and
        // libigl's "ignored non-comment line" warning is never triggered.
        // For all other formats (.off, .stl, .ply, ...) fall back to libigl —
        // those formats don't have 'l' elements so no warning would appear anyway.
        std::string ext = mesh_path;
        { size_t dot = ext.rfind('.'); ext = (dot != std::string::npos) ? ext.substr(dot+1) : ""; }
        for (char & c : ext) c = (char)std::tolower((unsigned char)c);

        bool ok = (ext == "obj") ? load_obj_vfl(mesh_path, VO, FO, l_edges)
                                 : igl::read_triangle_mesh(mesh_path, VO, FO);
        if (!ok) return;
    }
    std::cout << "Loaded: |V|=" << VO.rows() << "  |F|=" << FO.rows()
              << "  |l|=" << l_edges.size() << "\n";

    detect_stale_chains(l_edges, FO);

    gTargetFaces = tarF;

    {
        MatrixXi FF; VectorXi C;

        // Derive stem once for all exports in this block.
        std::string stem = mesh_path;
        { size_t sl = stem.find_last_of("/\\"); if (sl != std::string::npos) stem = stem.substr(sl+1); }
        { size_t dot = stem.rfind('.'); if (dot != std::string::npos) stem = stem.substr(0, dot); }

        // Export raw mesh before any re-winding.
        const std::string raw_path = out_dir + "raw_" + stem + ".obj";
        if (!igl::writeOBJ(raw_path, VO, FO))
            fprintf(stderr, "[RAW] writeOBJ failed: %s\n", raw_path.c_str());
        else
            fprintf(stderr, "[RAW] raw mesh -> %s  (%d verts, %d faces)\n",
                    raw_path.c_str(), (int)VO.rows(), (int)FO.rows());

        int n_flipped = orient_faces_consistently(VO, FO, FF, C);
        // Record which faces were re-wound (CW → CCW) before overwriting FO
        gFaceFlipped.resize(FO.rows());
        for (int f = 0; f < FO.rows(); f++)
            gFaceFlipped(f) = (FF(f,0) != FO(f,0)) ? 1 : 0;
        FO = FF;
        std::cout << "orient_faces_consistently: " << n_flipped
                  << " / " << FO.rows() << " faces re-wound\n";

        // Export the consistently-oriented mesh so the re-winding can be inspected.
        const std::string oriented_path = out_dir + "oriented_" + stem + ".obj";
        if (!igl::writeOBJ(oriented_path, VO, FO))
            fprintf(stderr, "[ORIENT] writeOBJ failed: %s\n", oriented_path.c_str());
        else
            fprintf(stderr, "[ORIENT] oriented mesh -> %s  (%d faces re-wound)\n",
                    oriented_path.c_str(), n_flipped);
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
            gE, gQuadrics, gQSlimV1, gQSlimV2, gVF, gCostFn, gPreFn, gPostFn);
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

// ---- live face count (mirrors save_simplified_mesh: excludes dead + all-inf cap faces) ----
static int count_live_faces()
{
    int live = 0;
    for (int f = 0; f < gF.rows(); f++) {
        if (is_face_dead(gF, f)) continue;
        int v0 = gF(f,0), v1 = gF(f,1), v2 = gF(f,2);
        if (std::isinf(gV(v0,0)) || std::isinf(gV(v1,0)) || std::isinf(gV(v2,0))) continue;
        live++;
    }
    return live;
}

// ---- one step ----
bool do_next_step()
{
    if (gFinished) return false;

    face_flip_tracker_pre_update();
    vertex_watch_pre_step();

    for (int tries = 0; tries < 1'000'000; tries++) {
        if (gQ.empty() || std::get<0>(gQ.top()) == std::numeric_limits<double>::infinity()) {
            gFinished = true;
            std::cerr << "[FINISHED] queue_exhausted  live_faces=" << count_live_faces()
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
            // s = survivor (lower vertex index, kept + repositioned)
            // d = absorbed  (higher index, all face refs remapped to s, then gone)
            // f1, f2 = the two flap faces NULLed out by this collapse
            // gV.row(s) is already at the new placement position at this point
            int s = std::min(gE(e,0), gE(e,1));
            int d = std::max(gE(e,0), gE(e,1));
            if (gStructGateLog) {
                const int nS = (int)gVertexStructIDs.size();
                auto sids = [&](int v) -> std::string {
                    if (v >= nS) return "INF";
                    std::string r = "{";
                    for (int x : gVertexStructIDs[v]) { r += std::to_string(x); r += ','; }
                    if (r.size() > 1) r.back() = '}'; else r += '}';
                    return r;
                };
                bool match = (s < nS && d < nS && gVertexStructIDs[s] == gVertexStructIDs[d]);
                fprintf(gStructGateLog,
                    "[STRUCT-COLLAPSE #%d] e=%d  s=%d ids=%s  d=%d ids=%s  match=%d  seam=%d\n",
                    gCollapseCount, e, s, sids(s).c_str(), d, sids(d).c_str(),
                    match ? 1 : 0, gLastCollapseWasSeam ? 1 : 0);
                fflush(gStructGateLog);
            }
            vertex_watch_check_collapse(s, d);
            sample_tracker_update();
            face_flip_tracker_post_update();
            fprintf(stderr,
                "[COLLAPSE #%d] e=%d  kept=v%d  gone=v%d"
                "  new_pos=(%.4g,%.4g,%.4g)"
                "  deleted_faces=[f%d,f%d]"
                "  seam=%d/%d\n",
                gCollapseCount, e, s, d,
                gV(s,0), gV(s,1), gV(s,2),
                f1, f2,
                gSeamCollapseCount, gCollapseCount);

            int live = count_live_faces();
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
    std::cerr << "[FINISHED] max_tries_exceeded  live_faces=" << count_live_faces()
              << "  target=" << gTargetFaces
              << "  collapses=" << gCollapseCount
              << "  seam_attempts=" << gSeamAttemptCount
              << "  seam_collapses=" << gSeamCollapseCount << "\n";
    return false;
}

// ---- main ----
int main(int argc, char * argv[])
{
    // Parse all named arguments; every arg has a default so none are required.
    std::string meshPath         = "bunny.obj";
    int         targetFaces      = 285;
    std::string namedMode;
    int         gNSamplesTotal   = -1;   // -1 = not given → sampling disabled
    std::string namedOutDir;
    bool        validityChecks   = false;
    std::string matstructPath;      // optional: path to .ma_struct file for struct-ID collapse gating
    bool        matStructCheck = false;  // --mat_struct_check: enable struct-ID collapse gate
    std::string traceVerticesPath;  // optional: text file with one fine_vertex_id per line
    int         trackFaceFlip     = -1;  // --track_face_flip <idx>


    //usage
    // [--mesh_path PATH]       default: bunny.obj
    // [--target_faces N]       default: 285
    // [--mode midpoint|qslim|meshlab]   default: qslim
    // [--n_samples_total N]    optional — omit to disable sampling entirely
    // [--output_dir PATH]      default: .
    // [--validity-checks]
    // [--trace_vertices PATH]  text file: one fine_vertex_id per line; enables per-step walk trace


    for (int i = 1; i < argc; ++i) {
        if (!argv[i]) continue;
        std::string a = argv[i];
        if (a == "--validity-checks") {
            validityChecks = true;
        } else if (a == "--mat_struct_check") {
            matStructCheck = true;
        } else if (i + 1 < argc) {
            if      (a == "--mesh_path")        meshPath          = argv[i+1];
            else if (a == "--target_faces")     targetFaces       = std::stoi(argv[i+1]);
            else if (a == "--mode")             namedMode         = argv[i+1];
            else if (a == "--n_samples_total")  gNSamplesTotal    = std::stoi(argv[i+1]);
            else if (a == "--output_dir")       namedOutDir       = argv[i+1];
            else if (a == "--matstruct_path")   matstructPath     = argv[i+1];
            else if (a == "--trace_vertices")   traceVerticesPath = argv[i+1];
            else if (a == "--track_face_flip")  trackFaceFlip     = std::stoi(argv[i+1]);
            else { continue; }
            ++i;
        }
    }

    if (matStructCheck && matstructPath.empty()) {
        std::cerr << "[ERROR] --mat_struct_check requires --matstruct_path <path/to/file.ma_struct>\n";
        return 1;
    }

    SSP_validity_checks_enable(validityChecks);
    std::cout << "Validity checks: " << (validityChecks ? "ENABLED" : "DISABLED") << "\n";

    if (!namedMode.empty()) {
        if      (namedMode == "midpoint") gDecType = 0;
        else if (namedMode == "qslim")    gDecType = 1;
        else if (namedMode == "meshlab")  gDecType = 2;
        else {
            std::cerr << "unknown --mode '" << namedMode << "' — use midpoint, qslim or meshlab\n"
                      << "usage: collapse_viz_bin"
                         "  [--mesh_path PATH]  [--target_faces N]"
                         "  [--mode midpoint|qslim|meshlab]"
                         "  [--n_samples_total N]  [--output_dir PATH]  [--validity-checks]\n";
            return 1;
        }
    }

    if (gDecType == 2) {
        if (gMlCfg.read("meshlab_qem.ini"))
            std::cout << "Loaded meshlab_qem.ini\n";
        else
            std::cout << "meshlab_qem.ini not found — using defaults\n";
    }

    // Build output paths before init_ssp so out_dir is available inside it.
    std::string stem = meshPath;
    {
        size_t slash = stem.find_last_of("/\\");
        if (slash != std::string::npos) stem = stem.substr(slash + 1);
        size_t dot = stem.rfind('.');
        if (dot != std::string::npos) stem = stem.substr(0, dot);
    }
    std::string out_dir = namedOutDir.empty() ? "." : namedOutDir;
    if (!out_dir.empty() && out_dir.back() != '/' && out_dir.back() != '\\')
        out_dir += '/';
    gStem   = stem;
    gOutDir = out_dir;

    {
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);
        if (ec)
            fprintf(stderr, "[OUT] warning: could not create output dir '%s': %s\n",
                    out_dir.c_str(), ec.message().c_str());
        else
            printf("[OUT] output dir: %s\n", out_dir.c_str());
    }

    init_ssp(meshPath.c_str(), targetFaces, out_dir);

    // Load per-vertex struct IDs from .ma_struct file (optional).
    if (!matstructPath.empty()) {
        if (load_matstruct(matstructPath, gVertexStructIDs)) {
            std::cout << "Struct IDs loaded from " << matstructPath
                      << "  (" << gVertexStructIDs.size() << " vertices)\n";
            if (matStructCheck)
                std::cout << "  --mat_struct_check ON: collapses blocked when struct ID sets differ\n";
            else
                std::cout << "  --mat_struct_check OFF: struct ID gate inactive\n";
        } else {
            std::cerr << "[ERROR] load_matstruct failed for: " << matstructPath << "\n";
            if (matStructCheck) {
                std::cerr << "  --mat_struct_check is ON but struct IDs could not be loaded — aborting.\n";
                return 1;
            }
        }
    }

    // Helper: format a std::set<int> as "{1,2,3}" for logging.
    auto struct_ids_str = [](const std::set<int>& s) -> std::string {
        std::string r = "{";
        for (int x : s) { r += std::to_string(x); r += ','; }
        if (r.size() > 1) r.back() = '}'; else r += '}';
        return r;
    };

    // Struct ID gate: block collapse (v0, v1) when their struct ID sets differ.
    // Only active when both --matstruct_path and --mat_struct_check are provided.
    // Skips the infinity cap vertex (index >= gVertexStructIDs.size()).
    if (!gVertexStructIDs.empty() && matStructCheck) {
        // Try out_dir first; fall back to cwd so the file always lands somewhere writable.
        std::string gate_log_path = out_dir + "struct_gate_log.txt";
        gStructGateLog = fopen(gate_log_path.c_str(), "w");
        if (!gStructGateLog) {
            gate_log_path = "struct_gate_log.txt";
            gStructGateLog = fopen(gate_log_path.c_str(), "w");
        }
        if (gStructGateLog) {
            fprintf(gStructGateLog,
                "# struct-ID gate log\n"
                "# format: [GATE] collapse=#  e=<edge>  u=<vtx> ids=<set>  v=<vtx> ids=<set>  PASS|BLOCK\n"
                "# [STRUCT-COLLAPSE] lines log every successful collapse's struct IDs\n\n");
            fflush(gStructGateLog);
            std::cout << "  struct_gate_log -> " << gate_log_path << "\n";
        } else {
            std::cerr << "[ERROR] could not open struct_gate_log at " << gate_log_path
                      << " — gate will run but decisions won't be logged\n";
        }

        auto orig_pre = gPreFn;
        gPreFn = [orig_pre, struct_ids_str](
            const MatrixXd & V, const MatrixXi & F, const MatrixXi & E,
            const VectorXi & EMAP, const MatrixXi & EF, const MatrixXi & EI,
            const igl::min_heap<std::tuple<double,int,int>> & Q,
            const VectorXi & EQ, const MatrixXd & C, const int e) -> bool
        {
            if (!orig_pre(V, F, E, EMAP, EF, EI, Q, EQ, C, e)) return false;
            const int u = E(e,0), v = E(e,1);
            const int nStruct = (int)gVertexStructIDs.size();
            if (u >= nStruct || v >= nStruct) return true;  // infinity cap vertex — always allow
            bool ok = (gVertexStructIDs[u] == gVertexStructIDs[v]);
            if (gStructGateLog) {
                fprintf(gStructGateLog,
                    "[GATE] collapse=#%d  e=%d  u=%d ids=%s  v=%d ids=%s  %s\n",
                    gCollapseCount + 1, e, u,
                    struct_ids_str(gVertexStructIDs[u]).c_str(),
                    v,
                    struct_ids_str(gVertexStructIDs[v]).c_str(),
                    ok ? "PASS" : "BLOCK");
                fflush(gStructGateLog);
            }
            return ok;
        };
    }

    SSP_qslim_enable_log(true);   // activate ML_QEM_LOG output now that init cost pass is done
    if (gDecType == 2)
        meshlab_enable_cost_logging();

    // Lock: no collapse may involve a stale chain vertex (protects l-element chain edges).
    // Any collapse where either endpoint is a stale vertex is permanently blocked.
    if (!gStaleVertexSet.empty()) {
        auto stale_pre = gPreFn;
        gPreFn = [stale_pre](
            const MatrixXd & V, const MatrixXi & F, const MatrixXi & E,
            const VectorXi & EMAP, const MatrixXi & EF, const MatrixXi & EI,
            const igl::min_heap<std::tuple<double,int,int>> & Q,
            const VectorXi & EQ, const MatrixXd & C, const int e) -> bool
        {
            if (!stale_pre(V, F, E, EMAP, EF, EI, Q, EQ, C, e)) return false;
            return !gStaleVertexSet.count(E(e,0)) && !gStaleVertexSet.count(E(e,1));
        };
        std::cout << "[STALE] pre-collapse lock active: "
                  << gStaleVertexSet.size() << " vertices protected\n";
    }

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
    const std::string c2f_path              = out_dir + "c2f_"               + stem + ".txt";
    const std::string bundle_path           = out_dir + "correspondence_"     + stem + ".c2f";
    const std::string samples_fine_path     = out_dir + "samples_fine_"       + stem + ".txt";
    const std::string samples_coarse_path   = out_dir + "samples_coarse_"     + stem + ".txt";
    const std::string samples_vertices_path = out_dir + "samples_vertices_"   + stem + ".txt";

    if (gNSamplesTotal >= 0) {
        if (!traceVerticesPath.empty()) {
            const std::string trace_out = out_dir + "fine_samples_log_" + stem + ".txt";
            sample_tracker_set_trace(traceVerticesPath, trace_out);
        }
        sample_tracker_init(gNSamplesTotal);
    } else {
        std::cout << "Sampling disabled (--n_samples_total not given).\n";
    }
    if (trackFaceFlip >= 0)
        face_flip_tracker_init(trackFaceFlip);

    print_seam_edge_costs(out_dir + "seam_edge_costs_" + stem + ".txt");
    SSP_seam_log_open((out_dir + "seam_diag_"         + stem + ".txt").c_str());
    SSP_rej_log_open ((out_dir + "collapse_rejections_" + stem + ".txt").c_str());
    dc_log_open      ((out_dir + "dc_log_"             + stem + ".txt").c_str());

#ifdef C2F_VIZ_DIAGNOSTIC
    polyscope::init();

    // Polyscope point cloud: one point per original MAT vertex, coloured by struct ID set.
    // Each unique set<int> combination maps to a unique scalar via FNV-1a hash.
    if (!gVertexStructIDs.empty()) {
        const int nVO = (int)gVO.rows();
        Eigen::VectorXd structScalar(nVO);
        for (int i = 0; i < nVO; ++i) {
            uint32_t h = 2166136261u;
            for (int id : gVertexStructIDs[i]) {
                h ^= (uint32_t)id;
                h *= 16777619u;
            }
            structScalar(i) = (double)h;
        }
        auto* pc = polyscope::registerPointCloud("mat_struct_ids", gVO);
        pc->setPointRadius(0.004, true);
        pc->addScalarQuantity("struct_hash", structScalar)->setEnabled(false);
    }

    polyscope::state::userCallback = ui_callback;
    update_display();
    polyscope::show();
#else
    // Headless: run all collapses to target without any GUI.
    while (!gFinished)
        do_next_step();
#endif

    SSP_seam_log_close();
    SSP_rej_log_close();
    dc_log_close();

    // Export the simplified mesh regardless of how many collapses happened.
    save_simplified_mesh(out_dir + "simplified_" + stem + ".obj");

    // Auto-save on exit regardless of C2F_VIZ_DIAGNOSTIC and regardless of
    // whether decimation reached the target face count.
    if (!gDecInfo.empty()) {
        coarse_fine_compute_and_save(c2f_path);
        coarse_fine_save_bundle(c2f_path, bundle_path);
    }

    sample_tracker_save(samples_fine_path, samples_coarse_path, samples_vertices_path);
    sample_tracker_export_deformed_mesh(out_dir + "deformed_fine_mesh_" + stem + ".obj");

    return 0;
}
