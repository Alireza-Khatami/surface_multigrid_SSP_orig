#include "sheet_seam_viz.h"

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>
#include <polyscope/pick.h>

#include <igl/remove_unreferenced.h>
#include <igl/collapse_edge.h>  // IGL_COLLAPSE_EDGE_NULL

#include <single_collapse_data.h>

#include <Eigen/Dense>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <cmath>

using namespace Eigen;

// ---- globals defined in main.cpp ----
extern MatrixXd gV;
extern MatrixXi gF;
extern VectorXi gFaceSheetID;
extern int      gNumSheets;
extern std::vector<std::vector<int>> gVF;
extern std::vector<std::pair<int,int>> gSeamEdgeList;

// ---- internal state ----

// 8 perceptually distinct colors, one per sheet (cycles if numSheets > 8)
static const std::array<std::array<float,3>, 8> kSheetPalette = {{
    {0.90f, 0.30f, 0.20f},  // red
    {0.20f, 0.53f, 0.87f},  // blue
    {0.20f, 0.72f, 0.38f},  // green
    {0.93f, 0.70f, 0.08f},  // yellow
    {0.70f, 0.28f, 0.84f},  // purple
    {0.95f, 0.50f, 0.10f},  // orange
    {0.28f, 0.82f, 0.82f},  // cyan
    {0.92f, 0.38f, 0.68f},  // pink
}};

static bool gShowSheetColors   = true;
static bool gShowSeamEdges     = true;
static bool gShowSeamOnering   = true;
static int  gPickedSeamEdgeIdx = -1;  // index into gSeamEdgeList; -1 = nothing picked

// Track last pick so we only react on changes, not every frame
static std::pair<polyscope::Structure*, size_t> gLastPick = {nullptr, SIZE_MAX};

// Names of currently registered per-sheet one-ring meshes (for cleanup)
static std::vector<std::string> gActiveOneringNames;

// ---- local mesh helpers (mirror of the ones in visualizer.cpp) ----

static MatrixXd safe_V_ss()
{
    MatrixXd V = gV.leftCols(3);
    for (int v = 0; v < V.rows(); v++)
        if (std::isinf(V(v,0)) || std::isinf(V(v,1)) || std::isinf(V(v,2)))
            V.row(v).setZero();
    return V;
}

// Returns live (non-null, non-infinity) faces together with their original gF row indices.
// remove_unreferenced preserves face order, so origIdx[fi] == gF row for Fc.row(fi).
static MatrixXi live_faces_ex_ss(std::vector<int>& origIdx)
{
    origIdx.clear();
    std::vector<std::array<int,3>> rows;
    for (int f = 0; f < gF.rows(); f++) {
        int v0 = gF(f,0), v1 = gF(f,1), v2 = gF(f,2);
        if (v0 == IGL_COLLAPSE_EDGE_NULL) continue;
        if (std::isinf(gV(v0,0)) || std::isinf(gV(v1,0)) || std::isinf(gV(v2,0))) continue;
        rows.push_back({v0, v1, v2});
        origIdx.push_back(f);
    }
    MatrixXi F((int)rows.size(), 3);
    for (int i = 0; i < (int)rows.size(); i++)
        F.row(i) << rows[i][0], rows[i][1], rows[i][2];
    return F;
}

// ---- public API ----

void clear_seam_onering()
{
    for (auto & nm : gActiveOneringNames)
        polyscope::removeStructure(nm, /*errorIfAbsent=*/false);
    gActiveOneringNames.clear();
}

void update_seam_onering_display()
{
    clear_seam_onering();  // always clean up first, even if we won't draw anything
    if (!gShowSeamOnering || gPickedSeamEdgeIdx < 0 ||
        gPickedSeamEdgeIdx >= (int)gSeamEdgeList.size()) return;

    const int vi = gSeamEdgeList[gPickedSeamEdgeIdx].first;
    const int vj = gSeamEdgeList[gPickedSeamEdgeIdx].second;
    const int nFSheet = (int)gFaceSheetID.size();

    // Collect one-ring faces of vi ∪ vj, real faces only, grouped by sheet ID
    std::map<int, std::set<int>> sheetFaces;
    for (int v : {vi, vj}) {
        if (v >= (int)gVF.size()) continue;
        for (int f : gVF[v]) {
            if (gF(f,0) == IGL_COLLAPSE_EDGE_NULL) continue;
            if (std::isinf(gV(gF(f,0),0)) || std::isinf(gV(gF(f,1),0)) ||
                std::isinf(gV(gF(f,2),0))) continue;
            if (f >= nFSheet) continue;
            sheetFaces[gFaceSheetID(f)].insert(f);
        }
    }

    MatrixXd Vd = safe_V_ss();
    int paletteIdx = 0;
    for (auto & kv : sheetFaces) {
        const int sid = kv.first;
        const auto & fset = kv.second;
        if (fset.empty()) continue;

        MatrixXi Fs((int)fset.size(), 3);
        int fi = 0;
        for (int f : fset) Fs.row(fi++) = gF.row(f);

        MatrixXd Vc; MatrixXi Fc; VectorXi I1, I2;
        igl::remove_unreferenced(Vd, Fs, Vc, Fc, I1, I2);

        // Push vertices outward along averaged face normals to prevent z-fighting
        // with the main mesh (which occupies the same surface).
        {
            MatrixXd VN = MatrixXd::Zero(Vc.rows(), 3);
            for (int fi = 0; fi < Fc.rows(); fi++) {
                Vector3d a = Vc.row(Fc(fi,0)), b = Vc.row(Fc(fi,1)), c_ = Vc.row(Fc(fi,2));
                Vector3d n = (b - a).cross(c_ - a);
                for (int k = 0; k < 3; k++) VN.row(Fc(fi,k)) += n.transpose();
            }
            double bbox_diag = (Vc.colwise().maxCoeff() - Vc.colwise().minCoeff()).norm();
            double eps = std::max(bbox_diag * 5e-4, 1e-7);
            for (int v = 0; v < Vc.rows(); v++) {
                double nl = VN.row(v).norm();
                if (nl > 1e-10) Vc.row(v) += (eps / nl) * VN.row(v);
            }
        }

        int cidx = (paletteIdx++) % (int)kSheetPalette.size();
        auto & col = kSheetPalette[cidx];
        std::string nm = "onering_sheet_" + std::to_string(sid);
        polyscope::registerSurfaceMesh(nm, Vc, Fc)
            ->setSurfaceColor({col[0], col[1], col[2]})
            ->setEdgeWidth(2.0)->setSmoothShade(false);
        gActiveOneringNames.push_back(nm);
    }
}

void update_sheet_display()
{
    // Sheet-colored mesh: one face color per sheet ID
    polyscope::removeStructure("sheet_view", false);
    if (gShowSheetColors && gFaceSheetID.size() > 0) {
        std::vector<int> origIdx;
        MatrixXi Flive = live_faces_ex_ss(origIdx);
        MatrixXd Vd = safe_V_ss();
        MatrixXd Vc; MatrixXi Fc; VectorXi I1, I2;
        igl::remove_unreferenced(Vd, Flive, Vc, Fc, I1, I2);
        // remove_unreferenced only renumbers vertices; face order is preserved,
        // so origIdx[fi] is the correct gF row for Fc.row(fi).
        MatrixXd faceColors(Fc.rows(), 3);
        for (int fi = 0; fi < Fc.rows(); fi++) {
            int orig = origIdx[fi];
            int sid  = (orig < (int)gFaceSheetID.size()) ? gFaceSheetID(orig) : 0;
            int cidx = sid % (int)kSheetPalette.size();
            faceColors(fi,0) = kSheetPalette[cidx][0];
            faceColors(fi,1) = kSheetPalette[cidx][1];
            faceColors(fi,2) = kSheetPalette[cidx][2];
        }
        auto * sm = polyscope::registerSurfaceMesh("sheet_view", Vc, Fc);
        sm->setEdgeWidth(0.5)->setSmoothShade(false);
        sm->addFaceColorQuantity("sheet", faceColors)->setEnabled(true);
    }

    // Seam edges: yellow curve network; clicking an edge triggers the one-ring view.
    // Pick index layout for this network: nodes 0..2N-1, edges 2N..3N-1
    // (N seam edges → 2N nodes because we use 2 separate nodes per edge).
    polyscope::removeStructure("seam_edges", false);
    if (gShowSeamEdges && !gSeamEdgeList.empty()) {
        const int N = (int)gSeamEdgeList.size();
        MatrixXd Vd = safe_V_ss();
        MatrixXd nodes(2*N, 3);
        MatrixXi edges(N, 2);
        for (int i = 0; i < N; i++) {
            int u = gSeamEdgeList[i].first, v = gSeamEdgeList[i].second;
            nodes.row(2*i)   = Vd.row(u);
            nodes.row(2*i+1) = Vd.row(v);
            edges.row(i) << 2*i, 2*i+1;
        }
        polyscope::registerCurveNetwork("seam_edges", nodes, edges)
            ->setRadius(0.00024f)   // 4× the default edge_radius used elsewhere
            ->setColor({1.0f, 0.85f, 0.05f});
    }
}

void sheet_seam_pick_check()
{
    if (!gShowSeamEdges || gSeamEdgeList.empty()) return;
    if (!polyscope::pick::haveSelection()) return;

    auto sel = polyscope::pick::getSelection();
    if (sel == gLastPick) return;
    gLastPick = sel;

    polyscope::Structure* structure = sel.first;
    size_t localIdx = sel.second;
    if (!structure || structure->name != "seam_edges") return;

    // Curve network pick layout: nodes first (0..2N-1), then edges (2N..3N-1).
    // Node 2i and 2i+1 are the endpoints of edge i, so both map to the same edge.
    const int numNodes = 2 * (int)gSeamEdgeList.size();
    int edgeIdx;
    if ((int)localIdx < numNodes)
        edgeIdx = (int)localIdx / 2;   // endpoint node → parent edge
    else
        edgeIdx = (int)localIdx - numNodes;
    if (edgeIdx < 0 || edgeIdx >= (int)gSeamEdgeList.size()) return;

    gPickedSeamEdgeIdx = edgeIdx;
    update_seam_onering_display();
}

void sheet_seam_imgui_section()
{
    if (!ImGui::CollapsingHeader("Sheet & Seam Inspection")) return;

    bool sheetVis = false;
    sheetVis |= ImGui::Checkbox("Sheet colors", &gShowSheetColors);
    ImGui::SameLine();
    sheetVis |= ImGui::Checkbox("Seam edges (yellow)", &gShowSeamEdges);
    if (sheetVis) {
        update_sheet_display();
        if (!gShowSeamEdges) {
            // losing the curve network invalidates any active pick
            gPickedSeamEdgeIdx = -1;
            gLastPick = {nullptr, SIZE_MAX};
            clear_seam_onering();
            polyscope::pick::resetSelection();
        }
    }

    bool oneringToggle = ImGui::Checkbox("Show one-ring on seam pick", &gShowSeamOnering);
    if (oneringToggle) update_seam_onering_display();

    ImGui::Separator();
    if (gPickedSeamEdgeIdx >= 0 && gPickedSeamEdgeIdx < (int)gSeamEdgeList.size()) {
        const int vi = gSeamEdgeList[gPickedSeamEdgeIdx].first;
        const int vj = gSeamEdgeList[gPickedSeamEdgeIdx].second;
        ImGui::TextColored({1.0f, 0.85f, 0.05f, 1.0f}, "Picked seam: v%d — v%d", vi, vj);

        // Count distinct sheets in the one-ring
        const int nFSheet = (int)gFaceSheetID.size();
        std::set<int> sheets;
        for (int v : {vi, vj}) {
            if (v >= (int)gVF.size()) continue;
            for (int f : gVF[v]) {
                if (gF(f,0) == IGL_COLLAPSE_EDGE_NULL) continue;
                if (std::isinf(gV(gF(f,0),0))) continue;
                if (f < nFSheet) sheets.insert(gFaceSheetID(f));
            }
        }
        ImGui::Text("Sheets in one-ring: %d", (int)sheets.size());

        // Color swatch + label for each sheet
        for (int sid : sheets) {
            int cidx = sid % (int)kSheetPalette.size();
            auto & c = kSheetPalette[cidx];
            ImGui::ColorButton(("##sid" + std::to_string(sid)).c_str(),
                               {c[0], c[1], c[2], 1.0f},
                               ImGuiColorEditFlags_NoTooltip, {12, 12});
            ImGui::SameLine();
            ImGui::Text("sheet %d", sid);
        }

        if (ImGui::Button("Clear pick")) {
            gPickedSeamEdgeIdx = -1;
            gLastPick = {nullptr, SIZE_MAX};
            clear_seam_onering();
            polyscope::pick::resetSelection();
        }
    } else {
        ImGui::TextDisabled("Click a yellow seam edge to inspect its one-ring.");
        ImGui::TextDisabled("Each color = one sheet's faces around that edge.");
    }

    ImGui::Separator();
    ImGui::Text("Mesh sheets: %d  |  Seam edges: %d", gNumSheets, (int)gSeamEdgeList.size());
}
