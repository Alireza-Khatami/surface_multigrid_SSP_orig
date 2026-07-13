#include "coarse_fine_viz.h"

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>

#include <igl/collapse_edge.h>       // IGL_COLLAPSE_EDGE_NULL

#include <query_coarse_to_fine.h>
#include <single_collapse_data.h>

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
static bool gC2FLoaded = false;
static char gC2FPath[512] = "coarse_to_fine.txt";

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
            if (gF(f, 0) == IGL_COLLAPSE_EDGE_NULL) continue;
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
            if (gF(f, 0) == IGL_COLLAPSE_EDGE_NULL) continue;
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

        // Find fi in FIdx_post to get the vertex state AT the time of that collapse.
        // Using current gF is wrong: non-active-sheet collapses after dIdx may have
        // remapped vertices in fi (d→s in Pass 2) without updating decIM[fi].
        // FIdx_post captures the post-collapse state of fi at dIdx, which is what
        // subsetVIdx at that collapse actually contains.
        int post_row = -1;
        for (int r = 0; r < (int)sd_ptr->FIdx_post.size(); r++)
            if (sd_ptr->FIdx_post(r) == fi) { post_row = r; break; }

        if (post_row < 0) { useCurrentGF(); continue; }

        const SheetData & sd = *sd_ptr;
        const int gv0 = sd.subsetVIdx(sd.FUV_post(post_row, 0));
        const int gv1 = sd.subsetVIdx(sd.FUV_post(post_row, 1));
        const int gv2 = sd.subsetVIdx(sd.FUV_post(post_row, 2));

        // Rotate so vi is first.
        int col = -1;
        if (gv0 == vi) col = 0;
        else if (gv1 == vi) col = 1;
        else if (gv2 == vi) col = 2;

        if (col < 0) {
            // vi not in post-collapse vertices (vi introduced by non-active collapse after dIdx).
            useCurrentGF(); continue;
        }

        const int vs[3] = {gv0, gv1, gv2};
        BF(i, 0) = vs[col];
        BF(i, 1) = vs[(col + 1) % 3];
        BF(i, 2) = vs[(col + 2) % 3];
        BC(i, 0) = 1.0; BC(i, 1) = 0.0; BC(i, 2) = 0.0;
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

void coarse_fine_clear()
{
    polyscope::removeStructure("fine_mesh",  /*errorIfAbsent=*/false);
    polyscope::removeStructure("c2f_lines",  /*errorIfAbsent=*/false);
    gC2FLoaded = false;
}

void coarse_fine_load_and_show(const std::string & path)
{
    std::ifstream in(path);
    if (!in) { std::cerr << "[c2f] Cannot read " << path << "\n"; return; }

    int N; in >> N;
    if (N <= 0) { std::cerr << "[c2f] Empty correspondence file.\n"; return; }

    // Two nodes per correspondence (coarse pos, fine pos) + one edge.
    MatrixXd nodes(2 * N, 3);
    MatrixXi edges(N, 2);
    int loaded = 0;

    for (int i = 0; i < N; i++) {
        int vi, fv0, fv1, fv2;
        double bc0, bc1, bc2;
        if (!(in >> vi >> bc0 >> bc1 >> bc2 >> fv0 >> fv1 >> fv2)) break;

        // Guard against out-of-range indices (should not happen with correct files).
        if (vi  >= gV.rows()  || fv0 >= gVO.rows() ||
            fv1 >= gVO.rows() || fv2 >= gVO.rows()) continue;

        nodes.row(2 * loaded)     = gV.row(vi).leftCols(3);
        nodes.row(2 * loaded + 1) = bc0 * gVO.row(fv0)
                                  + bc1 * gVO.row(fv1)
                                  + bc2 * gVO.row(fv2);
        edges.row(loaded) << 2 * loaded, 2 * loaded + 1;
        loaded++;
    }

    if (loaded == 0) { std::cerr << "[c2f] No valid entries.\n"; return; }

    // Trim to actual count if some entries were skipped.
    nodes.conservativeResize(2 * loaded, 3);
    edges.conservativeResize(loaded, 2);

    // Fine (original) mesh — semi-transparent so lines are visible through it.
    polyscope::registerSurfaceMesh("fine_mesh", gVO, gFO)
        ->setSurfaceColor({0.55f, 0.55f, 0.55f})
        ->setEdgeWidth(0.3f)
        ->setSmoothShade(false)
        ->setTransparency(0.6f);

    // Correspondence lines: coarse vertex → fine mesh point.
    polyscope::registerCurveNetwork("c2f_lines", nodes, edges)
        ->setRadius(0.00004f)
        ->setColor({1.0f, 0.35f, 0.05f});

    gC2FLoaded = true;
    std::cout << "[c2f] Loaded " << loaded << " correspondences from " << path << "\n";
}

void coarse_fine_imgui_section()
{
    if (!ImGui::CollapsingHeader("Coarse → Fine Correspondence")) return;

    if (!gFinished) {
        ImGui::TextDisabled("Decimate to target first.");
        return;
    }

    ImGui::SetNextItemWidth(230);
    ImGui::InputText("##c2fpath", gC2FPath, sizeof(gC2FPath));
    ImGui::SameLine();
    if (ImGui::Button("Save##c2f"))
        coarse_fine_compute_and_save(std::string(gC2FPath));

    if (ImGui::Button("Load & Show##c2f")) {
        coarse_fine_clear();
        coarse_fine_load_and_show(std::string(gC2FPath));
    }
    if (gC2FLoaded) {
        ImGui::SameLine();
        if (ImGui::Button("Hide##c2f"))
            coarse_fine_clear();
    }
}
