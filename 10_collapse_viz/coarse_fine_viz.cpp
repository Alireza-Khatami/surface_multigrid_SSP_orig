#include "coarse_fine_viz.h"

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/structure.h>

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
static bool  gC2FLoaded  = false;
static char  gC2FPath[512] = "coarse_to_fine.txt";
static float gC2FzOffset = 0.0f;

// Stored after load so the Z-offset slider can re-register without re-reading the file.
static MatrixXd gCoarseMeshV;  // all gV vertices (no Z offset, infinity zeroed out)
static MatrixXi gCoarseMeshF;  // live coarse faces (null and infinity faces removed)
static MatrixXd gCorrVectors;  // per-vertex (fine_pos - coarse_pos), zero if no correspondence

// Names of structures that were enabled before c2f load, so Hide can restore them.
static std::vector<std::pair<std::string,std::string>> gPrevEnabled; // (type, name)

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
    MatrixXi tmpF(nFO, 3);
    int nLive = 0;
    for (int f = 0; f < nFO; f++) {
        if (gF(f, 0) == IGL_COLLAPSE_EDGE_NULL) continue;
        bool has_inf = false;
        for (int c = 0; c < 3; c++)
            if (std::isinf(gV(gF(f, c), 0))) { has_inf = true; break; }
        if (has_inf) continue;
        tmpF.row(nLive++) = gF.row(f);
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

    const uint32_t magic = 0xC2F50001;
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

    std::cerr << "[bundle] Saved NC=" << NC << " FC=" << FC
              << " NF=" << NF << " FF=" << FF << " → " << bundlePath << "\n";
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
        if (gF(f, 0) == IGL_COLLAPSE_EDGE_NULL) continue;
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

void coarse_fine_imgui_section()
{
    if (!ImGui::CollapsingHeader("Coarse → Fine Correspondence")) return;

    if (!gFinished) {
        ImGui::TextDisabled("Decimate to target first.");
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
