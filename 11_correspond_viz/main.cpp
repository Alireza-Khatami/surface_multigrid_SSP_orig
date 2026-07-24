#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>
#include <polyscope/pick.h>
#include <polyscope/structure.h>

#include <Eigen/Dense>

#include <single_collapse_data.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>

#include "bundle.h"
#include "face_sample_viz.h"

using namespace Eigen;

// -----------------------------------------------------------------------
// Bundle format (little-endian binary, extension .c2f):
//   magic   uint32  0xC2F50001 = v1 (no SSP data)
//                   0xC2F50002 = v2 (includes SSP query data)
//   NC      uint32  compact coarse vertex count
//   FC      uint32  coarse face count
//   NF      uint32  fine vertex count
//   FF      uint32  fine face count
//   coarse vertices  NC×3 double
//   coarse faces     FC×3 uint32
//   fine vertices    NF×3 double
//   fine faces       FF×3 uint32
//   correspondence   NC × (bc0 bc1 bc2 double  +  fv0 fv1 fv2 uint32)
//   [v2 only] SSP query data
// -----------------------------------------------------------------------

static bool load_bundle(const std::string & path, Bundle & b)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "[bundle] Cannot read " << path << "\n"; return false; }

    auto read32 = [&](uint32_t & v) { in.read((char*)&v, 4); };
    auto readD  = [&](double   & v) { in.read((char*)&v, 8); };

    uint32_t magic; read32(magic);
    if (magic != 0xC2F50001 && magic != 0xC2F50002) {
        std::cerr << "[bundle] Bad magic 0x" << std::hex << magic << "\n"; return false;
    }
    const bool v2 = (magic == 0xC2F50002);

    uint32_t NC, FC, NF, FF;
    read32(NC); read32(FC); read32(NF); read32(FF);

    b.coarseV.resize(NC, 3);
    for (uint32_t i = 0; i < NC; i++) {
        readD(b.coarseV(i,0)); readD(b.coarseV(i,1)); readD(b.coarseV(i,2));
    }

    b.coarseF.resize(FC, 3);
    for (uint32_t f = 0; f < FC; f++) {
        uint32_t v0, v1, v2; read32(v0); read32(v1); read32(v2);
        b.coarseF(f,0) = v0; b.coarseF(f,1) = v1; b.coarseF(f,2) = v2;
    }

    b.fineV.resize(NF, 3);
    for (uint32_t i = 0; i < NF; i++) {
        readD(b.fineV(i,0)); readD(b.fineV(i,1)); readD(b.fineV(i,2));
    }

    b.fineF.resize(FF, 3);
    for (uint32_t f = 0; f < FF; f++) {
        uint32_t v0, v1, v2; read32(v0); read32(v1); read32(v2);
        b.fineF(f,0) = v0; b.fineF(f,1) = v1; b.fineF(f,2) = v2;
    }

    // Read correspondence (derive corrVec from bc+fv).
    b.corrVec.resize(NC, 3);
    for (uint32_t i = 0; i < NC; i++) {
        double bc0, bc1, bc2; readD(bc0); readD(bc1); readD(bc2);
        uint32_t fv0, fv1, fv2; read32(fv0); read32(fv1); read32(fv2);
        RowVector3d fine_pos = bc0 * b.fineV.row(fv0)
                             + bc1 * b.fineV.row(fv1)
                             + bc2 * b.fineV.row(fv2);
        b.corrVec.row(i) = fine_pos - b.coarseV.row(i);
    }

    // ---- v2: SSP query data ----
    if (v2) {
        auto read32s = [&](int32_t & v) { in.read((char*)&v, 4); };

        uint32_t nV_total, nF_decIM, nFO;
        read32(nV_total); read32(nF_decIM); read32(nFO);
        b.nV_total = (int)nV_total;
        b.nF_total = (int)nF_decIM;

        b.vtxMap.resize(NC);
        for (uint32_t i = 0; i < NC; i++) {
            int32_t v; read32s(v); b.vtxMap[i] = v;
        }

        b.faceMap.resize(FC);
        for (uint32_t f = 0; f < FC; f++) {
            int32_t v; read32s(v); b.faceMap[f] = v;
        }

        b.faceSheetID.resize(nFO);
        for (uint32_t f = 0; f < nFO; f++) {
            int32_t v; read32s(v); b.faceSheetID(f) = v;
        }

        b.decIM.resize(nF_decIM);
        for (uint32_t f = 0; f < nF_decIM; f++) {
            uint32_t cnt; read32(cnt);
            b.decIM[f].resize(cnt);
            for (uint32_t k = 0; k < cnt; k++) {
                int32_t v; read32s(v); b.decIM[f][k] = v;
            }
        }

        uint32_t nDec; read32(nDec);
        b.decInfo.resize(nDec);
        for (uint32_t d = 0; d < nDec; d++) {
            uint32_t nSheets; read32(nSheets);
            b.decInfo[d].sheets.resize(nSheets);
            for (uint32_t s = 0; s < nSheets; s++) {
                SheetData & sd = b.decInfo[d].sheets[s];
                int32_t sid; read32s(sid); sd.global_sheet_id = sid;

                uint32_t svSz; read32(svSz);
                sd.subsetVIdx.resize(svSz);
                for (uint32_t j = 0; j < svSz; j++) {
                    int32_t v; read32s(v); sd.subsetVIdx(j) = v;
                }

                uint32_t uvRows; read32(uvRows);
                sd.UV_pre.resize(uvRows, 2);
                for (uint32_t r = 0; r < uvRows; r++) {
                    readD(sd.UV_pre(r, 0)); readD(sd.UV_pre(r, 1));
                }
                sd.UV_post.resize(uvRows, 2);
                for (uint32_t r = 0; r < uvRows; r++) {
                    readD(sd.UV_post(r, 0)); readD(sd.UV_post(r, 1));
                }

                uint32_t fuvRows; read32(fuvRows);
                sd.FUV_pre.resize(fuvRows, 3);
                for (uint32_t r = 0; r < fuvRows; r++) {
                    int32_t a, bv, c; read32s(a); read32s(bv); read32s(c);
                    sd.FUV_pre(r,0)=a; sd.FUV_pre(r,1)=bv; sd.FUV_pre(r,2)=c;
                }
                sd.FIdx_pre.resize(fuvRows);
                for (uint32_t r = 0; r < fuvRows; r++) {
                    int32_t v; read32s(v); sd.FIdx_pre(r) = v;
                }
            }
        }

        b.hasSspData = true;
        std::cerr << "[bundle] Loaded v2 SSP data: nDec=" << nDec
                  << "  nF_decIM=" << nF_decIM << "\n";
    }

    if (!in) { std::cerr << "[bundle] Read error in " << path << "\n"; return false; }
    std::cerr << "[bundle] Loaded NC=" << NC << " FC=" << FC
              << " NF=" << NF << " FF=" << FF << "\n";
    return true;
}

// ---- global viz state ----
Bundle gBundle;
float  gZOffset        = 1.0f;
static int     gSampleStep     = 1;
static bool    gShowArrows     = true;
static int     gSelectedSample = -1;

// ---- helpers ----
static void coarse_fine_pts(int coarseIdx,
                             RowVector3d & cPt, RowVector3d & fPt, RowVector3d & arrow)
{
    cPt   = gBundle.coarseV.row(coarseIdx);
    cPt(2) += (double)gZOffset;
    fPt   = gBundle.coarseV.row(coarseIdx) + gBundle.corrVec.row(coarseIdx);
    arrow = gBundle.corrVec.row(coarseIdx);
    arrow(2) -= (double)gZOffset;
}

// -----------------------------------------------------------------------
// Selection viz: single highlighted coarse point + arrow + fine dest.
// -----------------------------------------------------------------------
static bool gSelRegistered = false;

static void rebuild_selection_viz()
{
    if (gSelRegistered) {
        polyscope::removeStructure("sel_coarse");
        polyscope::removeStructure("sel_fine");
        gSelRegistered = false;
    }

    if (gSelectedSample < 0) return;

    int step      = std::max(1, gSampleStep);
    int coarseIdx = gSelectedSample * step;
    int NC        = (int)gBundle.coarseV.rows();
    if (coarseIdx >= NC) return;

    RowVector3d cPt, fPt, arrow;
    coarse_fine_pts(coarseIdx, cPt, fPt, arrow);

    MatrixXd cM(1,3), fM(1,3), aM(1,3);
    cM.row(0) = cPt; fM.row(0) = fPt; aM.row(0) = arrow;

    auto* sc = polyscope::registerPointCloud("sel_coarse", cM);
    sc->setPointColor({1.0f, 0.15f, 0.15f});
    sc->setPointRadius(0.012, true);

    auto* vq = sc->addVectorQuantity("to_fine", aM, polyscope::VectorType::AMBIENT);
    vq->setEnabled(true);
    vq->setVectorColor({1.0f, 0.0f, 0.0f});
    vq->setVectorLengthScale(1.0, false);

    auto* sf = polyscope::registerPointCloud("sel_fine", fM);
    sf->setPointColor({1.0f, 0.15f, 0.15f});
    sf->setPointRadius(0.012, true);

    gSelRegistered = true;
}

// -----------------------------------------------------------------------

static void rebuild_coarse_viz()
{
    MatrixXd V = gBundle.coarseV;
    V.col(2).array() += (double)gZOffset;

    auto * ps = polyscope::registerSurfaceMesh("coarse_mesh", V, gBundle.coarseF);
    ps->setSurfaceColor({0.25f, 0.52f, 0.95f});
    ps->setEdgeWidth(0.5f);
    ps->setSmoothShade(false);
}

static void rebuild_sample_viz()
{
    int NC   = (int)gBundle.coarseV.rows();
    int step = std::max(1, gSampleStep);

    std::vector<int> idx;
    idx.reserve((NC + step - 1) / step);
    for (int i = 0; i < NC; i += step)
        idx.push_back(i);

    int NS = (int)idx.size();
    MatrixXd coarsePts(NS, 3), finePts(NS, 3), arrowVecs(NS, 3);

    for (int j = 0; j < NS; j++) {
        int i = idx[j];

        coarsePts.row(j)  = gBundle.coarseV.row(i);
        coarsePts(j, 2)  += (double)gZOffset;

        finePts.row(j) = gBundle.coarseV.row(i) + gBundle.corrVec.row(i);

        arrowVecs.row(j)  = gBundle.corrVec.row(i);
        arrowVecs(j, 2)  -= (double)gZOffset;
    }

    auto* cp = polyscope::registerPointCloud("sample_coarse", coarsePts);
    cp->setPointColor({1.0f, 0.85f, 0.0f});
    cp->setPointRadius(0.006, true);

    auto* vq = cp->addVectorQuantity("to_fine", arrowVecs, polyscope::VectorType::AMBIENT);
    vq->setEnabled(gShowArrows);
    vq->setVectorColor({1.0f, 0.35f, 0.05f});
    vq->setVectorLengthScale(1.0, false);

    auto* fp = polyscope::registerPointCloud("sample_fine", finePts);
    fp->setPointColor({0.15f, 0.85f, 0.40f});
    fp->setPointRadius(0.006, true);
}

static void rebuild_all()
{
    rebuild_coarse_viz();
    rebuild_sample_viz();
    rebuild_selection_viz();
    rebuild_face_sample_viz();
}

int main(int argc, char ** argv)
{
    std::string bundlePath = "c2f_bundle.c2f";
    if (argc > 1) bundlePath = argv[1];

    if (!load_bundle(bundlePath, gBundle)) return 1;

    polyscope::init();
    polyscope::options::programName = "Coarse-to-Fine Correspondence";

    polyscope::registerSurfaceMesh("fine_mesh", gBundle.fineV, gBundle.fineF)
        ->setSurfaceColor({0.55f, 0.55f, 0.55f})
        ->setEdgeWidth(0.3f)
        ->setSmoothShade(false)
        ->setTransparency(0.6f);

    rebuild_all();

    int NC = (int)gBundle.coarseV.rows();

    polyscope::state::userCallback = [NC]() {

        // ---- pick detection ----
        {
            int newSel  = -1;
            int newFace = -1;
            if (polyscope::pick::haveSelection()) {
                auto sel = polyscope::pick::getSelection();
                polyscope::Structure* structure = sel.first;
                size_t localInd = sel.second;
                if (structure && structure->name == "sample_coarse")
                    newSel = (int)localInd;
                if (structure && structure->name == "coarse_mesh") {
                    auto* sm = dynamic_cast<polyscope::SurfaceMesh*>(structure);
                    if (sm) {
                        size_t nV = sm->nVertices();
                        size_t nF = sm->nFaces();
                        if (localInd >= nV && localInd < nV + nF)
                            newFace = (int)(localInd - nV);
                    }
                }
            }
            if (newSel != gSelectedSample) {
                gSelectedSample = newSel;
                rebuild_selection_viz();
            }
            if (newFace != gSelectedFace) {
                gSelectedFace = newFace;
                rebuild_face_sample_viz();
            }
        }

        // ---- UI window ----
        ImGui::SetNextWindowSize({340, 230}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Correspondence");

        bool changed = false;

        ImGui::SetNextItemWidth(220);
        if (ImGui::SliderFloat("Z offset (coarse)", &gZOffset, -2.0f, 2.0f))
            changed = true;

        ImGui::SetNextItemWidth(220);
        if (ImGui::SliderInt("Sample every X pts", &gSampleStep, 1, 40)) {
            gSelectedSample = -1;
            changed = true;
        }

        if (ImGui::Checkbox("Show arrows", &gShowArrows))
            changed = true;

        int shown = (NC + std::max(1, gSampleStep) - 1) / std::max(1, gSampleStep);
        ImGui::Text("Showing %d / %d coarse vertices", shown, NC);
        if (gSelectedSample >= 0) {
            int ci = gSelectedSample * std::max(1, gSampleStep);
            ImGui::Text("Selected: sample[%d] -> coarse[%d]", gSelectedSample, ci);
            if (ImGui::Button("Clear selection")) {
                gSelectedSample = -1;
                rebuild_selection_viz();
            }
        }

        if (changed) rebuild_all();

        ImGui::Separator();
        ImGui::Text("Face sampling (click a coarse face)");
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputInt("n samples", &gFaceSampleN)) {
            gFaceSampleN = std::max(1, gFaceSampleN);
            if (gSelectedFace >= 0) rebuild_face_sample_viz();
        }
        {
            bool visChanged = false;
            if (ImGui::Checkbox("coarse samples", &gShowFaceSampleCoarse)) visChanged = true;
            ImGui::SameLine();
            if (ImGui::Checkbox("fine samples",   &gShowFaceSampleFine))   visChanged = true;
            if (visChanged) update_face_sample_visibility();
        }
        if (gSelectedFace >= 0) {
            ImGui::Text("Face %d  |  %d samples", gSelectedFace, gFaceSampleN);
            if (ImGui::Button("Clear face")) {
                gSelectedFace = -1;
                rebuild_face_sample_viz();
            }
        }

        ImGui::End();
    };

    polyscope::show();
    return 0;
}
