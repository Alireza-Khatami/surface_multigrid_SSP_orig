#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/structure.h>

#include <Eigen/Dense>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>

using namespace Eigen;

// -----------------------------------------------------------------------
// Bundle format (little-endian binary, extension .c2f):
//   magic   uint32  0xC2F50001
//   NC      uint32  compact coarse vertex count
//   FC      uint32  coarse face count
//   NF      uint32  fine vertex count
//   FF      uint32  fine face count
//   coarse vertices  NC×3 double
//   coarse faces     FC×3 uint32
//   fine vertices    NF×3 double
//   fine faces       FF×3 uint32
//   correspondence   NC × (bc0 bc1 bc2 double  +  fv0 fv1 fv2 uint32)
// -----------------------------------------------------------------------

struct Bundle {
    MatrixXd coarseV;   // NC × 3
    MatrixXi coarseF;   // FC × 3
    MatrixXd fineV;     // NF × 3
    MatrixXi fineF;     // FF × 3
    MatrixXd corrVec;   // NC × 3  (fine_pos - coarse_pos)
};

static bool load_bundle(const std::string & path, Bundle & b)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "[bundle] Cannot read " << path << "\n"; return false; }

    auto read32 = [&](uint32_t & v) { in.read((char*)&v, 4); };
    auto readD  = [&](double   & v) { in.read((char*)&v, 8); };

    uint32_t magic; read32(magic);
    if (magic != 0xC2F50001) {
        std::cerr << "[bundle] Bad magic 0x" << std::hex << magic << "\n"; return false;
    }

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

    // Read correspondence, build world-space vectors.
    b.corrVec.resize(NC, 3);
    for (uint32_t i = 0; i < NC; i++) {
        double bc0, bc1, bc2; readD(bc0); readD(bc1); readD(bc2);
        uint32_t fv0, fv1, fv2; read32(fv0); read32(fv1); read32(fv2);

        RowVector3d fine_pos = bc0 * b.fineV.row(fv0)
                             + bc1 * b.fineV.row(fv1)
                             + bc2 * b.fineV.row(fv2);
        b.corrVec.row(i) = fine_pos - b.coarseV.row(i);
    }

    if (!in) { std::cerr << "[bundle] Read error in " << path << "\n"; return false; }
    std::cerr << "[bundle] Loaded NC=" << NC << " FC=" << FC
              << " NF=" << NF << " FF=" << FF << "\n";
    return true;
}

// ---- global viz state ----
static Bundle  gBundle;
static float   gZOffset = 0.0f;

static void rebuild_coarse_viz()
{
    MatrixXd V = gBundle.coarseV;
    V.col(2).array() += (double)gZOffset;

    auto * ps = polyscope::registerSurfaceMesh("coarse_mesh", V, gBundle.coarseF);
    ps->setSurfaceColor({0.25f, 0.52f, 0.95f});
    ps->setEdgeWidth(0.5f);
    ps->setSmoothShade(false);

    // Adjust vectors: coarse moved in Z, so subtract offset from Z component
    // so tips remain on the stationary fine mesh.
    MatrixXd corrAdj = gBundle.corrVec;
    corrAdj.col(2).array() -= (double)gZOffset;

    auto * vq = ps->addVertexVectorQuantity("correspondence", corrAdj,
                                             polyscope::VectorType::AMBIENT);
    vq->setEnabled(true);
    vq->setVectorColor({1.0f, 0.35f, 0.05f});
    vq->setVectorLengthScale(1.0, false);
}

int main(int argc, char ** argv)
{
    std::string bundlePath = "c2f_bundle.c2f";
    if (argc > 1) bundlePath = argv[1];

    if (!load_bundle(bundlePath, gBundle)) return 1;

    polyscope::init();
    polyscope::options::programName = "Coarse-to-Fine Correspondence";

    // Fine mesh — semi-transparent background reference.
    polyscope::registerSurfaceMesh("fine_mesh", gBundle.fineV, gBundle.fineF)
        ->setSurfaceColor({0.55f, 0.55f, 0.55f})
        ->setEdgeWidth(0.3f)
        ->setSmoothShade(false)
        ->setTransparency(0.6f);

    rebuild_coarse_viz();

    polyscope::state::userCallback = []() {
        ImGui::SetNextWindowSize({320, 100}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Correspondence");

        ImGui::SetNextItemWidth(220);
        if (ImGui::SliderFloat("Z offset (coarse)", &gZOffset, -2.0f, 2.0f))
            rebuild_coarse_viz();

        ImGui::End();
    };

    polyscope::show();
    return 0;
}
