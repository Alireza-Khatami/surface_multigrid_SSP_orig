#include "coarse_subdiv_c2f.h"
#include "subdiv_sample_tracker/subdiv_relax.h"  // subdiv_long_path

#include <query_coarse_to_fine.h>
#include <single_collapse_data.h>

#include <igl/writeOBJ.h>
#include <Eigen/Dense>

#include <cstdio>
#include <vector>

using namespace Eigen;

// ---- globals from main.cpp ----
extern MatrixXd gV;
extern MatrixXd gVO;
extern MatrixXi gF;
extern MatrixXi gFO;
extern VectorXi gFaceSheetID;
extern std::vector<std::vector<int>> gDecIM;
extern std::vector<single_collapse_data> gDecInfo;

static void write_obj(const std::string & path, const MatrixXd & V, const MatrixXi & F, const char * what)
{
    if (!igl::writeOBJ(subdiv_long_path(path), V, F))
        fprintf(stderr, "[coarse_subdiv] writeOBJ failed: %s\n", path.c_str());
    else
        fprintf(stderr, "[coarse_subdiv] %s -> %s\n", what, path.c_str());
}

CoarseSubdivC2F coarse_subdiv_c2f_build(const CoarseMeshCompaction & cmc, int64_t nTarget)
{
    CoarseSubdivC2F C;
    C.S = build_subdiv_mesh(cmc.Vbase, cmc.Fout, nTarget);
    const SubdivMesh & S = C.S;
    const int n    = (int)S.V.rows();
    const int nDec = (int)gDecInfo.size();

    // One query per subdivided vertex on a coarse face: gF row + bary, with the
    // bary in the FUV_pre column order of the last collapse that touched the
    // face (what query_coarse_to_fine expects), or in gF order if no collapse did.
    std::vector<int> qVert;
    MatrixXd BC(n, 3);
    MatrixXi BF(n, 3);
    VectorXi FIdx(n);
    int nUnmatched = 0;
    for (int i = 0; i < n; ++i) {
        const int cf = S.fineFace[i];
        if (cf < 0) continue;
        const int fi = cmc.faceOrigIdx(cf);
        int cur[3];
        for (int c = 0; c < 3; ++c) cur[c] = cmc.newToOld(cmc.Fout(cf, c));

        int dIdx = -1;
        if (fi < (int)gDecIM.size())
            for (int k = (int)gDecIM[fi].size() - 1; k >= 0; --k)
                if (gDecIM[fi][k] < nDec) { dIdx = gDecIM[fi][k]; break; }

        const int q = (int)qVert.size();
        if (dIdx < 0) {
            for (int c = 0; c < 3; ++c) { BF(q, c) = cur[c]; BC(q, c) = S.fineBary(i, c); }
        } else {
            const int sid = fi < gFaceSheetID.size() ? gFaceSheetID(fi) : 0;
            const SheetData * sd = nullptr;
            for (const SheetData & s : gDecInfo[dIdx].sheets)
                if (s.global_sheet_id == sid) { sd = &s; break; }
            if (!sd && gDecInfo[dIdx].sheets.size() == 1) sd = &gDecInfo[dIdx].sheets[0];
            int row = -1;
            if (sd && sd->b.size() >= 2)
                for (int r = 0; r < (int)sd->FIdx_pre.size(); ++r)
                    if (sd->FIdx_pre(r) == fi) { row = r; break; }
            bool ok = row >= 0;
            if (ok) {
                // After collapse dIdx the face has the survivor where FUV_pre has the absorbed vertex.
                const int gs = sd->subsetVIdx(sd->b(0)), gd = sd->subsetVIdx(sd->b(1));
                for (int c = 0; c < 3 && ok; ++c) {
                    const int g = sd->subsetVIdx(sd->FUV_pre(row, c));
                    const int post = (g == gd) ? gs : g;
                    int k = 0;
                    while (k < 3 && cur[k] != post) ++k;
                    if (k == 3) { ok = false; break; }
                    BF(q, c) = g;
                    BC(q, c) = S.fineBary(i, k);
                }
            }
            if (!ok) { ++nUnmatched; continue; }
        }
        FIdx(q) = fi;
        qVert.push_back(i);
    }
    const int nq = (int)qVert.size();
    BC.conservativeResize(nq, 3);
    BF.conservativeResize(nq, 3);
    FIdx.conservativeResize(nq);

    // SSP never renumbers vertices or faces: identity maps.
    const VectorXi IM  = VectorXi::LinSpaced((int)gV.rows(), 0, (int)gV.rows() - 1);
    const VectorXi IMF = VectorXi::LinSpaced((int)gF.rows(), 0, (int)gF.rows() - 1);
    query_coarse_to_fine(gDecInfo, IM, gDecIM, IMF, gFaceSheetID, BC, BF, FIdx);

    // Fine positions; vertices not queried keep their coarse position.
    C.P = S.V;
    C.fineFace.assign(n, -1);
    C.fineBary = MatrixXd::Zero(n, 3);
    int nOffFace = 0;  // query ended with corners that are not its fine face's corners
    for (int q = 0; q < nq; ++q) {
        const int i = qVert[q];
        C.P.row(i) = BC(q, 0) * gVO.row(BF(q, 0)).leftCols(3)
                   + BC(q, 1) * gVO.row(BF(q, 1)).leftCols(3)
                   + BC(q, 2) * gVO.row(BF(q, 2)).leftCols(3);
        const bool onFace = FIdx(q) >= 0 && FIdx(q) < gFO.rows() &&
                            BF(q, 0) == gFO(FIdx(q), 0) && BF(q, 1) == gFO(FIdx(q), 1) && BF(q, 2) == gFO(FIdx(q), 2);
        if (!onFace) { ++nOffFace; continue; }
        C.fineFace[i] = FIdx(q);
        C.fineBary.row(i) = BC.row(q);
    }

    fprintf(stderr,
        "[coarse_subdiv] %d levels: |V| %lld -> %d, |F| %lld -> %lld (target %lld); "
        "%d mapped to fine, %d on no coarse face, %d unmatched, %d ended off their fine face\n",
        S.nLevels, (long long)cmc.Vbase.rows(), n, (long long)cmc.Fout.rows(), (long long)S.F.rows(),
        (long long)nTarget, nq, n - nq - nUnmatched, nUnmatched, nOffFace);
    return C;
}

void coarse_subdiv_c2f_write(const CoarseSubdivC2F & C, int64_t maxObjVerts,
                             const std::string & coarseObjPath, const std::string & fineObjPath)
{
    const int64_t n = C.S.V.rows();
    if (n > maxObjVerts) {
        fprintf(stderr, "[coarse_subdiv] skipping OBJs: %lld vertices > %lld\n", (long long)n, (long long)maxObjVerts);
        return;
    }
    write_obj(coarseObjPath, C.S.V, C.S.F, "subdivided coarse mesh");
    write_obj(fineObjPath,   C.P,   C.S.F, "subdivided coarse mesh at fine correspondences");
}
