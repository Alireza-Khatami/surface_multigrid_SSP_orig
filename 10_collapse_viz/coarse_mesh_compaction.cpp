#include "coarse_mesh_compaction.h"
#include "face_dead.h"

#include <igl/remove_unreferenced.h>

#include <algorithm>
#include <stdexcept>
#include <string>

using namespace Eigen;

static CoarseMeshCompaction build_compact_coarse_mesh(const MatrixXd & gV, const MatrixXi & gF)
{
    std::vector<int> live;
    live.reserve(gF.rows());
    for (int f = 0; f < gF.rows(); f++)
        if (is_face_live(gF, gV, f)) live.push_back(f);

    MatrixXi Flive((int)live.size(), 3);
    for (int i = 0; i < (int)live.size(); i++)
        Flive.row(i) = gF.row(live[i]);

    // remove_unreferenced: I = old->new (-1 if dropped), J = new->old. Face order is preserved.
    CoarseMeshCompaction out;
    VectorXi I, J;
    igl::remove_unreferenced(gV.leftCols(3), Flive, out.Vbase, out.Fout, I, J);
    out.oldToNew    = I;
    out.newToOld    = J;
    out.faceOrigIdx = Map<VectorXi>(live.data(), (int)live.size());
    out.NC          = (int)out.Vbase.rows();
    return out;
}

static void extend_with_stale_chains(CoarseMeshCompaction & cmc, const MatrixXd & gV,
                                     const std::vector<std::vector<int>> & gStaleChains)
{
    std::vector<int> added; // gV ids of appended naked vertices, in assignment order
    for (size_t ci = 0; ci < gStaleChains.size(); ci++) {
        if (gStaleChains[ci].empty())
            throw std::runtime_error("[cmc] stale chain " + std::to_string(ci) + " is empty");
        for (int vid : gStaleChains[ci]) {
            if (vid < 0 || vid >= cmc.oldToNew.size())
                throw std::runtime_error("[cmc] stale chain " + std::to_string(ci) +
                                         " has out-of-range vertex id " + std::to_string(vid) +
                                         " (gV rows = " + std::to_string(cmc.oldToNew.size()) + ")");
            if (cmc.oldToNew(vid) >= 0) continue; // already a face vertex or seen earlier
            cmc.oldToNew(vid) = cmc.NC + (int)added.size();
            added.push_back(vid);
        }
    }

    if (!added.empty()) {
        const int n0 = (int)cmc.Vbase.rows(), nA = (int)added.size();
        cmc.Vbase.conservativeResize(n0 + nA, 3);
        cmc.newToOld.conservativeResize(n0 + nA);
        for (int i = 0; i < nA; i++) {
            cmc.Vbase.row(n0 + i) = gV.row(added[i]).leftCols(3);
            cmc.newToOld(n0 + i)  = added[i];
        }
    }

    cmc.staleChains.clear();
    cmc.staleChains.reserve(gStaleChains.size());
    for (const auto & chain : gStaleChains) {
        std::vector<int> compact;
        compact.reserve(chain.size());
        for (int vid : chain) compact.push_back(cmc.oldToNew(vid));
        cmc.staleChains.push_back(std::move(compact));
    }
}

CoarseMeshCompaction build_final_coarse_mesh(const MatrixXd & gV, const MatrixXi & gF,
                                             const std::vector<std::vector<int>> & gStaleChains)
{
    CoarseMeshCompaction cmc = build_compact_coarse_mesh(gV, gF);
    extend_with_stale_chains(cmc, gV, gStaleChains);
    return cmc;
}

// ---- CoarseFaceLookup ----

static std::array<int,3> sorted3(int a, int b, int c)
{
    std::array<int,3> k{a, b, c};
    std::sort(k.begin(), k.end());
    return k;
}

static std::pair<int,int> sorted2(int a, int b) { return {std::min(a, b), std::max(a, b)}; }

CoarseFaceLookup::CoarseFaceLookup(const CoarseMeshCompaction & c)
    : cmc(c), faceByVertex(c.Vbase.rows(), -1)
{
    int maxRow = -1;
    for (int f = 0; f < cmc.faceOrigIdx.size(); f++) maxRow = std::max(maxRow, cmc.faceOrigIdx(f));
    gFRowToFace.assign(maxRow + 1, -1);

    for (int f = 0; f < cmc.Fout.rows(); f++) {
        const int a = cmc.Fout(f,0), b = cmc.Fout(f,1), d = cmc.Fout(f,2);
        gFRowToFace[cmc.faceOrigIdx(f)] = f;
        faceByVerts.emplace(sorted3(a, b, d), f);
        faceByEdge.emplace(sorted2(a, b), f);
        faceByEdge.emplace(sorted2(b, d), f);
        faceByEdge.emplace(sorted2(d, a), f);
        for (int v : {a, b, d})
            if (faceByVertex[v] < 0) faceByVertex[v] = f;
    }
}

CoarseSample CoarseFaceLookup::resolve(int gF_row, const RowVector3i & gv_corners,
                                       const RowVector3d & bary) const
{
    auto fail = [&](const std::string & why) {
        throw std::runtime_error("[cmc] cannot resolve sample (gF row " + std::to_string(gF_row) +
                                 ", gV " + std::to_string(gv_corners(0)) + " " +
                                 std::to_string(gv_corners(1)) + " " + std::to_string(gv_corners(2)) +
                                 "): " + why);
    };

    // Sum weights per distinct compact vertex (corners can repeat after collapses).
    std::vector<std::pair<int,double>> w;
    for (int c = 0; c < 3; c++) {
        const int gv = gv_corners(c);
        if (gv < 0 || gv >= cmc.oldToNew.size()) fail("corner out of range");
        const int cv = cmc.oldToNew(gv);
        if (cv < 0 || cv >= cmc.NC) fail("corner is not a live face vertex");
        auto it = std::find_if(w.begin(), w.end(), [&](const auto & p) { return p.first == cv; });
        if (it == w.end()) w.push_back({cv, bary(c)});
        else               it->second += bary(c);
    }

    auto faceHasAll = [&](int f) {
        if (f < 0) return false;
        for (const auto & p : w)
            if (cmc.Fout(f,0) != p.first && cmc.Fout(f,1) != p.first && cmc.Fout(f,2) != p.first)
                return false;
        return true;
    };

    // Prefer the tracker's own face; fall back to any face holding the same corners.
    int face = (gF_row >= 0 && gF_row < (int)gFRowToFace.size()) ? gFRowToFace[gF_row] : -1;
    if (!faceHasAll(face)) {
        face = -1;
        if (w.size() == 3) {
            auto it = faceByVerts.find(sorted3(w[0].first, w[1].first, w[2].first));
            if (it != faceByVerts.end()) face = it->second;
        } else if (w.size() == 2) {
            auto it = faceByEdge.find(sorted2(w[0].first, w[1].first));
            if (it != faceByEdge.end()) face = it->second;
        } else {
            face = faceByVertex[w[0].first];
        }
    }
    if (face < 0) fail("no live compact face contains its corners");

    CoarseSample s{face, Vector3d::Zero()};
    for (int j = 0; j < 3; j++)
        for (const auto & p : w)
            if (cmc.Fout(face, j) == p.first) s.bary(j) = p.second;
    return s;
}
