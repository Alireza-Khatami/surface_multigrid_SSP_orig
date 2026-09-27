// Standalone checks for build_subdiv_mesh + build_struct_sets.
// Usage: subdiv_mesh_test <mesh.obj> <n_target> [<file.ma_struct>]
//
// Every check is independent of the construction path where possible:
// positions are recomputed from carriers, areas/normals from geometry,
// edge valences from a closed-form count, struct IDs against load_matstruct.

#include "../subdiv_mesh.h"
#include "../subdiv_struct_ids.h"
#include "../../load_matstruct.h"

#include <igl/readOBJ.h>
#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <unordered_map>

// tests/legacy_load_matstruct_ref.cpp
bool legacy_load_matstruct(const std::string & fname, std::vector<std::set<int>> & vertex_struct_ids);

using namespace Eigen;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static Vector3d tri_n(const MatrixXd & V, int a, int b, int c)
{
    const Vector3d pa = V.row(a).transpose(), pb = V.row(b).transpose(), pc = V.row(c).transpose();
    return (pb - pa).cross(pc - pa);
}

static uint64_t key(int a, int b) { if (a > b) std::swap(a, b); return (uint64_t)(uint32_t)a << 32 | (uint32_t)b; }

int main(int argc, char ** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s mesh.obj n_target [ma_struct]\n", argv[0]); return 2; }
    MatrixXd VO, TC, N; MatrixXi FO, FTC, FN;
    if (!igl::readOBJ(argv[1], VO, TC, N, FO, FTC, FN)) { fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }
    const int64_t nTarget = std::stoll(argv[2]);
    const int nVO = (int)VO.rows(), nFO = (int)FO.rows();

    auto t0 = std::chrono::steady_clock::now();
    SubdivMesh M;
    try { M = build_subdiv_mesh(VO, FO, nTarget); }
    catch (const std::exception & e) { fprintf(stderr, "EXCEPTION: %s\n", e.what()); return 1; }
    auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "build: %.2f s\n", std::chrono::duration<double>(t1 - t0).count());

    const int64_t Vs = M.V.rows(), Fs = M.F.rows();
    const int L = M.nLevels;
    const double scale = (VO.colwise().maxCoeff() - VO.colwise().minCoeff()).norm();

    // ---- counts ----
    CHECK(Fs == (int64_t)nFO << (2 * L), "face count %lld != nFO*4^L", (long long)Fs);
    CHECK(Vs >= nTarget || L == 0, "vertex count below target");
    if (L > 0) {
        // Level L-1 would have been below target.
        int64_t v = nVO;
        // closed form: V_l = V_{l-1} + E_{l-1}; E_l = 2 E_{l-1} + 3 F_{l-1}; F_l = 4 F_{l-1}
        int64_t E = M.origEdges.rows(), F = nFO;
        for (int l = 0; l < L; ++l) { const int64_t v2 = v + E; E = 2 * E + 3 * F; F = 4 * F; if (l == L - 1) { CHECK(v < nTarget, "one level too many"); } v = v2; }
        CHECK(v == Vs, "vertex count %lld != closed form %lld", (long long)Vs, (long long)v);
    }
    CHECK(M.carrierType.size() == (size_t)Vs && M.fineFace.size() == (size_t)Vs, "array sizes");
    CHECK(M.faceOrig.size() == (size_t)Fs, "faceOrig size");

    // ---- nesting: each level built on its own == the prefix / recovered faces of the final mesh ----
    CHECK((int)M.levelVerts.size() == L + 1 && M.levelVerts.back() == Vs, "levelVerts");
    for (int l = 0; l <= L && Vs <= 3000000; ++l) {
        const SubdivMesh Ml = build_subdiv_mesh(VO, FO, M.levelVerts[l]);
        CHECK(Ml.nLevels == l, "rebuild at level %d gave %d levels", l, Ml.nLevels);
        CHECK(Ml.F == subdiv_level_faces(M, l), "level %d faces differ from the recovered faces", l);
        CHECK(Ml.V == M.V.topRows(Ml.V.rows()), "level %d vertices are not the prefix", l);
        CHECK(Ml.carrierType == std::vector<uint8_t>(M.carrierType.begin(), M.carrierType.begin() + Ml.V.rows()),
              "level %d carriers are not the prefix", l);
    }

    // ---- original vertices are the prefix, exactly ----
    for (int v = 0; v < nVO; ++v) {
        CHECK(M.carrierType[v] == SUBDIV_CARRIER_VERTEX && M.carrierIndex[v] == v, "prefix carrier %d", v);
        CHECK(M.V.row(v) == VO.row(v), "prefix position %d", v);
    }
    for (int64_t v = nVO; v < Vs; ++v)
        CHECK(M.carrierType[v] != SUBDIV_CARRIER_VERTEX, "new vertex %lld has VERTEX carrier", (long long)v);

    // ---- positions recomputed from carriers (independent of fineFace/fineBary) ----
    double maxPosErr = 0.0, maxBaryErr = 0.0;
    for (int64_t v = 0; v < Vs; ++v) {
        const int idx = M.carrierIndex[v];
        RowVector3d p;
        switch (M.carrierType[v]) {
        case SUBDIV_CARRIER_VERTEX: p = VO.row(idx); break;
        case SUBDIV_CARRIER_EDGE:
            p = M.carrierCoord(v, 0) * VO.row(M.origEdges(idx, 0)) + M.carrierCoord(v, 1) * VO.row(M.origEdges(idx, 1));
            CHECK(M.carrierCoord(v, 0) > 0 && M.carrierCoord(v, 1) > 0 && M.carrierCoord(v, 2) == 0
                  && M.carrierCoord(v, 0) + M.carrierCoord(v, 1) == 1.0, "edge coords %lld", (long long)v);
            break;
        default:
            p = M.carrierCoord(v, 0) * VO.row(FO(idx, 0)) + M.carrierCoord(v, 1) * VO.row(FO(idx, 1))
              + M.carrierCoord(v, 2) * VO.row(FO(idx, 2));
            CHECK(M.carrierCoord.row(v).minCoeff() > 0 && M.carrierCoord.row(v).sum() == 1.0, "face coords %lld", (long long)v);
        }
        maxPosErr = std::max(maxPosErr, (p - M.V.row(v)).norm());
        const int f = M.fineFace[v];
        if (f >= 0) {
            CHECK(M.fineBary.row(v).minCoeff() >= 0.0 && M.fineBary.row(v).sum() == 1.0, "fine bary %lld", (long long)v);
            const RowVector3d q = M.fineBary(v, 0) * VO.row(FO(f, 0)) + M.fineBary(v, 1) * VO.row(FO(f, 1))
                                + M.fineBary(v, 2) * VO.row(FO(f, 2));
            maxBaryErr = std::max(maxBaryErr, (q - p).norm());
        }
    }
    fprintf(stderr, "max |carrier pos - V| = %.3g, max |fine bary pos - carrier pos| = %.3g (scale %.3g)\n",
            maxPosErr, maxBaryErr, scale);
    CHECK(maxPosErr <= 1e-12 * scale && maxBaryErr <= 1e-12 * scale, "position mismatch");

    // ---- geometry: sub faces tile each original face exactly, same orientation ----
    std::vector<double> areaSum(nFO, 0.0);
    int nFlipped = 0, nDegenerate = 0;
    for (int64_t f = 0; f < Fs; ++f) {
        const int a = M.F(f, 0), b = M.F(f, 1), c = M.F(f, 2);
        CHECK(a != b && b != c && c != a, "repeated corner in sub face %lld", (long long)f);
        const int fo = M.faceOrig[f];
        const Vector3d n = tri_n(M.V, a, b, c);
        const Vector3d no = tri_n(VO, FO(fo, 0), FO(fo, 1), FO(fo, 2));
        areaSum[fo] += 0.5 * n.norm();
        if (n.norm() == 0.0) ++nDegenerate;
        else if (n.dot(no) <= 0.0) ++nFlipped;
    }
    double maxAreaRel = 0.0;
    for (int f = 0; f < nFO; ++f) {
        const double A = 0.5 * tri_n(VO, FO(f, 0), FO(f, 1), FO(f, 2)).norm();
        if (A > 0) maxAreaRel = std::max(maxAreaRel, std::abs(areaSum[f] - A) / A);
    }
    fprintf(stderr, "area tiling max rel err = %.3g, flipped sub faces = %d, degenerate = %d\n",
            maxAreaRel, nFlipped, nDegenerate);
    CHECK(maxAreaRel < 1e-9, "sub faces do not tile original faces");
    CHECK(nFlipped == 0, "orientation not preserved");

    // ---- topology: every sub edge's valence follows from its endpoint carriers ----
    // Endpoint carriers span a set U of original vertices (a face-interior
    // endpoint marks the edge as private to that face):
    //   an endpoint inside face f -> edge interior to f alone      -> valence 2
    //   |U| == 2 (both on one original edge e)                     -> valence(e)
    //   |U| == 3 (crosses face interior between boundary points)   -> 2 x #faces on U
    {
        std::unordered_map<uint64_t, int> val0, valL;
        for (int f = 0; f < nFO; ++f) for (int c = 0; c < 3; ++c) val0[key(FO(f, c), FO(f, (c + 1) % 3))]++;
        std::map<std::array<int, 3>, int> faceMult;
        for (int f = 0; f < nFO; ++f) {
            std::array<int, 3> k = { FO(f, 0), FO(f, 1), FO(f, 2) };
            std::sort(k.begin(), k.end());
            faceMult[k]++;
        }
        valL.reserve((size_t)(Fs * 2));
        for (int64_t f = 0; f < Fs; ++f) for (int c = 0; c < 3; ++c) valL[key(M.F(f, c), M.F(f, (c + 1) % 3))]++;

        auto span = [&](int v, std::set<int> & U) -> int {  // returns face if FACE carrier, else -1
            const int idx = M.carrierIndex[v];
            switch (M.carrierType[v]) {
            case SUBDIV_CARRIER_VERTEX: U.insert(idx); return -1;
            case SUBDIV_CARRIER_EDGE:   U.insert(M.origEdges(idx, 0)); U.insert(M.origEdges(idx, 1)); return -1;
            default:                    return idx;
            }
        };
        int64_t nBad = 0;
        std::map<int, int64_t> hL;
        for (auto & kv : valL) {
            const int u = (int)(kv.first >> 32), w = (int)(kv.first & 0xffffffffu);
            std::set<int> U;
            const int fu = span(u, U), fw = span(w, U);
            int expect = -1;
            if (fu >= 0 || fw >= 0) {
                if (fu >= 0 && fw >= 0 && fu != fw) expect = -2;  // two different face interiors: impossible
                else expect = 2;
            } else if (U.size() == 2) {
                auto it = val0.find(key(*U.begin(), *U.rbegin()));
                expect = (it == val0.end()) ? -3 : it->second;
            } else if (U.size() == 3) {
                std::array<int, 3> k; std::copy(U.begin(), U.end(), k.begin());
                auto it = faceMult.find(k);
                expect = (it == faceMult.end()) ? -4 : 2 * it->second;
            }
            if (kv.second != expect) {
                if (nBad < 5) fprintf(stderr, "  edge (%d,%d): valence %d, expected %d\n", u, w, kv.second, expect);
                ++nBad;
            }
            hL[kv.second]++;
        }
        CHECK(nBad == 0, "%lld sub edges have the wrong valence", (long long)nBad);
        fprintf(stderr, "edge valence histogram (level %d):", L);
        for (auto & kv : hL) fprintf(stderr, " %d:%lld", kv.first, (long long)kv.second);
        fprintf(stderr, "  (duplicate-face groups: %d)\n",
                (int)std::count_if(faceMult.begin(), faceMult.end(), [](auto & kv) { return kv.second > 1; }));
    }

    // ---- struct IDs ----
    if (argc > 3) {
        MatStruct E;
        std::string err;
        if (!load_matstruct(argv[3], VO, FO, E, &err)) { fprintf(stderr, "FAIL: %s\n", err.c_str()); return 1; }
        StructPalette P; std::vector<int32_t> sid;
        auto t2 = std::chrono::steady_clock::now();
        build_struct_sets(M, FO, &E, P, sid);
        auto t3 = std::chrono::steady_clock::now();
        fprintf(stderr, "struct sets: %.2f s\n", std::chrono::duration<double>(t3 - t2).count());

        // Reference: the pre-unification reader, verbatim.
        std::vector<std::set<int>> ref;
        CHECK(legacy_load_matstruct(argv[3], ref), "legacy_load_matstruct");
        CHECK(ref == E.vertexIds, "shared reader's vertex sets differ from the old load_matstruct");
        auto setOf = [&](int64_t v) {
            const int k = sid[v];
            return std::set<int>(P.ids.begin() + P.offsets[k], P.ids.begin() + P.offsets[k + 1]);
        };
        int nMismatch = 0;
        for (int v = 0; v < nVO; ++v) nMismatch += (setOf(v) != ref[v]);
        CHECK(nMismatch == 0, "%d original vertices differ from load_matstruct", nMismatch);

        // Independent rule for new vertices: an edge-carried vertex gets the ids
        // shared by both edge endpoints that are seam/boundary ids of that .ma edge
        // plus sheet ids of the incident faces; a face-carried vertex gets its face's
        // sheet ids. Check the weaker, construction-independent property that every
        // new vertex's set is a subset of both carrier endpoints' load_matstruct sets
        // (for an edge) or of all three corners' sets (for a face), and that sheet-
        // only interiors carry no seam/boundary/junction bits.
        int nNotSubset = 0, nBadMask = 0;
        for (int64_t v = nVO; v < Vs; ++v) {
            const std::set<int> s = setOf(v);
            const int idx = M.carrierIndex[v];
            std::vector<int> corners;
            if (M.carrierType[v] == SUBDIV_CARRIER_EDGE) corners = { M.origEdges(idx, 0), M.origEdges(idx, 1) };
            else corners = { FO(idx, 0), FO(idx, 1), FO(idx, 2) };
            for (int c : corners)
                for (int id : s) if (!ref[c].count(id)) { ++nNotSubset; break; }
            const uint8_t m = P.typeMask[sid[v]];
            if (M.carrierType[v] == SUBDIV_CARRIER_FACE && (m & ~STRUCT_MASK_SHEET)) ++nBadMask;
            if (m & STRUCT_MASK_JUNCTION) ++nBadMask;
        }
        CHECK(nNotSubset == 0, "%d new vertices have ids not on their carrier's corners", nNotSubset);
        CHECK(nBadMask == 0, "%d new vertices have an impossible type mask", nBadMask);

        std::map<uint8_t, int64_t> maskHist; int64_t nEmpty = 0;
        for (int64_t v = 0; v < Vs; ++v) { maskHist[P.typeMask[sid[v]]]++; nEmpty += (P.offsets[sid[v] + 1] == P.offsets[sid[v]]); }
        fprintf(stderr, "type-mask histogram:");
        for (auto & kv : maskHist) fprintf(stderr, " 0x%x:%lld", kv.first, (long long)kv.second);
        fprintf(stderr, "  (empty sets: %lld)\n", (long long)nEmpty);
    }

    fprintf(stderr, g_fail ? "==> %d CHECKS FAILED\n" : "==> all checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}
