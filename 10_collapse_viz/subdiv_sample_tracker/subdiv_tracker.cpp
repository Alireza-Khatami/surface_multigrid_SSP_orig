#include "subdiv_tracker.h"

#include <single_collapse_data.h>

#include <igl/writeOBJ.h>
#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace Eigen;

// ---- globals from main.cpp ----
extern MatrixXd gV;
extern MatrixXd gVO;
extern MatrixXi gF;
extern MatrixXi gFO;
extern std::vector<single_collapse_data> gDecInfo;
extern std::vector<std::set<int>> gVertexStructIDs;

namespace {

bool                 gEnabled = false;
SubdivMesh           gM;
StructPalette        gPal;
std::vector<int32_t> gSet;

// Per sample (= subdivided vertex). gFace < 0: untracked (vertex on no face).
std::vector<int32_t> gFace;   // gF row
std::vector<double>  gBC;     // 3 per sample, in gBF corner order
std::vector<int32_t> gBF;     // 3 per sample, gV ids at the time of the last cast

std::vector<std::vector<int32_t>> gBucket;  // gF row -> samples on it
std::vector<int32_t> gRedirect;             // gV id -> survivor (itself while live)
std::vector<int32_t> gFaceStamp;            // gF row -> last collapse that recast it
int                  gCollapses = 0;
int64_t              gRequested = 0;

struct Stats {
    int64_t casts = 0;
    int64_t nonIdentityOrder = 0;   // sample bary order != FUV_pre corner order
    int64_t cornerMismatch = 0;     // sample corners != pre face corners
    int64_t sdMismatch = 0;         // a sheet's (s, d) != the collapse's (s, d)
    int64_t faceInTwoSheets = 0;    // a pre face listed by two sheets of one collapse
    int64_t stranded = 0;           // samples on pre faces of a sheet with no post faces
    int64_t outside = 0;            // query outside every post face (clamped)
    double  maxOutside = 0.0;       // largest -min(bary) among those
} gStats;

int gLogBudget = 20;
#define TRK_LOG(...) do { if (gLogBudget > 0) { --gLogBudget; fprintf(stderr, __VA_ARGS__); } } while (0)

int resolve(int v)
{
    while (gRedirect[v] != v) {
        gRedirect[v] = gRedirect[gRedirect[v]];  // path halving
        v = gRedirect[v];
    }
    return v;
}

void fail(const std::string & msg) { throw std::runtime_error("[subdiv_tracker] " + msg); }

} // namespace

// ---------------------------------------------------------------- init

void subdiv_tracker_init(int64_t nTarget, const std::string & matstructPath)
{
    gEnabled = false;
    gStats = Stats();
    gCollapses = 0;
    gLogBudget = 20;
    gRequested = nTarget;

    const int nFO = (int)gFO.rows();
    if (nFO == 0) fail("empty fine mesh");
    if (gF.rows() < nFO) fail("gF has fewer rows than gFO");
    for (int f = 0; f < nFO; ++f)
        if (gF.row(f) != gFO.row(f))
            fail("gF row " + std::to_string(f) + " differs from gFO row (seeding assumes they match)");

    gM = build_subdiv_mesh(gVO, gFO, nTarget);

    if (!matstructPath.empty()) {
        const MatStructElements E = load_matstruct_elements(matstructPath, gVO, gFO, gM.origEdges);
        if (gVertexStructIDs.size() == E.vertexIds.size()) {
            for (size_t v = 0; v < E.vertexIds.size(); ++v)
                if (std::set<int>(E.vertexIds[v].begin(), E.vertexIds[v].end()) != gVertexStructIDs[v])
                    fail("struct IDs of vertex " + std::to_string(v) + " differ from load_matstruct");
        }
        build_struct_sets(gM, gFO, &E, gPal, gSet);
    } else {
        build_struct_sets(gM, gFO, nullptr, gPal, gSet);
    }

    const size_t Vs = gM.carrierType.size();
    gFace.assign(Vs, -1);
    gBC.assign(3 * Vs, 0.0);
    gBF.assign(3 * Vs, -1);
    gBucket.assign(gF.rows(), {});
    gFaceStamp.assign(gF.rows(), 0);
    gRedirect.resize(gV.rows());
    std::iota(gRedirect.begin(), gRedirect.end(), 0);

    size_t nUntracked = 0;
    for (size_t i = 0; i < Vs; ++i) {
        const int f = gM.fineFace[i];
        if (f < 0) { ++nUntracked; continue; }
        gFace[i] = f;
        for (int c = 0; c < 3; ++c) {
            gBC[3 * i + c] = gM.fineBary((Index)i, c);
            gBF[3 * i + c] = gFO(f, c);
        }
        gBucket[f].push_back((int32_t)i);
    }

    gEnabled = true;
    fprintf(stderr, "[subdiv_tracker] init: %zu samples (%zu untracked: vertices on no face), "
            "%d levels, %lld faces\n", Vs, nUntracked, gM.nLevels, (long long)gM.F.rows());
}

bool subdiv_tracker_enabled() { return gEnabled; }

// ---------------------------------------------------------------- update

void subdiv_tracker_update(int s, int d)
{
    if (!gEnabled || gDecInfo.empty()) return;
    const single_collapse_data & D = gDecInfo.back();
    const int ci = ++gCollapses;

    const int nV = (int)gRedirect.size();
    if (s < 0 || d < 0 || s >= nV || d >= nV || s == d) fail("bad collapse pair");
    if (resolve(s) != s || resolve(d) != d) fail("collapse endpoint already absorbed");

    for (const SheetData & sd : D.sheets) {
        if (sd.b.size() < 2) continue;
        const int gs = sd.subsetVIdx(sd.b(0)), gd = sd.subsetVIdx(sd.b(1));
        if (gs != s || gd != d) {
            ++gStats.sdMismatch;
            TRK_LOG("[subdiv_tracker] collapse %d: sheet %d reports (s,d)=(%d,%d), collapse is (%d,%d)\n",
                    ci, sd.global_sheet_id, gs, gd, s, d);
        }
    }

    std::vector<std::pair<int32_t, int32_t>> items;  // (sample, pre row)
    std::vector<double> pa0, pa1, pv0x, pv0y, pv1x, pv1y, pd00, pd01, pd11, pden;

    for (const SheetData & sd : D.sheets) {
        const int nPre = (int)sd.FIdx_pre.size();
        if (nPre == 0) continue;

        items.clear();
        for (int r = 0; r < nPre; ++r) {
            const int f = sd.FIdx_pre(r);
            if (f < 0 || f >= (int)gBucket.size()) fail("pre face out of range");
            if (gFaceStamp[f] == ci) {
                ++gStats.faceInTwoSheets;
                TRK_LOG("[subdiv_tracker] collapse %d: face %d is a pre face of two sheets\n", ci, f);
                continue;
            }
            if (sd.FIdx_post.size() == 0) {
                gStats.stranded += (int64_t)gBucket[f].size();
                continue;
            }
            gFaceStamp[f] = ci;
            for (int32_t si : gBucket[f]) items.push_back({ si, r });
            gBucket[f].clear();
        }
        if (items.empty()) continue;

        // Per post face: the quantities compute_barycentric() derives, same formulas.
        const int nPost = (int)sd.FUV_post.rows();
        pa0.resize(nPost); pa1.resize(nPost);
        pv0x.resize(nPost); pv0y.resize(nPost); pv1x.resize(nPost); pv1y.resize(nPost);
        pd00.resize(nPost); pd01.resize(nPost); pd11.resize(nPost); pden.resize(nPost);
        for (int j = 0; j < nPost; ++j) {
            const double ax = sd.UV_post(sd.FUV_post(j, 0), 0), ay = sd.UV_post(sd.FUV_post(j, 0), 1);
            const double bx = sd.UV_post(sd.FUV_post(j, 1), 0), by = sd.UV_post(sd.FUV_post(j, 1), 1);
            const double cx = sd.UV_post(sd.FUV_post(j, 2), 0), cy = sd.UV_post(sd.FUV_post(j, 2), 1);
            pa0[j] = ax; pa1[j] = ay;
            pv0x[j] = bx - ax; pv0y[j] = by - ay;
            pv1x[j] = cx - ax; pv1y[j] = cy - ay;
            pd00[j] = pv0x[j] * pv0x[j] + pv0y[j] * pv0y[j];
            pd01[j] = pv0x[j] * pv1x[j] + pv0y[j] * pv1y[j];
            pd11[j] = pv1x[j] * pv1x[j] + pv1y[j] * pv1y[j];
            pden[j] = pd00[j] * pd11[j] - pd01[j] * pd01[j];
        }

        for (const auto & it : items) {
            const int32_t si = it.first;
            const int r = it.second;
            double * bc = &gBC[3 * (size_t)si];
            int32_t * bf = &gBF[3 * (size_t)si];

            // Express the sample's bary in FUV_pre(r) corner order, matching by vertex id.
            int rc[3];
            for (int c = 0; c < 3; ++c) rc[c] = resolve(bf[c]);
            double bpre[3];
            bool ok = rc[0] != rc[1] && rc[1] != rc[2] && rc[2] != rc[0];
            bool identity = true;
            for (int k = 0; k < 3 && ok; ++k) {
                const int g = sd.subsetVIdx(sd.FUV_pre(r, k));
                int c = 0;
                while (c < 3 && rc[c] != g) ++c;
                if (c == 3) { ok = false; break; }
                bpre[k] = bc[c];
                identity = identity && (c == k);
            }
            if (!ok) {
                ++gStats.cornerMismatch;
                TRK_LOG("[subdiv_tracker] collapse %d: sample %d corners (%d,%d,%d) do not match pre face %d\n",
                        ci, si, rc[0], rc[1], rc[2], sd.FIdx_pre(r));
                for (int k = 0; k < 3; ++k) bpre[k] = bc[k];
            } else if (!identity) {
                ++gStats.nonIdentityOrder;
            }

            // Query point in UV_pre.
            const int l0 = sd.FUV_pre(r, 0), l1 = sd.FUV_pre(r, 1), l2 = sd.FUV_pre(r, 2);
            const double qx = bpre[0] * sd.UV_pre(l0, 0) + bpre[1] * sd.UV_pre(l1, 0) + bpre[2] * sd.UV_pre(l2, 0);
            const double qy = bpre[0] * sd.UV_pre(l0, 1) + bpre[1] * sd.UV_pre(l1, 1) + bpre[2] * sd.UV_pre(l2, 1);

            // Face of UV_post that best contains it: smallest max(-bary), first wins
            // (same selection as the old trackers, including best = 0 when no face
            // scores below 1).
            auto baryOf = [&](int j, double & u, double & v, double & w) {
                const double v2x = qx - pa0[j], v2y = qy - pa1[j];
                const double d20 = v2x * pv0x[j] + v2y * pv0y[j];
                const double d21 = v2x * pv1x[j] + v2y * pv1y[j];
                v = (pd11[j] * d20 - pd01[j] * d21) / pden[j];
                w = (pd00[j] * d21 - pd01[j] * d20) / pden[j];
                u = 1.0 - (v + w);
            };
            int best = 0; double minD = 1.0;
            for (int j = 0; j < nPost; ++j) {
                double u, v, w;
                baryOf(j, u, v, w);
                const double dj = -std::min(u, std::min(v, w));
                if (dj < minD) { minD = dj; best = j; }
            }
            double bu, bv, bw;
            baryOf(best, bu, bv, bw);
            if (minD > 1e-12) {
                ++gStats.outside;
                gStats.maxOutside = std::max(gStats.maxOutside, minD);
            }

            double B[3] = { std::max(0.0, bu), std::max(0.0, bv), std::max(0.0, bw) };
            const double sum = B[0] + B[1] + B[2];
            if (sum > 1e-12) { B[0] /= sum; B[1] /= sum; B[2] /= sum; }

            const int nf = sd.FIdx_post(best);
            gFace[si] = nf;
            for (int k = 0; k < 3; ++k) {
                bc[k] = B[k];
                bf[k] = sd.subsetVIdx(sd.FUV_post(best, k));
            }
            gBucket[nf].push_back(si);
            ++gStats.casts;
        }
    }

    gRedirect[d] = s;
}

// ---------------------------------------------------------------- accessors

const SubdivMesh &            subdiv_tracker_mesh()     { return gM; }
const StructPalette &         subdiv_tracker_palette()  { return gPal; }
const std::vector<int32_t> &  subdiv_tracker_set_ids()  { return gSet; }
int  subdiv_tracker_collapses()      { return gCollapses; }
bool subdiv_tracker_is_tracked(int i) { return gEnabled && i >= 0 && i < (int)gFace.size() && gFace[i] >= 0; }
int  subdiv_tracker_cur_face(int i)   { return subdiv_tracker_is_tracked(i) ? gFace[i] : -1; }

Vector3d subdiv_tracker_cur_bary(int i)
{
    if (!subdiv_tracker_is_tracked(i)) return Vector3d::Zero();
    return Vector3d(gBC[3 * (size_t)i], gBC[3 * (size_t)i + 1], gBC[3 * (size_t)i + 2]);
}

Vector3i subdiv_tracker_cur_corners(int i)
{
    if (!subdiv_tracker_is_tracked(i)) return Vector3i(-1, -1, -1);
    return Vector3i(resolve(gBF[3 * (size_t)i]), resolve(gBF[3 * (size_t)i + 1]), resolve(gBF[3 * (size_t)i + 2]));
}

Vector3d subdiv_tracker_cur_pos(int i)
{
    if (!gEnabled || i < 0 || i >= (int)gFace.size()) return Vector3d::Zero();
    if (gFace[i] < 0) return gM.V.row(i).transpose();
    Vector3d p = Vector3d::Zero();
    for (int c = 0; c < 3; ++c)
        p += gBC[3 * (size_t)i + c] * gV.row(resolve(gBF[3 * (size_t)i + c])).leftCols(3).transpose();
    return p;
}

void subdiv_tracker_cur_positions(MatrixXd & P)
{
    const int n = (int)gFace.size();
    P.resize(n, 3);
    for (int i = 0; i < n; ++i) P.row(i) = subdiv_tracker_cur_pos(i).transpose();
}

// ---------------------------------------------------------------- save
//
// subdiv_<stem>.sdt, little-endian. Header (224 bytes):
//   0   char[8]  magic "SUBDIVT\0"
//   8   uint32   version (1)
//   12  uint32   header_bytes (224)
//   16  uint32   n_levels
//   20  uint32   reserved (0)
//   24  uint64   n_samples_requested
//   32  uint64   n_sub_verts      (Vs)
//   40  uint64   n_sub_faces      (Fs)
//   48  uint64   n_fine_verts     (|gVO|)
//   56  uint64   n_fine_faces     (|gFO|)
//   64  uint64   n_orig_edges     (nE)
//   72  uint64   n_coarse_verts   (|cmc.Vbase|)
//   80  uint64   n_coarse_faces   (|cmc.Fout|)
//   88  uint64   n_palette        (P)
//   96  uint64   n_palette_ids
//   104 uint64   n_arrays (14)
//   112 uint64   offsets[14]      byte offset of each array; each is 8-byte aligned
// Arrays:
//   0  sub_V             float64 Vs x 3   subdivided vertex positions on the fine mesh
//   1  sub_F             int32   Fs x 3   subdivided faces (Laplacian connectivity)
//   2  sub_face_orig     int32   Fs       fine face (gFO row) containing each sub face
//   3  orig_edges        int32   nE x 2   fine-mesh edges (min, max); EDGE carrier index
//   4  carrier_type      uint8   Vs       0 VERTEX, 1 EDGE, 2 FACE
//   5  carrier_index     int32   Vs       gVO vertex / orig_edges row / gFO face
//   6  fine_face         int32   Vs       gFO face the vertex lies on (-1: on no face)
//   7  fine_bary         float64 Vs x 3   in gFO.row(fine_face) corner order; for an EDGE
//                                         carrier two entries are the linear (1-t, t)
//   8  coarse_face       int32   Vs       face of the compact coarse mesh (cmc.Fout =
//                                         simplified_<stem>.obj face order); -1 untracked
//   9  coarse_bary       float64 Vs x 3   in cmc.Fout.row(coarse_face) corner order
//   10 struct_set_id     int32   Vs       index into the palette
//   11 palette_offsets   int32   P+1      set k = palette_ids[offsets[k] .. offsets[k+1])
//   12 palette_ids       int32   n_palette_ids
//   13 palette_type_mask uint8   P        bit0 sheet, bit1 seam, bit2 boundary, bit3 junction

namespace {

struct BinWriter {
    std::ofstream f;
    uint64_t pos = 0;
    void raw(const void * p, size_t n)
    {
        f.write(static_cast<const char *>(p), (std::streamsize)n);
        pos += n;
    }
    void pad8()
    {
        static const char z[8] = {};
        const size_t n = (size_t)((8 - pos % 8) % 8);
        if (n) raw(z, n);
    }
    template <class T> void val(T v) { raw(&v, sizeof(T)); }
};

} // namespace

void subdiv_tracker_save(const CoarseFaceLookup & lookup, const std::string & path)
{
    if (!gEnabled) return;
    const CoarseMeshCompaction & cmc = lookup.cmc;
    const size_t Vs = gFace.size();

    if (gCollapses != (int)gDecInfo.size())
        fail("tracker saw " + std::to_string(gCollapses) + " collapses, gDecInfo has "
             + std::to_string(gDecInfo.size()));

    // ---- resolve on the compact coarse mesh + consistency checks ----
    std::vector<int32_t> cFace(Vs, -1);
    std::vector<double>  cBary(3 * Vs, 0.0);
    int64_t nFaceMatch = 0, nFaceDead = 0, nFaceOther = 0;
    double maxBarySumErr = 0.0, minBary = 0.0, maxPosDiff = 0.0;
    for (size_t i = 0; i < Vs; ++i) {
        if (gFace[i] < 0) continue;
        const int f = gFace[i];
        RowVector3i corners;
        RowVector3d bary;
        for (int c = 0; c < 3; ++c) {
            corners(c) = resolve(gBF[3 * i + c]);
            bary(c) = gBC[3 * i + c];
        }
        maxBarySumErr = std::max(maxBarySumErr, std::abs(bary.sum() - 1.0));
        minBary = std::min(minBary, bary.minCoeff());

        // The sample's corners must be exactly its gF face's current corners.
        std::array<int, 3> a = { gF(f, 0), gF(f, 1), gF(f, 2) }, b = { corners(0), corners(1), corners(2) };
        std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
        if (gF(f, 0) < 0 || gF(f, 0) == gF(f, 1)) ++nFaceDead;
        else if (a == b) ++nFaceMatch;
        else ++nFaceOther;

        const CoarseSample cs = lookup.resolve(f, corners, bary);
        cFace[i] = cs.face;
        for (int c = 0; c < 3; ++c) cBary[3 * i + c] = cs.bary(c);

        const Vector3d pg = subdiv_tracker_cur_pos((int)i);
        Vector3d pc = Vector3d::Zero();
        for (int c = 0; c < 3; ++c) pc += cs.bary(c) * cmc.Vbase.row(cmc.Fout(cs.face, c)).leftCols(3).transpose();
        maxPosDiff = std::max(maxPosDiff, (pg - pc).norm());
    }

    fprintf(stderr,
        "[subdiv_tracker] %d collapses, %lld casts | order!=FUV_pre %lld | corner mismatch %lld | "
        "(s,d) mismatch %lld | face in 2 sheets %lld | stranded %lld | outside post ring %lld (max %.3g)\n",
        gCollapses, (long long)gStats.casts, (long long)gStats.nonIdentityOrder,
        (long long)gStats.cornerMismatch, (long long)gStats.sdMismatch,
        (long long)gStats.faceInTwoSheets, (long long)gStats.stranded,
        (long long)gStats.outside, gStats.maxOutside);
    fprintf(stderr,
        "[subdiv_tracker] final: corners==gF face %lld, on dead face %lld, other %lld | "
        "max |sum(bary)-1| %.3g, min bary %.3g | max |pos(gV) - pos(compact)| %.3g\n",
        (long long)nFaceMatch, (long long)nFaceDead, (long long)nFaceOther,
        maxBarySumErr, minBary, maxPosDiff);

    // ---- write ----
    const int kArrays = 14;
    const uint32_t kHeader = 112 + 8 * kArrays;
    const uint64_t nE = (uint64_t)gM.origEdges.rows();
    const uint64_t P = (uint64_t)gPal.size();

    // Row-major copies where Eigen storage is column-major.
    std::vector<int32_t> subF((size_t)gM.F.rows() * 3), origE((size_t)nE * 2);
    for (Index f = 0; f < gM.F.rows(); ++f) for (int c = 0; c < 3; ++c) subF[3 * f + c] = gM.F(f, c);
    for (Index e = 0; e < (Index)nE; ++e) for (int c = 0; c < 2; ++c) origE[2 * e + c] = gM.origEdges(e, c);
    std::vector<double> subV(Vs * 3), fineB(Vs * 3);
    for (size_t i = 0; i < Vs; ++i) for (int c = 0; c < 3; ++c) {
        subV[3 * i + c]  = gM.V((Index)i, c);
        fineB[3 * i + c] = gM.fineBary((Index)i, c);
    }

    struct Arr { const void * p; size_t bytes; };
    const Arr arrs[kArrays] = {
        { subV.data(),              subV.size() * 8 },
        { subF.data(),              subF.size() * 4 },
        { gM.faceOrig.data(),       gM.faceOrig.size() * 4 },
        { origE.data(),             origE.size() * 4 },
        { gM.carrierType.data(),    gM.carrierType.size() },
        { gM.carrierIndex.data(),   gM.carrierIndex.size() * 4 },
        { gM.fineFace.data(),       gM.fineFace.size() * 4 },
        { fineB.data(),             fineB.size() * 8 },
        { cFace.data(),             cFace.size() * 4 },
        { cBary.data(),             cBary.size() * 8 },
        { gSet.data(),              gSet.size() * 4 },
        { gPal.offsets.data(),      gPal.offsets.size() * 4 },
        { gPal.ids.data(),          gPal.ids.size() * 4 },
        { gPal.typeMask.data(),     gPal.typeMask.size() },
    };
    uint64_t offsets[kArrays];
    uint64_t at = kHeader;
    for (int k = 0; k < kArrays; ++k) {
        at = (at + 7) / 8 * 8;
        offsets[k] = at;
        at += arrs[k].bytes;
    }

    BinWriter w;
    w.f.open(path, std::ios::binary);
    if (!w.f) fail("cannot write " + path);
    const char magic[8] = { 'S', 'U', 'B', 'D', 'I', 'V', 'T', '\0' };
    w.raw(magic, 8);
    w.val<uint32_t>(1);
    w.val<uint32_t>(kHeader);
    w.val<uint32_t>((uint32_t)gM.nLevels);
    w.val<uint32_t>(0);
    w.val<uint64_t>((uint64_t)gRequested);
    w.val<uint64_t>(Vs);
    w.val<uint64_t>((uint64_t)gM.F.rows());
    w.val<uint64_t>((uint64_t)gVO.rows());
    w.val<uint64_t>((uint64_t)gFO.rows());
    w.val<uint64_t>(nE);
    w.val<uint64_t>((uint64_t)cmc.Vbase.rows());
    w.val<uint64_t>((uint64_t)cmc.Fout.rows());
    w.val<uint64_t>(P);
    w.val<uint64_t>((uint64_t)gPal.ids.size());
    w.val<uint64_t>((uint64_t)kArrays);
    for (int k = 0; k < kArrays; ++k) w.val<uint64_t>(offsets[k]);
    if (w.pos != kHeader) fail("header size mismatch");
    for (int k = 0; k < kArrays; ++k) {
        w.pad8();
        if (w.pos != offsets[k]) fail("array offset mismatch");
        w.raw(arrs[k].p, arrs[k].bytes);
    }
    w.f.close();
    if (!w.f) fail("write failed for " + path);
    fprintf(stderr, "[subdiv_tracker] wrote %s (%.1f MB)\n", path.c_str(), w.pos / 1048576.0);
}

void subdiv_tracker_export_fine_obj(const std::string & path, int64_t maxVerts)
{
    if (!gEnabled) return;
    if ((int64_t)gM.V.rows() > maxVerts) {
        fprintf(stderr, "[subdiv_tracker] skipping %s: %lld vertices > %lld\n",
                path.c_str(), (long long)gM.V.rows(), (long long)maxVerts);
        return;
    }
    if (!igl::writeOBJ(path, gM.V, gM.F))
        fprintf(stderr, "[subdiv_tracker] writeOBJ failed: %s\n", path.c_str());
    else
        fprintf(stderr, "[subdiv_tracker] fine subdivided mesh -> %s\n", path.c_str());
}

void subdiv_tracker_export_deformed_obj(const std::string & path, int64_t maxVerts)
{
    if (!gEnabled) return;
    if ((int64_t)gFace.size() > maxVerts) {
        fprintf(stderr, "[subdiv_tracker] skipping %s: %zu vertices > %lld\n",
                path.c_str(), gFace.size(), (long long)maxVerts);
        return;
    }
    MatrixXd P;
    subdiv_tracker_cur_positions(P);
    if (!igl::writeOBJ(path, P, gM.F))
        fprintf(stderr, "[subdiv_tracker] writeOBJ failed: %s\n", path.c_str());
    else
        fprintf(stderr, "[subdiv_tracker] deformed subdivided mesh -> %s\n", path.c_str());
}
