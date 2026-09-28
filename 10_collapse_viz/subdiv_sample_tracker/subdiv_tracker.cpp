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
#include <map>
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

namespace {

bool                 gEnabled = false;
SubdivMesh           gM;
StructPalette        gPal;
std::vector<int32_t> gSet;
bool                 gRelaxed = false;
MatrixXd             gVseed;   // positions before relaxation (relaxed init only)
RelaxGraph           gGraph;
std::string          gRelaxMethod;
std::vector<int64_t> gAnchors;      // solve_project curve anchors (subdivided vertex ids)
std::vector<int32_t> gAnchorKey;    // per anchor: smallest seam/boundary struct id of its set

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
    int64_t placementChecked = 0;   // post-ring corners compared with gV
    int64_t placementUnchecked = 0; // sheets without V_post
    int64_t placementNaN = 0;
    double  maxPlacementDiff = 0.0; // max |V_post corner - gV| after the collapse
} gStats;

int gLogBudget = 20;
int gOutsideLogBudget = 200;        // per (collapse, sheet) lines for samples outside the post ring
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

void subdiv_tracker_init(int64_t nTarget, const MatStruct * ms, bool relax, const std::string & method,
                         int curveAnchors, double curveAnchorTol)
{
    gEnabled = false;
    gStats = Stats();
    gCollapses = 0;
    gLogBudget = 20;
    gOutsideLogBudget = 200;
    gRequested = nTarget;

    const int nFO = (int)gFO.rows();
    if (nFO == 0) fail("empty fine mesh");
    if (gF.rows() < nFO) fail("gF has fewer rows than gFO");
    for (int f = 0; f < nFO; ++f)
        if (gF.row(f) != gFO.row(f))
            fail("gF row " + std::to_string(f) + " differs from gFO row (seeding assumes they match)");

    gM = build_subdiv_mesh(gVO, gFO, nTarget);

    build_struct_sets(gM, gFO, ms, gPal, gSet);

    gRelaxed = false;
    gVseed.resize(0, 3);
    gGraph = RelaxGraph();
    gRelaxMethod.clear();
    gAnchors.clear();
    gAnchorKey.clear();
    if (relax) {
        gGraph = build_relax_graph(gM, gPal, gSet, ms);
        gVseed = gM.V;
        const MeshQuality q0 = subdiv_mesh_quality(gM.V, gM.F);
        if (method != "newton" && method != "solve_project") fail("unknown relax method " + method);
        const bool experiment = method == "solve_project";
        RelaxOptions expOpt;
        expOpt.curveAnchors = curveAnchors;
        expOpt.curveAnchorTol = curveAnchorTol;
        const RelaxReport R = experiment ? subdiv_relax_solve_project(gM, gVO, gFO, ms, gPal, gSet, gGraph, expOpt)
                                         : subdiv_relax(gM, gVO, gFO, ms, gPal, gSet, gGraph);
        const double tolStep = 1e-10;
        // The experiment is not expected to reach a resting state; it only reports.
        if (!experiment && (!R.converged || R.jacobiStepMove > tolStep))
            fail("relaxation did not converge (one relaxation step still moves "
                 + std::to_string(R.jacobiStepMove) + " x diag)");
        if (R.seedOffStructure || R.fixedMoved || R.posMismatch || R.badBary || R.offStructure)
            fail("relaxation consistency checks failed");
        const MeshQuality q1 = subdiv_mesh_quality(gM.V, gM.F, &gVseed);
        fprintf(stderr,
            "[subdiv_tracker] relaxation (%s), before -> after: edge CV %.4f -> %.4f | min angle %.3f -> %.3f, "
            "p1 %.3f -> %.3f, p5 %.3f -> %.3f, median %.3f -> %.3f deg | degenerate %lld -> %lld | "
            "flipped vs seed %lld | move max %.3g mean %.3g (x diag)\n",
            method.c_str(), q0.edgeCV, q1.edgeCV, q0.minAngle, q1.minAngle, q0.p1, q1.p1, q0.p5, q1.p5, q0.median, q1.median,
            (long long)q0.degenerate, (long long)q1.degenerate, (long long)q1.flippedVsRef, R.maxMove, R.meanMove);
        gRelaxed = true;
        gRelaxMethod = method;
        gAnchors = R.anchors;
        for (int64_t v : gAnchors) {
            int32_t key = -1;
            const int k = gSet[v];
            for (int32_t a = gPal.offsets[k]; a < gPal.offsets[k + 1]; ++a) {
                const int id = gPal.ids[a];
                const int t = ms ? ms->structType.at(id) : -1;
                if ((t == 1 || t == 2) && (key < 0 || id < key)) key = id;
            }
            gAnchorKey.push_back(key);
        }
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

        int64_t nOutHere = 0; double maxOutHere = 0.0;
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
                ++nOutHere; maxOutHere = std::max(maxOutHere, minD);
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

        // Where do samples fall outside the post ring? Signed UV areas of the
        // pre and post patches (they should match) and inverted post faces.
        if (nOutHere > 0 && gOutsideLogBudget > 0) {
            --gOutsideLogBudget;
            auto triArea = [](const MatrixXd & UV, int a, int b, int c) {
                return 0.5 * ((UV(b, 0) - UV(a, 0)) * (UV(c, 1) - UV(a, 1)) - (UV(c, 0) - UV(a, 0)) * (UV(b, 1) - UV(a, 1)));
            };
            double aPre = 0, aPost = 0, minPre = 1e300, minPost = 1e300;
            int invPre = 0, invPost = 0;
            for (int r = 0; r < nPre; ++r) {
                const double a = triArea(sd.UV_pre, sd.FUV_pre(r, 0), sd.FUV_pre(r, 1), sd.FUV_pre(r, 2));
                aPre += a; minPre = std::min(minPre, a); invPre += a <= 0;
            }
            for (int j = 0; j < nPost; ++j) {
                const double a = triArea(sd.UV_post, sd.FUV_post(j, 0), sd.FUV_post(j, 1), sd.FUV_post(j, 2));
                aPost += a; minPost = std::min(minPost, a); invPost += a <= 0;
            }
            fprintf(stderr,
                "[subdiv_tracker] outside: collapse %d (s %d, d %d) sheet %d | %lld of %zu samples outside, max %.3g | "
                "lscm_case %d flap %d dc %d sym %d | pre %d faces area %.6g (min %.3g, inverted %d) | "
                "post %d faces area %.6g (min %.3g, inverted %d)\n",
                ci, s, d, sd.global_sheet_id, (long long)nOutHere, items.size(), maxOutHere,
                D.lscm_case ? *D.lscm_case : -1, D.numFlapFaces, (int)sd.has_double_cover, sd.dc_uv_symmetric,
                nPre, aPre, minPre, invPre, nPost, aPost, minPost, invPost);
        }
    }

    // The UV_post each sample was cast into was built from V_post, the ring with
    // the survivor at the decimation's placement point (whatever the cost
    // function chose). Check that geometry is the one the collapse actually
    // produced: every post-face corner's V_post row must equal gV now.
    for (const SheetData & sd : D.sheets) {
        if (sd.V_post.rows() == 0) { ++gStats.placementUnchecked; continue; }
        for (int j = 0; j < sd.FUV_post.rows(); ++j)
            for (int k = 0; k < 3; ++k) {
                const int l = sd.FUV_post(j, k);
                int g = sd.subsetVIdx(l);
                g = (g == d) ? s : resolve(g);
                const double diff = (sd.V_post.row(l).leftCols(3) - gV.row(g).leftCols(3)).norm();
                if (!(diff <= gStats.maxPlacementDiff)) {
                    if (std::isnan(diff)) ++gStats.placementNaN;
                    else gStats.maxPlacementDiff = diff;
                }
                ++gStats.placementChecked;
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
// subdiv_<stem>.sdt, little-endian. Header (224 bytes; 232 when relaxed):
//   0   char[8]  magic "SUBDIVT\0"
//   8   uint32   version (1; 2 when relaxed)
//   12  uint32   header_bytes (112 + 8 * n_arrays)
//   16  uint32   n_levels
//   20  uint32   relaxed (0 / 1; "reserved (0)" in version 1)
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
//   104 uint64   n_arrays (14; 15 when relaxed)
//   112 uint64   offsets[n_arrays] byte offset of each array; each is 8-byte aligned
// Arrays:
//   0  sub_V             float64 Vs x 3   subdivided vertex positions on the fine mesh
//                                         (after relaxation when relaxed)
//   1  sub_F             int32   Fs x 3   subdivided faces (Laplacian connectivity)
//   2  sub_face_orig     int32   Fs       fine face (gFO row) containing each sub face
//   3  orig_edges        int32   nE x 2   fine-mesh edges (min, max); EDGE carrier index
//   4  carrier_type      uint8   Vs       0 VERTEX, 1 EDGE, 2 FACE, of the seed position
//                                         (before relaxation); the struct ids come from it
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
//   14 sub_V_seed        float64 Vs x 3   relaxed only: positions before relaxation

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
        "[subdiv_tracker] placement: %lld post-ring corners vs gV, max diff %.3g (NaN %lld, sheets without V_post %lld)\n",
        (long long)gStats.placementChecked, gStats.maxPlacementDiff,
        (long long)gStats.placementNaN, (long long)gStats.placementUnchecked);
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
    const int kArrays = gRelaxed ? 15 : 14;
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
    std::vector<double> seedV;
    if (gRelaxed) {
        seedV.resize(Vs * 3);
        for (size_t i = 0; i < Vs; ++i) for (int c = 0; c < 3; ++c) seedV[3 * i + c] = gVseed((Index)i, c);
    }
    const Arr arrs[15] = {
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
        { seedV.data(),             seedV.size() * 8 },
    };
    uint64_t offsets[15];
    uint64_t at = kHeader;
    for (int k = 0; k < kArrays; ++k) {
        at = (at + 7) / 8 * 8;
        offsets[k] = at;
        at += arrs[k].bytes;
    }

    BinWriter w;
    w.f.open(subdiv_long_path(path), std::ios::binary);
    if (!w.f) fail("cannot write " + path);
    const char magic[8] = { 'S', 'U', 'B', 'D', 'I', 'V', 'T', '\0' };
    w.raw(magic, 8);
    w.val<uint32_t>(gRelaxed ? 2 : 1);
    w.val<uint32_t>(kHeader);
    w.val<uint32_t>((uint32_t)gM.nLevels);
    w.val<uint32_t>(gRelaxed ? 1 : 0);
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
    if (!igl::writeOBJ(subdiv_long_path(path), gM.V, gM.F))
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
    if (!igl::writeOBJ(subdiv_long_path(path), P, gM.F))
        fprintf(stderr, "[subdiv_tracker] writeOBJ failed: %s\n", path.c_str());
    else
        fprintf(stderr, "[subdiv_tracker] deformed subdivided mesh -> %s\n", path.c_str());
}

void subdiv_tracker_export_seed_obj(const std::string & path, int64_t maxVerts)
{
    if (!gEnabled || !gRelaxed) return;
    if ((int64_t)gVseed.rows() > maxVerts) {
        fprintf(stderr, "[subdiv_tracker] skipping %s: %lld vertices > %lld\n",
                path.c_str(), (long long)gVseed.rows(), (long long)maxVerts);
        return;
    }
    if (!igl::writeOBJ(subdiv_long_path(path), gVseed, gM.F))
        fprintf(stderr, "[subdiv_tracker] writeOBJ failed: %s\n", path.c_str());
    else
        fprintf(stderr, "[subdiv_tracker] seed (unrelaxed) subdivided mesh -> %s\n", path.c_str());
}

void subdiv_tracker_export_graph(const std::string & path)
{
    if (!gEnabled || !gRelaxed) return;
    save_relax_graph(path, gVseed, gGraph, gPal, gSet);
}

bool subdiv_tracker_relaxed() { return gRelaxed; }
const MatrixXd & subdiv_tracker_seed_positions() { return gVseed; }
const RelaxGraph & subdiv_tracker_graph() { return gGraph; }

std::string subdiv_tracker_relax_tag()
{
    return gRelaxed ? "relaxed_" + gRelaxMethod + "_" : std::string();
}

namespace {

// Unit icosphere, one subdivision (42 vertices, 80 faces).
void icosphere(std::vector<Vector3d> & V, std::vector<std::array<int, 3>> & F)
{
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    V = { {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
          {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1} };
    F = { {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
          {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8},
          {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1} };
    std::map<std::pair<int, int>, int> mid;
    std::vector<std::array<int, 3>> F2;
    auto midpoint = [&](int a, int b) {
        const auto key = std::make_pair(std::min(a, b), std::max(a, b));
        auto it = mid.find(key);
        if (it != mid.end()) return it->second;
        V.push_back(0.5 * (V[a] + V[b]));
        return mid[key] = (int)V.size() - 1;
    };
    for (const auto & f : F) {
        const int a = midpoint(f[0], f[1]), b = midpoint(f[1], f[2]), c = midpoint(f[2], f[0]);
        F2.push_back({ f[0], a, c }); F2.push_back({ f[1], b, a });
        F2.push_back({ f[2], c, b }); F2.push_back({ a, b, c });
    }
    F = F2;
    for (auto & v : V) v.normalize();
}

// Distinct, stable color per struct id (golden-ratio hue walk).
std::array<uint8_t, 3> struct_color(int id)
{
    const double h = std::fmod(0.13 + 0.61803398875 * (double)(id < 0 ? 0 : id), 1.0) * 6.0;
    const double s = 0.85, v = 0.95, c = v * s, x = c * (1 - std::abs(std::fmod(h, 2.0) - 1)), m = v - c;
    double r = 0, g = 0, b = 0;
    switch ((int)h) {
    case 0: r = c; g = x; break; case 1: r = x; g = c; break; case 2: g = c; b = x; break;
    case 3: g = x; b = c; break; case 4: r = x; b = c; break; default: r = c; b = x; break;
    }
    return { (uint8_t)std::lround(255 * (r + m)), (uint8_t)std::lround(255 * (g + m)), (uint8_t)std::lround(255 * (b + m)) };
}

} // namespace

void subdiv_tracker_export_anchor_ply(const std::string & path, int64_t maxVerts)
{
    if (!gEnabled || !gRelaxed || gAnchors.empty()) return;
    if ((int64_t)gM.V.rows() > maxVerts) {
        fprintf(stderr, "[subdiv_tracker] skipping %s: %lld vertices > %lld\n",
                path.c_str(), (long long)gM.V.rows(), (long long)maxVerts);
        return;
    }
    std::vector<Vector3d> SV;
    std::vector<std::array<int, 3>> SF;
    icosphere(SV, SF);
    const double diag = (gVO.leftCols(3).colwise().maxCoeff() - gVO.leftCols(3).colwise().minCoeff()).norm();
    const double r = 0.004 * diag;

    const int64_t nMeshV = gM.V.rows(), nMeshF = gM.F.rows();
    const int64_t nV = nMeshV + (int64_t)gAnchors.size() * (int64_t)SV.size();
    const int64_t nF = nMeshF + (int64_t)gAnchors.size() * (int64_t)SF.size();
    std::ofstream f(subdiv_long_path(path), std::ios::binary);
    if (!f) fail("cannot write " + path);
    f << "ply\nformat binary_little_endian 1.0\n"
      << "comment subdivided mesh after relaxation (" << gRelaxMethod << ", grey) + "
      << gAnchors.size() << " curve anchors as spheres, colored by seam/boundary struct id\n"
      << "element vertex " << nV << "\nproperty float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
      << "element face " << nF << "\nproperty list uchar int vertex_indices\nend_header\n";
    auto vert = [&](const Vector3d & p, const std::array<uint8_t, 3> & c) {
        const float xyz[3] = { (float)p(0), (float)p(1), (float)p(2) };
        f.write((const char *)xyz, sizeof(xyz));
        f.write((const char *)c.data(), 3);
    };
    auto face = [&](int32_t a, int32_t b, int32_t c) {
        const uint8_t n = 3;
        const int32_t idx[3] = { a, b, c };
        f.write((const char *)&n, 1);
        f.write((const char *)idx, sizeof(idx));
    };
    const std::array<uint8_t, 3> grey = { 185, 185, 185 };
    for (int64_t i = 0; i < nMeshV; ++i) vert(gM.V.row(i).transpose(), grey);
    for (size_t a = 0; a < gAnchors.size(); ++a) {
        const Vector3d c = gM.V.row(gAnchors[a]).transpose();
        const auto col = struct_color(gAnchorKey[a]);
        for (const auto & v : SV) vert(c + r * v, col);
    }
    for (int64_t k = 0; k < nMeshF; ++k) face(gM.F(k, 0), gM.F(k, 1), gM.F(k, 2));
    for (size_t a = 0; a < gAnchors.size(); ++a) {
        const int32_t base = (int32_t)(nMeshV + (int64_t)a * (int64_t)SV.size());
        for (const auto & t : SF) face(base + t[0], base + t[1], base + t[2]);
    }
    f.close();
    if (!f) fail("write failed for " + path);
    std::set<int32_t> keys(gAnchorKey.begin(), gAnchorKey.end());
    fprintf(stderr, "[subdiv_tracker] relaxed mesh + %zu anchor spheres (%zu seam/boundary structs) -> %s\n",
            gAnchors.size(), keys.size(), path.c_str());
}
