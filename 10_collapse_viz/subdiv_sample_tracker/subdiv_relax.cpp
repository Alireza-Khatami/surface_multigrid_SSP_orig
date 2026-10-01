#include "subdiv_relax.h"

#include <igl/parallel_for.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>

#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

using namespace Eigen;

namespace {

void fail(const std::string & msg) { throw std::runtime_error("[subdiv_relax] " + msg); }

double now_s()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- palette helpers

// Per palette set: its sheet ids and its seam/boundary ids, sorted.
struct SetIds {
    std::vector<std::vector<int>> sheet, curve;
};

SetIds split_palette(const StructPalette & pal, const MatStruct * ms)
{
    SetIds S;
    S.sheet.resize(pal.size());
    S.curve.resize(pal.size());
    if (!ms) return S;
    for (int k = 0; k < pal.size(); ++k)
        for (int32_t a = pal.offsets[k]; a < pal.offsets[k + 1]; ++a) {
            const int id = pal.ids[a];
            const int t = ms->structType.at(id);
            if (t == 0) S.sheet[k].push_back(id);
            else if (t == 1 || t == 2) S.curve[k].push_back(id);
        }
    return S;  // palette ids are sorted within a set, so these are too
}

bool intersects(const std::vector<int> & a, const std::vector<int> & b)
{
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) return true;
        if (a[i] < b[j]) ++i; else ++j;
    }
    return false;
}

uint8_t role_of(uint8_t mask)
{
    if (mask & STRUCT_MASK_JUNCTION) return RELAX_JUNCTION;
    if (mask & (STRUCT_MASK_SEAM | STRUCT_MASK_BOUNDARY)) return RELAX_CURVE;
    return RELAX_SHEET;
}

// ---------------------------------------------------------------- closest points

inline Vector3d P3(const MatrixXd & V, int i) { return V.row(i).leftCols<3>().transpose(); }

// Closest point on segment ab: parameter t in [0,1] (point = (1-t) a + t b).
inline double seg_t(const Vector3d & p, const Vector3d & a, const Vector3d & b)
{
    const Vector3d ab = b - a;
    const double l2 = ab.squaredNorm();
    if (!(l2 > 0.0)) return 0.0;
    return std::min(1.0, std::max(0.0, (p - a).dot(ab) / l2));
}

// Closest point on triangle abc as barycentric (Ericson, RTCD 5.1.5), robust
// to degenerate triangles (then the best of the three edges).
Vector3d tri_bary(const Vector3d & p, const Vector3d & a, const Vector3d & b, const Vector3d & c)
{
    const Vector3d ab = b - a, ac = c - a;
    const double lmax = std::max({ ab.squaredNorm(), ac.squaredNorm(), (c - b).squaredNorm() });
    if (!(ab.cross(ac).squaredNorm() > 1e-24 * lmax * lmax)) {
        Vector3d best(1, 0, 0);
        double bd = std::numeric_limits<double>::infinity();
        const Vector3d * P[3] = { &a, &b, &c };
        for (int e = 0; e < 3; ++e) {
            const int i = e, j = (e + 1) % 3;
            const double t = seg_t(p, *P[i], *P[j]);
            const double d = (p - ((1 - t) * *P[i] + t * *P[j])).squaredNorm();
            if (d < bd) { bd = d; best.setZero(); best(i) = 1 - t; best(j) = t; }
        }
        return best;
    }
    const Vector3d ap = p - a;
    const double d1 = ab.dot(ap), d2 = ac.dot(ap);
    if (d1 <= 0 && d2 <= 0) return Vector3d(1, 0, 0);
    const Vector3d bp = p - b;
    const double d3 = ab.dot(bp), d4 = ac.dot(bp);
    if (d3 >= 0 && d4 <= d3) return Vector3d(0, 1, 0);
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) { const double v = d1 / (d1 - d3); return Vector3d(1 - v, v, 0); }
    const Vector3d cp = p - c;
    const double d5 = ab.dot(cp), d6 = ac.dot(cp);
    if (d6 >= 0 && d5 <= d6) return Vector3d(0, 0, 1);
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) { const double w = d2 / (d2 - d6); return Vector3d(1 - w, 0, w); }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return Vector3d(0, 1 - w, w);
    }
    const double denom = 1.0 / (va + vb + vc);
    const double v = vb * denom, w = vc * denom;
    return Vector3d(1 - v - w, v, w);
}

// Position of (face, bary) exactly as build_subdiv_mesh computes it.
inline Vector3d interp(const MatrixXd & VO, const MatrixXi & FO, int f, const Vector3d & b)
{
    return (b(0) * VO.row(FO(f, 0)).leftCols<3>()
          + b(1) * VO.row(FO(f, 1)).leftCols<3>()
          + b(2) * VO.row(FO(f, 2)).leftCols<3>()).transpose();
}

// ---------------------------------------------------------------- BVH

// Bounding-volume hierarchy over triangles or segments (c2 < 0). Queries are
// exact nearest (squared distance, ties broken by the lower global id), so the
// result does not depend on traversal order.
struct PrimBVH {
    std::vector<std::array<int32_t, 3>> corner;
    std::vector<int32_t> gid;           // FO row (triangles) / origEdges row (segments)
    struct Node { AlignedBox3d box; int32_t left = -1, right = -1, begin = 0, end = 0; };
    std::vector<Node> nodes;
    std::vector<int32_t> order;

    void build(const MatrixXd & V)
    {
        const int n = (int)gid.size();
        order.resize(n);
        for (int i = 0; i < n; ++i) order[i] = i;
        std::vector<AlignedBox3d> pb(n);
        std::vector<Vector3d> cen(n);
        for (int i = 0; i < n; ++i) {
            AlignedBox3d b;
            for (int c = 0; c < 3; ++c) if (corner[i][c] >= 0) b.extend(P3(V, corner[i][c]));
            pb[i] = b;
            cen[i] = b.center();
        }
        nodes.clear();
        nodes.reserve(2 * (size_t)n / 3 + 2);
        if (n == 0) return;
        build_rec(0, n, pb, cen);
    }

    int build_rec(int begin, int end, const std::vector<AlignedBox3d> & pb, const std::vector<Vector3d> & cen)
    {
        const int id = (int)nodes.size();
        nodes.emplace_back();
        AlignedBox3d box, cbox;
        for (int i = begin; i < end; ++i) { box.extend(pb[order[i]]); cbox.extend(cen[order[i]]); }
        nodes[id].box = box;
        nodes[id].begin = begin;
        nodes[id].end = end;
        if (end - begin <= 4) return id;
        int axis;
        cbox.sizes().maxCoeff(&axis);
        const int mid = (begin + end) / 2;
        std::nth_element(order.begin() + begin, order.begin() + mid, order.begin() + end,
            [&](int a, int b) { return cen[a](axis) < cen[b](axis) || (cen[a](axis) == cen[b](axis) && a < b); });
        const int l = build_rec(begin, mid, pb, cen);
        const int r = build_rec(mid, end, pb, cen);
        nodes[id].left = l;
        nodes[id].right = r;
        return id;
    }

    // Closest primitive to p. On entry (bestD, bestGid) is the incumbent (use
    // +inf / INT_MAX for none); updated only by a strictly better candidate.
    // Returns the local primitive index or -1 if the incumbent stands.
    int query(const MatrixXd & V, const Vector3d & p, double & bestD, int & bestGid, Vector3d & bestBary) const
    {
        if (nodes.empty()) return -1;
        int found = -1;
        int stack[128];
        int sp = 0;
        stack[sp++] = 0;
        while (sp) {
            const Node & nd = nodes[stack[--sp]];
            if (nd.box.squaredExteriorDistance(p) > bestD) continue;
            if (nd.left < 0) {
                for (int k = nd.begin; k < nd.end; ++k) {
                    const int i = order[k];
                    Vector3d b;
                    Vector3d q;
                    const auto & c = corner[i];
                    if (c[2] < 0) {
                        const Vector3d a = P3(V, c[0]), bb = P3(V, c[1]);
                        const double t = seg_t(p, a, bb);
                        b = Vector3d(1 - t, t, 0);
                        q = (1 - t) * a + t * bb;
                    } else {
                        const Vector3d a = P3(V, c[0]), bb = P3(V, c[1]), cc = P3(V, c[2]);
                        b = tri_bary(p, a, bb, cc);
                        q = b(0) * a + b(1) * bb + b(2) * cc;
                    }
                    const double d = (p - q).squaredNorm();
                    if (d < bestD || (d == bestD && gid[i] < bestGid)) {
                        bestD = d; bestGid = gid[i]; bestBary = b; found = i;
                    }
                }
            } else {
                if (sp + 2 > 128) fail("BVH too deep");
                // Visit the nearer child first (pushed last).
                const double dl = nodes[nd.left].box.squaredExteriorDistance(p);
                const double dr = nodes[nd.right].box.squaredExteriorDistance(p);
                if (dl <= dr) { stack[sp++] = nd.right; stack[sp++] = nd.left; }
                else          { stack[sp++] = nd.left;  stack[sp++] = nd.right; }
            }
        }
        return found;
    }
};

// ---------------------------------------------------------------- projector

struct ProjResult {
    int face = -1;         // FO row
    Vector3d bary;         // in FO.row(face) corner order
    int edge = -1;         // origEdges row (curve vertices)
    Vector3d pos;
};

struct Projector {
    const MatrixXd & VO;
    const MatrixXi & FO;
    const MatrixXi & origEdges;
    std::vector<int32_t> edgeFace;              // lowest-index face on each orig edge
    std::vector<PrimBVH> trees;
    std::unordered_map<int, int> sheetTree, curveTree;  // struct id -> tree
    int globalTree = -1;
    std::vector<std::vector<int>> targets;      // per palette set: trees to project onto
    std::vector<uint8_t> isCurveSet;            // per palette set: targets are curve trees
    const MatStruct * ms;
    const SetIds & S;
    std::vector<std::vector<int>> faceSheets;   // FO row -> sheet ids (sorted), for membership tests
    std::vector<std::vector<int>> edgeCurves;   // orig edge -> curve ids (sorted)
    // Incidence, for choosing the active element at a corner / crease.
    std::vector<std::vector<int32_t>> vertFaces;   // VO row -> faces using it
    std::vector<std::vector<int32_t>> edgeFaces;   // orig edge -> faces on it
    std::vector<std::vector<int32_t>> vertEdges;   // VO row -> orig edges on it

    Projector(const MatrixXd & VO_, const MatrixXi & FO_, const MatrixXi & E_, const MatStruct * ms_,
              const StructPalette & pal, const SetIds & S_, const std::vector<uint8_t> & setRole)
        : VO(VO_), FO(FO_), origEdges(E_), ms(ms_), S(S_)
    {
        const int nF = (int)FO.rows(), nE = (int)origEdges.rows();
        edgeFace.assign(nE, -1);
        edgeFaces.assign(nE, {});
        vertFaces.assign(VO.rows(), {});
        vertEdges.assign(VO.rows(), {});
        for (int f = 0; f < nF; ++f)
            for (int c = 0; c < 3; ++c) {
                const int e = subdiv_find_edge(origEdges, FO(f, c), FO(f, (c + 1) % 3));
                if (e < 0) fail("face edge missing from origEdges");
                if (edgeFace[e] < 0) edgeFace[e] = f;
                edgeFaces[e].push_back(f);
                vertFaces[FO(f, c)].push_back(f);
            }
        for (int e = 0; e < nE; ++e) {
            vertEdges[origEdges(e, 0)].push_back(e);
            vertEdges[origEdges(e, 1)].push_back(e);
        }

        faceSheets.assign(nF, {});
        edgeCurves.assign(nE, {});
        if (ms) {
            for (int f = 0; f < nF; ++f) faceSheets[f] = ms->faceIds[f];
            for (int e = 0; e < nE; ++e) {
                auto it = ms->edgeIds.find(matstruct_edge_key(origEdges(e, 0), origEdges(e, 1)));
                if (it != ms->edgeIds.end()) edgeCurves[e] = it->second;
            }
        }

        std::unordered_map<int, std::vector<int>> sheetFaces, curveEdges;
        for (int f = 0; f < nF; ++f) for (int id : faceSheets[f]) sheetFaces[id].push_back(f);
        for (int e = 0; e < nE; ++e) for (int id : edgeCurves[e]) curveEdges[id].push_back(e);

        auto add_tree = [&](const std::vector<int> & prims, bool seg) {
            PrimBVH t;
            for (int p : prims) {
                if (seg) t.corner.push_back({ origEdges(p, 0), origEdges(p, 1), -1 });
                else     t.corner.push_back({ FO(p, 0), FO(p, 1), FO(p, 2) });
                t.gid.push_back(p);
            }
            t.build(VO);
            trees.push_back(std::move(t));
            return (int)trees.size() - 1;
        };
        std::vector<int> ids;
        for (auto & kv : sheetFaces) ids.push_back(kv.first);
        std::sort(ids.begin(), ids.end());
        for (int id : ids) sheetTree[id] = add_tree(sheetFaces[id], false);
        ids.clear();
        for (auto & kv : curveEdges) ids.push_back(kv.first);
        std::sort(ids.begin(), ids.end());
        for (int id : ids) curveTree[id] = add_tree(curveEdges[id], true);

        targets.assign(pal.size(), {});
        isCurveSet.assign(pal.size(), 0);
        for (int k = 0; k < pal.size(); ++k) {
            if (setRole[k] == RELAX_CURVE) {
                isCurveSet[k] = 1;
                for (int id : S.curve[k]) { auto it = curveTree.find(id); if (it != curveTree.end()) targets[k].push_back(it->second); }
            } else if (setRole[k] == RELAX_SHEET) {
                for (int id : S.sheet[k]) { auto it = sheetTree.find(id); if (it != sheetTree.end()) targets[k].push_back(it->second); }
                if (S.sheet[k].empty()) {
                    if (globalTree < 0) {
                        std::vector<int> all(nF);
                        for (int f = 0; f < nF; ++f) all[f] = f;
                        globalTree = add_tree(all, false);
                    }
                    targets[k].push_back(globalTree);
                }
            }
        }
    }

    // Is face f / edge e part of palette set k's targets?
    bool face_in(int k, int f) const
    {
        if (S.sheet[k].empty()) return true;  // global tree
        return intersects(faceSheets[f], S.sheet[k]);
    }
    bool edge_in(int k, int e) const { return intersects(edgeCurves[e], S.curve[k]); }

    // Project p for a vertex of set k. (curFace, curEdge) is the current
    // location, used as the initial incumbent when it is a valid target.
    ProjResult project(int k, const Vector3d & p, int curFace, int curEdge) const
    {
        ProjResult R;
        double bestD = std::numeric_limits<double>::infinity();
        int bestGid = std::numeric_limits<int>::max();
        Vector3d bestBary(1, 0, 0);
        const bool curve = isCurveSet[k] != 0;

        if (curve) {
            if (curEdge >= 0 && edge_in(k, curEdge)) {
                const Vector3d a = P3(VO, origEdges(curEdge, 0)), b = P3(VO, origEdges(curEdge, 1));
                const double t = seg_t(p, a, b);
                bestD = (p - ((1 - t) * a + t * b)).squaredNorm();
                bestGid = curEdge;
                bestBary = Vector3d(1 - t, t, 0);
            }
        } else if (curFace >= 0 && face_in(k, curFace)) {
            const Vector3d a = P3(VO, FO(curFace, 0)), b = P3(VO, FO(curFace, 1)), c = P3(VO, FO(curFace, 2));
            bestBary = tri_bary(p, a, b, c);
            bestD = (p - (bestBary(0) * a + bestBary(1) * b + bestBary(2) * c)).squaredNorm();
            bestGid = curFace;
        }
        for (int t : targets[k]) trees[t].query(VO, p, bestD, bestGid, bestBary);
        if (bestGid == std::numeric_limits<int>::max()) fail("projection found no target");

        if (curve) {
            const int e = bestGid;
            const int f = edgeFace[e];
            R.edge = e;
            R.face = f;
            R.bary.setZero();
            for (int c = 0; c < 3; ++c) {
                if (FO(f, c) == origEdges(e, 0)) R.bary(c) = bestBary(0);
                if (FO(f, c) == origEdges(e, 1)) R.bary(c) = bestBary(1);
            }
        } else {
            R.face = bestGid;
            R.bary = bestBary;
        }
        // Clean rounding: no negatives, exact sum 1 is not representable in
        // general, so only renormalise when off by more than an ulp-level.
        for (int c = 0; c < 3; ++c) if (R.bary(c) < 0) R.bary(c) = 0;
        const double s = R.bary.sum();
        if (std::abs(s - 1.0) > 1e-15) R.bary /= s;
        R.pos = interp(VO, FO, R.face, R.bary);
        return R;
    }
};

} // namespace

// ---------------------------------------------------------------- graph

RelaxGraph build_relax_graph(const SubdivMesh & M, const StructPalette & pal,
                             const std::vector<int32_t> & setId, const MatStruct * ms)
{
    const double t0 = now_s();
    RelaxGraph G;
    const int64_t Vs = M.V.rows();
    if ((int64_t)setId.size() != Vs) fail("setId size mismatch");

    const SetIds S = split_palette(pal, ms);
    std::vector<uint8_t> setRole(pal.size());
    for (int k = 0; k < pal.size(); ++k) setRole[k] = ms ? role_of(pal.typeMask[k]) : (uint8_t)RELAX_SHEET;

    G.role.resize(Vs);
    for (int64_t i = 0; i < Vs; ++i) {
        G.role[i] = setRole[setId[i]];
        ++G.nRole[G.role[i]];
        if (ms && G.role[i] == RELAX_SHEET && S.sheet[setId[i]].empty()) ++G.sheetWithoutSheetId;
    }

    // Unique undirected edges of M.F.
    std::vector<uint64_t> keys;
    keys.reserve((size_t)M.F.rows() * 3);
    for (Index f = 0; f < M.F.rows(); ++f)
        for (int c = 0; c < 3; ++c) {
            uint32_t a = (uint32_t)M.F(f, c), b = (uint32_t)M.F(f, (c + 1) % 3);
            if (a > b) std::swap(a, b);
            keys.push_back((uint64_t)a << 32 | b);
        }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    G.nUndirected = (int64_t)keys.size();

    // Rule cache over palette pairs.
    std::unordered_map<uint64_t, uint8_t> cache;  // bit0 share sheet, bit1 share curve
    auto share = [&](int ka, int kb) -> uint8_t {
        const uint64_t key = (uint64_t)(uint32_t)std::min(ka, kb) << 32 | (uint32_t)std::max(ka, kb);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
        uint8_t r = 0;
        if (intersects(S.sheet[ka], S.sheet[kb])) r |= 1;
        if (intersects(S.curve[ka], S.curve[kb])) r |= 2;
        if (S.sheet[ka].empty() && S.sheet[kb].empty()) r |= 4;
        cache.emplace(key, r);
        return r;
    };
    // keep(i <- j): does j pull on i?
    auto keep = [&](int64_t i, int64_t j) -> bool {
        const uint8_t ri = G.role[i], rj = G.role[j];
        if (!ms) return true;
        if (ri == RELAX_JUNCTION) return false;
        const uint8_t s = share(setId[i], setId[j]);
        if (ri == RELAX_CURVE) return rj != RELAX_SHEET && (s & 2);
        return (s & 1) || (s & 4);
    };
    auto kind_of = [&](int64_t i, int64_t j) -> int {
        const uint8_t ri = G.role[i], rj = G.role[j];
        if (ri == RELAX_SHEET) return rj == RELAX_SHEET ? 0 : rj == RELAX_CURVE ? 1 : 2;
        return rj == RELAX_CURVE ? 3 : 4;
    };

    std::vector<int64_t> deg(Vs + 1, 0);
    std::vector<uint8_t> dir(keys.size(), 0);  // bit0: a<-b kept, bit1: b<-a kept
    for (size_t e = 0; e < keys.size(); ++e) {
        const int64_t a = (int64_t)(keys[e] >> 32), b = (int64_t)(keys[e] & 0xffffffffu);
        uint8_t d = 0;
        if (keep(a, b)) { d |= 1; ++deg[a]; ++G.kind[kind_of(a, b)]; }
        if (keep(b, a)) { d |= 2; ++deg[b]; ++G.kind[kind_of(b, a)]; }
        if (G.role[a] == RELAX_SHEET && G.role[b] == RELAX_SHEET && d != 3) ++G.droppedSheetSheet;
        dir[e] = d;
    }
    G.rowOffs.assign(Vs + 1, 0);
    for (int64_t i = 0; i < Vs; ++i) G.rowOffs[i + 1] = G.rowOffs[i] + deg[i];
    G.cols.resize((size_t)G.rowOffs[Vs]);
    std::vector<int64_t> fill(G.rowOffs.begin(), G.rowOffs.end() - 1);
    for (size_t e = 0; e < keys.size(); ++e) {
        const int32_t a = (int32_t)(keys[e] >> 32), b = (int32_t)(keys[e] & 0xffffffffu);
        if (dir[e] & 1) G.cols[fill[a]++] = b;
        if (dir[e] & 2) G.cols[fill[b]++] = a;
    }
    for (int64_t i = 0; i < Vs; ++i) {
        std::sort(G.cols.begin() + G.rowOffs[i], G.cols.begin() + G.rowOffs[i + 1]);
        if (G.rowOffs[i + 1] == G.rowOffs[i]) {
            if (G.role[i] == RELAX_SHEET) ++G.isolatedSheet;
            else if (G.role[i] == RELAX_CURVE) ++G.isolatedCurve;
        }
    }

    fprintf(stderr,
        "[subdiv_relax] graph: %lld vertices (sheet %lld, curve %lld, junction %lld), %lld undirected edges, "
        "%lld directed entries (%.2f s)\n"
        "[subdiv_relax]   kept: S<-S %lld, S<-C %lld, S<-J %lld, C<-C %lld, C<-J %lld | "
        "dropped sheet-sheet %lld | sheet without sheet id %lld | isolated sheet %lld, curve %lld\n",
        (long long)Vs, (long long)G.nRole[0], (long long)G.nRole[1], (long long)G.nRole[2],
        (long long)G.nUndirected, (long long)G.cols.size(), now_s() - t0,
        (long long)G.kind[0], (long long)G.kind[1], (long long)G.kind[2], (long long)G.kind[3], (long long)G.kind[4],
        (long long)G.droppedSheetSheet, (long long)G.sheetWithoutSheetId,
        (long long)G.isolatedSheet, (long long)G.isolatedCurve);
    return G;
}

// .slg layout (little endian, arrays 8-byte aligned):
//   header (120 bytes): magic "SUBDIVG\0", u32 version = 1, u32 header_bytes,
//     u64 Vs, u64 nnz, u64 P, u64 n_palette_ids, u64 n_arrays = 8, u64 offsets[8]
//   0 V                 f64  Vs x 3   seed positions (before relaxation)
//   1 role              u8   Vs       0 sheet, 1 curve (seam/boundary), 2 junction
//   2 struct_set_id     i32  Vs
//   3 row_offsets       i64  Vs+1     row i = the vertices that pull on i
//   4 cols              i32  nnz      ascending within a row
//   5 palette_offsets   i32  P+1
//   6 palette_ids       i32  n_palette_ids
//   7 palette_type_mask u8   P
void save_relax_graph(const std::string & path, const MatrixXd & V, const RelaxGraph & G,
                      const StructPalette & pal, const std::vector<int32_t> & setId)
{
    const uint64_t Vs = (uint64_t)V.rows();
    std::vector<double> v(Vs * 3);
    for (uint64_t i = 0; i < Vs; ++i) for (int c = 0; c < 3; ++c) v[3 * i + c] = V((Index)i, c);

    struct Arr { const void * p; size_t bytes; };
    const int kArrays = 8;
    const Arr arrs[kArrays] = {
        { v.data(),               v.size() * 8 },
        { G.role.data(),          G.role.size() },
        { setId.data(),           setId.size() * 4 },
        { G.rowOffs.data(),       G.rowOffs.size() * 8 },
        { G.cols.data(),          G.cols.size() * 4 },
        { pal.offsets.data(),     pal.offsets.size() * 4 },
        { pal.ids.data(),         pal.ids.size() * 4 },
        { pal.typeMask.data(),    pal.typeMask.size() },
    };
    const uint32_t kHeader = 56 + 8 * kArrays;
    uint64_t offsets[kArrays], at = kHeader;
    for (int k = 0; k < kArrays; ++k) { at = (at + 7) / 8 * 8; offsets[k] = at; at += arrs[k].bytes; }

    std::ofstream f(subdiv_long_path(path), std::ios::binary);
    if (!f) fail("cannot write " + path);
    uint64_t pos = 0;
    auto raw = [&](const void * p, size_t n) { f.write((const char *)p, (std::streamsize)n); pos += n; };
    auto u32 = [&](uint32_t x) { raw(&x, 4); };
    auto u64 = [&](uint64_t x) { raw(&x, 8); };
    const char magic[8] = { 'S', 'U', 'B', 'D', 'I', 'V', 'G', '\0' };
    raw(magic, 8);
    u32(1); u32(kHeader);
    u64(Vs); u64((uint64_t)G.cols.size()); u64((uint64_t)pal.size()); u64((uint64_t)pal.ids.size());
    u64((uint64_t)kArrays);
    for (int k = 0; k < kArrays; ++k) u64(offsets[k]);
    if (pos != kHeader) fail("graph header size mismatch");
    static const char z[8] = {};
    for (int k = 0; k < kArrays; ++k) {
        const size_t pad = (size_t)((8 - pos % 8) % 8);
        if (pad) raw(z, pad);
        if (pos != offsets[k]) fail("graph array offset mismatch");
        raw(arrs[k].p, arrs[k].bytes);
    }
    f.close();
    if (!f) fail("write failed for " + path);
    fprintf(stderr, "[subdiv_relax] graph -> %s (%.1f MB)\n", path.c_str(), pos / 1048576.0);
}

// ---------------------------------------------------------------- relaxation

RelaxReport subdiv_relax(SubdivMesh & M, const MatrixXd & VO, const MatrixXi & FO, const MatStruct * ms,
                         const StructPalette & pal, const std::vector<int32_t> & setId,
                         const RelaxGraph & G, const RelaxOptions & opt)
{
    const double tStart = now_s();
    RelaxReport R;
    const int64_t Vs = M.V.rows();
    if ((int64_t)G.role.size() != Vs) fail("graph / mesh size mismatch");
    if (!opt.holdFixed.empty() && (int64_t)opt.holdFixed.size() != Vs) fail("holdFixed / mesh size mismatch");
    const double diag = (VO.leftCols(3).colwise().maxCoeff() - VO.leftCols(3).colwise().minCoeff()).norm();
    const double tolAbs = opt.tol * diag;

    const SetIds S = split_palette(pal, ms);
    std::vector<uint8_t> setRole(pal.size());
    for (int k = 0; k < pal.size(); ++k) setRole[k] = ms ? role_of(pal.typeMask[k]) : (uint8_t)RELAX_SHEET;
    const double tp = now_s();
    const Projector proj(VO, FO, M.origEdges, ms, pal, S, setRole);
    if (opt.verbose)
        fprintf(stderr, "[subdiv_relax] projector: %zu trees (%.2f s)\n", proj.trees.size(), now_s() - tp);

    const MatrixXd Vseed = M.V;
    MatrixXd X = M.V;
    std::vector<int32_t> face = M.fineFace;
    MatrixXd bary = M.fineBary;
    std::vector<int32_t> edge(Vs, -1);
    for (int64_t i = 0; i < Vs; ++i)
        if (G.role[i] == RELAX_CURVE && M.carrierType[i] == SUBDIV_CARRIER_EDGE) edge[i] = M.carrierIndex[i];

    std::vector<uint8_t> isFree(Vs, 0);
    for (int64_t i = 0; i < Vs; ++i) {
        const bool f = G.role[i] != RELAX_JUNCTION && face[i] >= 0
                    && G.rowOffs[i + 1] > G.rowOffs[i] && !proj.targets[setId[i]].empty()
                    && (opt.holdFixed.empty() || !opt.holdFixed[i]);
        isFree[i] = f;
        if (f) ++R.nFree; else ++R.nFixed;
    }

    // A connected group of free vertices of one role that nothing fixed pulls on
    // (a curve loop with no junction, a sheet with no rim) can slide as a whole,
    // so its fixed point is not unique. Pin its lowest-index vertex (an original
    // MAT vertex when the group has one, since those come first).
    {
        std::vector<uint8_t> seen(Vs, 0);
        for (int64_t s0 = 0; s0 < Vs; ++s0) {
            if (!isFree[s0] || seen[s0]) continue;
            const uint8_t cls = G.role[s0];
            std::vector<int64_t> st{ s0 };
            seen[s0] = 1;
            bool anchored = false;
            int64_t lowest = s0;
            while (!st.empty()) {
                const int64_t v = st.back(); st.pop_back();
                lowest = std::min(lowest, v);
                for (int64_t q = G.rowOffs[v]; q < G.rowOffs[v + 1]; ++q) {
                    const int j = G.cols[q];
                    if (!isFree[j] || G.role[j] != cls) { anchored = true; continue; }
                    if (!seen[j]) { seen[j] = 1; st.push_back(j); }
                }
            }
            if (!anchored) {
                isFree[lowest] = 0; --R.nFree; ++R.nFixed; ++R.nPinned;
                if (G.role[lowest] == RELAX_CURVE) ++R.nPinnedCurve;
            }
        }
        if (opt.verbose)
            fprintf(stderr, "[subdiv_relax] pinned %lld vertices of unanchored groups (%lld curve loops)\n",
                    (long long)R.nPinned, (long long)R.nPinnedCurve);
    }

    // Project the free vertices once at their seed: they are on their structure,
    // so this must not move them beyond rounding. It also fills edge[] for curve
    // vertices carried by an original vertex.
    {
        std::vector<double> mv(Vs, 0.0);
        igl::parallel_for(Vs, [&](int64_t i) {
            if (!isFree[i]) return;
            const ProjResult r = proj.project(setId[i], X.row(i).transpose(), face[i], edge[i]);
            mv[i] = (r.pos - X.row(i).transpose()).norm();
            if (mv[i] > 1e-12 * diag) return;  // not on its own structure: fixed below
            face[i] = r.face; bary.row(i) = r.bary.transpose(); edge[i] = r.edge; X.row(i) = r.pos.transpose();
        }, 1000);
        double m = 0.0;
        for (int64_t i = 0; i < Vs; ++i) {
            if (!isFree[i]) continue;
            if (mv[i] > 1e-12 * diag) { isFree[i] = 0; --R.nFree; ++R.nFixed; ++R.seedOffStructure; }
            else m = std::max(m, mv[i]);
        }
        if (opt.verbose)
            fprintf(stderr, "[subdiv_relax] seed self-projection: max move %.3g (x diag), %lld vertices not on "
                    "their own structure (kept fixed)\n", m / diag, (long long)R.seedOffStructure);
    }

    // Tangent frame of vertex i: 3 x dim orthonormal columns (dim 0..3), chosen
    // by active set from the pull r = sum_j (x_j - x_i):
    //   curve vertex inside an edge            -> that edge's direction
    //   curve vertex on a polyline kink         -> the incident curve edge the pull
    //                                              points along; none -> held (dim 0)
    //   sheet vertex inside a face              -> that face's plane
    //   sheet vertex on an edge / fine vertex   -> the incident sheet face the pull
    //                                              enters; else the incident edge it
    //                                              points along; else held (dim 0)
    // Without this a vertex sitting exactly on a kink or crease keeps the frame of
    // whichever element the projection returned and cannot follow its pull.
    const double kZero = 1e-12;
    auto face_frame = [&](int f, Matrix<double, 3, Dynamic> & T, Vector3d * nOut) -> bool {
        const Vector3d a = P3(VO, FO(f, 0)), b = P3(VO, FO(f, 1)), c = P3(VO, FO(f, 2));
        Vector3d n = (b - a).cross(c - a);
        const double l = n.norm();
        const double lmax = std::max({ (b - a).squaredNorm(), (c - a).squaredNorm(), (c - b).squaredNorm() });
        if (!(l * l > 1e-24 * lmax * lmax)) return false;
        n /= l;
        const Vector3d t1 = (std::abs(n(0)) < 0.9 ? Vector3d::UnitX() : Vector3d::UnitY()).cross(n).normalized();
        T.resize(3, 2);
        T.col(0) = t1;
        T.col(1) = n.cross(t1);
        if (nOut) *nOut = n;
        return true;
    };
    auto line_frame = [&](int v0, int v1, Matrix<double, 3, Dynamic> & T) -> bool {
        const Vector3d d = P3(VO, v1) - P3(VO, v0);
        const double l = d.norm();
        if (!(l > 0)) return false;
        T = d / l;
        return true;
    };
    // Best incident edge of fine vertex v (among `edges`) that the pull points
    // along, as a line frame; false if the pull points along none of them.
    auto best_edge = [&](int v, const std::vector<int32_t> & edges, const Vector3d & r,
                         Matrix<double, 3, Dynamic> & T) -> bool {
        double best = 0.0;
        int be = -1;
        for (int e : edges) {
            const int w = M.origEdges(e, 0) == v ? M.origEdges(e, 1) : M.origEdges(e, 0);
            const Vector3d d = P3(VO, w) - P3(VO, v);
            const double l = d.norm();
            if (!(l > 0)) continue;
            const double s = d.dot(r) / l;
            if (s > best || (s == best && be >= 0 && e < be)) { best = s; be = e; }
        }
        return be >= 0 && line_frame(M.origEdges(be, 0), M.origEdges(be, 1), T);
    };
    auto frame = [&](int64_t i, const Vector3d & r, Matrix<double, 3, Dynamic> & T) {
        const int k = setId[i];
        const int f = face[i];
        const Vector3d b = bary.row(i).transpose();
        if (G.role[i] == RELAX_CURVE) {
            const int e = edge[i];
            const int v0 = M.origEdges(e, 0), v1 = M.origEdges(e, 1);
            double w0 = 0.0, w1 = 0.0;
            for (int c = 0; c < 3; ++c) {
                if (FO(f, c) == v0) w0 = b(c);
                if (FO(f, c) == v1) w1 = b(c);
            }
            if (w0 > kZero && w1 > kZero) {
                if (line_frame(v0, v1, T)) return;
                T = Matrix3d::Identity();
                return;
            }
            const int v = w0 >= w1 ? v0 : v1;  // on the polyline vertex v
            std::vector<int32_t> cand;
            for (int e2 : proj.vertEdges[v]) if (proj.edge_in(k, e2)) cand.push_back(e2);
            if (!best_edge(v, cand, r, T)) T.resize(3, 0);
            return;
        }
        int nz = 0, zc[3];
        for (int c = 0; c < 3; ++c) if (!(b(c) > kZero)) zc[nz++] = c;
        if (nz == 0) {
            if (!face_frame(f, T, nullptr)) T = Matrix3d::Identity();
            return;
        }
        if (nz == 1) {
            // On the edge opposite corner zc[0].
            const int p = FO(f, (zc[0] + 1) % 3), q = FO(f, (zc[0] + 2) % 3);
            const int e = subdiv_find_edge(M.origEdges, p, q);
            const Vector3d P = P3(VO, p), dir = (P3(VO, q) - P).normalized();
            double best = 0.0;
            int bf = -1;
            for (int g : proj.edgeFaces[e]) {
                if (!proj.face_in(k, g)) continue;
                Matrix<double, 3, Dynamic> Tg;
                Vector3d n;
                if (!face_frame(g, Tg, &n)) continue;
                int third = -1;
                for (int c = 0; c < 3; ++c) if (FO(g, c) != p && FO(g, c) != q) third = FO(g, c);
                Vector3d w = P3(VO, third) - P;
                w -= w.dot(dir) * dir;              // in g's plane, perpendicular to the edge, into g
                const double wl = w.norm();
                if (!(wl > 0)) continue;
                const double s = (r - r.dot(n) * n).dot(w / wl);
                if (s > best || (s == best && bf >= 0 && g < bf)) { best = s; bf = g; }
            }
            if (bf >= 0 && face_frame(bf, T, nullptr)) return;
            if (line_frame(p, q, T)) return;
            T.resize(3, 0);
            return;
        }
        // On fine vertex v.
        const int v = FO(f, b(0) > kZero ? 0 : b(1) > kZero ? 1 : 2);
        double best = 0.0;
        int bf = -1;
        for (int g : proj.vertFaces[v]) {
            if (!proj.face_in(k, g)) continue;
            Matrix<double, 3, Dynamic> Tg;
            Vector3d n;
            if (!face_frame(g, Tg, &n)) continue;
            int cv = 0;
            while (FO(g, cv) != v) ++cv;
            const Vector3d V0 = P3(VO, v);
            const Vector3d e1 = P3(VO, FO(g, (cv + 1) % 3)) - V0, e2 = P3(VO, FO(g, (cv + 2) % 3)) - V0;
            const Vector3d rg = r - r.dot(n) * n;
            // rg = s e1 + t e2; the pull enters g iff s > 0 and t > 0.
            Matrix2d Gm;
            Gm << e1.dot(e1), e1.dot(e2), e1.dot(e2), e2.dot(e2);
            const Vector2d st = Gm.ldlt().solve(Vector2d(e1.dot(rg), e2.dot(rg)));
            if (st(0) > 0 && st(1) > 0 && rg.norm() > best) { best = rg.norm(); bf = g; }
        }
        if (bf >= 0 && face_frame(bf, T, nullptr)) return;
        std::vector<int32_t> cand;
        for (int e2 : proj.vertEdges[v]) {
            bool inSheet = false;
            for (int g : proj.edgeFaces[e2]) if (proj.face_in(k, g)) { inSheet = true; break; }
            if (inSheet) cand.push_back(e2);
        }
        if (!best_edge(v, cand, r, T)) T.resize(3, 0);
    };

    std::vector<double> move(Vs, 0.0);
    std::vector<int32_t> nFace(Vs), nEdge(Vs);
    MatrixXd nX(Vs, 3), nBary(Vs, 3);

    // Commit projected candidates for the listed vertices; returns max move.
    auto commit = [&](const std::vector<int64_t> & list) {
        double m = 0.0;
        for (int64_t i : list) {
            m = std::max(m, move[i]);
            X.row(i) = nX.row(i); face[i] = nFace[i]; edge[i] = nEdge[i]; bary.row(i) = nBary.row(i);
        }
        return m;
    };

    if (opt.solver == RelaxSolver::Newton) {
        const int64_t maxIter = opt.maxIter > 0 ? opt.maxIter : 100000;
        for (int pass = 0; pass < 2; ++pass) {
            const uint8_t cls = pass == 0 ? RELAX_CURVE : RELAX_SHEET;
            std::vector<int64_t> list;
            for (int64_t i = 0; i < Vs; ++i) if (isFree[i] && G.role[i] == cls) list.push_back(i);
            int64_t & iters = pass == 0 ? R.itersCurve : R.itersSheet;
            double & delta = pass == 0 ? R.deltaCurve : R.deltaSheet;
            bool conv = list.empty();
            if (conv) delta = 0.0;

            std::vector<int64_t> var(Vs, -1), slot(Vs, -1);
            for (size_t a = 0; a < list.size(); ++a) slot[list[a]] = (int64_t)a;
            std::vector<Matrix<double, 3, Dynamic>> T(list.size());
            SimplicialLDLT<SparseMatrix<double>> ldlt;
            bool analyzed = false;
            std::vector<int> lastDims;

            // Energy change of the candidate positions nX (listed vertices) against
            // X, E = 1/2 sum over the kept edges of |x_i - x_j|^2, per edge from
            // differences so that it stays accurate when the change is tiny. An
            // edge between two listed vertices appears in both rows (weight 1/2).
            std::vector<double> dEi(list.size());
            auto energy_change = [&]() {
                igl::parallel_for((int64_t)list.size(), [&](int64_t a) {
                    const int64_t i = list[a];
                    double s = 0.0;
                    for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) {
                        const int j = G.cols[q];
                        const bool both = slot[j] >= 0;
                        const Vector3d d  = (X.row(i) - X.row(j)).transpose();
                        const Vector3d dn = (nX.row(i) - (both ? nX.row(j) : X.row(j))).transpose();
                        s += (both ? 0.25 : 0.5) * (dn - d).dot(dn + d);
                    }
                    dEi[a] = s;
                }, 1000);
                double s = 0.0;
                for (double v : dEi) s += v;
                return s;
            };

            // Plain projected step x <- Pi(x + 0.5 L x) of the listed vertices into
            // nX; returns its max move.
            auto plain_step = [&]() {
                igl::parallel_for((int64_t)list.size(), [&](int64_t a) {
                    const int64_t i = list[a];
                    Vector3d mean = Vector3d::Zero();
                    for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) mean += X.row(G.cols[q]).transpose();
                    mean /= (double)(G.rowOffs[i + 1] - G.rowOffs[i]);
                    const Vector3d x = X.row(i).transpose();
                    const ProjResult pr = proj.project(setId[i], x + 0.5 * (mean - x), face[i], edge[i]);
                    move[i] = (pr.pos - x).norm();
                    nX.row(i) = pr.pos.transpose(); nFace[i] = pr.face; nEdge[i] = pr.edge; nBary.row(i) = pr.bary.transpose();
                }, 1000);
                double m = 0.0;
                for (int64_t i : list) m = std::max(m, move[i]);
                return m;
            };

            // Newton rounds alternate with plain steps. Newton converges fast for the
            // smooth, global part but its frames (the current edge / face) cannot
            // carry a vertex over a curve corner or a sheet crease; the plain step
            // uses the full pull and does. Done when a plain step moves nothing,
            // which is the definition of "stopped changing".
            for (;;) {
            conv = list.empty();
            while (!conv && iters < maxIter) {
                const double ti = now_s();
                // Frames and variable layout.
                int64_t nVar = 0;
                std::vector<int> dims(list.size());
                for (size_t a = 0; a < list.size(); ++a) {
                    const int64_t i = list[a];
                    Vector3d r = Vector3d::Zero();
                    for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) r += (X.row(G.cols[q]) - X.row(i)).transpose();
                    frame(i, r, T[a]);
                    dims[a] = (int)T[a].cols();
                    var[list[a]] = nVar;
                    nVar += dims[a];
                }

                // Tangent-plane Hessian of E (Gauss-Newton with the constraint
                // linearised): deg_i u_i - sum_j T_i^T T_j u_j = T_i^T sum_j (x_j - x_i).
                std::vector<Triplet<double>> trip;
                trip.reserve((size_t)nVar * 7);
                VectorXd rhs = VectorXd::Zero(nVar);
                for (size_t a = 0; a < list.size(); ++a) {
                    const int64_t i = list[a];
                    const int64_t deg = G.rowOffs[i + 1] - G.rowOffs[i];
                    const double diagv = (double)deg * (1.0 + 1e-10);
                    Vector3d r = Vector3d::Zero();
                    for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) {
                        const int j = G.cols[q];
                        r += (X.row(j) - X.row(i)).transpose();
                        if (slot[j] >= 0) {
                            // All entries, zeros included: the pattern must depend only
                            // on dims and the graph, since analyzePattern is reused.
                            const MatrixXd B = -(T[a].transpose() * T[slot[j]]);
                            for (int uu = 0; uu < B.rows(); ++uu)
                                for (int w = 0; w < B.cols(); ++w)
                                    trip.emplace_back(var[i] + uu, var[j] + w, B(uu, w));
                        }
                    }
                    for (int uu = 0; uu < dims[a]; ++uu) trip.emplace_back(var[i] + uu, var[i] + uu, diagv);
                    rhs.segment(var[i], dims[a]) = T[a].transpose() * r;
                }
                SparseMatrix<double> A(nVar, nVar);
                A.setFromTriplets(trip.begin(), trip.end());
                trip.clear(); trip.shrink_to_fit();
                if (iters == 0) {
                    const double asym = (SparseMatrix<double>(A.transpose()) - A).norm();
                    if (asym > 1e-12 * A.norm()) fail("tangent system is not symmetric (" + std::to_string(asym) + ")");
                }
                if (!analyzed || dims != lastDims) { ldlt.analyzePattern(A); analyzed = true; lastDims = dims; }
                ldlt.factorize(A);
                if (ldlt.info() != Success) fail("tangent system factorization failed");
                const VectorXd u = ldlt.solve(rhs);
                if (ldlt.info() != Success || !u.allFinite()) fail("tangent system solve failed");
                const double slope = rhs.dot(u);  // -dE/dalpha at alpha = 0 (> 0: A is SPD)

                // Backtracking line search on E (Armijo). E must decrease on every
                // accepted step, so the iteration cannot cycle (a valley crease would
                // otherwise make a vertex bounce between two faces' tangent planes).
                double alpha = 1.0, m = 0.0, dE = 0.0;
                int halvings = 0;
                bool accepted = false;
                for (; halvings <= 50; ++halvings, alpha *= 0.5) {
                    igl::parallel_for((int64_t)list.size(), [&](int64_t a) {
                        const int64_t i = list[a];
                        const Vector3d y = X.row(i).transpose() + alpha * (T[a] * u.segment(var[i], dims[a]));
                        const ProjResult pr = proj.project(setId[i], y, face[i], edge[i]);
                        move[i] = (pr.pos - X.row(i).transpose()).norm();
                        nX.row(i) = pr.pos.transpose(); nFace[i] = pr.face; nEdge[i] = pr.edge; nBary.row(i) = pr.bary.transpose();
                    }, 1000);
                    m = 0.0;
                    for (int64_t i : list) m = std::max(m, move[i]);
                    if (halvings == 0 && m <= tolAbs) { accepted = true; break; }  // full step moves nothing
                    dE = energy_change();
                    if (dE <= -1e-4 * alpha * slope) { accepted = true; break; }
                }
                ++iters;
                if (!accepted) {
                    // No decrease along the projected Newton direction even for tiny
                    // steps: Newton is done for this round (the plain step decides).
                    delta = 0.0;
                    conv = true;
                    ++R.lineSearchStalls;
                    break;
                }
                commit(list);
                delta = m / diag;
                conv = halvings == 0 && m <= tolAbs;
                R.halvings += halvings;
                if (opt.verbose && (iters <= 10 || iters % 10 == 0 || conv))
                    fprintf(stderr, "[subdiv_relax] %s iter %lld: %lld free, %lld unknowns, max move %.3g (x diag), "
                            "step %.3g, dE %.3g (%.2f s)\n",
                            pass == 0 ? "curve" : "sheet", (long long)iters, (long long)list.size(),
                            (long long)nVar, delta, alpha, dE, now_s() - ti);
            }
            if (list.empty()) break;
            const double mp = plain_step();
            delta = mp / diag;
            if (mp <= tolAbs) { conv = true; break; }
            if (iters >= maxIter) { conv = false; break; }
            commit(list);
            ++iters;
            ++R.plainSteps;
            if (opt.verbose && R.plainSteps % 100 == 1)
                fprintf(stderr, "[subdiv_relax] %s iter %lld: plain step, max move %.3g (x diag)\n",
                        pass == 0 ? "curve" : "sheet", (long long)iters, delta);
            }
            if (!conv) {
                fprintf(stderr, "[subdiv_relax] WARNING: %s pass did not converge in %lld iterations (last max move %.3g x diag)\n",
                        pass == 0 ? "curve" : "sheet", (long long)maxIter, delta);
                std::vector<int64_t> top = list;
                std::sort(top.begin(), top.end(), [&](int64_t a, int64_t b) { return move[a] > move[b]; });
                for (size_t t = 0; t < std::min<size_t>(8, top.size()); ++t) {
                    const int64_t i = top[t];
                    fprintf(stderr, "[subdiv_relax]   v %lld move %.3g face %d bary (%.3g %.3g %.3g) edge %d deg %lld\n",
                            (long long)i, move[i] / diag, face[i], bary(i, 0), bary(i, 1), bary(i, 2), edge[i],
                            (long long)(G.rowOffs[i + 1] - G.rowOffs[i]));
                }
            }
            R.converged = (pass == 0) ? conv : (R.converged && conv);
        }
    } else {
        const int64_t maxIter = opt.maxIter > 0 ? opt.maxIter : 1000000;
        std::vector<int64_t> list;
        for (int64_t i = 0; i < Vs; ++i) if (isFree[i]) list.push_back(i);
        bool conv = list.empty();
        while (!conv && R.itersSheet < maxIter) {
            igl::parallel_for((int64_t)list.size(), [&](int64_t a) {
                const int64_t i = list[a];
                Vector3d mean = Vector3d::Zero();
                for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) mean += X.row(G.cols[q]).transpose();
                mean /= (double)(G.rowOffs[i + 1] - G.rowOffs[i]);
                const Vector3d x = X.row(i).transpose();
                const ProjResult pr = proj.project(setId[i], x + opt.lambda * (mean - x), face[i], edge[i]);
                move[i] = (pr.pos - x).norm();
                nX.row(i) = pr.pos.transpose(); nFace[i] = pr.face; nEdge[i] = pr.edge; nBary.row(i) = pr.bary.transpose();
            }, 1000);
            const double m = commit(list);
            ++R.itersSheet;
            R.deltaSheet = m / diag;
            conv = m <= tolAbs;
            if (opt.verbose && (R.itersSheet % 10000 == 0 || conv))
                fprintf(stderr, "[subdiv_relax] jacobi iter %lld: max move %.3g (x diag)\n",
                        (long long)R.itersSheet, R.deltaSheet);
        }
        if (!conv) fprintf(stderr, "[subdiv_relax] WARNING: jacobi did not converge in %lld iterations\n", (long long)maxIter);
        R.converged = conv;
    }

    // ---- write back ----
    M.V = X;
    M.fineFace = face;
    M.fineBary = bary;

    // ---- checks ----
    double sumMove = 0.0;
    for (int64_t i = 0; i < Vs; ++i) {
        const double mv = (X.row(i) - Vseed.row(i)).norm();
        R.maxMove = std::max(R.maxMove, mv);
        sumMove += mv;
        if (!isFree[i]) continue;
        const Vector3d b = bary.row(i).transpose();
        if (b.minCoeff() < 0 || std::abs(b.sum() - 1.0) > 1e-12) ++R.badBary;
        if (interp(VO, FO, face[i], b) != X.row(i).transpose()) ++R.posMismatch;
        const int k = setId[i];
        if (G.role[i] == RELAX_CURVE) {
            const int e = edge[i];
            bool ok = e >= 0 && proj.edge_in(k, e);
            for (int c = 0; c < 3 && ok; ++c)
                if (b(c) != 0.0 && FO(face[i], c) != M.origEdges(e, 0) && FO(face[i], c) != M.origEdges(e, 1)) ok = false;
            if (!ok) ++R.offStructure;
        } else if (!proj.face_in(k, face[i])) {
            ++R.offStructure;
        }
    }
    R.maxMove /= diag;
    R.meanMove = Vs ? sumMove / Vs / diag : 0.0;
    for (int64_t i = 0; i < Vs; ++i)
        if (!isFree[i] && (X.row(i) != Vseed.row(i))) ++R.fixedMoved;

    // The plan's definition of "stopped changing": one step x <- Pi(x + lambda L x)
    // (lambda = 0.5) from the result must not move any vertex.
    {
        std::vector<double> mv(Vs, 0.0);
        igl::parallel_for(Vs, [&](int64_t i) {
            if (!isFree[i]) return;
            Vector3d mean = Vector3d::Zero();
            for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) mean += X.row(G.cols[q]).transpose();
            mean /= (double)(G.rowOffs[i + 1] - G.rowOffs[i]);
            const Vector3d x = X.row(i).transpose();
            mv[i] = (proj.project(setId[i], x + 0.5 * (mean - x), face[i], edge[i]).pos - x).norm();
        }, 1000);
        for (int64_t i = 0; i < Vs; ++i) {
            if (mv[i] > R.jacobiStepMove) { R.jacobiStepMove = mv[i]; R.jacobiStepVertex = i; }
        }
        R.jacobiStepMove /= diag;
    }

    // Fixed-point residual.
    for (int64_t i = 0; i < Vs; ++i) {
        if (!isFree[i]) continue;
        const Vector3d b = bary.row(i).transpose();
        bool interior;
        if (G.role[i] == RELAX_CURVE) {
            int nz = 0; double mn = 1.0;
            for (int c = 0; c < 3; ++c) if (b(c) != 0.0) { ++nz; mn = std::min(mn, b(c)); }
            interior = nz == 2 && mn > 1e-9;
        } else {
            interior = b.minCoeff() > 1e-9;
        }
        if (!interior) continue;
        Vector3d mean = Vector3d::Zero();
        double len = 0.0;
        for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) {
            mean += X.row(G.cols[q]).transpose();
            len += (X.row(G.cols[q]) - X.row(i)).norm();
        }
        const double deg = (double)(G.rowOffs[i + 1] - G.rowOffs[i]);
        Matrix<double, 3, Dynamic> T;
        frame(i, mean - deg * X.row(i).transpose(), T);
        mean /= deg; len /= deg;
        if (!(len > 0)) continue;
        const double res = (T.transpose() * (mean - X.row(i).transpose())).norm() / len;
        if (G.role[i] == RELAX_CURVE) { R.residualCurve = std::max(R.residualCurve, res); ++R.nResidualCurve; }
        else                          { R.residualSheet = std::max(R.residualSheet, res); ++R.nResidualSheet; }
    }

    fprintf(stderr,
        "[subdiv_relax] %s: %s | curve iters %lld (last %.3g), sheet iters %lld (last %.3g) | "
        "free %lld, fixed %lld | move max %.3g mean %.3g (x diag) | halvings %lld, stalls %lld, plain steps %lld, pinned %lld (%.2f s)\n"
        "[subdiv_relax]   one step x <- Pi(x + 0.5 L x) from the result: max move %.3g (x diag) at v %lld\n"
        "[subdiv_relax]   fixed-point residual |T^T L x| / edge: sheet %.3g (%lld interior), curve %.3g (%lld interior)\n"
        "[subdiv_relax]   checks: seed off structure %lld, fixed moved %lld, pos != interp %lld, bad bary %lld, "
        "off own structure %lld\n",
        opt.solver == RelaxSolver::Newton ? "newton" : "jacobi", R.converged ? "converged" : "NOT CONVERGED",
        (long long)R.itersCurve, R.deltaCurve, (long long)R.itersSheet, R.deltaSheet,
        (long long)R.nFree, (long long)R.nFixed, R.maxMove, R.meanMove,
        (long long)R.halvings, (long long)R.lineSearchStalls, (long long)R.plainSteps, (long long)R.nPinned, now_s() - tStart,
        R.jacobiStepMove, (long long)R.jacobiStepVertex,
        R.residualSheet, (long long)R.nResidualSheet, R.residualCurve, (long long)R.nResidualCurve,
        (long long)R.seedOffStructure, (long long)R.fixedMoved, (long long)R.posMismatch, (long long)R.badBary, (long long)R.offStructure);
    return R;
}

std::string subdiv_long_path(const std::string & path)
{
#ifdef _WIN32
    namespace fs = std::filesystem;
    const std::string abs = fs::absolute(fs::path(path)).lexically_normal().make_preferred().string();
    if (abs.rfind("\\\\", 0) == 0) return abs;  // already long-path form, or a UNC path
    return "\\\\?\\" + abs;
#else
    return path;
#endif
}

// ---------------------------------------------------------------- quality

MeshQuality subdiv_mesh_quality(const MatrixXd & V, const MatrixXi & F, const MatrixXd * Vref)
{
    MeshQuality Q;
    const double diag = (V.leftCols(3).colwise().maxCoeff() - V.leftCols(3).colwise().minCoeff()).norm();
    double s = 0.0, s2 = 0.0;
    int64_t n = 0;
    std::vector<float> minAng((size_t)F.rows());
    for (Index f = 0; f < F.rows(); ++f) {
        const Vector3d p[3] = { P3(V, F(f, 0)), P3(V, F(f, 1)), P3(V, F(f, 2)) };
        double ang = 180.0;
        for (int c = 0; c < 3; ++c) {
            const Vector3d u = p[(c + 1) % 3] - p[c], w = p[(c + 2) % 3] - p[c];
            // each undirected edge counted from the face with the edge c->c+1
            const double l = u.norm();
            s += l; s2 += l * l; ++n;
            const double nu = u.norm(), nw = w.norm();
            const double a = (nu > 0 && nw > 0) ? std::acos(std::max(-1.0, std::min(1.0, u.dot(w) / (nu * nw)))) * 180.0 / M_PI : 0.0;
            ang = std::min(ang, a);
        }
        minAng[(size_t)f] = (float)ang;
        const Vector3d nrm = (p[1] - p[0]).cross(p[2] - p[0]);
        if (0.5 * nrm.norm() <= 1e-14 * diag * diag) ++Q.degenerate;
        if (Vref) {
            const Vector3d r0 = P3(*Vref, F(f, 0)), r1 = P3(*Vref, F(f, 1)), r2 = P3(*Vref, F(f, 2));
            if (nrm.dot((r1 - r0).cross(r2 - r0)) < 0) ++Q.flippedVsRef;
        }
    }
    const double mean = n ? s / n : 0.0;
    Q.edgeCV = mean > 0 ? std::sqrt(std::max(0.0, s2 / n - mean * mean)) / mean : 0.0;
    if (!minAng.empty()) {
        std::sort(minAng.begin(), minAng.end());
        auto pct = [&](double q) { return (double)minAng[(size_t)std::min<double>(minAng.size() - 1, q * minAng.size())]; };
        Q.minAngle = minAng.front(); Q.p1 = pct(0.01); Q.p5 = pct(0.05); Q.median = pct(0.5);
    }
    return Q;
}
