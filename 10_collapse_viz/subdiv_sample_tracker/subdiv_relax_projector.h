#pragma once
// Projection of a point onto its own MAT structure (sheet faces / seam and
// boundary edges), plus the palette helpers it needs. Extracted from
// subdiv_relax_solve_project.cpp so the coarse-subdivision relaxation
// (coarse_subdiv_relax.cpp) can use the same projector. Header-only: every
// function is inline, so each user includes it without duplicate definitions.
#include "subdiv_relax.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace subdiv_proj {

using Eigen::AlignedBox3d;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Vector3d;

inline void proj_fail(const std::string & msg) { throw std::runtime_error("[subdiv_relax] " + msg); }

// ---------------------------------------------------------------- palette helpers

// Per palette set: its sheet ids and its seam/boundary ids, sorted.
struct SetIds {
    std::vector<std::vector<int>> sheet, curve;
};

inline SetIds split_palette(const StructPalette & pal, const MatStruct * ms)
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

inline bool intersects(const std::vector<int> & a, const std::vector<int> & b)
{
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) return true;
        if (a[i] < b[j]) ++i; else ++j;
    }
    return false;
}

inline uint8_t role_of(uint8_t mask)
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
inline Vector3d tri_bary(const Vector3d & p, const Vector3d & a, const Vector3d & b, const Vector3d & c)
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
                if (sp + 2 > 128) proj_fail("BVH too deep");
                // Visit the nearer child first (pushed last).
                const double dl = nodes[nd.left].box.squaredExteriorDistance(p);
                const double dr = nodes[nd.right].box.squaredExteriorDistance(p);
                if (dl <= dr) { stack[sp++] = nd.right; stack[sp++] = nd.left; }
                else          { stack[sp++] = nd.left;  stack[sp++] = nd.right; }
            }
        }
        return found;
    }

    // Structural self-check; returns the number of violations (expected 0):
    // order is a permutation, every child box lies inside its parent's box, the
    // children split the parent's range, and every leaf box contains the boxes
    // of its primitives.
    int64_t structure_errors(const MatrixXd & V) const
    {
        const int n = (int)gid.size();
        int64_t err = 0;
        std::vector<uint8_t> seen(n, 0);
        for (int k : order) { if (k < 0 || k >= n || seen[k]) ++err; else seen[k] = 1; }
        if ((int)order.size() != n) ++err;
        if (n == 0) return err;
        if (nodes.empty() || nodes[0].begin != 0 || nodes[0].end != n) ++err;
        std::vector<uint8_t> covered(n, 0);
        for (const Node & nd : nodes) {
            if (nd.left < 0) {
                for (int k = nd.begin; k < nd.end; ++k) {
                    const int i = order[k];
                    ++covered[i];
                    AlignedBox3d b;
                    for (int c = 0; c < 3; ++c) if (corner[i][c] >= 0) b.extend(P3(V, corner[i][c]));
                    if (!nd.box.contains(b)) ++err;
                }
            } else {
                const Node & l = nodes[nd.left], & r = nodes[nd.right];
                if (!nd.box.contains(l.box) || !nd.box.contains(r.box)) ++err;
                if (l.begin != nd.begin || l.end != r.begin || r.end != nd.end) ++err;
            }
        }
        for (uint8_t c : covered) if (c != 1) ++err;
        return err;
    }

    // Brute-force closest primitive (same arithmetic and tie rule as query), the
    // reference query() must match exactly.
    void brute_force(const MatrixXd & V, const Vector3d & p, double & bestD, int & bestGid) const
    {
        for (int i = 0; i < (int)gid.size(); ++i) {
            const auto & c = corner[i];
            Vector3d q;
            if (c[2] < 0) {
                const Vector3d a = P3(V, c[0]), bb = P3(V, c[1]);
                const double t = seg_t(p, a, bb);
                q = (1 - t) * a + t * bb;
            } else {
                const Vector3d a = P3(V, c[0]), bb = P3(V, c[1]), cc = P3(V, c[2]);
                const Vector3d b = tri_bary(p, a, bb, cc);
                q = b(0) * a + b(1) * bb + b(2) * cc;
            }
            const double d = (p - q).squaredNorm();
            if (d < bestD || (d == bestD && gid[i] < bestGid)) { bestD = d; bestGid = gid[i]; }
        }
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
                if (e < 0) proj_fail("face edge missing from origEdges");
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
        if (bestGid == std::numeric_limits<int>::max()) proj_fail("projection found no target");

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

} // namespace subdiv_proj
