// EXPERIMENT (copied from subdiv_relax.cpp): relax by one 3D linear solve of
// L x = 0, then project. Selected with --subdiv_relax_method solve_project.
//
// Same graph, projection, pinning and checks as subdiv_relax.cpp; only the
// solver differs. Instead of iterating until one plain step moves nothing, it
// solves the 3D Laplacian once per pass (curves, then sheets with the curves
// fixed) and projects each vertex onto its own structure. The plain-step check
// at the end reports how far that result is from a resting state.
#include "subdiv_relax.h"
#include "subdiv_relax_projector.h"

#include <igl/parallel_for.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>

#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <functional>
#include <queue>
#include <set>
#include <unordered_map>

using namespace Eigen;

namespace {

using namespace subdiv_proj;

void fail(const std::string & msg) { throw std::runtime_error("[subdiv_relax] " + msg); }

double now_s()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

} // namespace

// ---------------------------------------------------------------- relaxation

RelaxReport subdiv_relax_solve_project(SubdivMesh & M, const MatrixXd & VO, const MatrixXi & FO, const MatStruct * ms,
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

    // Curve anchors: in every connected group of curve vertices (joined by kept
    // curve-curve edges), fix opt.curveAnchors vertices at their seed positions,
    // spread evenly by farthest-point sampling on seed arc length (graph
    // distance along the curve). Without them a closed loop's 3D solve of
    // L x = 0 collapses the loop to its single pinned vertex.
    if (opt.curveAnchors > 0) {
        std::vector<uint8_t> seen(Vs, 0);
        std::vector<double> dist(Vs, std::numeric_limits<double>::infinity());
        for (int64_t s0 = 0; s0 < Vs; ++s0) {
            if (G.role[s0] != RELAX_CURVE || seen[s0]) continue;
            // Collect the group.
            std::vector<int64_t> members{ s0 }, st{ s0 };
            seen[s0] = 1;
            while (!st.empty()) {
                const int64_t v = st.back(); st.pop_back();
                for (int64_t q = G.rowOffs[v]; q < G.rowOffs[v + 1]; ++q) {
                    const int j = G.cols[q];
                    if (G.role[j] != RELAX_CURVE || seen[j]) continue;
                    seen[j] = 1; members.push_back(j); st.push_back(j);
                }
            }
            // Farthest-point sampling: first anchor = lowest index (a fixed vertex
            // if the group was pinned), then the vertex farthest from all anchors.
            const int64_t first = *std::min_element(members.begin(), members.end());
            for (int64_t v : members) dist[v] = std::numeric_limits<double>::infinity();
            int64_t next = first;
            for (int k = 0; k < opt.curveAnchors && next >= 0; ++k) {
                R.anchors.push_back(next);
                if (isFree[next]) { isFree[next] = 0; --R.nFree; ++R.nFixed; ++R.nCurveAnchors; }
                // Dijkstra from the new anchor, keeping the minimum over anchors.
                using QE = std::pair<double, int64_t>;
                std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
                dist[next] = 0.0;
                pq.push({ 0.0, next });
                while (!pq.empty()) {
                    const auto [d, v] = pq.top(); pq.pop();
                    if (d > dist[v]) continue;
                    for (int64_t q = G.rowOffs[v]; q < G.rowOffs[v + 1]; ++q) {
                        const int j = G.cols[q];
                        if (G.role[j] != RELAX_CURVE) continue;
                        const double nd = d + (Vseed.row(v) - Vseed.row(j)).norm();
                        if (nd < dist[j]) { dist[j] = nd; pq.push({ nd, j }); }
                    }
                }
                double best = 0.0;
                next = -1;
                for (int64_t v : members)
                    if (dist[v] > best || (dist[v] == best && next >= 0 && v < next)) { best = dist[v]; next = v; }
                if (best <= 0.0) next = -1;
            }
        }
        if (opt.verbose)
            fprintf(stderr, "[subdiv_relax] curve anchors: %lld vertices fixed (%d per curve group)\n",
                    (long long)R.nCurveAnchors, opt.curveAnchors);
    }

    // Adaptive curve anchors (used when no fixed count is given). Between two
    // fixed vertices the 3D solve puts the curve vertices on the straight chord
    // joining them, so a stretch is fine exactly when the seam stays within tol
    // of that chord. Douglas-Peucker on the seed polyline of every chain:
    //   chain terminals: junctions, branch vertices (3+ curve neighbours), free
    //   ends, and one vertex per closed loop with no other terminal;
    //   inside a chain, the vertex farthest from the chord between the current
    //   ends becomes an anchor if it is more than tol * diag away; recurse.
    // Short or straight seams get no interior anchors, tightly curved ones many.
    if (opt.curveAnchors <= 0 && opt.curveAnchorTol > 0) {
        const double eps = opt.curveAnchorTol * diag;
        auto isC = [&](int64_t v) { return G.role[v] == RELAX_CURVE; };
        auto cnbrs = [&](int64_t v, std::vector<int64_t> & out) {
            out.clear();
            for (int64_t q = G.rowOffs[v]; q < G.rowOffs[v + 1]; ++q) if (isC(G.cols[q])) out.push_back(G.cols[q]);
        };
        std::vector<uint8_t> anchor(Vs, 0), seen(Vs, 0), isTerm(Vs, 0);
        auto add_anchor = [&](int64_t v) {
            if (!isC(v) || anchor[v]) return;
            anchor[v] = 1;
            R.anchors.push_back(v);
            if (isFree[v]) { isFree[v] = 0; --R.nFree; ++R.nFixed; ++R.nCurveAnchors; }
        };
        // Douglas-Peucker on seed positions of chain[lo..hi] (both ends fixed).
        std::function<void(const std::vector<int64_t> &, size_t, size_t)> dp =
            [&](const std::vector<int64_t> & ch, size_t lo, size_t hi) {
                if (hi <= lo + 1) return;
                const Vector3d a = Vseed.row(ch[lo]).transpose(), b = Vseed.row(ch[hi]).transpose();
                double best = -1.0;
                size_t bi = lo;
                for (size_t k = lo + 1; k < hi; ++k) {
                    const Vector3d p = Vseed.row(ch[k]).transpose();
                    const double t = seg_t(p, a, b);
                    const double d = (p - ((1 - t) * a + t * b)).norm();
                    if (d > best) { best = d; bi = k; }
                }
                if (best <= eps) return;
                add_anchor(ch[bi]);
                dp(ch, lo, bi);
                dp(ch, bi, hi);
            };

        std::vector<int64_t> nb, nb2;
        int64_t nChains = 0, nLoops = 0, nTerminals = 0;
        for (int64_t s0 = 0; s0 < Vs; ++s0) {
            if (!isC(s0) || seen[s0]) continue;
            // Group members.
            std::vector<int64_t> members{ s0 }, st{ s0 };
            seen[s0] = 1;
            while (!st.empty()) {
                const int64_t v = st.back(); st.pop_back();
                cnbrs(v, nb);
                for (int64_t j : nb) if (!seen[j]) { seen[j] = 1; members.push_back(j); st.push_back(j); }
            }
            // A vertex next to a junction: a chain through it ends at that junction.
            auto junction_of = [&](int64_t v) -> int64_t {
                for (int64_t q = G.rowOffs[v]; q < G.rowOffs[v + 1]; ++q)
                    if (G.role[G.cols[q]] == RELAX_JUNCTION) return G.cols[q];
                return -1;
            };
            // Terminals (anchored): branch vertices (3+ curve neighbours) and free
            // ends (at most one curve neighbour, no junction).
            std::vector<int64_t> terms, starts;
            for (int64_t v : members) {
                cnbrs(v, nb);
                const bool hasJ = junction_of(v) >= 0;
                if (nb.size() >= 3 || (nb.size() <= 1 && !hasJ)) terms.push_back(v);
                else if (nb.size() <= 1 && hasJ) starts.push_back(v);  // path end at a junction
            }
            if (terms.empty() && starts.empty()) {  // closed loop: one terminal
                terms.push_back(*std::min_element(members.begin(), members.end()));
                ++nLoops;
            }
            for (int64_t t : terms) { add_anchor(t); isTerm[t] = 1; }
            nTerminals += (int64_t)terms.size();
            starts.insert(starts.begin(), terms.begin(), terms.end());

            // Walk every chain once: from a start along each curve neighbour until a
            // terminal, the start itself (loop), or a dead end (closed by its junction).
            std::set<std::pair<int64_t, int64_t>> used;  // directed steps already walked
            for (int64_t s : starts) {
                cnbrs(s, nb);
                for (int64_t first : nb) {
                    if (used.count({ s, first })) continue;
                    std::vector<int64_t> ch;
                    if (!isTerm[s]) ch.push_back(junction_of(s));
                    ch.push_back(s);
                    used.insert({ s, first });
                    int64_t prev = s, cur = first;
                    for (;;) {
                        ch.push_back(cur);
                        if (isTerm[cur] || cur == s) break;
                        cnbrs(cur, nb2);
                        int64_t nxt = -1;
                        for (int64_t j : nb2) if (j != prev) { nxt = j; break; }
                        if (nxt < 0) {
                            const int64_t j = junction_of(cur);
                            if (j >= 0) ch.push_back(j); else add_anchor(cur);
                            break;
                        }
                        used.insert({ cur, nxt });
                        prev = cur; cur = nxt;
                    }
                    used.insert({ cur, prev });  // not again from the far end
                    ++nChains;
                    dp(ch, 0, ch.size() - 1);
                }
            }
            for (int64_t t : terms) isTerm[t] = 0;
        }
        if (opt.verbose)
            fprintf(stderr, "[subdiv_relax] adaptive curve anchors (tol %.3g x diag): %lld anchors, %lld chains, "
                    "%lld closed loops, %lld branch/free-end terminals\n",
                    opt.curveAnchorTol, (long long)R.anchors.size(), (long long)nChains, (long long)nLoops,
                    (long long)nTerminals);
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

    // ---- solve L x = 0 in 3D, then project (curves first, then sheets) ----
    // One linear solve per pass, no iteration: every free vertex is placed at
    // the 3D average of its pullers (fixed vertices as boundary values), and the
    // result is then projected onto the vertex's own structure. Curves do not
    // depend on sheets, so the curve pass comes first and the sheet pass uses
    // the projected curve positions as boundary values.
    // (A joint solve of seams + sheets at once was tried and was worse on the
    // fine subdivision: the sheets then see the seams before projection. See
    // md_files/subdiv_relax_solve_project_experiments.md, commit 84b44fc.)
    //
    // opt.jointSolve: instead, one system for all free curve and sheet vertices
    // at once (non-symmetric, since sheets are pulled by curves but not the
    // reverse; sparse LU), then every vertex is projected. It differs from the
    // two passes only in that the sheets see the curves before projection.
    if (opt.jointSolve) {
        const double ti = now_s();
        std::vector<int64_t> list;
        for (int64_t i = 0; i < Vs; ++i) if (isFree[i]) list.push_back(i);
        std::vector<int64_t> slot(Vs, -1);
        for (size_t a = 0; a < list.size(); ++a) slot[list[a]] = (int64_t)a;
        const int64_t n = (int64_t)list.size();
        std::vector<Triplet<double>> trip;
        trip.reserve((size_t)n * 7);
        MatrixXd rhs = MatrixXd::Zero(n, 3);
        for (int64_t a = 0; a < n; ++a) {
            const int64_t i = list[a];
            trip.emplace_back(a, a, (double)(G.rowOffs[i + 1] - G.rowOffs[i]));
            for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) {
                const int j = G.cols[q];
                if (slot[j] >= 0) trip.emplace_back(a, slot[j], -1.0);
                else              rhs.row(a) += X.row(j);
            }
        }
        SparseMatrix<double> A(n, n);
        A.setFromTriplets(trip.begin(), trip.end());
        trip.clear(); trip.shrink_to_fit();
        SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> lu;
        lu.compute(A);
        if (lu.info() != Success) fail("joint 3D Laplacian factorization failed");
        const MatrixXd Y = lu.solve(rhs);
        if (lu.info() != Success || !Y.allFinite()) fail("joint 3D Laplacian solve failed");
        const double relRes = (A * Y - rhs).norm() / std::max(1e-300, rhs.norm());

        std::vector<double> offDist(list.size(), 0.0);
        igl::parallel_for(n, [&](int64_t a) {
            const int64_t i = list[a];
            const Vector3d y = Y.row(a).transpose();
            const ProjResult pr = proj.project(setId[i], y, face[i], edge[i]);
            offDist[a] = (pr.pos - y).norm();
            move[i] = (pr.pos - X.row(i).transpose()).norm();
            nX.row(i) = pr.pos.transpose(); nFace[i] = pr.face; nEdge[i] = pr.edge; nBary.row(i) = pr.bary.transpose();
        }, 1000);
        const double m = commit(list);
        R.itersCurve = R.itersSheet = 1;
        R.deltaCurve = R.deltaSheet = m / diag;
        double off[2] = { 0, 0 }, offSum[2] = { 0, 0 };
        int64_t cnt[2] = { 0, 0 };
        for (int64_t a = 0; a < n; ++a) {
            const int c = G.role[list[a]] == RELAX_CURVE ? 0 : 1;
            off[c] = std::max(off[c], offDist[a]); offSum[c] += offDist[a]; ++cnt[c];
        }
        if (opt.verbose)
            fprintf(stderr, "[subdiv_relax] joint: 3D solve of %lld unknowns x 3 (LU, rel. residual %.2g), then projection: "
                    "off the MAT curves max %.3g mean %.3g, sheets max %.3g mean %.3g (x diag), max move %.3g (x diag) (%.2f s)\n",
                    (long long)n, relRes, off[0] / diag, cnt[0] ? offSum[0] / cnt[0] / diag : 0.0,
                    off[1] / diag, cnt[1] ? offSum[1] / cnt[1] / diag : 0.0, m / diag, now_s() - ti);
    }
    for (int pass = 0; pass < (opt.jointSolve ? 0 : 2); ++pass) {
        const double ti = now_s();
        const uint8_t cls = pass == 0 ? RELAX_CURVE : RELAX_SHEET;
        std::vector<int64_t> list;
        for (int64_t i = 0; i < Vs; ++i) if (isFree[i] && G.role[i] == cls) list.push_back(i);
        int64_t & iters = pass == 0 ? R.itersCurve : R.itersSheet;
        double & delta = pass == 0 ? R.deltaCurve : R.deltaSheet;
        if (list.empty()) { delta = 0.0; continue; }

        std::vector<int64_t> slot(Vs, -1);
        for (size_t a = 0; a < list.size(); ++a) slot[list[a]] = (int64_t)a;
        const int64_t n = (int64_t)list.size();
        std::vector<Triplet<double>> trip;
        trip.reserve((size_t)n * 7);
        MatrixXd rhs = MatrixXd::Zero(n, 3);
        for (int64_t a = 0; a < n; ++a) {
            const int64_t i = list[a];
            const int64_t deg = G.rowOffs[i + 1] - G.rowOffs[i];
            trip.emplace_back(a, a, (double)deg);
            for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) {
                const int j = G.cols[q];
                if (slot[j] >= 0) trip.emplace_back(a, slot[j], -1.0);
                else              rhs.row(a) += X.row(j);
            }
        }
        SparseMatrix<double> A(n, n);
        A.setFromTriplets(trip.begin(), trip.end());
        trip.clear(); trip.shrink_to_fit();
        const double asym = (SparseMatrix<double>(A.transpose()) - A).norm();
        if (asym > 1e-12 * A.norm()) fail("3D Laplacian block is not symmetric");
        SimplicialLDLT<SparseMatrix<double>> ldlt;
        ldlt.compute(A);
        if (ldlt.info() != Success) fail("3D Laplacian factorization failed");
        const MatrixXd Y = ldlt.solve(rhs);
        if (ldlt.info() != Success || !Y.allFinite()) fail("3D Laplacian solve failed");

        // How far the 3D solution is from the MAT before projection.
        std::vector<double> offDist(list.size(), 0.0);
        igl::parallel_for(n, [&](int64_t a) {
            const int64_t i = list[a];
            const Vector3d y = Y.row(a).transpose();
            const ProjResult pr = proj.project(setId[i], y, face[i], edge[i]);
            offDist[a] = (pr.pos - y).norm();
            move[i] = (pr.pos - X.row(i).transpose()).norm();
            nX.row(i) = pr.pos.transpose(); nFace[i] = pr.face; nEdge[i] = pr.edge; nBary.row(i) = pr.bary.transpose();
        }, 1000);
        const double m = commit(list);
        iters = 1;
        delta = m / diag;
        double offMax = 0.0, offSum = 0.0;
        for (double d : offDist) { offMax = std::max(offMax, d); offSum += d; }
        if (opt.verbose)
            fprintf(stderr, "[subdiv_relax] %s: 3D solve of %lld unknowns x 3, then projection: "
                    "3D solution off the MAT max %.3g mean %.3g (x diag), max move %.3g (x diag) (%.2f s)\n",
                    pass == 0 ? "curve" : "sheet", (long long)n, offMax / diag, offSum / n / diag, delta, now_s() - ti);
    }
    // One-shot method: "converged" only means it ran. Whether the result is a
    // resting state is what the plain-step check below reports.
    R.converged = true;
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
        "solve+project", R.converged ? "converged" : "NOT CONVERGED",
        (long long)R.itersCurve, R.deltaCurve, (long long)R.itersSheet, R.deltaSheet,
        (long long)R.nFree, (long long)R.nFixed, R.maxMove, R.meanMove,
        (long long)R.halvings, (long long)R.lineSearchStalls, (long long)R.plainSteps, (long long)R.nPinned, now_s() - tStart,
        R.jacobiStepMove, (long long)R.jacobiStepVertex,
        R.residualSheet, (long long)R.nResidualSheet, R.residualCurve, (long long)R.nResidualCurve,
        (long long)R.seedOffStructure, (long long)R.fixedMoved, (long long)R.posMismatch, (long long)R.badBary, (long long)R.offStructure);
    return R;
}
