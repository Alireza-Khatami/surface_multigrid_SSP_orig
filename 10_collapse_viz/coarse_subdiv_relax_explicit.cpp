#include "coarse_subdiv_relax_explicit.h"
#include "subdiv_sample_tracker/subdiv_relax_projector.h"

#include <igl/parallel_for.h>
#include <igl/writeOBJ.h>
#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Eigen;
using namespace subdiv_proj;

namespace {

void explicit_fail(const std::string & msg) { throw std::runtime_error("[relax_explicit] " + msg); }

double now_s()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

Vector3d tri_normal(const MatrixXd & P, const MatrixXi & F, int64_t f)
{
    const Vector3d a = P.row(F(f, 0)).transpose(), b = P.row(F(f, 1)).transpose(), c = P.row(F(f, 2)).transpose();
    return (b - a).cross(c - a);
}

} // namespace

RelaxReport subdiv_relax_explicit(SubdivMesh & M, const MatrixXd & VO, const MatrixXi & FO, const MatStruct * ms,
                                  const StructPalette & pal, const std::vector<int32_t> & setId,
                                  const RelaxGraph & G, const ExplicitRelaxOptions & opt)
{
    const double tStart = now_s();
    RelaxReport R;
    const int64_t Vs = M.V.rows(), nF = M.F.rows();
    if ((int64_t)G.role.size() != Vs) explicit_fail("graph / mesh size mismatch");
    if (!opt.holdFixed.empty() && (int64_t)opt.holdFixed.size() != Vs) explicit_fail("holdFixed / mesh size mismatch");
    const bool haveRef = opt.foldRef.size() > 0;
    if (haveRef && (opt.foldRef.rows() != nF || opt.foldRef.cols() != 3)) explicit_fail("foldRef / face count mismatch");
    if (opt.noNewFolds && !haveRef) explicit_fail("noNewFolds needs foldRef");
    if (!(opt.lambda > 0 && opt.lambda <= 1)) explicit_fail("lambda must be in (0, 1]");
    const bool weighted = !opt.weights.empty();
    if (weighted && opt.weights.size() != G.cols.size()) explicit_fail("weights / graph size mismatch");
    // Weight of graph entry q (row i): w_q, or 1 when uniform or the row's weights sum to 0.
    std::vector<uint8_t> uniformRow(Vs, 1);
    int64_t zeroRows = 0;
    if (weighted)
        for (int64_t i = 0; i < Vs; ++i) {
            double s = 0.0;
            for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) s += opt.weights[q];
            uniformRow[i] = !(s > 0);
            if (uniformRow[i] && G.rowOffs[i + 1] > G.rowOffs[i]) ++zeroRows;
        }
    auto w = [&](int64_t i, int64_t q) { return uniformRow[i] ? 1.0 : opt.weights[q]; };
    const double diag = (VO.leftCols(3).colwise().maxCoeff() - VO.leftCols(3).colwise().minCoeff()).norm();
    const double tolAbs = opt.tol * diag;

    // ---- projector onto each vertex's own structure of the fine mesh
    const SetIds S = split_palette(pal, ms);
    std::vector<uint8_t> setRole(pal.size());
    for (int k = 0; k < pal.size(); ++k) setRole[k] = ms ? role_of(pal.typeMask[k]) : (uint8_t)RELAX_SHEET;
    const Projector proj(VO, FO, M.origEdges, ms, pal, S, setRole);

    // ---- state: position + location on the fine mesh (face, bary; edge for curves)
    const MatrixXd Vseed = M.V;
    MatrixXd X = M.V, bary = M.fineBary;
    std::vector<int32_t> face = M.fineFace, edge(Vs, -1);
    for (int64_t i = 0; i < Vs; ++i)
        if (G.role[i] == RELAX_CURVE && M.carrierType[i] == SUBDIV_CARRIER_EDGE) edge[i] = M.carrierIndex[i];

    // ---- free vertices: not a junction, located, pulled by someone, has a target,
    //      not held; and the seed already on its own structure.
    std::vector<uint8_t> isFree(Vs, 0);
    for (int64_t i = 0; i < Vs; ++i)
        isFree[i] = G.role[i] != RELAX_JUNCTION && face[i] >= 0 && G.rowOffs[i + 1] > G.rowOffs[i]
                 && !proj.targets[setId[i]].empty() && (opt.holdFixed.empty() || !opt.holdFixed[i]);
    for (int64_t i = 0; i < Vs; ++i) {
        if (!isFree[i]) continue;
        const ProjResult r = proj.project(setId[i], X.row(i).transpose(), face[i], edge[i]);
        if ((r.pos - X.row(i).transpose()).norm() > 1e-12 * diag) { isFree[i] = 0; ++R.seedOffStructure; }
    }
    // A connected group of free vertices that nothing fixed pulls on can slide as
    // a whole: pin its lowest-index vertex.
    {
        std::vector<uint8_t> seen(Vs, 0);
        for (int64_t s0 = 0; s0 < Vs; ++s0) {
            if (!isFree[s0] || seen[s0]) continue;
            std::vector<int64_t> st{ s0 };
            seen[s0] = 1;
            bool anchored = false;
            int64_t lowest = s0;
            while (!st.empty()) {
                const int64_t v = st.back(); st.pop_back();
                lowest = std::min(lowest, v);
                for (int64_t q = G.rowOffs[v]; q < G.rowOffs[v + 1]; ++q) {
                    const int j = G.cols[q];
                    if (!isFree[j]) { anchored = true; continue; }
                    if (!seen[j]) { seen[j] = 1; st.push_back(j); }
                }
            }
            if (!anchored) { isFree[lowest] = 0; ++R.nPinned; }
        }
    }
    std::vector<int64_t> list;
    for (int64_t i = 0; i < Vs; ++i) if (isFree[i]) list.push_back(i);
    R.nFree = (int64_t)list.size();
    R.nFixed = Vs - R.nFree;

    // ---- metrics
    auto energy = [&](const MatrixXd & P) {  // 1/2 sum over graph edges of w_ij |x_i - x_j|^2 (each once)
        double e = 0.0;
        for (int64_t i = 0; i < Vs; ++i)
            for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q)
                if (G.cols[q] > i) e += 0.5 * w(i, q) * (P.row(i) - P.row(G.cols[q])).squaredNorm();
        return e;
    };
    auto count_folded = [&](const MatrixXd & P) {
        int64_t n = 0;
        if (haveRef) for (int64_t f = 0; f < nF; ++f) if (tri_normal(P, M.F, f).dot(opt.foldRef.row(f).transpose()) <= 0) ++n;
        return n;
    };
    auto count_degenerate = [&](const MatrixXd & P) {
        int64_t n = 0;
        for (int64_t f = 0; f < nF; ++f) if (0.5 * tri_normal(P, M.F, f).norm() <= 1e-14 * diag * diag) ++n;
        return n;
    };
    auto snapshot = [&](int64_t it) {
        if (opt.snapshotPrefix.empty()) return;
        if (std::find(opt.snapshotIters.begin(), opt.snapshotIters.end(), it) == opt.snapshotIters.end()) return;
        const std::string path = opt.snapshotPrefix + "it" + std::to_string(it) + ".obj";
        if (!igl::writeOBJ(subdiv_long_path(path), X, M.F))
            fprintf(stderr, "[relax_explicit] writeOBJ failed: %s\n", path.c_str());
    };
    if (haveRef) R.foldedSeed = count_folded(X);
    proj.reset_stats();  // projection statistics of the iterations only (not the seed check)
    ProjStats lastLog;
    fprintf(stderr, "[relax_explicit] %lld free, %lld fixed (%lld pinned, %lld seeds off structure) | lambda %.3g, "
                    "tol %.3g x diag, max %lld iterations | %s projection | no new folds %s | %s weights%s | "
                    "energy %.6g, folded %lld\n",
            (long long)R.nFree, (long long)R.nFixed, (long long)R.nPinned, (long long)R.seedOffStructure, opt.lambda,
            opt.tol, (long long)opt.maxIter, opt.localProjection ? "local" : "global", opt.noNewFolds ? "on" : "off",
            weighted ? "given" : "uniform",
            weighted ? (" (" + std::to_string(zeroRows) + " rows with zero weight sum -> uniform)").c_str() : "",
            energy(X), (long long)R.foldedSeed);

    // ---- iterate
    MatrixXd nX = X, nBary = bary;
    std::vector<int32_t> nFace = face, nEdge = edge;
    std::vector<double> move(Vs, 0.0);
    std::vector<uint8_t> newFold(nF, 0);
    auto shares_vertex = [&](int f, int g) {
        for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) if (FO(f, a) == FO(g, b)) return true;
        return false;
    };
    int64_t it = 0;
    double lastMove = 0.0;
    for (; it < opt.maxIter; ++it) {
        // Candidate step for every free vertex, all from the same X (Jacobi).
        igl::parallel_for((int64_t)list.size(), [&](int64_t a) {
            const int64_t i = list[a];
            Vector3d mean = Vector3d::Zero();
            double ws = 0.0;
            for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) {
                const double wq = w(i, q);
                mean += wq * X.row(G.cols[q]).transpose();
                ws += wq;
            }
            mean /= ws;
            const Vector3d x = X.row(i).transpose();
            const Vector3d y = x + opt.lambda * (mean - x);
            const ProjResult pr = opt.localProjection ? proj.project_local(setId[i], y, face[i], edge[i])
                                                      : proj.project(setId[i], y, face[i], edge[i]);
            move[i] = (pr.pos - x).norm();
            nX.row(i) = pr.pos.transpose(); nFace[i] = pr.face; nEdge[i] = pr.edge; nBary.row(i) = pr.bary.transpose();
        }, 1000);

        // No-new-folds: hold back every moved vertex of a triangle that is unfolded
        // at X and folded at the candidate, until there is none.
        if (opt.noNewFolds) {
            for (;;) {
                igl::parallel_for(nF, [&](int64_t f) {
                    const Vector3d ref = opt.foldRef.row(f).transpose();
                    newFold[f] = tri_normal(X, M.F, f).dot(ref) > 0 && tri_normal(nX, M.F, f).dot(ref) <= 0;
                }, 1000);
                int64_t held = 0;
                for (int64_t f = 0; f < nF; ++f) {
                    if (!newFold[f]) continue;
                    for (int c = 0; c < 3; ++c) {
                        const int v = M.F(f, c);
                        if (!isFree[v] || nX.row(v) == X.row(v)) continue;
                        nX.row(v) = X.row(v); nFace[v] = face[v]; nEdge[v] = edge[v]; nBary.row(v) = bary.row(v);
                        move[v] = 0.0;
                        ++held;
                    }
                }
                if (!held) break;
                R.foldReverts += held;
            }
        }

        // Commit.
        double m = 0.0;
        for (int64_t i : list) {
            m = std::max(m, move[i]);
            if (move[i] > 0) {
                ++R.projMoves;
                if (!shares_vertex(face[i], nFace[i])) { ++R.projJumps; R.projJumpMax = std::max(R.projJumpMax, move[i] / diag); }
            }
            X.row(i) = nX.row(i); face[i] = nFace[i]; edge[i] = nEdge[i]; bary.row(i) = nBary.row(i);
        }
        lastMove = m;
        snapshot(it + 1);
        if ((it + 1) % opt.logEvery == 0 || m <= tolAbs) {
            const ProjStats now = proj.stats(), d = now - lastLog;
            lastLog = now;
            fprintf(stderr, "[relax_explicit] iter %lld: max move %.3g (x diag), energy %.6g, folded %lld, degenerate %lld | "
                            "projection: %.2f%% on an edge/vertex, off-surface distance mean %.3g max %.3g (x diag) (%.1f s)\n",
                    (long long)(it + 1), m / diag, energy(X), (long long)count_folded(X), (long long)count_degenerate(X),
                    d.calls ? 100.0 * d.onBorder / d.calls : 0.0, d.calls ? d.distSum / d.calls / diag : 0.0,
                    now.distMax / diag, now_s() - tStart);
        }
        if (m <= tolAbs) { ++it; R.converged = true; break; }
    }
    R.itersSheet = it;
    R.deltaSheet = lastMove / diag;

    M.V = X;
    M.fineFace = face;
    M.fineBary = bary;
    if (haveRef) R.foldedResult = count_folded(X);
    R.localCalls = proj.nLocal; R.localGrown = proj.nLocalGrown; R.localGlobal = proj.nLocalGlobal;
    const ProjStats ps = proj.stats();

    // ---- checks (as subdiv_relax)
    double sumMove = 0.0;
    for (int64_t i = 0; i < Vs; ++i) {
        const double mv = (X.row(i) - Vseed.row(i)).norm();
        R.maxMove = std::max(R.maxMove, mv);
        sumMove += mv;
        if (!isFree[i]) { if (X.row(i) != Vseed.row(i)) ++R.fixedMoved; continue; }
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

    fprintf(stderr,
        "[relax_explicit] %s after %lld iterations (last max move %.3g x diag) | move max %.3g mean %.3g (x diag) | "
        "folded %lld -> %lld, moves held back %lld (%.1f s)\n"
        "[relax_explicit]   projection: %lld vertex moves, %lld jumps, largest jump %.3g (x diag); local calls %lld, "
        "grew past the first ring %lld, fell back to global %lld\n"
        "[relax_explicit]   projection over all iterations: %lld calls, %.2f%% on an edge/vertex of their "
        "triangle (or an end of their edge), off-surface distance mean %.3g max %.3g (x diag), barycentric fix-ups %lld\n"
        "[relax_explicit]   checks: seed off structure %lld, fixed moved %lld, pos != interp %lld, bad bary %lld, "
        "off own structure %lld\n",
        R.converged ? "converged" : "NOT CONVERGED", (long long)R.itersSheet, R.deltaSheet, R.maxMove, R.meanMove,
        (long long)R.foldedSeed, (long long)R.foldedResult, (long long)R.foldReverts, now_s() - tStart,
        (long long)R.projMoves, (long long)R.projJumps, R.projJumpMax, (long long)R.localCalls,
        (long long)R.localGrown, (long long)R.localGlobal,
        (long long)ps.calls, ps.calls ? 100.0 * ps.onBorder / ps.calls : 0.0,
        ps.calls ? ps.distSum / ps.calls / diag : 0.0, ps.distMax / diag, (long long)ps.baryFix,
        (long long)R.seedOffStructure, (long long)R.fixedMoved, (long long)R.posMismatch, (long long)R.badBary,
        (long long)R.offStructure);
    return R;
}
