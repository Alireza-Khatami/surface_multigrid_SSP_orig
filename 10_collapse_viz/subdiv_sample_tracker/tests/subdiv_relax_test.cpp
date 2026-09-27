// Standalone checks for build_relax_graph + subdiv_relax.
// Usage: subdiv_relax_test <mesh.obj> <file.ma_struct> <n_target> [newton|jacobi|both] [out_prefix]
//
//  - graph: every row re-checked against the rule, computed straight from the
//    MatStruct (struct types per id), not from the palette split the builder uses;
//    two-way S-S / C-C, completeness over the mesh edges.
//  - relaxation: the module's own checks must all be 0; a second relaxation of
//    the result must not move anything (it is a fixed point); with "both" the
//    Newton and Jacobi fixed points are compared.

#include "../subdiv_mesh.h"
#include "../subdiv_struct_ids.h"
#include "../subdiv_relax.h"
#include "../../load_matstruct.h"

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <set>
#include <string>

using namespace Eigen;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; if (g_fail < 30) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); } } } while (0)

static double tri_area2(const MatrixXd & V, int a, int b, int c)
{
    const Vector3d pa = V.row(a).transpose(), pb = V.row(b).transpose(), pc = V.row(c).transpose();
    return (pb - pa).cross(pc - pa).norm();
}

static void print_quality(const char * label, const MeshQuality & q)
{
    fprintf(stderr, "  %-8s edge CV %.4f | min angle %.3f, p1 %.3f, p5 %.3f, median %.3f deg | degenerate %lld | flipped vs seed %lld\n",
            label, q.edgeCV, q.minAngle, q.p1, q.p5, q.median, (long long)q.degenerate, (long long)q.flippedVsRef);
}

int main(int argc, char ** argv)
{
    if (argc < 4) { fprintf(stderr, "usage: %s mesh.obj file.ma_struct n_target [newton|jacobi|both] [out_prefix]\n", argv[0]); return 2; }
    const std::string mode = argc > 4 ? argv[4] : "newton";
    const std::string out = argc > 5 ? argv[5] : "";
    MatrixXd VO, TC, N; MatrixXi FO, FTC, FN;
    if (!igl::readOBJ(argv[1], VO, TC, N, FO, FTC, FN)) { fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }
    MatStruct ms;
    std::string err;
    if (!load_matstruct(argv[2], VO, FO, ms, &err)) { fprintf(stderr, "ma_struct: %s\n", err.c_str()); return 2; }
    const int64_t nTarget = std::stoll(argv[3]);

    SubdivMesh M = build_subdiv_mesh(VO, FO, nTarget);
    StructPalette pal;
    std::vector<int32_t> setId;
    build_struct_sets(M, FO, &ms, pal, setId);
    const int64_t Vs = M.V.rows();

    // ---------------- graph ----------------
    const RelaxGraph G = build_relax_graph(M, pal, setId, &ms);

    // Independent rule: per vertex its id set by type, from ms.structType.
    auto ids_of = [&](int64_t v, int want /*0 sheet, 1 curve, 3 junction*/) {
        std::vector<int> r;
        for (int32_t a = pal.offsets[setId[v]]; a < pal.offsets[setId[v] + 1]; ++a) {
            const int t = ms.structType.at(pal.ids[a]);
            if ((want == 1 && (t == 1 || t == 2)) || t == want) r.push_back(pal.ids[a]);
        }
        return r;
    };
    auto share = [](const std::vector<int> & a, const std::vector<int> & b) {
        for (int x : a) if (std::find(b.begin(), b.end(), x) != b.end()) return true;
        return false;
    };
    std::vector<uint8_t> role(Vs);
    for (int64_t v = 0; v < Vs; ++v)
        role[v] = !ids_of(v, 3).empty() ? RELAX_JUNCTION : !ids_of(v, 1).empty() ? RELAX_CURVE : RELAX_SHEET;
    int64_t roleMismatch = 0;
    for (int64_t v = 0; v < Vs; ++v) roleMismatch += role[v] != G.role[v];
    CHECK(roleMismatch == 0, "%lld role mismatches", (long long)roleMismatch);

    std::set<std::pair<int, int>> meshEdges;
    for (Index f = 0; f < M.F.rows(); ++f)
        for (int c = 0; c < 3; ++c) {
            int a = M.F(f, c), b = M.F(f, (c + 1) % 3);
            if (a > b) std::swap(a, b);
            meshEdges.insert({ a, b });
        }
    CHECK((int64_t)meshEdges.size() == G.nUndirected, "undirected edge count");
    auto expect = [&](int i, int j) {  // j pulls on i ?
        if (role[i] == RELAX_JUNCTION) return false;
        if (role[i] == RELAX_CURVE) return role[j] != RELAX_SHEET && share(ids_of(i, 1), ids_of(j, 1));
        return share(ids_of(i, 0), ids_of(j, 0)) || (ids_of(i, 0).empty() && ids_of(j, 0).empty());
    };
    auto has = [&](int i, int j) {
        return std::binary_search(G.cols.begin() + G.rowOffs[i], G.cols.begin() + G.rowOffs[i + 1], j);
    };
    int64_t nExpected = 0, wrong = 0, oneWaySS = 0, oneWayCC = 0;
    for (const auto & e : meshEdges) {
        const int a = e.first, b = e.second;
        const bool ab = expect(a, b), ba = expect(b, a);
        nExpected += ab + ba;
        if (ab != has(a, b) || ba != has(b, a)) ++wrong;
        if (role[a] == RELAX_SHEET && role[b] == RELAX_SHEET && has(a, b) != has(b, a)) ++oneWaySS;
        if (role[a] == RELAX_CURVE && role[b] == RELAX_CURVE && has(a, b) != has(b, a)) ++oneWayCC;
    }
    CHECK(wrong == 0, "%lld mesh edges with a wrong direction set", (long long)wrong);
    CHECK(nExpected == (int64_t)G.cols.size(), "entry count %lld vs expected %lld", (long long)G.cols.size(), (long long)nExpected);
    CHECK(oneWaySS == 0 && oneWayCC == 0, "one-way S-S %lld / C-C %lld", (long long)oneWaySS, (long long)oneWayCC);
    int64_t junctionRows = 0, notMeshEdge = 0;
    for (int64_t i = 0; i < Vs; ++i)
        for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) {
            if (role[i] == RELAX_JUNCTION) ++junctionRows;
            const int a = (int)std::min<int64_t>(i, G.cols[q]), b = (int)std::max<int64_t>(i, G.cols[q]);
            if (!meshEdges.count({ a, b })) ++notMeshEdge;
        }
    CHECK(junctionRows == 0, "junction rows not empty");
    CHECK(notMeshEdge == 0, "entries that are not mesh edges");
    fprintf(stderr, "graph check: %lld entries re-derived, role mismatch %lld, wrong %lld, one-way S-S %lld C-C %lld\n",
            (long long)nExpected, (long long)roleMismatch, (long long)wrong, (long long)oneWaySS, (long long)oneWayCC);

    // Anchoring of curve and sheet components: a component with no fixed
    // neighbour (junction for curves; curve/junction for sheets) has no unique
    // fixed point.
    {
        auto analyse = [&](uint8_t cls, const char * name) {
            std::vector<int64_t> comp(Vs, -1);
            int64_t nComp = 0, nFree = 0, nFreeVerts = 0, nLoopish = 0, maxSize = 0;
            for (int64_t s0 = 0; s0 < Vs; ++s0) {
                if (G.role[s0] != cls || comp[s0] >= 0) continue;
                std::vector<int64_t> st{ s0 }, members;
                comp[s0] = nComp;
                bool anchored = false;
                int64_t ends = 0;
                while (!st.empty()) {
                    const int64_t v = st.back(); st.pop_back();
                    members.push_back(v);
                    int64_t same = 0;
                    for (int64_t q = G.rowOffs[v]; q < G.rowOffs[v + 1]; ++q) {
                        const int j = G.cols[q];
                        if (G.role[j] != cls) { anchored = true; continue; }
                        ++same;
                        if (comp[j] < 0) { comp[j] = nComp; st.push_back(j); }
                    }
                    if (same <= 1) ++ends;
                }
                if (!anchored) {
                    ++nFree; nFreeVerts += (int64_t)members.size();
                    if (ends == 0) ++nLoopish;
                    maxSize = std::max<int64_t>(maxSize, (int64_t)members.size());
                }
                ++nComp;
            }
            fprintf(stderr, "%s components: %lld, unanchored %lld (%lld vertices, largest %lld, %lld without an end vertex)\n",
                    name, (long long)nComp, (long long)nFree, (long long)nFreeVerts, (long long)maxSize, (long long)nLoopish);
        };
        analyse(RELAX_CURVE, "curve");
        analyse(RELAX_SHEET, "sheet");
        // Curve vertices with exactly one pulling neighbour that is not a junction:
        // the free end of an open curve piece.
        int64_t freeEnds = 0;
        for (int64_t i = 0; i < Vs; ++i)
            if (G.role[i] == RELAX_CURVE && G.rowOffs[i + 1] - G.rowOffs[i] == 1 && G.role[G.cols[G.rowOffs[i]]] == RELAX_CURVE) ++freeEnds;
        fprintf(stderr, "curve vertices pulled by a single curve neighbour (free ends): %lld\n", (long long)freeEnds);
    }

    if (!out.empty()) {
        save_relax_graph(out + "graph.slg", M.V, G, pal, setId);
        igl::writeOBJ(out + "seed.obj", M.V, M.F);
    }

    // ---------------- relaxation ----------------
    const MatrixXd Vseed = M.V;
    const MeshQuality q0 = subdiv_mesh_quality(M.V, M.F, &Vseed);

    auto check_report = [&](const char * label, const RelaxReport & R) {
        CHECK(R.converged, "%s: not converged", label);
        CHECK(R.jacobiStepMove <= 1e-10, "%s: one Jacobi step from the result moves %.3g (v %lld)", label,
              R.jacobiStepMove, (long long)R.jacobiStepVertex);
        CHECK(R.seedOffStructure == 0, "%s: %lld seeds off structure", label, (long long)R.seedOffStructure);
        CHECK(R.fixedMoved == 0 && R.posMismatch == 0 && R.badBary == 0 && R.offStructure == 0,
              "%s: checks fixed %lld pos %lld bary %lld off %lld", label, (long long)R.fixedMoved,
              (long long)R.posMismatch, (long long)R.badBary, (long long)R.offStructure);
    };

    SubdivMesh Mn = M, Mj = M;
    if (mode == "newton" || mode == "both") {
        auto t0 = std::chrono::steady_clock::now();
        const RelaxReport R = subdiv_relax(Mn, VO, FO, &ms, pal, setId, G);
        fprintf(stderr, "newton: %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        check_report("newton", R);
        const MeshQuality q1 = subdiv_mesh_quality(Mn.V, Mn.F, &Vseed);
        fprintf(stderr, "quality:\n");
        print_quality("seed", q0);
        print_quality("relaxed", q1);

        // Junctions / fixed vertices bit-identical (independent of the module's check).
        int64_t jMoved = 0;
        for (int64_t i = 0; i < Vs; ++i) if (G.role[i] == RELAX_JUNCTION && Mn.V.row(i) != Vseed.row(i)) ++jMoved;
        CHECK(jMoved == 0, "%lld junctions moved", (long long)jMoved);

        // Fixed point: relaxing the result again must not move anything.
        SubdivMesh M2 = Mn;
        RelaxOptions o; o.verbose = false;
        const RelaxReport R2 = subdiv_relax(M2, VO, FO, &ms, pal, setId, G, o);
        check_report("newton re-run", R2);
        const double d2 = (M2.V - Mn.V).rowwise().norm().maxCoeff()
                        / (VO.colwise().maxCoeff() - VO.colwise().minCoeff()).norm();
        fprintf(stderr, "re-run: curve iters %lld, sheet iters %lld, max change %.3g (x diag)\n",
                (long long)R2.itersCurve, (long long)R2.itersSheet, d2);
        CHECK(d2 <= 1e-10, "re-run moved %.3g", d2);

        if (!out.empty()) igl::writeOBJ(out + "relaxed.obj", Mn.V, Mn.F);

        // Degenerate triangles after relaxation: who collapsed onto whom.
        {
            const double diag = (VO.colwise().maxCoeff() - VO.colwise().minCoeff()).norm();
            int shown = 0;
            for (Index f = 0; f < Mn.F.rows() && shown < 5; ++f) {
                const int a = Mn.F(f, 0), b = Mn.F(f, 1), c = Mn.F(f, 2);
                const Vector3d pa = Mn.V.row(a).transpose(), pb = Mn.V.row(b).transpose(), pc = Mn.V.row(c).transpose();
                if (0.5 * (pb - pa).cross(pc - pa).norm() > 1e-14 * diag * diag) continue;
                ++shown;
                fprintf(stderr, "degenerate face %lld (orig face %d): roles %d %d %d | |ab| %.3g |bc| %.3g |ca| %.3g (x diag) | "
                        "fine faces %d %d %d | seed area %.3g\n",
                        (long long)f, M.faceOrig[f], G.role[a], G.role[b], G.role[c],
                        (pb - pa).norm() / diag, (pc - pb).norm() / diag, (pa - pc).norm() / diag,
                        Mn.fineFace[a], Mn.fineFace[b], Mn.fineFace[c],
                        0.5 * tri_area2(Vseed, a, b, c) / (diag * diag));
            }
        }
    }
    if (mode == "lambdas") {
        // Path dependence of the literal scheme: Jacobi with two step sizes.
        SubdivMesh Ma = M, Mb = M;
        RelaxOptions oa; oa.solver = RelaxSolver::Jacobi; oa.lambda = 0.5;
        RelaxOptions ob = oa; ob.lambda = 0.25;
        check_report("jacobi 0.5", subdiv_relax(Ma, VO, FO, &ms, pal, setId, G, oa));
        check_report("jacobi 0.25", subdiv_relax(Mb, VO, FO, &ms, pal, setId, G, ob));
        const double diag = (VO.colwise().maxCoeff() - VO.colwise().minCoeff()).norm();
        const VectorXd d = (Ma.V - Mb.V).rowwise().norm() / diag;
        int64_t over = 0;
        for (Index i = 0; i < d.size(); ++i) over += d(i) > 1e-8;
        fprintf(stderr, "jacobi 0.5 vs 0.25: max %.3g, mean %.3g (x diag), %lld vertices > 1e-8\n", d.maxCoeff(), d.mean(), (long long)over);
    }
    if (mode == "jacobi" || mode == "both") {
        auto t0 = std::chrono::steady_clock::now();
        RelaxOptions o; o.solver = RelaxSolver::Jacobi;
        const RelaxReport R = subdiv_relax(Mj, VO, FO, &ms, pal, setId, G, o);
        fprintf(stderr, "jacobi: %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        check_report("jacobi", R);
    }
    if (mode == "both") {
        const double diag = (VO.colwise().maxCoeff() - VO.colwise().minCoeff()).norm();
        const VectorXd d = (Mn.V - Mj.V).rowwise().norm() / diag;
        int64_t over = 0;
        for (Index i = 0; i < d.size(); ++i) over += d(i) > 1e-8;
        double byRole[3] = {};
        for (Index i = 0; i < d.size(); ++i) byRole[G.role[i]] = std::max(byRole[G.role[i]], d(i));
        fprintf(stderr, "newton vs jacobi: max %.3g, mean %.3g (x diag), %lld vertices > 1e-8 | max sheet %.3g curve %.3g junction %.3g\n",
                d.maxCoeff(), d.mean(), (long long)over, byRole[0], byRole[1], byRole[2]);
        auto energy = [&](const MatrixXd & X, int role) {
            double e = 0.0;
            for (int64_t i = 0; i < Vs; ++i) {
                if (role >= 0 && G.role[i] != role) continue;
                for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) e += 0.5 * (X.row(i) - X.row(G.cols[q])).squaredNorm();
            }
            return e;
        };
        fprintf(stderr, "energy (sum over rows): curve rows newton %.10g jacobi %.10g | sheet rows newton %.10g jacobi %.10g\n",
                energy(Mn.V, RELAX_CURVE), energy(Mj.V, RELAX_CURVE), energy(Mn.V, RELAX_SHEET), energy(Mj.V, RELAX_SHEET));
        print_quality("newton", subdiv_mesh_quality(Mn.V, Mn.F, &Vseed));
        print_quality("jacobi", subdiv_mesh_quality(Mj.V, Mj.F, &Vseed));
        std::vector<Index> idx(d.size());
        for (Index i = 0; i < d.size(); ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](Index a, Index b) { return d(a) > d(b); });
        for (int t = 0; t < 6; ++t) {
            const Index i = idx[t];
            fprintf(stderr, "  v %lld role %d carrier %d diff %.3g | newton face %d bary (%.3f %.3f %.3f) | jacobi face %d bary (%.3f %.3f %.3f)\n",
                    (long long)i, G.role[i], M.carrierType[i], d(i), Mn.fineFace[i], Mn.fineBary(i, 0), Mn.fineBary(i, 1), Mn.fineBary(i, 2),
                    Mj.fineFace[i], Mj.fineBary(i, 0), Mj.fineBary(i, 1), Mj.fineBary(i, 2));
        }
        // Newton started from Jacobi's result: if Jacobi had truly converged,
        // Newton must not move it.
        {
            SubdivMesh M3 = Mj;
            RelaxOptions o; o.verbose = false;
            subdiv_relax(M3, VO, FO, &ms, pal, setId, G, o);
            const VectorXd dj = (M3.V - Mj.V).rowwise().norm() / diag;
            const VectorXd dn = (M3.V - Mn.V).rowwise().norm() / diag;
            fprintf(stderr, "newton from jacobi's result: moves it max %.3g | then differs from newton-from-seed max %.3g | energy curve %.10g sheet %.10g\n",
                    dj.maxCoeff(), dn.maxCoeff(), energy(M3.V, RELAX_CURVE), energy(M3.V, RELAX_SHEET));
        }
        // Not a check: the fixed point is not unique (curve loops have several
        // resting positions a polyline segment apart; even Jacobi with two step
        // sizes ends on different ones). Both results are verified fixed points
        // by check_report; the energies above say which one relaxed further.
    }

    fprintf(stderr, g_fail ? "RESULT: %d FAILURES\n" : "RESULT: all checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}
