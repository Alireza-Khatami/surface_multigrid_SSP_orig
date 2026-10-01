#include "coarse_subdiv_relax.h"
#include "coarse_subdiv_relax_explicit.h"
#include "collapse_structure_tracker/simp_viz_tracker.h"
#include "subdiv_sample_tracker/subdiv_relax.h"
#include "subdiv_sample_tracker/subdiv_relax_projector.h"
#include "subdiv_sample_tracker/subdiv_struct_ids.h"

#include <igl/writeOBJ.h>
#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace Eigen;
using namespace subdiv_proj;

// ---- globals from main.cpp ----
extern MatrixXd gVO;
extern MatrixXi gFO;

static void relax_fail(const std::string & msg) { throw std::runtime_error("[coarse_subdiv_relax] " + msg); }

// Structure IDs of the compact coarse mesh, as a MatStruct indexed like it
// (vertices = cmc.Vbase rows, faces = cmc.Fout rows), so build_struct_sets can
// assign them to the subdivided vertices by its usual carrier rules.
static MatStruct coarse_matstruct(const CoarseMeshCompaction & cmc, const MatStruct & ms)
{
    const int NCE = (int)cmc.Vbase.rows(), FC = (int)cmc.Fout.rows(), nFine = (int)gVO.rows();
    MatStruct cms;
    cms.nv = NCE;
    cms.nf = FC;
    cms.structType = ms.structType;
    cms.hasStructSection = ms.hasStructSection;

    // Vertices: struct IDs tracked through the collapses. With the gate they equal
    // the fine vertex's own IDs; count any difference.
    cms.vertexIds.resize(NCE);
    int idsDiffer = 0;
    for (int i = 0; i < NCE; ++i) {
        const int gv = cmc.newToOld(i);
        cms.vertexIds[i] = simp_viz_tracker_struct_ids(gv);
        if (gv < (int)ms.vertexIds.size() && cms.vertexIds[i] != ms.vertexIds[gv]) ++idsDiffer;
    }

    // Faces: a coarse face is its gF row, which keeps the fine face's sheet.
    cms.faceIds.resize(FC);
    for (int f = 0; f < FC; ++f) cms.faceIds[f] = ms.faceIds[cmc.faceOrigIdx(f)];

    // Edges: owner of each fine vertex = the coarse vertex it was merged into.
    std::vector<int> owner(nFine, -1);
    int ownerConflicts = 0;
    for (int i = 0; i < NCE; ++i)
        for (int u : simp_viz_tracker_ancestors(cmc.newToOld(i))) {
            if (u < 0 || u >= nFine) continue;
            if (owner[u] >= 0 && owner[u] != i) ++ownerConflicts;
            owner[u] = i;
        }
    std::unordered_set<uint64_t> coarseEdges;
    for (int f = 0; f < FC; ++f)
        for (int c = 0; c < 3; ++c)
            coarseEdges.insert(matstruct_edge_key(cmc.Fout(f, c), cmc.Fout(f, (c + 1) % 3)));

    int fineCurveEdges = 0, inside = 0, noOwner = 0, notCoarseEdge = 0;
    for (const auto & e : ms.maEdges) {
        auto it = ms.edgeIds.find(matstruct_edge_key(e[0], e[1]));
        if (it == ms.edgeIds.end() || it->second.empty()) continue;
        ++fineCurveEdges;
        const int a = (e[0] >= 0 && e[0] < nFine) ? owner[e[0]] : -1;
        const int b = (e[1] >= 0 && e[1] < nFine) ? owner[e[1]] : -1;
        if (a < 0 || b < 0) { ++noOwner; continue; }
        if (a == b) { ++inside; continue; }  // collapsed into one coarse vertex
        const uint64_t key = matstruct_edge_key(a, b);
        if (!coarseEdges.count(key)) { ++notCoarseEdge; continue; }
        std::vector<int> & ids = cms.edgeIds[key];
        ids.insert(ids.end(), it->second.begin(), it->second.end());
    }
    int notOnEndpoints = 0;
    for (auto & kv : cms.edgeIds) {
        std::vector<int> & ids = kv.second;
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        const int a = (int)(kv.first >> 32), b = (int)(kv.first & 0xffffffffu);
        for (int id : ids)
            if (!cms.vertexIds[a].count(id) || !cms.vertexIds[b].count(id)) { ++notOnEndpoints; break; }
    }
    fprintf(stderr,
        "[coarse_subdiv_relax] coarse struct IDs: %d vertices (%d differ from the fine vertex's IDs), %d faces, "
        "%zu curve edges from %d fine curve edges (%d collapsed inside a coarse vertex, %d without owner, "
        "%d not a coarse edge); %d owner conflicts, %d curve edges whose IDs are not on both endpoints\n",
        NCE, idsDiffer, FC, cms.edgeIds.size(), fineCurveEdges, inside, noOwner, notCoarseEdge,
        ownerConflicts, notOnEndpoints);
    if (idsDiffer || noOwner || notCoarseEdge || ownerConflicts || notOnEndpoints)
        relax_fail("coarse structure IDs are inconsistent (see counts above)");
    return cms;
}

// Distinct colour per structure id (golden-ratio hue walk); grey for -1.
static std::array<uint8_t, 3> struct_color(int id)
{
    if (id < 0) return { 128, 128, 128 };
    const double h = std::fmod(0.1 + id * 0.6180339887498949, 1.0) * 6.0, s = 0.75, v = 0.95;
    const int i = (int)h;
    const double f = h - i, p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
    double r, g, b;
    switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
    }
    return { (uint8_t)std::lround(255 * r), (uint8_t)std::lround(255 * g), (uint8_t)std::lround(255 * b) };
}

// Smallest id in both sorted lists, -1 if none.
static int first_common(const std::vector<int> & a, const std::vector<int> & b)
{
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) return a[i];
        if (a[i] < b[j]) ++i; else ++j;
    }
    return -1;
}

// Binary PLY of graph edges as an edge soup: every edge has its own two
// vertices, so per-vertex colours show the edge's structure in any viewer.
//   vertex: x y z (float), red green blue (uchar), vid (int, subdivided vertex),
//           fixed (uchar, 1 = held during the relaxation)
//   edge  : vertex1 vertex2 (int), red green blue (uchar), struct_id (int)
// Junction files (edges empty) hold one vertex per junction vertex.
struct GraphPly {
    struct E { int64_t a, b; int id; };
    std::vector<E> edges;
    std::vector<std::pair<int64_t, int>> points;  // (vertex, struct id)
};

static void write_graph_ply(const std::string & path, const GraphPly & g, const MatrixXd & V,
                            const std::vector<uint8_t> & fixed)
{
    std::ofstream f(subdiv_long_path(path), std::ios::binary);
    if (!f) relax_fail("cannot write " + path);
    const size_t nv = g.points.size() + 2 * g.edges.size();
    f << "ply\nformat binary_little_endian 1.0\n"
      << "element vertex " << nv << "\nproperty float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty int vid\nproperty uchar fixed\n"
      << "element edge " << g.edges.size() << "\nproperty int vertex1\nproperty int vertex2\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty int struct_id\nend_header\n";
    auto put_vertex = [&](int64_t i, int id) {
        const float p[3] = { (float)V(i, 0), (float)V(i, 1), (float)V(i, 2) };
        const auto c = struct_color(id);
        const int32_t vid = (int32_t)i;
        const uint8_t fx = fixed[i];
        f.write((const char *)p, sizeof p);
        f.write((const char *)c.data(), 3);
        f.write((const char *)&vid, 4);
        f.write((const char *)&fx, 1);
    };
    for (const auto & pt : g.points) put_vertex(pt.first, pt.second);
    for (const auto & e : g.edges) { put_vertex(e.a, e.id); put_vertex(e.b, e.id); }
    int32_t k = (int32_t)g.points.size();
    for (const auto & e : g.edges) {
        const int32_t ab[2] = { k, k + 1 };
        const auto c = struct_color(e.id);
        const int32_t id = e.id;
        f.write((const char *)ab, sizeof ab);
        f.write((const char *)c.data(), 3);
        f.write((const char *)&id, 4);
        k += 2;
    }
    if (!f) relax_fail("write failed: " + path);
}

// The relaxation graph at the seed positions, split by row role, one file each:
//   sheets_   : edges of SHEET rows (i <- j sharing a sheet id), coloured by that sheet id
//   curves_   : edges of CURVE rows (i <- j sharing a seam/boundary id, j curve or
//               junction), coloured by that curve id
//   junctions_: the junction vertices (their rows are empty: never move), coloured
//               by junction id
// Each undirected pair appears once per file.
static void export_relax_graph_ply(const std::string & dir, const std::string & stem, const RelaxGraph & G,
                                   const MatrixXd & V, const StructPalette & pal, const SetIds & S,
                                   const std::vector<int32_t> & setId, const MatStruct & ms,
                                   const std::vector<uint8_t> & fixed)
{
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(subdiv_long_path(dir)), ec);
    if (ec) relax_fail("cannot create " + dir + ": " + ec.message());

    GraphPly sheets, curves, junctions;
    std::unordered_set<uint64_t> seenS, seenC;
    std::unordered_set<int> sheetIds, curveIds, junctionIds;
    int64_t noCommon[2] = {};
    const int64_t Vs = V.rows();
    for (int64_t i = 0; i < Vs; ++i) {
        const int k = setId[i];
        if (G.role[i] == RELAX_JUNCTION) {
            int jid = -1;
            for (int32_t a = pal.offsets[k]; a < pal.offsets[k + 1]; ++a)
                if (ms.structType.count(pal.ids[a]) && ms.structType.at(pal.ids[a]) == 3) { jid = pal.ids[a]; break; }
            junctions.points.push_back({ i, jid });
            junctionIds.insert(jid);
            continue;
        }
        const bool sheet = G.role[i] == RELAX_SHEET;
        for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) {
            const int64_t j = G.cols[q];
            const uint64_t key = ((uint64_t)std::min(i, j) << 32) | (uint64_t)std::max(i, j);
            if (!(sheet ? seenS : seenC).insert(key).second) continue;
            const int id = sheet ? first_common(S.sheet[k], S.sheet[setId[j]]) : first_common(S.curve[k], S.curve[setId[j]]);
            if (id < 0) ++noCommon[sheet ? 0 : 1];
            (sheet ? sheets : curves).edges.push_back({ i, j, id });
            (sheet ? sheetIds : curveIds).insert(id);
        }
    }
    const std::string base = dir + "/";
    write_graph_ply(base + "sheets_" + stem + ".ply", sheets, V, fixed);
    write_graph_ply(base + "curves_" + stem + ".ply", curves, V, fixed);
    write_graph_ply(base + "junctions_" + stem + ".ply", junctions, V, fixed);
    fprintf(stderr,
        "[coarse_subdiv_relax] Laplacian graph PLYs -> %s: sheets %zu edges / %zu ids, curves %zu edges / %zu ids, "
        "junctions %zu vertices / %zu ids; edges with no shared id: sheet %lld, curve %lld (expected 0)\n",
        dir.c_str(), sheets.edges.size(), sheetIds.size(), curves.edges.size(), curveIds.size(),
        junctions.points.size(), junctionIds.size(), (long long)noCommon[0], (long long)noCommon[1]);
}

// Checks every projector tree: structure (boxes nest, leaves contain their
// primitives, each primitive once) and nPts closest-point queries against brute
// force (identical distance and primitive, same tie rule). Query points: around
// random primitives at three scales of the tree's box diagonal (on / near / far).
static void verify_projector_bvh(const Projector & proj, int nPts)
{
    int64_t structErr = 0, mismatch = 0, nq = 0;
    for (size_t t = 0; t < proj.trees.size(); ++t) {
        const PrimBVH & tr = proj.trees[t];
        structErr += tr.structure_errors(proj.VO);
        if (tr.gid.empty()) continue;
        const double tdiag = tr.nodes[0].box.diagonal().norm();
        std::mt19937 rng((uint32_t)t);
        std::uniform_int_distribution<int> pick(0, (int)tr.gid.size() - 1);
        std::normal_distribution<double> nd(0.0, 1.0);
        const double scale[3] = { 0.0, 0.02, 0.5 };
        for (int s = 0; s < nPts; ++s) {
            const auto & c = tr.corner[pick(rng)];
            Vector3d p = Vector3d::Zero();
            int m = 0;
            for (int a = 0; a < 3; ++a) if (c[a] >= 0) { p += P3(proj.VO, c[a]); ++m; }
            p /= m;
            p += scale[s % 3] * tdiag * Vector3d(nd(rng), nd(rng), nd(rng));
            double d0 = std::numeric_limits<double>::infinity(), d1 = d0;
            int g0 = std::numeric_limits<int>::max(), g1 = g0;
            Vector3d b;
            tr.brute_force(proj.VO, p, d0, g0);
            tr.query(proj.VO, p, d1, g1, b);
            if (d0 != d1 || g0 != g1) ++mismatch;
            ++nq;
        }
    }
    fprintf(stderr, "[coarse_subdiv_relax] projector BVH check: %zu trees, %lld structure errors, "
                    "%lld of %lld queries differ from brute force\n",
            proj.trees.size(), (long long)structErr, (long long)mismatch, (long long)nq);
    if (structErr || mismatch) relax_fail("projector BVH check failed");
}

void coarse_subdiv_relax_export(const CoarseMeshCompaction & cmc, const CoarseSubdivC2F & C,
                                const MatStruct & ms, const CoarseSubdivRelaxConfig & cfg,
                                int64_t maxObjVerts, const std::string & objPath,
                                const std::string & graphDir, const std::string & graphStem)
{
    const std::string & method = cfg.method;
    const bool isExplicit = method == "explicit";
    if (method != "newton" && method != "solve_project" && !isExplicit) relax_fail("unknown relax method " + method);
    const int64_t Vs = C.S.V.rows();
    const double diag = (gVO.leftCols(3).colwise().maxCoeff() - gVO.leftCols(3).colwise().minCoeff()).norm();

    // 1. Struct IDs of the subdivided vertices, from the coarse carriers.
    const MatStruct cms = coarse_matstruct(cmc, ms);
    StructPalette pal;
    std::vector<int32_t> setId;
    build_struct_sets(C.S, cmc.Fout, &cms, pal, setId);

    // 2. The same vertices and faces, located on the fine mesh.
    SubdivMesh M = C.S;
    M.V = C.P;
    M.fineFace = C.fineFace;
    M.fineBary = C.fineBary;
    M.origEdges = subdiv_unique_edges(gFO);
    std::fill(M.carrierType.begin(), M.carrierType.end(), (uint8_t)SUBDIV_CARRIER_FACE);
    for (int64_t i = 0; i < Vs; ++i) M.carrierIndex[i] = M.fineFace[i];

    // 3. Seeds onto their own fine structure: the query result is on the fine
    //    surface, but not exactly on (e.g.) a seam edge, and the relaxation keeps
    //    any seed that is not on its structure fixed.
    const SetIds S = split_palette(pal, &ms);
    std::vector<uint8_t> setRole(pal.size());
    for (int k = 0; k < pal.size(); ++k) setRole[k] = role_of(pal.typeMask[k]);
    const Projector proj(gVO, gFO, M.origEdges, &ms, pal, S, setRole);
    verify_projector_bvh(proj, 30);

    std::unordered_map<int, std::vector<int>> junctionVerts;  // junction id -> fine vertices
    for (const MatStructEntry & st : ms.structs)
        if (st.type == 3) junctionVerts[st.id] = st.elements;

    double snapMax[3] = {}, snapSum[3] = {};
    int64_t snapN[3] = {}, noTarget = 0, unmapped = 0;
    for (int64_t i = 0; i < Vs; ++i) {
        if (M.fineFace[i] < 0) { ++unmapped; continue; }
        const int k = setId[i];
        const uint8_t role = setRole[k];
        const Vector3d p = M.V.row(i).transpose();
        Vector3d q;
        if (role == RELAX_JUNCTION) {
            // Nearest fine vertex of the vertex's junction(s).
            int best = -1;
            double bd = std::numeric_limits<double>::infinity();
            for (int32_t a = pal.offsets[k]; a < pal.offsets[k + 1]; ++a) {
                auto it = junctionVerts.find(pal.ids[a]);
                if (it == junctionVerts.end()) continue;
                for (int v : it->second) {
                    const double d = (P3(gVO, v) - p).squaredNorm();
                    if (d < bd) { bd = d; best = v; }
                }
            }
            if (best < 0 || proj.vertFaces[best].empty()) { ++noTarget; continue; }
            const int f = proj.vertFaces[best][0];
            M.fineFace[i] = f;
            M.fineBary.row(i).setZero();
            for (int c = 0; c < 3; ++c) if (gFO(f, c) == best) M.fineBary(i, c) = 1.0;
            q = P3(gVO, best);
        } else {
            if (proj.targets[k].empty()) { ++noTarget; continue; }
            const ProjResult r = proj.project(k, p, M.fineFace[i], -1);
            M.fineFace[i] = r.face;
            M.fineBary.row(i) = r.bary.transpose();
            if (role == RELAX_CURVE) { M.carrierType[i] = SUBDIV_CARRIER_EDGE; M.carrierIndex[i] = r.edge; }
            q = r.pos;
        }
        M.V.row(i) = q.transpose();
        const double d = (q - p).norm() / diag;
        snapMax[role] = std::max(snapMax[role], d);
        snapSum[role] += d;
        ++snapN[role];
    }
    const char * roleName[3] = { "sheet", "curve", "junction" };
    for (int r = 0; r < 3; ++r)
        fprintf(stderr, "[coarse_subdiv_relax] seed snap onto own structure, %s: %lld vertices, move max %.3g mean %.3g (x diag)\n",
                roleName[r], (long long)snapN[r], snapMax[r], snapN[r] ? snapSum[r] / snapN[r] : 0.0);
    fprintf(stderr, "[coarse_subdiv_relax] %lld vertices not mapped to fine, %lld without a structure target (kept fixed)\n",
            (long long)unmapped, (long long)noTarget);

    // 4. Relax with the existing graph and solver.
    RelaxGraph G = build_relax_graph(M, pal, setId, &ms);
    if (cfg.jointPass || isExplicit) {
        // Symmetric graph: every row holds all its mesh neighbours (the plain
        // two-way adjacency), roles kept. Curve vertices then also feel the
        // sheet vertices next to them; they still only slide along their curve.
        RelaxGraph J = build_relax_graph(M, pal, setId, nullptr);
        J.role = G.role;
        G = std::move(J);
    }
    const MatrixXd Vseed = M.V;
    const MeshQuality q0 = subdiv_mesh_quality(M.V, M.F);
    RelaxOptions opt;
    opt.curveAnchorTol = cfg.curveAnchorTol;
    opt.maxIter = cfg.maxIter;
    opt.localProjection = cfg.localProjection;
    opt.jointPass = cfg.jointPass;
    opt.jointSolve = cfg.jointSolve;
    if (cfg.jointSolve && method != "solve_project")
        fprintf(stderr, "[coarse_subdiv_relax] WARNING: the joint solve is solve_project only; ignored by %s\n", method.c_str());
    if (cfg.jointPass && method == "solve_project")
        fprintf(stderr, "[coarse_subdiv_relax] WARNING: the joint pass is Newton only; solve_project keeps two passes "
                        "(on the symmetric graph)\n");
    if (cfg.perCoarseFace) {
        // C.S carriers are those of the coarse mesh (M's were overwritten above).
        opt.holdFixed.resize(Vs);
        int64_t held = 0;
        for (int64_t i = 0; i < Vs; ++i) {
            opt.holdFixed[i] = C.S.carrierType[i] != SUBDIV_CARRIER_FACE;
            held += opt.holdFixed[i];
        }
        fprintf(stderr, "[coarse_subdiv_relax] per coarse face: %lld vertices on coarse vertices / edges held at their seeds\n",
                (long long)held);
    }
    MatrixXd foldRef;  // empty unless built below
    if (cfg.noNewFolds || isExplicit) {
        // Reference normal of each subdivided triangle: its coarse face's normal,
        // signed to agree with the majority of that coarse face's seed triangles
        // (some coarse faces are oriented against the fine sheet under them, which
        // is not a fold). Folded = disagrees with that majority.
        const int64_t nF = M.F.rows();
        const int FC = (int)cmc.Fout.rows();
        auto n_of = [&](const MatrixXd & P, const Vector3i & t) {
            const Vector3d a = P.row(t(0)).transpose(), b = P.row(t(1)).transpose(), c = P.row(t(2)).transpose();
            return Vector3d((b - a).cross(c - a));
        };
        std::vector<Vector3d> cn(FC, Vector3d::Zero());
        std::vector<int64_t> vote(FC, 0);
        for (int64_t f = 0; f < nF; ++f) {
            const int cf = C.S.faceOrig[f];
            if (cf < 0 || cf >= FC) relax_fail("subdivided face without a coarse face");
            const Vector3d nc = n_of(C.S.V, M.F.row(f).transpose());
            cn[cf] += nc;
            vote[cf] += n_of(M.V, M.F.row(f).transpose()).dot(nc) >= 0 ? 1 : -1;
        }
        foldRef.resize(nF, 3);
        for (int64_t f = 0; f < nF; ++f) {
            const int cf = C.S.faceOrig[f];
            foldRef.row(f) = ((vote[cf] >= 0 ? 1.0 : -1.0) * cn[cf].normalized()).transpose();
        }
        opt.foldRef = foldRef;
        if (cfg.noNewFolds && method == "solve_project")
            fprintf(stderr, "[coarse_subdiv_relax] WARNING: the no-new-folds rule is ignored by solve_project\n");
        if (!cfg.noNewFolds) opt.foldRef.resize(0, 3);  // explicit: built for logging only
    }
    if (!graphDir.empty()) {
        std::vector<uint8_t> fixed(Vs);
        for (int64_t i = 0; i < Vs; ++i)
            fixed[i] = G.role[i] == RELAX_JUNCTION || (!opt.holdFixed.empty() && opt.holdFixed[i]);
        export_relax_graph_ply(graphDir, graphStem, G, M.V, pal, S, setId, ms, fixed);
    }
    RelaxReport R;
    if (isExplicit) {
        ExplicitRelaxOptions eo;
        eo.lambda = cfg.explicitLambda;
        eo.maxIter = cfg.explicitMaxIter;
        eo.tol = cfg.explicitTol;
        eo.localProjection = !cfg.explicitGlobalProj;
        eo.holdFixed = opt.holdFixed;
        eo.noNewFolds = cfg.noNewFolds;
        eo.foldRef = foldRef;
        eo.snapshotPrefix = objPath.substr(0, objPath.size() - 4) + "_";
        R = subdiv_relax_explicit(M, gVO, gFO, &ms, pal, setId, G, eo);
    } else {
        R = method == "newton" ? subdiv_relax(M, gVO, gFO, &ms, pal, setId, G, opt)
                               : subdiv_relax_solve_project(M, gVO, gFO, &ms, pal, setId, G, opt);
    }
    const MeshQuality q1 = subdiv_mesh_quality(M.V, M.F, &Vseed);
    fprintf(stderr,
        "[coarse_subdiv_relax] relaxation (%s), before -> after: edge CV %.4f -> %.4f | min angle %.3f -> %.3f, "
        "p1 %.3f -> %.3f, p5 %.3f -> %.3f, median %.3f -> %.3f deg | degenerate %lld -> %lld | "
        "flipped vs seed %lld | move max %.3g mean %.3g (x diag)\n",
        (method + (cfg.perCoarseFace ? ", per coarse face" : "") + (cfg.noNewFolds ? ", no new folds" : "")
         + (cfg.localProjection ? ", local projection" : "") + (cfg.jointPass ? ", joint pass" : "")
         + (cfg.jointSolve ? ", joint solve" : "")).c_str(), q0.edgeCV, q1.edgeCV, q0.minAngle, q1.minAngle, q0.p1, q1.p1, q0.p5, q1.p5, q0.median, q1.median,
        (long long)q0.degenerate, (long long)q1.degenerate, (long long)q1.flippedVsRef, R.maxMove, R.meanMove);
    if (R.seedOffStructure || R.fixedMoved || R.posMismatch || R.badBary || R.offStructure)
        relax_fail("relaxation consistency checks failed");

    // 5. Export.
    if (Vs > maxObjVerts) {
        fprintf(stderr, "[coarse_subdiv_relax] skipping %s: %lld vertices > %lld\n",
                objPath.c_str(), (long long)Vs, (long long)maxObjVerts);
        return;
    }
    if (!igl::writeOBJ(subdiv_long_path(objPath), M.V, M.F))
        fprintf(stderr, "[coarse_subdiv_relax] writeOBJ failed: %s\n", objPath.c_str());
    else
        fprintf(stderr, "[coarse_subdiv_relax] relaxed subdivided coarse mesh on the fine MAT -> %s\n", objPath.c_str());
}
