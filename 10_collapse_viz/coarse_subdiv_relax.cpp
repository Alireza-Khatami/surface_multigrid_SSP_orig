#include "coarse_subdiv_relax.h"
#include "collapse_structure_tracker/simp_viz_tracker.h"
#include "subdiv_sample_tracker/subdiv_relax.h"
#include "subdiv_sample_tracker/subdiv_relax_projector.h"
#include "subdiv_sample_tracker/subdiv_struct_ids.h"

#include <igl/writeOBJ.h>
#include <Eigen/Dense>

#include <algorithm>
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

void coarse_subdiv_relax_export(const CoarseMeshCompaction & cmc, const CoarseSubdivC2F & C,
                                const MatStruct & ms, const std::string & method, double curveAnchorTol,
                                int64_t maxIter,
                                int64_t maxObjVerts, const std::string & objPath)
{
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
    const RelaxGraph G = build_relax_graph(M, pal, setId, &ms);
    const MatrixXd Vseed = M.V;
    const MeshQuality q0 = subdiv_mesh_quality(M.V, M.F);
    RelaxOptions opt;
    opt.curveAnchorTol = curveAnchorTol;
    opt.maxIter = maxIter;
    if (method != "newton" && method != "solve_project") relax_fail("unknown relax method " + method);
    const RelaxReport R = method == "newton" ? subdiv_relax(M, gVO, gFO, &ms, pal, setId, G, opt)
                                             : subdiv_relax_solve_project(M, gVO, gFO, &ms, pal, setId, G, opt);
    const MeshQuality q1 = subdiv_mesh_quality(M.V, M.F, &Vseed);
    fprintf(stderr,
        "[coarse_subdiv_relax] relaxation (%s), before -> after: edge CV %.4f -> %.4f | min angle %.3f -> %.3f, "
        "p1 %.3f -> %.3f, p5 %.3f -> %.3f, median %.3f -> %.3f deg | degenerate %lld -> %lld | "
        "flipped vs seed %lld | move max %.3g mean %.3g (x diag)\n",
        method.c_str(), q0.edgeCV, q1.edgeCV, q0.minAngle, q1.minAngle, q0.p1, q1.p1, q0.p5, q1.p5, q0.median, q1.median,
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
