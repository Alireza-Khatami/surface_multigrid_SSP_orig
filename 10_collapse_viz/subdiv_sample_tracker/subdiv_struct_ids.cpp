#include "subdiv_struct_ids.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <stdexcept>

using namespace Eigen;

static void fail(const std::string & msg)
{
    throw std::runtime_error("[subdiv_struct_ids] " + msg);
}

static void sort_unique(std::vector<int> & v)
{
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

MatStructElements load_matstruct_elements(const std::string & path,
                                          const MatrixXd & VO,
                                          const MatrixXi & FO,
                                          const MatrixXi & origEdges)
{
    std::ifstream f(path);
    if (!f) fail("cannot open " + path);

    int nv, ne, nf;
    if (!(f >> nv >> ne >> nf)) fail("bad header in " + path);
    if (nv != VO.rows() || nf != FO.rows())
        fail("counts do not match the mesh: .ma has " + std::to_string(nv) + " vertices / "
             + std::to_string(nf) + " faces, mesh has " + std::to_string(VO.rows()) + " / "
             + std::to_string(FO.rows()));

    double maxPosDiff = 0.0;
    for (int i = 0; i < nv; ++i) {
        char ch; double x, y, z, r;
        if (!(f >> ch >> x >> y >> z >> r) || ch != 'v') fail("bad vertex line " + std::to_string(i));
        maxPosDiff = std::max({ maxPosDiff, std::abs(x - VO(i, 0)), std::abs(y - VO(i, 1)),
                                std::abs(z - VO(i, 2)) });
    }

    std::vector<std::array<int, 2>> maEdges(ne);
    for (int i = 0; i < ne; ++i) {
        char ch;
        if (!(f >> ch >> maEdges[i][0] >> maEdges[i][1]) || ch != 'e')
            fail("bad edge line " + std::to_string(i));
    }

    for (int i = 0; i < nf; ++i) {
        char ch; std::array<int, 3> a;
        if (!(f >> ch >> a[0] >> a[1] >> a[2]) || ch != 'f') fail("bad face line " + std::to_string(i));
        std::array<int, 3> b = { FO(i, 0), FO(i, 1), FO(i, 2) };
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (a != b) fail(".ma face " + std::to_string(i) + " has different corners than mesh face "
                         + std::to_string(i));
    }

    MatStructElements E;
    E.faceIds.assign(nf, {});
    E.edgeIds.assign(origEdges.rows(), {});
    E.vertexIds.assign(nv, {});

    int numStructs = 0;
    if (!(f >> numStructs)) {
        fprintf(stderr, "[subdiv_struct_ids] no struct section in %s\n", path.c_str());
        return E;
    }

    int nUnknownType = 0;
    for (int s = 0; s < numStructs; ++s) {
        int sid, type, count;
        if (!(f >> sid >> type >> count)) fail("truncated struct header " + std::to_string(s));
        if (E.structType.count(sid)) fail("duplicate struct id " + std::to_string(sid));
        E.structType[sid] = type;

        for (int j = 0; j < count; ++j) {
            int el;
            if (!(f >> el)) fail("truncated struct " + std::to_string(sid));
            switch (type) {
            case 0:  // SHEET: face index
                if (el < 0 || el >= nf) fail("sheet face out of range in struct " + std::to_string(sid));
                E.faceIds[el].push_back(sid);
                for (int c = 0; c < 3; ++c) E.vertexIds[FO(el, c)].push_back(sid);
                break;
            case 1:  // SEAM: edge index
            case 2: {  // BOUNDARY: edge index
                if (el < 0 || el >= ne) fail("edge out of range in struct " + std::to_string(sid));
                const int a = maEdges[el][0], b = maEdges[el][1];
                if (a < 0 || a >= nv || b < 0 || b >= nv) fail("edge vertex out of range");
                E.vertexIds[a].push_back(sid);
                E.vertexIds[b].push_back(sid);
                const int e = subdiv_find_edge(origEdges, a, b);
                if (e < 0) ++E.nEdgesWithoutFace;  // e.g. a stale edge: no subdivided vertex on it
                else       E.edgeIds[e].push_back(sid);
                break;
            }
            case 3:  // JUNCTION: vertex index
                if (el < 0 || el >= nv) fail("junction vertex out of range in struct " + std::to_string(sid));
                E.vertexIds[el].push_back(sid);
                break;
            default:
                ++nUnknownType;
            }
        }
    }

    for (auto & v : E.faceIds)   sort_unique(v);
    for (auto & v : E.edgeIds)   sort_unique(v);
    for (auto & v : E.vertexIds) sort_unique(v);

    int nFacesNoSheet = 0;
    for (const auto & v : E.faceIds) nFacesNoSheet += v.empty();

    fprintf(stderr,
        "[subdiv_struct_ids] %s: %d structs, max |.ma pos - mesh pos| = %.3g, "
        "%d/%d faces in no sheet, %d seam/boundary edges on no face, %d elements of unknown type\n",
        path.c_str(), numStructs, maxPosDiff, nFacesNoSheet, nf, E.nEdgesWithoutFace, nUnknownType);
    return E;
}

void build_struct_sets(const SubdivMesh & M,
                       const MatrixXi & FO,
                       const MatStructElements * elems,
                       StructPalette & palette,
                       std::vector<int32_t> & setId)
{
    palette = StructPalette();
    const size_t Vs = M.carrierType.size();

    std::map<std::vector<int>, int32_t> interned;
    auto intern = [&](const std::vector<int> & ids) -> int32_t {
        auto it = interned.find(ids);
        if (it != interned.end()) return it->second;
        const int32_t k = (int32_t)palette.typeMask.size();
        uint8_t mask = 0;
        for (int id : ids) {
            const int type = elems ? elems->structType.at(id) : -1;
            if (type >= 0 && type <= 3) mask |= (uint8_t)(1u << type);
        }
        palette.ids.insert(palette.ids.end(), ids.begin(), ids.end());
        palette.offsets.push_back((int32_t)palette.ids.size());
        palette.typeMask.push_back(mask);
        interned.emplace(ids, k);
        return k;
    };

    setId.assign(Vs, -1);
    if (!elems) {
        const int32_t empty = intern({});
        std::fill(setId.begin(), setId.end(), empty);
        return;
    }

    // Faces incident to each original edge.
    std::vector<std::vector<int>> edgeFaces(M.origEdges.rows());
    for (int f = 0; f < FO.rows(); ++f)
        for (int c = 0; c < 3; ++c)
            edgeFaces[subdiv_find_edge(M.origEdges, FO(f, c), FO(f, (c + 1) % 3))].push_back(f);

    std::vector<int32_t> vCache(elems->vertexIds.size(), -1);
    std::vector<int32_t> eCache(M.origEdges.rows(), -1);
    std::vector<int32_t> fCache(FO.rows(), -1);

    for (size_t v = 0; v < Vs; ++v) {
        const int idx = M.carrierIndex[v];
        switch (M.carrierType[v]) {
        case SUBDIV_CARRIER_VERTEX:
            if (vCache[idx] < 0) vCache[idx] = intern(elems->vertexIds[idx]);
            setId[v] = vCache[idx];
            break;
        case SUBDIV_CARRIER_EDGE:
            if (eCache[idx] < 0) {
                std::vector<int> ids = elems->edgeIds[idx];
                for (int f : edgeFaces[idx])
                    ids.insert(ids.end(), elems->faceIds[f].begin(), elems->faceIds[f].end());
                sort_unique(ids);
                eCache[idx] = intern(ids);
            }
            setId[v] = eCache[idx];
            break;
        case SUBDIV_CARRIER_FACE:
            if (fCache[idx] < 0) fCache[idx] = intern(elems->faceIds[idx]);
            setId[v] = fCache[idx];
            break;
        }
    }

    size_t maxSet = 0;
    for (int k = 0; k < palette.size(); ++k)
        maxSet = std::max(maxSet, (size_t)(palette.offsets[k + 1] - palette.offsets[k]));
    fprintf(stderr, "[subdiv_struct_ids] palette: %d distinct sets, %zu ids total, largest set %zu\n",
            palette.size(), palette.ids.size(), maxSet);
}
