#include "subdiv_struct_ids.h"

#include <algorithm>
#include <cstdio>
#include <map>

using namespace Eigen;

static void sort_unique(std::vector<int> & v)
{
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

void build_struct_sets(const SubdivMesh & M,
                       const MatrixXi & FO,
                       const MatStruct * ms,
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
            const int type = ms ? ms->structType.at(id) : -1;
            if (type >= 0 && type <= 3) mask |= (uint8_t)(1u << type);
        }
        palette.ids.insert(palette.ids.end(), ids.begin(), ids.end());
        palette.offsets.push_back((int32_t)palette.ids.size());
        palette.typeMask.push_back(mask);
        interned.emplace(ids, k);
        return k;
    };

    setId.assign(Vs, -1);
    if (!ms) {
        const int32_t empty = intern({});
        std::fill(setId.begin(), setId.end(), empty);
        return;
    }

    // Faces incident to each original edge.
    std::vector<std::vector<int>> edgeFaces(M.origEdges.rows());
    for (int f = 0; f < FO.rows(); ++f)
        for (int c = 0; c < 3; ++c)
            edgeFaces[subdiv_find_edge(M.origEdges, FO(f, c), FO(f, (c + 1) % 3))].push_back(f);

    std::vector<int32_t> vCache(ms->vertexIds.size(), -1);
    std::vector<int32_t> eCache(M.origEdges.rows(), -1);
    std::vector<int32_t> fCache(FO.rows(), -1);

    for (size_t v = 0; v < Vs; ++v) {
        const int idx = M.carrierIndex[v];
        switch (M.carrierType[v]) {
        case SUBDIV_CARRIER_VERTEX:
            if (vCache[idx] < 0)
                vCache[idx] = intern(std::vector<int>(ms->vertexIds[idx].begin(), ms->vertexIds[idx].end()));
            setId[v] = vCache[idx];
            break;
        case SUBDIV_CARRIER_EDGE:
            if (eCache[idx] < 0) {
                std::vector<int> ids;
                auto it = ms->edgeIds.find(matstruct_edge_key(M.origEdges(idx, 0), M.origEdges(idx, 1)));
                if (it != ms->edgeIds.end()) ids = it->second;
                for (int f : edgeFaces[idx])
                    ids.insert(ids.end(), ms->faceIds[f].begin(), ms->faceIds[f].end());
                sort_unique(ids);
                eCache[idx] = intern(ids);
            }
            setId[v] = eCache[idx];
            break;
        case SUBDIV_CARRIER_FACE:
            if (fCache[idx] < 0) fCache[idx] = intern(ms->faceIds[idx]);
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
