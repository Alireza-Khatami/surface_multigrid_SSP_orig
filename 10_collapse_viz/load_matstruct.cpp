#include "load_matstruct.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <unordered_set>
#include <utility>

uint64_t matstruct_edge_key(int a, int b)
{
    if (a > b) std::swap(a, b);
    return (uint64_t)(uint32_t)a << 32 | (uint32_t)b;
}

static void sort_unique(std::vector<int> & v)
{
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

bool load_matstruct(const std::string & fname,
                    const Eigen::MatrixXd & VO,
                    const Eigen::MatrixXi & FO,
                    MatStruct & out,
                    std::string * err)
{
    out = MatStruct();
    auto fail = [&](const std::string & msg) {
        if (err) *err = "[load_matstruct] " + fname + ": " + msg;
        out = MatStruct();
        return false;
    };

    std::ifstream f(fname);
    if (!f) return fail("cannot open");

    int nv, ne, nf;
    if (!(f >> nv >> ne >> nf) || nv <= 0 || ne < 0 || nf < 0) return fail("bad header");
    if (nv != VO.rows() || nf != FO.rows())
        return fail("counts do not match the mesh: .ma has " + std::to_string(nv) + " vertices / "
                    + std::to_string(nf) + " faces, mesh has " + std::to_string(VO.rows()) + " / "
                    + std::to_string(FO.rows()));
    out.nv = nv; out.ne = ne; out.nf = nf;

    for (int i = 0; i < nv; ++i) {
        char ch; double x, y, z, r;
        if (!(f >> ch >> x >> y >> z >> r) || ch != 'v') return fail("bad vertex line " + std::to_string(i));
        out.maxPosDiff = std::max({ out.maxPosDiff, std::abs(x - VO(i, 0)), std::abs(y - VO(i, 1)),
                                    std::abs(z - VO(i, 2)) });
    }

    out.maEdges.resize(ne);
    for (int i = 0; i < ne; ++i) {
        char ch;
        if (!(f >> ch >> out.maEdges[i][0] >> out.maEdges[i][1]) || ch != 'e')
            return fail("bad edge line " + std::to_string(i));
        for (int k = 0; k < 2; ++k)
            if (out.maEdges[i][k] < 0 || out.maEdges[i][k] >= nv)
                return fail("edge " + std::to_string(i) + " has an out-of-range vertex");
    }

    for (int i = 0; i < nf; ++i) {
        char ch; std::array<int, 3> a;
        if (!(f >> ch >> a[0] >> a[1] >> a[2]) || ch != 'f') return fail("bad face line " + std::to_string(i));
        std::array<int, 3> b = { FO(i, 0), FO(i, 1), FO(i, 2) };
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (a != b) return fail("face " + std::to_string(i) + " has different corners than mesh face "
                                + std::to_string(i));
    }

    out.vertexIds.assign(nv, {});
    out.faceIds.assign(nf, {});

    int numStructs = 0;
    if (!(f >> numStructs)) {
        fprintf(stderr, "[load_matstruct] no struct section in %s\n", fname.c_str());
        return true;  // geometry only: every vertex has an empty set
    }
    out.hasStructSection = true;

    std::unordered_set<uint64_t> faceEdges;
    for (int fi = 0; fi < nf; ++fi)
        for (int c = 0; c < 3; ++c)
            faceEdges.insert(matstruct_edge_key(FO(fi, c), FO(fi, (c + 1) % 3)));

    out.structs.resize(numStructs);
    for (int s = 0; s < numStructs; ++s) {
        MatStructEntry & st = out.structs[s];
        int count;
        if (!(f >> st.id >> st.type >> count) || count < 0) return fail("truncated struct header " + std::to_string(s));
        if (out.structType.count(st.id)) return fail("duplicate struct id " + std::to_string(st.id));
        out.structType[st.id] = st.type;

        st.elements.resize(count);
        for (int j = 0; j < count; ++j) {
            int el;
            if (!(f >> el)) return fail("truncated struct " + std::to_string(st.id));
            st.elements[j] = el;
            switch (st.type) {
            case 0:  // SHEET: face index
                if (el < 0 || el >= nf) return fail("sheet face out of range in struct " + std::to_string(st.id));
                out.faceIds[el].push_back(st.id);
                for (int c = 0; c < 3; ++c) out.vertexIds[FO(el, c)].insert(st.id);
                break;
            case 1:    // SEAM: edge index
            case 2: {  // BOUNDARY: edge index
                if (el < 0 || el >= ne) return fail("edge out of range in struct " + std::to_string(st.id));
                const int a = out.maEdges[el][0], b = out.maEdges[el][1];
                out.vertexIds[a].insert(st.id);
                out.vertexIds[b].insert(st.id);
                const uint64_t k = matstruct_edge_key(a, b);
                out.edgeIds[k].push_back(st.id);
                if (!faceEdges.count(k)) ++out.nEdgesWithoutFace;
                break;
            }
            case 3:  // JUNCTION: vertex index
                if (el < 0 || el >= nv) return fail("junction vertex out of range in struct " + std::to_string(st.id));
                out.vertexIds[el].insert(st.id);
                break;
            default:
                break;  // unknown type: ignored
            }
        }
    }

    for (auto & v : out.faceIds) sort_unique(v);
    for (auto & kv : out.edgeIds) sort_unique(kv.second);

    int nFacesNoSheet = 0;
    for (const auto & v : out.faceIds) nFacesNoSheet += v.empty();
    fprintf(stderr,
        "[load_matstruct] %s: %d structs, max |.ma pos - mesh pos| = %.3g, "
        "%d/%d faces in no sheet, %d seam/boundary edges on no face\n",
        fname.c_str(), numStructs, out.maxPosDiff, nFacesNoSheet, nf, out.nEdgesWithoutFace);
    return true;
}
