#include "subdiv_mesh.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

using namespace Eigen;

static inline uint64_t edge_key(int a, int b)
{
    if (a > b) std::swap(a, b);
    return (uint64_t)(uint32_t)a << 32 | (uint32_t)b;
}

static void fail(const std::string & msg)
{
    throw std::runtime_error("[subdiv_mesh] " + msg);
}

MatrixXi subdiv_unique_edges(const MatrixXi & F)
{
    std::vector<uint64_t> keys;
    keys.reserve((size_t)F.rows() * 3);
    for (int f = 0; f < F.rows(); ++f)
        for (int c = 0; c < 3; ++c)
            keys.push_back(edge_key(F(f, c), F(f, (c + 1) % 3)));
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    MatrixXi E((Index)keys.size(), 2);
    for (size_t i = 0; i < keys.size(); ++i) {
        E((Index)i, 0) = (int)(keys[i] >> 32);
        E((Index)i, 1) = (int)(keys[i] & 0xffffffffu);
    }
    return E;
}

int subdiv_find_edge(const MatrixXi & edges, int a, int b)
{
    if (a > b) std::swap(a, b);
    int lo = 0, hi = (int)edges.rows();
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        const int ea = edges(mid, 0), eb = edges(mid, 1);
        if (ea < a || (ea == a && eb < b)) lo = mid + 1;
        else                               hi = mid;
    }
    return (lo < edges.rows() && edges(lo, 0) == a && edges(lo, 1) == b) ? lo : -1;
}

namespace {

// Growable per-vertex carrier storage used while building.
struct Carriers {
    std::vector<uint8_t> type;
    std::vector<int32_t> index;
    std::vector<double>  coord;  // 3 per vertex

    void push(uint8_t t, int32_t i, double c0, double c1, double c2)
    {
        type.push_back(t);
        index.push_back(i);
        coord.push_back(c0); coord.push_back(c1); coord.push_back(c2);
    }
};

int corner_of(const MatrixXi & FO, int f, int v)
{
    for (int c = 0; c < 3; ++c)
        if (FO(f, c) == v) return c;
    return -1;
}

// Barycentric coordinates of vertex v in FO.row(f) corner order. Throws if v's
// carrier is not contained in the closure of face f.
Vector3d to_face_bary(const Carriers & C, const MatrixXi & origEdges,
                      const MatrixXi & FO, int v, int f)
{
    Vector3d b = Vector3d::Zero();
    const int idx = C.index[v];
    switch (C.type[v]) {
    case SUBDIV_CARRIER_VERTEX: {
        const int c = corner_of(FO, f, idx);
        if (c < 0) fail("vertex " + std::to_string(v) + " (orig vertex " + std::to_string(idx)
                        + ") is not a corner of face " + std::to_string(f));
        b(c) = 1.0;
        break;
    }
    case SUBDIV_CARRIER_EDGE: {
        const int c0 = corner_of(FO, f, origEdges(idx, 0));
        const int c1 = corner_of(FO, f, origEdges(idx, 1));
        if (c0 < 0 || c1 < 0)
            fail("vertex " + std::to_string(v) + " (orig edge " + std::to_string(idx)
                 + ") is not on face " + std::to_string(f));
        b(c0) = C.coord[3 * (size_t)v + 0];
        b(c1) = C.coord[3 * (size_t)v + 1];
        break;
    }
    case SUBDIV_CARRIER_FACE:
        if (idx != f)
            fail("vertex " + std::to_string(v) + " lies inside face " + std::to_string(idx)
                 + ", not face " + std::to_string(f));
        b << C.coord[3 * (size_t)v + 0], C.coord[3 * (size_t)v + 1], C.coord[3 * (size_t)v + 2];
        break;
    default:
        fail("bad carrier type");
    }
    return b;
}

// Carrier of the point with barycentric b in face fo (b has dyadic entries, so
// the zero tests and the sum check are exact).
void classify(const Vector3d & b, int fo, const MatrixXi & FO, const MatrixXi & origEdges,
              uint8_t & type, int32_t & index, Vector3d & coord)
{
    if (b.sum() != 1.0 || b.minCoeff() < 0.0)
        fail("non-exact barycentric in face " + std::to_string(fo));

    int nz[3], n = 0;
    for (int c = 0; c < 3; ++c)
        if (b(c) != 0.0) nz[n++] = c;

    if (n == 3) {
        type = SUBDIV_CARRIER_FACE;
        index = fo;
        coord = b;
    } else if (n == 2) {
        const int a = FO(fo, nz[0]), bb = FO(fo, nz[1]);
        const int e = subdiv_find_edge(origEdges, a, bb);
        if (e < 0) fail("edge not found in face " + std::to_string(fo));
        type = SUBDIV_CARRIER_EDGE;
        index = e;
        // Weights in origEdges(e,0), origEdges(e,1) order.
        const double wa = b(nz[0]), wb = b(nz[1]);
        coord = (origEdges(e, 0) == a) ? Vector3d(wa, wb, 0.0) : Vector3d(wb, wa, 0.0);
    } else {
        fail("midpoint collapsed onto an original vertex in face " + std::to_string(fo));
    }
}

} // namespace

SubdivMesh build_subdiv_mesh(const MatrixXd & VO, const MatrixXi & FO, int64_t nTarget)
{
    const int nVO = (int)VO.rows();
    const int nFO = (int)FO.rows();

    for (int f = 0; f < nFO; ++f) {
        for (int c = 0; c < 3; ++c)
            if (FO(f, c) < 0 || FO(f, c) >= nVO)
                fail("face " + std::to_string(f) + " has out-of-range vertex");
        if (FO(f, 0) == FO(f, 1) || FO(f, 1) == FO(f, 2) || FO(f, 2) == FO(f, 0))
            fail("face " + std::to_string(f) + " has a repeated corner");
    }

    SubdivMesh M;
    M.nOrigVerts = nVO;
    M.levelVerts.push_back(nVO);
    M.origEdges  = subdiv_unique_edges(FO);

    Carriers C;
    C.type.reserve(nVO); C.index.reserve(nVO); C.coord.reserve(3 * (size_t)nVO);
    for (int v = 0; v < nVO; ++v)
        C.push(SUBDIV_CARRIER_VERTEX, v, 1.0, 0.0, 0.0);

    MatrixXi F = FO;
    std::vector<int32_t> faceOrig(nFO);
    for (int f = 0; f < nFO; ++f) faceOrig[f] = f;

    const int64_t kMaxIdx = std::numeric_limits<int32_t>::max();

    while ((int64_t)C.type.size() < nTarget && F.rows() > 0) {
        const int64_t Fs = F.rows();
        if (4 * Fs > kMaxIdx) fail("face count would overflow 32-bit indices");

        // Half-edge h = 3f + c is the edge F(f,c) -> F(f,(c+1)%3).
        std::vector<std::pair<uint64_t, int32_t>> he((size_t)(3 * Fs));
        for (int64_t f = 0; f < Fs; ++f)
            for (int c = 0; c < 3; ++c)
                he[(size_t)(3 * f + c)] = { edge_key(F((Index)f, c), F((Index)f, (c + 1) % 3)),
                                            (int32_t)(3 * f + c) };
        std::sort(he.begin(), he.end());

        // Midpoints are shared exactly as the original complex shares points:
        // a midpoint on an original edge is one vertex for every face around that
        // edge; a midpoint inside an original face belongs to that face alone.
        // (Distinct original faces on the same three vertices, which MAT meshes
        // contain, therefore get distinct interior vertices, while their shared
        // boundary stays shared.)
        std::vector<int32_t> mid((size_t)(3 * Fs), -1);
        std::vector<std::pair<int32_t, int32_t>> faceMid;  // (orig face, new vertex) within a group
        for (size_t i = 0; i < he.size();) {
            size_t j = i;
            while (j < he.size() && he[j].first == he[i].first) ++j;

            int32_t edgeVertex = -1;
            int32_t eIndex = -1; Vector3d eCoord;
            faceMid.clear();
            for (size_t k = i; k < j; ++k) {
                const int32_t h = he[k].second;
                const int f = h / 3, c = h % 3;
                const int u = F(f, c), w = F(f, (c + 1) % 3);
                const int fo = faceOrig[f];
                const Vector3d b = 0.5 * (to_face_bary(C, M.origEdges, FO, u, fo)
                                        + to_face_bary(C, M.origEdges, FO, w, fo));
                uint8_t type; int32_t index; Vector3d coord;
                classify(b, fo, FO, M.origEdges, type, index, coord);

                int32_t v = -1;
                if (type == SUBDIV_CARRIER_EDGE) {
                    if (!faceMid.empty()) fail("edge- and face-carried midpoints on one sub edge");
                    if (edgeVertex < 0) {
                        edgeVertex = (int32_t)C.type.size();
                        eIndex = index; eCoord = coord;
                        C.push(type, index, coord(0), coord(1), coord(2));
                    } else if (index != eIndex || coord != eCoord) {
                        fail("faces around an original edge disagree on a midpoint (level "
                             + std::to_string(M.nLevels) + ")");
                    }
                    v = edgeVertex;
                } else {
                    if (edgeVertex >= 0) fail("edge- and face-carried midpoints on one sub edge");
                    for (const auto & fm : faceMid)
                        if (fm.first == fo) { v = fm.second; break; }
                    if (v < 0) {
                        v = (int32_t)C.type.size();
                        faceMid.push_back({ fo, v });
                        C.push(type, index, coord(0), coord(1), coord(2));
                    }
                }
                if ((int64_t)C.type.size() >= kMaxIdx) fail("vertex count would overflow 32-bit indices");
                mid[(size_t)h] = v;
            }
            i = j;
        }

        MatrixXi F2((Index)(4 * Fs), 3);
        std::vector<int32_t> faceOrig2((size_t)(4 * Fs));
        for (int64_t f = 0; f < Fs; ++f) {
            const int a = F((Index)f, 0), b = F((Index)f, 1), c = F((Index)f, 2);
            const int m0 = mid[(size_t)(3 * f + 0)];  // a-b
            const int m1 = mid[(size_t)(3 * f + 1)];  // b-c
            const int m2 = mid[(size_t)(3 * f + 2)];  // c-a
            const Index r = (Index)(4 * f);
            F2.row(r + 0) << a,  m0, m2;
            F2.row(r + 1) << m0, b,  m1;
            F2.row(r + 2) << m2, m1, c;
            F2.row(r + 3) << m0, m1, m2;
            for (int k = 0; k < 4; ++k) faceOrig2[(size_t)(r + k)] = faceOrig[(size_t)f];
        }
        F.swap(F2);
        faceOrig.swap(faceOrig2);
        ++M.nLevels;
        M.levelVerts.push_back((int64_t)C.type.size());
    }

    // ---- fine-mesh face + barycentric per vertex ----
    std::vector<int32_t> vtxFace(nVO, -1);
    for (int f = 0; f < nFO; ++f)
        for (int c = 0; c < 3; ++c)
            if (vtxFace[FO(f, c)] < 0) vtxFace[FO(f, c)] = f;

    std::vector<int32_t> edgeFace(M.origEdges.rows(), -1);
    for (int f = 0; f < nFO; ++f)
        for (int c = 0; c < 3; ++c) {
            const int e = subdiv_find_edge(M.origEdges, FO(f, c), FO(f, (c + 1) % 3));
            if (edgeFace[e] < 0) edgeFace[e] = f;
        }

    const Index Vs = (Index)C.type.size();
    M.fineFace.assign((size_t)Vs, -1);
    M.fineBary = MatrixXd::Zero(Vs, 3);
    M.V.resize(Vs, 3);
    for (Index v = 0; v < Vs; ++v) {
        int f = -1;
        switch (C.type[v]) {
        case SUBDIV_CARRIER_VERTEX: f = vtxFace[C.index[v]];  break;
        case SUBDIV_CARRIER_EDGE:   f = edgeFace[C.index[v]]; break;
        case SUBDIV_CARRIER_FACE:   f = C.index[v];           break;
        }
        M.fineFace[v] = f;
        if (f < 0) {  // original vertex referenced by no face
            M.V.row(v) = VO.row(C.index[v]).leftCols(3);
            continue;
        }
        const Vector3d b = to_face_bary(C, M.origEdges, FO, (int)v, f);
        M.fineBary.row(v) = b.transpose();
        M.V.row(v) = b(0) * VO.row(FO(f, 0)).leftCols(3)
                   + b(1) * VO.row(FO(f, 1)).leftCols(3)
                   + b(2) * VO.row(FO(f, 2)).leftCols(3);
    }

    M.carrierCoord = Map<const Matrix<double, Dynamic, 3, RowMajor>>(C.coord.data(), Vs, 3);
    M.carrierType  = std::move(C.type);
    M.carrierIndex = std::move(C.index);
    M.F.swap(F);
    M.faceOrig.swap(faceOrig);

    fprintf(stderr, "[subdiv_mesh] %d levels: |V| %d -> %lld, |F| %d -> %lld (target %lld)\n",
            M.nLevels, nVO, (long long)Vs, nFO, (long long)M.F.rows(), (long long)nTarget);
    return M;
}

MatrixXi subdiv_level_faces(const SubdivMesh & M, int level)
{
    if (level < 0 || level > M.nLevels) fail("level out of range");
    const int k = M.nLevels - level;
    const int64_t block = (int64_t)1 << (2 * k);   // 4^k final faces per level-l face
    const int64_t off1  = (block - 1) / 3;         // row of the all-child-1 descendant
    const int64_t n     = M.F.rows() / block;
    MatrixXi Fl((Index)n, 3);
    for (int64_t r = 0; r < n; ++r) {
        const int64_t base = r * block;
        Fl((Index)r, 0) = M.F((Index)base, 0);
        Fl((Index)r, 1) = M.F((Index)(base + off1), 1);
        Fl((Index)r, 2) = M.F((Index)(base + 2 * off1), 2);
    }
    return Fl;
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
