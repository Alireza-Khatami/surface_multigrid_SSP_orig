#include "subdiv_relax.h"
#include "subdiv_relax_projector.h"

#include <igl/parallel_for.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>

#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

using namespace Eigen;

namespace {

void fail(const std::string & msg) { throw std::runtime_error("[subdiv_relax] " + msg); }

double now_s()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}


} // namespace

using namespace subdiv_proj;

// ---------------------------------------------------------------- graph

RelaxGraph build_relax_graph(const SubdivMesh & M, const StructPalette & pal,
                             const std::vector<int32_t> & setId, const MatStruct * ms)
{
    const double t0 = now_s();
    RelaxGraph G;
    const int64_t Vs = M.V.rows();
    if ((int64_t)setId.size() != Vs) fail("setId size mismatch");

    const SetIds S = split_palette(pal, ms);
    std::vector<uint8_t> setRole(pal.size());
    for (int k = 0; k < pal.size(); ++k) setRole[k] = ms ? role_of(pal.typeMask[k]) : (uint8_t)RELAX_SHEET;

    G.role.resize(Vs);
    for (int64_t i = 0; i < Vs; ++i) {
        G.role[i] = setRole[setId[i]];
        ++G.nRole[G.role[i]];
        if (ms && G.role[i] == RELAX_SHEET && S.sheet[setId[i]].empty()) ++G.sheetWithoutSheetId;
    }

    // Unique undirected edges of M.F.
    std::vector<uint64_t> keys;
    keys.reserve((size_t)M.F.rows() * 3);
    for (Index f = 0; f < M.F.rows(); ++f)
        for (int c = 0; c < 3; ++c) {
            uint32_t a = (uint32_t)M.F(f, c), b = (uint32_t)M.F(f, (c + 1) % 3);
            if (a > b) std::swap(a, b);
            keys.push_back((uint64_t)a << 32 | b);
        }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    G.nUndirected = (int64_t)keys.size();

    // Rule cache over palette pairs.
    std::unordered_map<uint64_t, uint8_t> cache;  // bit0 share sheet, bit1 share curve
    auto share = [&](int ka, int kb) -> uint8_t {
        const uint64_t key = (uint64_t)(uint32_t)std::min(ka, kb) << 32 | (uint32_t)std::max(ka, kb);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
        uint8_t r = 0;
        if (intersects(S.sheet[ka], S.sheet[kb])) r |= 1;
        if (intersects(S.curve[ka], S.curve[kb])) r |= 2;
        if (S.sheet[ka].empty() && S.sheet[kb].empty()) r |= 4;
        cache.emplace(key, r);
        return r;
    };
    // keep(i <- j): does j pull on i?
    auto keep = [&](int64_t i, int64_t j) -> bool {
        const uint8_t ri = G.role[i], rj = G.role[j];
        if (!ms) return true;
        if (ri == RELAX_JUNCTION) return false;
        const uint8_t s = share(setId[i], setId[j]);
        if (ri == RELAX_CURVE) return rj != RELAX_SHEET && (s & 2);
        return (s & 1) || (s & 4);
    };
    auto kind_of = [&](int64_t i, int64_t j) -> int {
        const uint8_t ri = G.role[i], rj = G.role[j];
        if (ri == RELAX_SHEET) return rj == RELAX_SHEET ? 0 : rj == RELAX_CURVE ? 1 : 2;
        return rj == RELAX_CURVE ? 3 : 4;
    };

    std::vector<int64_t> deg(Vs + 1, 0);
    std::vector<uint8_t> dir(keys.size(), 0);  // bit0: a<-b kept, bit1: b<-a kept
    for (size_t e = 0; e < keys.size(); ++e) {
        const int64_t a = (int64_t)(keys[e] >> 32), b = (int64_t)(keys[e] & 0xffffffffu);
        uint8_t d = 0;
        if (keep(a, b)) { d |= 1; ++deg[a]; ++G.kind[kind_of(a, b)]; }
        if (keep(b, a)) { d |= 2; ++deg[b]; ++G.kind[kind_of(b, a)]; }
        if (G.role[a] == RELAX_SHEET && G.role[b] == RELAX_SHEET && d != 3) ++G.droppedSheetSheet;
        dir[e] = d;
    }
    G.rowOffs.assign(Vs + 1, 0);
    for (int64_t i = 0; i < Vs; ++i) G.rowOffs[i + 1] = G.rowOffs[i] + deg[i];
    G.cols.resize((size_t)G.rowOffs[Vs]);
    std::vector<int64_t> fill(G.rowOffs.begin(), G.rowOffs.end() - 1);
    for (size_t e = 0; e < keys.size(); ++e) {
        const int32_t a = (int32_t)(keys[e] >> 32), b = (int32_t)(keys[e] & 0xffffffffu);
        if (dir[e] & 1) G.cols[fill[a]++] = b;
        if (dir[e] & 2) G.cols[fill[b]++] = a;
    }
    for (int64_t i = 0; i < Vs; ++i) {
        std::sort(G.cols.begin() + G.rowOffs[i], G.cols.begin() + G.rowOffs[i + 1]);
        if (G.rowOffs[i + 1] == G.rowOffs[i]) {
            if (G.role[i] == RELAX_SHEET) ++G.isolatedSheet;
            else if (G.role[i] == RELAX_CURVE) ++G.isolatedCurve;
        }
    }

    fprintf(stderr,
        "[subdiv_relax] graph: %lld vertices (sheet %lld, curve %lld, junction %lld), %lld undirected edges, "
        "%lld directed entries (%.2f s)\n"
        "[subdiv_relax]   kept: S<-S %lld, S<-C %lld, S<-J %lld, C<-C %lld, C<-J %lld | "
        "dropped sheet-sheet %lld | sheet without sheet id %lld | isolated sheet %lld, curve %lld\n",
        (long long)Vs, (long long)G.nRole[0], (long long)G.nRole[1], (long long)G.nRole[2],
        (long long)G.nUndirected, (long long)G.cols.size(), now_s() - t0,
        (long long)G.kind[0], (long long)G.kind[1], (long long)G.kind[2], (long long)G.kind[3], (long long)G.kind[4],
        (long long)G.droppedSheetSheet, (long long)G.sheetWithoutSheetId,
        (long long)G.isolatedSheet, (long long)G.isolatedCurve);
    return G;
}

// .slg layout (little endian, arrays 8-byte aligned):
//   header (120 bytes): magic "SUBDIVG\0", u32 version = 1, u32 header_bytes,
//     u64 Vs, u64 nnz, u64 P, u64 n_palette_ids, u64 n_arrays = 8, u64 offsets[8]
//   0 V                 f64  Vs x 3   seed positions (before relaxation)
//   1 role              u8   Vs       0 sheet, 1 curve (seam/boundary), 2 junction
//   2 struct_set_id     i32  Vs
//   3 row_offsets       i64  Vs+1     row i = the vertices that pull on i
//   4 cols              i32  nnz      ascending within a row
//   5 palette_offsets   i32  P+1
//   6 palette_ids       i32  n_palette_ids
//   7 palette_type_mask u8   P
void save_relax_graph(const std::string & path, const MatrixXd & V, const RelaxGraph & G,
                      const StructPalette & pal, const std::vector<int32_t> & setId)
{
    const uint64_t Vs = (uint64_t)V.rows();
    std::vector<double> v(Vs * 3);
    for (uint64_t i = 0; i < Vs; ++i) for (int c = 0; c < 3; ++c) v[3 * i + c] = V((Index)i, c);

    struct Arr { const void * p; size_t bytes; };
    const int kArrays = 8;
    const Arr arrs[kArrays] = {
        { v.data(),               v.size() * 8 },
        { G.role.data(),          G.role.size() },
        { setId.data(),           setId.size() * 4 },
        { G.rowOffs.data(),       G.rowOffs.size() * 8 },
        { G.cols.data(),          G.cols.size() * 4 },
        { pal.offsets.data(),     pal.offsets.size() * 4 },
        { pal.ids.data(),         pal.ids.size() * 4 },
        { pal.typeMask.data(),    pal.typeMask.size() },
    };
    const uint32_t kHeader = 56 + 8 * kArrays;
    uint64_t offsets[kArrays], at = kHeader;
    for (int k = 0; k < kArrays; ++k) { at = (at + 7) / 8 * 8; offsets[k] = at; at += arrs[k].bytes; }

    std::ofstream f(subdiv_long_path(path), std::ios::binary);
    if (!f) fail("cannot write " + path);
    uint64_t pos = 0;
    auto raw = [&](const void * p, size_t n) { f.write((const char *)p, (std::streamsize)n); pos += n; };
    auto u32 = [&](uint32_t x) { raw(&x, 4); };
    auto u64 = [&](uint64_t x) { raw(&x, 8); };
    const char magic[8] = { 'S', 'U', 'B', 'D', 'I', 'V', 'G', '\0' };
    raw(magic, 8);
    u32(1); u32(kHeader);
    u64(Vs); u64((uint64_t)G.cols.size()); u64((uint64_t)pal.size()); u64((uint64_t)pal.ids.size());
    u64((uint64_t)kArrays);
    for (int k = 0; k < kArrays; ++k) u64(offsets[k]);
    if (pos != kHeader) fail("graph header size mismatch");
    static const char z[8] = {};
    for (int k = 0; k < kArrays; ++k) {
        const size_t pad = (size_t)((8 - pos % 8) % 8);
        if (pad) raw(z, pad);
        if (pos != offsets[k]) fail("graph array offset mismatch");
        raw(arrs[k].p, arrs[k].bytes);
    }
    f.close();
    if (!f) fail("write failed for " + path);
    fprintf(stderr, "[subdiv_relax] graph -> %s (%.1f MB)\n", path.c_str(), pos / 1048576.0);
}

// ---------------------------------------------------------------- relaxation

RelaxReport subdiv_relax(SubdivMesh & M, const MatrixXd & VO, const MatrixXi & FO, const MatStruct * ms,
                         const StructPalette & pal, const std::vector<int32_t> & setId,
                         const RelaxGraph & G, const RelaxOptions & opt)
{
    const double tStart = now_s();
    RelaxReport R;
    const int64_t Vs = M.V.rows();
    if ((int64_t)G.role.size() != Vs) fail("graph / mesh size mismatch");
    if (!opt.holdFixed.empty() && (int64_t)opt.holdFixed.size() != Vs) fail("holdFixed / mesh size mismatch");
    if (opt.foldRef.size() && (opt.foldRef.rows() != M.F.rows() || opt.foldRef.cols() != 3)) fail("foldRef / face count mismatch");
    const double diag = (VO.leftCols(3).colwise().maxCoeff() - VO.leftCols(3).colwise().minCoeff()).norm();
    const double tolAbs = opt.tol * diag;

    const SetIds S = split_palette(pal, ms);
    std::vector<uint8_t> setRole(pal.size());
    for (int k = 0; k < pal.size(); ++k) setRole[k] = ms ? role_of(pal.typeMask[k]) : (uint8_t)RELAX_SHEET;
    const double tp = now_s();
    const Projector proj(VO, FO, M.origEdges, ms, pal, S, setRole);
    if (opt.verbose)
        fprintf(stderr, "[subdiv_relax] projector: %zu trees (%.2f s)\n", proj.trees.size(), now_s() - tp);
    // Projection of a step (not the seed check): global or local closest point.
    auto project_step = [&](int k, const Vector3d & y, int f, int e) {
        return opt.localProjection ? proj.project_local(k, y, f, e) : proj.project(k, y, f, e);
    };

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
                    if (!isFree[j] || (!opt.jointPass && G.role[j] != cls)) { anchored = true; continue; }
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

    // Fine faces f and g share a vertex (same face, or neighbours around a vertex).
    auto shares_vertex = [&](int f, int g) {
        for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) if (FO(f, a) == FO(g, b)) return true;
        return false;
    };
    // Commit projected candidates for the listed vertices; returns max move.
    auto commit = [&](const std::vector<int64_t> & list) {
        double m = 0.0;
        for (int64_t i : list) {
            m = std::max(m, move[i]);
            if (move[i] > 0) {
                ++R.projMoves;
                if (!shares_vertex(face[i], nFace[i])) {
                    ++R.projJumps;
                    R.projJumpMax = std::max(R.projJumpMax, move[i] / diag);
                }
            }
            X.row(i) = nX.row(i); face[i] = nFace[i]; edge[i] = nEdge[i]; bary.row(i) = nBary.row(i);
        }
        return m;
    };

    proj.reset_stats();  // projection statistics of the iterations only (not the seed check)

    // No-new-folds rule (opt.foldRef): triangle f is folded when n_f . ref_f <= 0.
    const bool foldRule = opt.foldRef.size() > 0;
    const int64_t nF = M.F.rows();
    auto tri_n = [&](const MatrixXd & P, int64_t f) {
        const Vector3d a = P.row(M.F(f, 0)).transpose(), b = P.row(M.F(f, 1)).transpose(), c = P.row(M.F(f, 2)).transpose();
        return Vector3d((b - a).cross(c - a));
    };
    auto count_folded = [&](const MatrixXd & P) {
        int64_t n = 0;
        for (int64_t f = 0; f < nF; ++f) if (tri_n(P, f).dot(opt.foldRef.row(f).transpose()) <= 0) ++n;
        return n;
    };
    if (foldRule) R.foldedSeed = count_folded(X);

    if (opt.solver == RelaxSolver::Newton) {
        const int64_t maxIter = opt.maxIter > 0 ? opt.maxIter : 100000;
        // Two passes (curves, then sheets with the curves fixed), or one joint pass.
        const int nPass = opt.jointPass ? 1 : 2;
        for (int pass = 0; pass < nPass; ++pass) {
            const uint8_t cls = pass == 0 ? RELAX_CURVE : RELAX_SHEET;
            const char * passName = opt.jointPass ? "joint" : pass == 0 ? "curve" : "sheet";
            std::vector<int64_t> list;
            for (int64_t i = 0; i < Vs; ++i)
                if (isFree[i] && (opt.jointPass ? G.role[i] != RELAX_JUNCTION : G.role[i] == cls)) list.push_back(i);
            int64_t & iters = pass == 0 ? R.itersCurve : R.itersSheet;
            double & delta = pass == 0 ? R.deltaCurve : R.deltaSheet;
            bool conv = list.empty();
            if (conv) delta = 0.0;

            std::vector<int64_t> var(Vs, -1), slot(Vs, -1);
            for (size_t a = 0; a < list.size(); ++a) slot[list[a]] = (int64_t)a;
            std::vector<Matrix<double, 3, Dynamic>> T(list.size());

            // No-new-folds rule: hold back (to X) every moved listed vertex of a
            // triangle that is unfolded at X but folded at the candidate nX, until
            // there is none. Holding a whole triangle back restores it, so this
            // terminates. Returns the number of vertices held back.
            std::vector<int64_t> watched;  // faces a move of this pass can change
            if (foldRule)
                for (int64_t f = 0; f < nF; ++f)
                    for (int c = 0; c < 3; ++c) if (slot[M.F(f, c)] >= 0) { watched.push_back(f); break; }
            std::vector<uint8_t> newFold(watched.size());
            auto block_new_folds = [&]() -> int64_t {
                if (!foldRule) return 0;
                int64_t held = 0;
                for (;;) {
                    igl::parallel_for((int64_t)watched.size(), [&](int64_t w) {
                        const int64_t f = watched[w];
                        const Vector3d ref = opt.foldRef.row(f).transpose();
                        if (!(tri_n(X, f).dot(ref) > 0)) { newFold[w] = 0; return; }
                        Vector3d q[3];
                        for (int c = 0; c < 3; ++c) {
                            const int v = M.F(f, c);
                            q[c] = slot[v] >= 0 ? Vector3d(nX.row(v).transpose()) : Vector3d(X.row(v).transpose());
                        }
                        newFold[w] = (q[1] - q[0]).cross(q[2] - q[0]).dot(ref) <= 0;
                    }, 1000);
                    int64_t n = 0;
                    for (size_t w = 0; w < watched.size(); ++w) {
                        if (!newFold[w]) continue;
                        for (int c = 0; c < 3; ++c) {
                            const int v = M.F(watched[w], c);
                            if (slot[v] < 0 || nX.row(v) == X.row(v)) continue;
                            nX.row(v) = X.row(v); nFace[v] = face[v]; nEdge[v] = edge[v]; nBary.row(v) = bary.row(v);
                            move[v] = 0.0;
                            ++n;
                        }
                    }
                    if (!n) return held;
                    held += n;
                }
            };
            SimplicialLDLT<SparseMatrix<double>> ldlt;
            bool analyzed = false;
            std::vector<int> lastDims;

            // Energy change of the candidate positions nX (listed vertices) against
            // X, E = 1/2 sum over the kept edges of |x_i - x_j|^2, per edge from
            // differences so that it stays accurate when the change is tiny. An
            // edge between two listed vertices appears in both rows (weight 1/2).
            std::vector<double> dEi(list.size());
            auto energy_change = [&]() {
                igl::parallel_for((int64_t)list.size(), [&](int64_t a) {
                    const int64_t i = list[a];
                    double s = 0.0;
                    for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) {
                        const int j = G.cols[q];
                        const bool both = slot[j] >= 0;
                        const Vector3d d  = (X.row(i) - X.row(j)).transpose();
                        const Vector3d dn = (nX.row(i) - (both ? nX.row(j) : X.row(j))).transpose();
                        s += (both ? 0.25 : 0.5) * (dn - d).dot(dn + d);
                    }
                    dEi[a] = s;
                }, 1000);
                double s = 0.0;
                for (double v : dEi) s += v;
                return s;
            };

            // Plain projected step x <- Pi(x + 0.5 L x) of the listed vertices into
            // nX; returns its max move.
            auto plain_step = [&]() {
                igl::parallel_for((int64_t)list.size(), [&](int64_t a) {
                    const int64_t i = list[a];
                    Vector3d mean = Vector3d::Zero();
                    for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) mean += X.row(G.cols[q]).transpose();
                    mean /= (double)(G.rowOffs[i + 1] - G.rowOffs[i]);
                    const Vector3d x = X.row(i).transpose();
                    const ProjResult pr = project_step(setId[i], x + 0.5 * (mean - x), face[i], edge[i]);
                    move[i] = (pr.pos - x).norm();
                    nX.row(i) = pr.pos.transpose(); nFace[i] = pr.face; nEdge[i] = pr.edge; nBary.row(i) = pr.bary.transpose();
                }, 1000);
                double m = 0.0;
                for (int64_t i : list) m = std::max(m, move[i]);
                return m;
            };

            // Newton rounds alternate with plain steps. Newton converges fast for the
            // smooth, global part but its frames (the current edge / face) cannot
            // carry a vertex over a curve corner or a sheet crease; the plain step
            // uses the full pull and does. Done when a plain step moves nothing,
            // which is the definition of "stopped changing".
            for (;;) {
            conv = list.empty();
            while (!conv && iters < maxIter) {
                const double ti = now_s();
                // Frames and variable layout.
                int64_t nVar = 0;
                std::vector<int> dims(list.size());
                for (size_t a = 0; a < list.size(); ++a) {
                    const int64_t i = list[a];
                    Vector3d r = Vector3d::Zero();
                    for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) r += (X.row(G.cols[q]) - X.row(i)).transpose();
                    frame(i, r, T[a]);
                    dims[a] = (int)T[a].cols();
                    var[list[a]] = nVar;
                    nVar += dims[a];
                }

                // Tangent-plane Hessian of E (Gauss-Newton with the constraint
                // linearised): deg_i u_i - sum_j T_i^T T_j u_j = T_i^T sum_j (x_j - x_i).
                std::vector<Triplet<double>> trip;
                trip.reserve((size_t)nVar * 7);
                VectorXd rhs = VectorXd::Zero(nVar);
                for (size_t a = 0; a < list.size(); ++a) {
                    const int64_t i = list[a];
                    const int64_t deg = G.rowOffs[i + 1] - G.rowOffs[i];
                    const double diagv = (double)deg * (1.0 + 1e-10);
                    Vector3d r = Vector3d::Zero();
                    for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) {
                        const int j = G.cols[q];
                        r += (X.row(j) - X.row(i)).transpose();
                        if (slot[j] >= 0) {
                            // All entries, zeros included: the pattern must depend only
                            // on dims and the graph, since analyzePattern is reused.
                            const MatrixXd B = -(T[a].transpose() * T[slot[j]]);
                            for (int uu = 0; uu < B.rows(); ++uu)
                                for (int w = 0; w < B.cols(); ++w)
                                    trip.emplace_back(var[i] + uu, var[j] + w, B(uu, w));
                        }
                    }
                    for (int uu = 0; uu < dims[a]; ++uu) trip.emplace_back(var[i] + uu, var[i] + uu, diagv);
                    rhs.segment(var[i], dims[a]) = T[a].transpose() * r;
                }
                SparseMatrix<double> A(nVar, nVar);
                A.setFromTriplets(trip.begin(), trip.end());
                trip.clear(); trip.shrink_to_fit();
                if (iters == 0) {
                    const double asym = (SparseMatrix<double>(A.transpose()) - A).norm();
                    if (asym > 1e-12 * A.norm()) fail("tangent system is not symmetric (" + std::to_string(asym) + ")");
                }
                if (!analyzed || dims != lastDims) { ldlt.analyzePattern(A); analyzed = true; lastDims = dims; }
                ldlt.factorize(A);
                if (ldlt.info() != Success) fail("tangent system factorization failed");
                const VectorXd u = ldlt.solve(rhs);
                if (ldlt.info() != Success || !u.allFinite()) fail("tangent system solve failed");
                const double slope = rhs.dot(u);  // -dE/dalpha at alpha = 0 (> 0: A is SPD)

                // Backtracking line search on E (Armijo). E must decrease on every
                // accepted step, so the iteration cannot cycle (a valley crease would
                // otherwise make a vertex bounce between two faces' tangent planes).
                double alpha = 1.0, m = 0.0, dE = 0.0;
                int halvings = 0;
                bool accepted = false;
                for (; halvings <= 50; ++halvings, alpha *= 0.5) {
                    igl::parallel_for((int64_t)list.size(), [&](int64_t a) {
                        const int64_t i = list[a];
                        const Vector3d y = X.row(i).transpose() + alpha * (T[a] * u.segment(var[i], dims[a]));
                        const ProjResult pr = project_step(setId[i], y, face[i], edge[i]);
                        move[i] = (pr.pos - X.row(i).transpose()).norm();
                        nX.row(i) = pr.pos.transpose(); nFace[i] = pr.face; nEdge[i] = pr.edge; nBary.row(i) = pr.bary.transpose();
                    }, 1000);
                    const int64_t held = block_new_folds();
                    R.foldReverts += held;
                    m = 0.0;
                    for (int64_t i : list) m = std::max(m, move[i]);
                    if (halvings == 0 && m <= tolAbs) { accepted = true; break; }  // full step moves nothing
                    dE = energy_change();
                    // Held-back vertices leave a partial step, for which the Armijo
                    // slope does not apply; any decrease is accepted then.
                    if (dE <= -1e-4 * alpha * slope || (held > 0 && dE < 0)) { accepted = true; break; }
                }
                ++iters;
                if (!accepted) {
                    // No decrease along the projected Newton direction even for tiny
                    // steps: Newton is done for this round (the plain step decides).
                    delta = 0.0;
                    conv = true;
                    ++R.lineSearchStalls;
                    break;
                }
                commit(list);
                delta = m / diag;
                conv = halvings == 0 && m <= tolAbs;
                R.halvings += halvings;
                if (opt.verbose && (iters <= 10 || iters % 10 == 0 || conv))
                    fprintf(stderr, "[subdiv_relax] %s iter %lld: %lld free, %lld unknowns, max move %.3g (x diag), "
                            "step %.3g, dE %.3g (%.2f s)\n",
                            passName, (long long)iters, (long long)list.size(),
                            (long long)nVar, delta, alpha, dE, now_s() - ti);
            }
            if (list.empty()) break;
            double mp = plain_step();
            if (foldRule) {
                R.foldReverts += block_new_folds();
                mp = 0.0;
                for (int64_t i : list) mp = std::max(mp, move[i]);
            }
            delta = mp / diag;
            if (mp <= tolAbs) { conv = true; break; }
            if (iters >= maxIter) { conv = false; break; }
            commit(list);
            ++iters;
            ++R.plainSteps;
            if (opt.verbose && R.plainSteps % 100 == 1)
                fprintf(stderr, "[subdiv_relax] %s iter %lld: plain step, max move %.3g (x diag)\n",
                        passName, (long long)iters, delta);
            }
            if (!conv) {
                fprintf(stderr, "[subdiv_relax] WARNING: %s pass did not converge in %lld iterations (last max move %.3g x diag)\n",
                        passName, (long long)maxIter, delta);
                std::vector<int64_t> top = list;
                std::sort(top.begin(), top.end(), [&](int64_t a, int64_t b) { return move[a] > move[b]; });
                for (size_t t = 0; t < std::min<size_t>(8, top.size()); ++t) {
                    const int64_t i = top[t];
                    fprintf(stderr, "[subdiv_relax]   v %lld move %.3g face %d bary (%.3g %.3g %.3g) edge %d deg %lld\n",
                            (long long)i, move[i] / diag, face[i], bary(i, 0), bary(i, 1), bary(i, 2), edge[i],
                            (long long)(G.rowOffs[i + 1] - G.rowOffs[i]));
                }
            }
            R.converged = (pass == 0) ? conv : (R.converged && conv);
        }
    } else {
        const int64_t maxIter = opt.maxIter > 0 ? opt.maxIter : 1000000;
        std::vector<int64_t> list;
        for (int64_t i = 0; i < Vs; ++i) if (isFree[i]) list.push_back(i);
        bool conv = list.empty();
        while (!conv && R.itersSheet < maxIter) {
            igl::parallel_for((int64_t)list.size(), [&](int64_t a) {
                const int64_t i = list[a];
                Vector3d mean = Vector3d::Zero();
                for (int64_t q = G.rowOffs[i]; q < G.rowOffs[i + 1]; ++q) mean += X.row(G.cols[q]).transpose();
                mean /= (double)(G.rowOffs[i + 1] - G.rowOffs[i]);
                const Vector3d x = X.row(i).transpose();
                const ProjResult pr = project_step(setId[i], x + opt.lambda * (mean - x), face[i], edge[i]);
                move[i] = (pr.pos - x).norm();
                nX.row(i) = pr.pos.transpose(); nFace[i] = pr.face; nEdge[i] = pr.edge; nBary.row(i) = pr.bary.transpose();
            }, 1000);
            const double m = commit(list);
            ++R.itersSheet;
            R.deltaSheet = m / diag;
            conv = m <= tolAbs;
            if (opt.verbose && (R.itersSheet % 10000 == 0 || conv))
                fprintf(stderr, "[subdiv_relax] jacobi iter %lld: max move %.3g (x diag)\n",
                        (long long)R.itersSheet, R.deltaSheet);
        }
        if (!conv) fprintf(stderr, "[subdiv_relax] WARNING: jacobi did not converge in %lld iterations\n", (long long)maxIter);
        R.converged = conv;
    }

    // ---- write back ----
    M.V = X;
    M.fineFace = face;
    M.fineBary = bary;
    if (foldRule) R.foldedResult = count_folded(X);
    R.localCalls = proj.nLocal; R.localGrown = proj.nLocalGrown; R.localGlobal = proj.nLocalGlobal;
    const ProjStats ps = proj.stats();  // before the final checks project again
    fprintf(stderr, "[subdiv_relax]   projection over all steps (incl. rejected line-search trials): %lld calls, "
                    "%.2f%% on an edge/vertex of their triangle (or an end of their edge), off-surface distance "
                    "mean %.3g max %.3g (x diag), barycentric fix-ups %lld\n",
            (long long)ps.calls, ps.calls ? 100.0 * ps.onBorder / ps.calls : 0.0,
            ps.calls ? ps.distSum / ps.calls / diag : 0.0, ps.distMax / diag, (long long)ps.baryFix);

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
            mv[i] = (project_step(setId[i], x + 0.5 * (mean - x), face[i], edge[i]).pos - x).norm();
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
        "off own structure %lld\n"
        "[subdiv_relax]   projection: %lld vertex moves, %lld jumps (new fine face shares no vertex with the old), "
        "largest jump %.3g (x diag)\n"
        "[subdiv_relax]   no-new-folds rule: %s, folded triangles seed %lld -> result %lld, moves held back %lld\n"
        "[subdiv_relax]   local projection: %s, %lld calls, %lld grew past the first ring, %lld fell back to global\n",
        opt.solver == RelaxSolver::Newton ? "newton" : "jacobi", R.converged ? "converged" : "NOT CONVERGED",
        (long long)R.itersCurve, R.deltaCurve, (long long)R.itersSheet, R.deltaSheet,
        (long long)R.nFree, (long long)R.nFixed, R.maxMove, R.meanMove,
        (long long)R.halvings, (long long)R.lineSearchStalls, (long long)R.plainSteps, (long long)R.nPinned, now_s() - tStart,
        R.jacobiStepMove, (long long)R.jacobiStepVertex,
        R.residualSheet, (long long)R.nResidualSheet, R.residualCurve, (long long)R.nResidualCurve,
        (long long)R.seedOffStructure, (long long)R.fixedMoved, (long long)R.posMismatch, (long long)R.badBary, (long long)R.offStructure,
        (long long)R.projMoves, (long long)R.projJumps, R.projJumpMax,
        foldRule ? "on" : "off", (long long)R.foldedSeed, (long long)R.foldedResult, (long long)R.foldReverts,
        opt.localProjection ? "on" : "off", (long long)R.localCalls, (long long)R.localGrown, (long long)R.localGlobal);
    return R;
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

// ---------------------------------------------------------------- quality

MeshQuality subdiv_mesh_quality(const MatrixXd & V, const MatrixXi & F, const MatrixXd * Vref)
{
    MeshQuality Q;
    const double diag = (V.leftCols(3).colwise().maxCoeff() - V.leftCols(3).colwise().minCoeff()).norm();
    double s = 0.0, s2 = 0.0;
    int64_t n = 0;
    std::vector<float> minAng((size_t)F.rows());
    for (Index f = 0; f < F.rows(); ++f) {
        const Vector3d p[3] = { P3(V, F(f, 0)), P3(V, F(f, 1)), P3(V, F(f, 2)) };
        double ang = 180.0;
        for (int c = 0; c < 3; ++c) {
            const Vector3d u = p[(c + 1) % 3] - p[c], w = p[(c + 2) % 3] - p[c];
            // each undirected edge counted from the face with the edge c->c+1
            const double l = u.norm();
            s += l; s2 += l * l; ++n;
            const double nu = u.norm(), nw = w.norm();
            const double a = (nu > 0 && nw > 0) ? std::acos(std::max(-1.0, std::min(1.0, u.dot(w) / (nu * nw)))) * 180.0 / M_PI : 0.0;
            ang = std::min(ang, a);
        }
        minAng[(size_t)f] = (float)ang;
        const Vector3d nrm = (p[1] - p[0]).cross(p[2] - p[0]);
        if (0.5 * nrm.norm() <= 1e-14 * diag * diag) ++Q.degenerate;
        if (Vref) {
            const Vector3d r0 = P3(*Vref, F(f, 0)), r1 = P3(*Vref, F(f, 1)), r2 = P3(*Vref, F(f, 2));
            if (nrm.dot((r1 - r0).cross(r2 - r0)) < 0) ++Q.flippedVsRef;
        }
    }
    const double mean = n ? s / n : 0.0;
    Q.edgeCV = mean > 0 ? std::sqrt(std::max(0.0, s2 / n - mean * mean)) / mean : 0.0;
    if (!minAng.empty()) {
        std::sort(minAng.begin(), minAng.end());
        auto pct = [&](double q) { return (double)minAng[(size_t)std::min<double>(minAng.size() - 1, q * minAng.size())]; };
        Q.minAngle = minAng.front(); Q.p1 = pct(0.01); Q.p5 = pct(0.05); Q.median = pct(0.5);
    }
    return Q;
}
