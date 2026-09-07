#include "stale_chains.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef C2F_VIZ_DIAGNOSTIC
#include <polyscope/polyscope.h>
#include <polyscope/curve_network.h>
extern Eigen::MatrixXd gVO;
#endif

using namespace Eigen;

// ---- globals ----
std::vector<std::vector<int>> gStaleChains;
std::unordered_set<int>       gStaleVertexSet;

#ifdef C2F_VIZ_DIAGNOSTIC
std::vector<uint8_t> gStaleChainVisible;
bool                 gStaleChainShowAll = true;
#endif

// ---- OBJ loader ----
bool load_obj_vfl(
    const std::string & path,
    MatrixXd & V,
    MatrixXi & F,
    std::vector<std::pair<int,int>> & l_edges)
{
    std::ifstream fh(path);
    if (!fh.is_open()) {
        fprintf(stderr, "[OBJ] cannot open '%s'\n", path.c_str());
        return false;
    }
    std::vector<std::array<double,3>> verts;
    std::vector<std::array<int,3>>   faces;
    std::string line;
    while (std::getline(fh, line)) {
        if (line.empty()) continue;
        size_t s = line.find_first_not_of(" \t\r");
        if (s == std::string::npos || line[s] == '#') continue;
        char token = line[s];
        if (token == 'v' && (s+1 < line.size()) && std::isspace((unsigned char)line[s+1])) {
            std::istringstream ss(line.substr(s+1));
            double x, y, z; ss >> x >> y >> z;
            verts.push_back({x, y, z});
        } else if (token == 'f' && (s+1 < line.size()) && std::isspace((unsigned char)line[s+1])) {
            std::istringstream ss(line.substr(s+1));
            std::array<int,3> tri;
            for (int k = 0; k < 3; k++) {
                std::string tok; ss >> tok;
                tri[k] = std::stoi(tok) - 1;  // 1-based → 0-based
            }
            faces.push_back(tri);
        } else if (token == 'l' && (s+1 < line.size()) && std::isspace((unsigned char)line[s+1])) {
            std::istringstream ss(line.substr(s+1));
            std::vector<int> vs;
            int vi;
            while (ss >> vi) vs.push_back(vi - 1);
            for (int i = 0; i + 1 < (int)vs.size(); i++)
                l_edges.push_back({vs[i], vs[i+1]});
        }
    }
    V.resize((int)verts.size(), 3);
    for (int i = 0; i < (int)verts.size(); i++)
        V.row(i) << verts[i][0], verts[i][1], verts[i][2];
    F.resize((int)faces.size(), 3);
    for (int i = 0; i < (int)faces.size(); i++)
        F.row(i) << faces[i][0], faces[i][1], faces[i][2];
    return true;
}

// ---- chain detection ----
void detect_stale_chains(
    const std::vector<std::pair<int,int>> & l_edges_raw,
    const MatrixXi & FO)
{
    std::vector<std::pair<int,int>> raw_edges = l_edges_raw;
    if (raw_edges.empty()) {
        std::cout << "[STALE] no l-elements found — no stale chains\n";
        return;
    }
    std::cout << "[STALE] " << raw_edges.size() << " l-edges parsed\n";

    // Build set of all face edges; naked l-edges must not appear in any triangle.
    std::set<std::pair<int,int>> face_edges;
    for (int f = 0; f < FO.rows(); f++)
        for (int c = 0; c < 3; c++) {
            int u = FO(f, c), v = FO(f, (c + 1) % 3);
            face_edges.insert({std::min(u, v), std::max(u, v)});
        }

    {
        std::vector<std::pair<int,int>> naked;
        naked.reserve(raw_edges.size());
        int n_face = 0;
        for (auto & e : raw_edges) {
            auto key = std::make_pair(std::min(e.first, e.second),
                                      std::max(e.first, e.second));
            if (face_edges.count(key)) { n_face++; continue; }
            naked.push_back(e);
        }
        if (n_face)
            std::cout << "[STALE] " << n_face << " l-edges skipped (also a face edge)\n";
        raw_edges = std::move(naked);
    }

    if (raw_edges.empty()) {
        std::cout << "[STALE] all l-edges were face edges — no stale chains\n";
        return;
    }
    std::cout << "[STALE] " << raw_edges.size() << " naked l-edges remain\n";

    // Build adjacency (deduplicated).
    std::map<int, std::vector<int>> adj;
    for (auto & e : raw_edges) {
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
    }
    for (auto & kv : adj) {
        auto & nb = kv.second;
        std::sort(nb.begin(), nb.end());
        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
    }

    // Build maximal chains.
    std::set<std::pair<int,int>> visited;
    auto mark_edge = [&](int u, int v) {
        visited.insert({std::min(u,v), std::max(u,v)});
    };
    auto edge_visited = [&](int u, int v) -> bool {
        return visited.count({std::min(u,v), std::max(u,v)}) > 0;
    };

    auto trace = [&](int start, int nxt) -> std::vector<int> {
        std::vector<int> chain = {start};
        int prev = start, cur = nxt;
        while (true) {
            chain.push_back(cur);
            mark_edge(prev, cur);
            const auto & nbrs = adj[cur];
            if ((int)nbrs.size() != 2) break;
            int next_v = (nbrs[0] != prev) ? nbrs[0] : nbrs[1];
            if (edge_visited(cur, next_v)) break;
            prev = cur; cur = next_v;
        }
        return chain;
    };

    // Pass 0: endpoints (degree 1), Pass 1: junctions (degree > 2).
    for (int pass = 0; pass < 2; pass++) {
        for (auto & kv : adj) {
            int v   = kv.first;
            int deg = (int)kv.second.size();
            if (pass == 0 ? deg != 1 : deg <= 2) continue;
            for (int nb : kv.second)
                if (!edge_visited(v, nb))
                    gStaleChains.push_back(trace(v, nb));
        }
    }

    // Pass 2: closed loops — unvisited edges among degree-2 vertices.
    for (auto & kv : adj) {
        if ((int)kv.second.size() != 2) continue;
        int v = kv.first;
        for (int nb : kv.second) {
            if (!edge_visited(v, nb)) {
                auto chain = trace(v, nb);
                if (chain.front() != chain.back())
                    chain.push_back(chain.front());
                gStaleChains.push_back(chain);
                break;
            }
        }
    }

    // Populate vertex set.
    for (const auto & chain : gStaleChains)
        for (int vid : chain)
            gStaleVertexSet.insert(vid);

    std::cout << "[STALE] " << gStaleChains.size() << " chains  ("
              << gStaleVertexSet.size() << " unique vertices protected)\n";
    for (int i = 0; i < (int)gStaleChains.size(); i++)
        printf("[STALE]   chain[%d]: %d vertices\n", i, (int)gStaleChains[i].size());
}

// ---- visualization (only when polyscope is available) ----
#ifdef C2F_VIZ_DIAGNOSTIC

std::array<float,3> stale_hsv_rgb(float h, float s, float v)
{
    float c = v*s, x = c*(1.f - std::fabs(std::fmod(h*6.f, 2.f) - 1.f)), m = v-c;
    float r,g,b;
    switch ((int)(h*6.f) % 6) {
        case 0: r=c;g=x;b=0;break; case 1:r=x;g=c;b=0;break;
        case 2: r=0;g=c;b=x;break; case 3:r=0;g=x;b=c;break;
        case 4: r=x;g=0;b=c;break; default:r=c;g=0;b=x;break;
    }
    return {r+m, g+m, b+m};
}

void update_stale_chains_display()
{
    if (gStaleChains.empty()) return;
    int nC = (int)gStaleChains.size();
    if ((int)gStaleChainVisible.size() != nC)
        gStaleChainVisible.assign(nC, 1);

    for (int ci = 0; ci < nC; ci++) {
        const auto & chain = gStaleChains[ci];
        int nV = (int)chain.size();
        if (nV < 2) continue;

        MatrixXd Vc(nV, 3);
        for (int k = 0; k < nV; k++) {
            int vid = chain[k];
            if (vid < (int)gVO.rows()) Vc.row(k) = gVO.row(vid);
            else                       Vc.row(k).setZero();
        }
        int nEdges = nV - 1;
        MatrixXi Ec(nEdges, 2);
        for (int k = 0; k < nEdges; k++) Ec.row(k) << k, k+1;

        auto col = stale_hsv_rgb((float)ci / (float)std::max(1, nC), 0.85f, 0.95f);
        char nm[64]; snprintf(nm, sizeof(nm), "stale_chain_%d", ci);
        polyscope::registerCurveNetwork(nm, Vc, Ec)
            ->setRadius(0.003, true)
            ->setColor({col[0], col[1], col[2]})
            ->setEnabled(gStaleChainShowAll && gStaleChainVisible[ci]);
    }
}

#endif // C2F_VIZ_DIAGNOSTIC
