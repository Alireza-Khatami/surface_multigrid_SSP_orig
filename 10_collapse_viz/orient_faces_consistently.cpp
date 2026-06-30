#include "orient_faces_consistently.h"

#include <Eigen/Dense>
#include <map>
#include <vector>
#include <queue>
#include <utility>

int orient_faces_consistently(
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    Eigen::MatrixXi       & FF,
    Eigen::VectorXi       & C)
{
    using namespace Eigen;
    using namespace std;

    const int nF = F.rows();
    FF = F;
    C  = VectorXi::Constant(nF, -1);

    // ------------------------------------------------------------------ //
    // Step 1: edge table
    // For each undirected edge {min_v, max_v}, collect all incident faces
    // as (face_idx, is_forward) where is_forward = the face traverses
    // min_v → max_v.
    // ------------------------------------------------------------------ //
    map<pair<int,int>, vector<pair<int,bool>>> edge_table;
    for (int fi = 0; fi < nF; fi++) {
        for (int k = 0; k < 3; k++) {
            int a = F(fi, k), b = F(fi, (k + 1) % 3);
            auto key = make_pair(min(a, b), max(a, b));
            edge_table[key].emplace_back(fi, a < b);
        }
    }

    // ------------------------------------------------------------------ //
    // Step 2: manifold-only face adjacency
    // Two faces are adjacent here only through an edge shared by EXACTLY 2
    // faces.  Boundary edges (1 face) and non-manifold edges (3+ faces) do
    // not propagate orientation.
    // is_consistent = true  ↔  the two faces traverse the shared edge in
    //                           opposite directions (already compatible).
    // ------------------------------------------------------------------ //
    vector<vector<pair<int,bool>>> adj(nF);
    for (auto & kv : edge_table) {
        const auto & faces = kv.second;
        if (faces.size() != 2) continue;
        int  fi      = faces[0].first;
        bool dir_i   = faces[0].second;
        int  fj      = faces[1].first;
        bool dir_j   = faces[1].second;
        bool consistent = (dir_i != dir_j);
        adj[fi].emplace_back(fj, consistent);
        adj[fj].emplace_back(fi, consistent);
    }

    // ------------------------------------------------------------------ //
    // Step 3: BFS — assign orientation decision without touching FF yet.
    // orientation[fi] = 0  →  keep  (same winding as component seed)
    // orientation[fi] = 1  →  flip  (reverse winding)
    // ------------------------------------------------------------------ //
    VectorXi orientation = VectorXi::Constant(nF, -1);
    int n_comp = 0;

    for (int seed = 0; seed < nF; seed++) {
        if (orientation(seed) != -1) continue;
        orientation(seed) = 0;
        C(seed)           = n_comp;
        queue<int> q;
        q.push(seed);
        while (!q.empty()) {
            int fi = q.front(); q.pop();
            for (auto & nb : adj[fi]) {
                int  fj         = nb.first;
                bool consistent = nb.second;
                if (orientation(fj) != -1) continue;
                orientation(fj) = consistent ? orientation(fi) : (1 - orientation(fi));
                C(fj)           = n_comp;
                q.push(fj);
            }
        }
        n_comp++;
    }

    // ------------------------------------------------------------------ //
    // Step 4: orient_outward per component.
    // After the BFS each component is internally consistent, but the seed
    // face determines which way its normals point.  We test the pending
    // (not yet committed) average normal of each component against the
    // outward direction (component centroid − overall mesh centroid) and
    // flip the whole component if its average normal points inward.
    // ------------------------------------------------------------------ //
    const Vector3d mesh_centroid = V.colwise().mean();

    for (int cid = 0; cid < n_comp; cid++) {
        Vector3d comp_centroid = Vector3d::Zero();
        Vector3d avg_normal    = Vector3d::Zero();
        int count = 0;

        for (int fi = 0; fi < nF; fi++) {
            if (C(fi) != cid) continue;

            // Vertices in their pending post-flip orientation
            int v0, v1, v2;
            if (orientation(fi) == 0) {
                v0 = F(fi, 0); v1 = F(fi, 1); v2 = F(fi, 2);
            } else {
                v0 = F(fi, 2); v1 = F(fi, 1); v2 = F(fi, 0);
            }

            const Vector3d p0 = V.row(v0).transpose();
            const Vector3d p1 = V.row(v1).transpose();
            const Vector3d p2 = V.row(v2).transpose();

            avg_normal    += (p1 - p0).cross(p2 - p0);
            comp_centroid += (p0 + p1 + p2) / 3.0;
            ++count;
        }
        if (count == 0) continue;
        comp_centroid /= count;

        // If the average normal points toward the mesh centroid, flip the
        // entire component.
        const Vector3d outward = comp_centroid - mesh_centroid;
        if (avg_normal.dot(outward) < 0.0) {
            for (int fi = 0; fi < nF; fi++) {
                if (C(fi) == cid)
                    orientation(fi) = 1 - orientation(fi);
            }
        }
    }

    // ------------------------------------------------------------------ //
    // Step 5: commit flips — single pass, no face is touched twice.
    // ------------------------------------------------------------------ //
    int n_flipped = 0;
    for (int fi = 0; fi < nF; fi++) {
        if (orientation(fi) == 1) {
            FF.row(fi) = F.row(fi).reverse().eval();
            ++n_flipped;
        }
    }

    return n_flipped;
}
