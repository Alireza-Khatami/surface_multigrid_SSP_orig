#pragma once
#include <Eigen/Core>
#include <array>
#include <map>
#include <utility>
#include <vector>

// The single dense numbering of the decimated mesh (gV/gF). Every coarse-side
// exporter must take its indices from one instance of this, built once.
//
// Indices 0..NC-1: vertices referenced by a live face (not dead, no vertex at
// infinity). Indices NC..Vbase.rows()-1: naked stale-chain vertices, appended
// in gStaleChains order, first-seen-gets-next-index.
struct CoarseMeshCompaction {
    Eigen::MatrixXd Vbase;       // compact vertex positions (NC + naked stale-chain rows)
    Eigen::MatrixXi Fout;        // FC x 3 compact faces (indices < NC)
    Eigen::VectorXi newToOld;    // compact -> gV id
    Eigen::VectorXi oldToNew;    // gV id -> compact, -1 if not live
    Eigen::VectorXi faceOrigIdx; // compact face -> gF row
    int NC = 0;                  // face-referenced vertex count
    std::vector<std::vector<int>> staleChains; // gStaleChains in compact indices
};

// Faces + stale chains. Throws on an empty or unmappable stale chain.
CoarseMeshCompaction build_final_coarse_mesh(const Eigen::MatrixXd & gV,
                                             const Eigen::MatrixXi & gF,
                                             const std::vector<std::vector<int>> & gStaleChains);

// A tracker sample expressed on the compact mesh.
struct CoarseSample {
    int             face; // index into cmc.Fout
    Eigen::Vector3d bary; // in cmc.Fout.row(face) corner order
};

// Converts tracker samples (gF row + bary on 3 gV corners) to CoarseSample.
// Holds a reference to cmc, which must outlive it.
class CoarseFaceLookup {
public:
    explicit CoarseFaceLookup(const CoarseMeshCompaction & cmc);

    // Throws if the sample's corners are not live or match no compact face.
    CoarseSample resolve(int gF_row, const Eigen::RowVector3i & gv_corners,
                         const Eigen::RowVector3d & bary) const;

    const CoarseMeshCompaction & cmc;

private:
    std::vector<int> gFRowToFace;                          // gF row -> compact face, -1 if dead
    std::map<std::array<int,3>, int> faceByVerts;          // sorted corners -> first face
    std::map<std::pair<int,int>, int> faceByEdge;          // sorted edge -> first face
    std::vector<int> faceByVertex;                         // compact vertex -> first face, -1 if naked
};
