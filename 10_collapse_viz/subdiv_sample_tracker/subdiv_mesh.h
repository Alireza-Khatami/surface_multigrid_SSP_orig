#pragma once
// Uniform midpoint (1->4, no smoothing) subdivision of the original fine mesh
// (gVO/gFO). The subdivided mesh is only a set of samples with fixed
// connectivity: it is never decimated.
//
// Every subdivided vertex records its carrier, the lowest-dimensional element
// of the original mesh containing it (original vertex, original edge, or
// interior of an original face), with exact coordinates on that carrier.
// All coordinates are dyadic rationals (k / 2^level), so they are exact in
// double precision and zero tests on them are exact.
//
// Index stability: subdivided vertices 0..|VO|-1 are the original vertices in
// their original order (including vertices referenced by no face). Each level
// appends its edge midpoints after the previous level's vertices, so the first
// |V_l| vertices of the final mesh are exactly the vertices of level l.

#include <Eigen/Core>
#include <cstdint>
#include <string>
#include <vector>

enum SubdivCarrier : uint8_t {
    SUBDIV_CARRIER_VERTEX = 0,  // carrierIndex = original vertex (VO row)
    SUBDIV_CARRIER_EDGE   = 1,  // carrierIndex = row of SubdivMesh::origEdges
    SUBDIV_CARRIER_FACE   = 2,  // carrierIndex = original face (FO row)
};

struct SubdivMesh {
    Eigen::MatrixXd V;            // Vs x 3 positions (interpolated from fineFace/fineBary)
    Eigen::MatrixXi F;            // Fs x 3, winding inherited from FO
    std::vector<int32_t> faceOrig; // subdivided face -> FO face that contains it

    // Unique undirected edges of FO as (min, max), sorted lexicographically.
    Eigen::MatrixXi origEdges;

    std::vector<uint8_t> carrierType;  // SubdivCarrier
    std::vector<int32_t> carrierIndex;
    // EDGE: (w0, w1, 0) weights of origEdges(e,0), origEdges(e,1).
    // FACE: barycentric in FO.row(f) corner order. VERTEX: (1, 0, 0).
    Eigen::MatrixXd carrierCoord;      // Vs x 3

    // Position on the fine mesh as one FO face + barycentric in its corner order.
    // VERTEX: first face (ascending row, then corner) that uses the vertex.
    // EDGE:   lowest-index face incident to the edge.
    // FACE:   the carrier face.
    // -1 / zero bary for an original vertex referenced by no face.
    std::vector<int32_t> fineFace;
    Eigen::MatrixXd fineBary;          // Vs x 3

    int nOrigVerts = 0;
    int nLevels    = 0;
    // levelVerts[l] = vertex count of level l (nested prefix), l = 0..nLevels.
    std::vector<int64_t> levelVerts;
};

// Faces of level l <= nLevels, recovered from the final faces: level-l face r
// owns final rows [r*4^k, (r+1)*4^k) with k = nLevels - l, and its corners are
// the corner-0 / corner-1 / corner-2 vertices reached by always descending into
// child 0 / 1 / 2. Uses only vertices < levelVerts[l].
Eigen::MatrixXi subdiv_level_faces(const SubdivMesh & M, int level);

// Subdivides (VO, FO) uniformly until the vertex count is >= nTarget
// (0 levels if |VO| >= nTarget). Throws std::runtime_error on invalid input
// (a face with a repeated corner, an out-of-range index) or if the result
// would not fit in 32-bit indices.
SubdivMesh build_subdiv_mesh(const Eigen::MatrixXd & VO,
                             const Eigen::MatrixXi & FO,
                             int64_t nTarget);

// Unique undirected edges of F as (min, max), sorted lexicographically.
Eigen::MatrixXi subdiv_unique_edges(const Eigen::MatrixXi & F);

// Index of edge (a, b) in `edges` (from subdiv_unique_edges), -1 if absent.
int subdiv_find_edge(const Eigen::MatrixXi & edges, int a, int b);

// Path for opening a file for writing. On Windows, returns the absolute path in
// long-path form (prefix backslash backslash ? backslash), so output paths over 260 characters (long run folders
// + long mesh stems) do not fail. Elsewhere, returns path unchanged.
std::string subdiv_long_path(const std::string & path);
