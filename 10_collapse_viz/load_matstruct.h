#pragma once
// The one reader for .ma_struct files. Everything that needs structure IDs
// (collapse gate, simp_viz tracker, subdivided-sample tracker, viewer) takes
// them from the MatStruct this fills, instead of parsing the file itself.
//
// File format:
//   <nv> <ne> <nf>
//   v x y z r          (nv lines)
//   e v0 v1            (ne lines)
//   f v0 v1 v2         (nf lines)
//   <num_structures>
//   <struct_id> <type_id> <count>
//   <elem_id_0> ... <elem_id_N>
//   ...
// type_id: 0 SHEET (faces), 1 SEAM (edges), 2 BOUNDARY (edges), 3 JUNCTION (vertices).
// Elements of any other type are ignored.

#include <Eigen/Core>

#include <array>
#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

struct MatStructEntry {
    int id   = -1;
    int type = -1;
    std::vector<int> elements;  // face / .ma edge / vertex indices, as listed
};

struct MatStruct {
    int nv = 0, ne = 0, nf = 0;
    std::vector<std::array<int, 2>> maEdges;  // "e v0 v1" lines, file order
    std::vector<MatStructEntry> structs;      // file order (empty without a struct section)
    std::unordered_map<int, int> structType;  // struct id -> type_id
    bool hasStructSection = false;

    // Derived, indexed like the fine mesh (the .ma is checked to describe it):
    // vertex v gets a struct's id when v is a corner of one of its SHEET faces,
    // an endpoint of one of its SEAM/BOUNDARY edges, or a listed JUNCTION vertex.
    std::vector<std::set<int>> vertexIds;
    std::vector<std::vector<int>> faceIds;    // per face: sheet ids, sorted
    // SEAM/BOUNDARY ids per edge, keyed by matstruct_edge_key(v0, v1), sorted.
    std::unordered_map<uint64_t, std::vector<int>> edgeIds;
    int nEdgesWithoutFace = 0;  // SEAM/BOUNDARY edges that bound no mesh face
    double maxPosDiff = 0.0;    // max |.ma vertex position - mesh vertex position|
};

// Order-independent key of edge (a, b).
uint64_t matstruct_edge_key(int a, int b);

// Reads fname into out and checks it describes the fine mesh (VO, FO): same
// vertex and face counts, and every .ma face has the same corner set as the
// FO row with the same index. Element indices must be in range. Returns false
// (with a message in *err) on any I/O or consistency error.
bool load_matstruct(const std::string & fname,
                    const Eigen::MatrixXd & VO,
                    const Eigen::MatrixXi & FO,
                    MatStruct & out,
                    std::string * err = nullptr);
