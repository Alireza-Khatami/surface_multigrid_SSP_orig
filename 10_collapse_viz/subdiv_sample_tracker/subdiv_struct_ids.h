#pragma once
// Structure IDs of subdivided vertices, from a .ma_struct file.
//
// Rules (consistent with load_matstruct for original vertices):
//   VERTEX v : every struct touching v (sheet faces, seam/boundary edges,
//              junction vertex) - the only carrier that can get junction IDs
//   EDGE e   : seam/boundary IDs of e  U  sheet IDs of every face incident to e
//   FACE f   : sheet IDs of f
//
// Storage: the distinct ID sets are interned once into a palette (CSR); each
// subdivided vertex stores one int32 index into it.

#include "subdiv_mesh.h"

#include <Eigen/Core>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Palette type-mask bits, one per .ma_struct type_id.
enum : uint8_t {
    STRUCT_MASK_SHEET    = 1 << 0,  // type_id 0
    STRUCT_MASK_SEAM     = 1 << 1,  // type_id 1
    STRUCT_MASK_BOUNDARY = 1 << 2,  // type_id 2
    STRUCT_MASK_JUNCTION = 1 << 3,  // type_id 3
};

struct StructPalette {
    std::vector<int32_t> offsets{0};  // P+1; set k is ids[offsets[k] .. offsets[k+1])
    std::vector<int32_t> ids;         // sorted ascending within each set
    std::vector<uint8_t> typeMask;    // P
    int size() const { return (int)typeMask.size(); }
};

// The .ma_struct element structure, indexed like the fine mesh.
struct MatStructElements {
    std::vector<std::vector<int>> faceIds;    // per FO face: sheet struct IDs
    std::vector<std::vector<int>> edgeIds;    // per origEdges row: seam/boundary struct IDs
    std::vector<std::vector<int>> vertexIds;  // per VO vertex (same rule as load_matstruct), sorted
    std::unordered_map<int, int>  structType; // struct ID -> type_id
    int nEdgesWithoutFace = 0;                // seam/boundary edges not on any FO face
};

// Parses `path` and checks it describes (VO, FO): same vertex and face counts,
// and every .ma face has the same corner set as the FO row with the same index.
// Throws std::runtime_error on I/O errors or any mismatch.
MatStructElements load_matstruct_elements(const std::string & path,
                                          const Eigen::MatrixXd & VO,
                                          const Eigen::MatrixXi & FO,
                                          const Eigen::MatrixXi & origEdges);

// Fills the palette and one set index per subdivided vertex. With elems ==
// nullptr (no .ma_struct), every vertex maps to a single empty set.
void build_struct_sets(const SubdivMesh & M,
                       const Eigen::MatrixXi & FO,
                       const MatStructElements * elems,
                       StructPalette & palette,
                       std::vector<int32_t> & setId);
