#pragma once
// Structure IDs of subdivided vertices, from the shared .ma_struct reader
// (load_matstruct.h).
//
// Rules (consistent with MatStruct::vertexIds for original vertices):
//   VERTEX v : MatStruct::vertexIds[v] - the only carrier that can get junction IDs
//   EDGE e   : seam/boundary IDs of e  U  sheet IDs of every face incident to e
//   FACE f   : sheet IDs of f
//
// Storage: the distinct ID sets are interned once into a palette (CSR); each
// subdivided vertex stores one int32 index into it.

#include "subdiv_mesh.h"
#include "../load_matstruct.h"

#include <Eigen/Core>
#include <cstdint>
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

// Fills the palette and one set index per subdivided vertex. ms must describe
// the mesh M was built from (FO). With ms == nullptr (no .ma_struct), every
// vertex maps to a single empty set.
void build_struct_sets(const SubdivMesh & M,
                       const Eigen::MatrixXi & FO,
                       const MatStruct * ms,
                       StructPalette & palette,
                       std::vector<int32_t> & setId);
