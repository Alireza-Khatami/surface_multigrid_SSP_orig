#pragma once
// Reads a .ma file with a trailing struct section (same format as
// load_matstruct.h) and returns the raw SEAM (type_id==1) and BOUNDARY
// (type_id==2) edges themselves — as (v0, v1, struct_id) triples — instead
// of collapsing them onto vertex struct-ID sets the way load_matstruct does.
//
// v0/v1 are fine-mesh vertex indices, same indexing convention as
// load_matstruct's vertex_struct_ids (i.e. they line up 1:1 with gVO rows).
//
// File format: see load_matstruct.h.

#include <string>
#include <vector>

struct MatStructEdge {
    int v0, v1;     // fine-mesh (gVO) vertex indices, endpoints of this edge
    int struct_id;  // which .ma_struct block this edge belongs to
};

// Fills boundary_edges / seam_edges from the struct section of `fname`.
// Returns false on I/O error. Leaves both vectors empty (but returns true)
// when the file has no struct section.
bool load_matstruct_edges(const std::string& fname,
                          std::vector<MatStructEdge>& boundary_edges,
                          std::vector<MatStructEdge>& seam_edges);
