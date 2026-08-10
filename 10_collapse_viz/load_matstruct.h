#pragma once
// Reads a .ma file with a trailing struct section and returns, for each
// vertex, the set of struct IDs it belongs to.
//
// A vertex accumulates a struct ID when:
//   - it is a corner of a SHEET face in that struct    (type_id == 0)
//   - it is an endpoint of a SEAM edge in that struct  (type_id == 1)
//   - it is an endpoint of a BOUNDARY edge in that struct (type_id == 2)
//   - it is directly listed as a JUNCTION vertex        (type_id == 3)
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

#include <set>
#include <vector>
#include <string>

// Fills vertex_struct_ids[v] with all struct IDs that touch vertex v.
// Returns false on I/O error.
bool load_matstruct(const std::string& fname,
                    std::vector<std::set<int>>& vertex_struct_ids);
