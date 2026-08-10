#include "load_matstruct.h"

#include <fstream>
#include <iostream>
#include <array>

bool load_matstruct(const std::string& fname,
                    std::vector<std::set<int>>& vertex_struct_ids)
{
    std::ifstream f(fname);
    if (!f) {
        std::cerr << "[load_matstruct] cannot open: " << fname << "\n";
        return false;
    }

    int nv, ne, nf;
    if (!(f >> nv >> ne >> nf) || nv <= 0) {
        std::cerr << "[load_matstruct] bad header in: " << fname << "\n";
        return false;
    }

    // Skip vertex lines ("v x y z r")
    for (int i = 0; i < nv; ++i) {
        char ch; double x, y, z, r;
        f >> ch >> x >> y >> z >> r;
    }

    // Read edges — need endpoint vertex indices to propagate seam struct IDs
    std::vector<std::array<int,2>> edge_verts(ne);
    for (int i = 0; i < ne; ++i) {
        char ch;
        f >> ch >> edge_verts[i][0] >> edge_verts[i][1];
    }

    // Read faces — need corner vertex indices to propagate sheet struct IDs
    std::vector<std::array<int,3>> face_verts(nf);
    for (int i = 0; i < nf; ++i) {
        char ch;
        f >> ch >> face_verts[i][0] >> face_verts[i][1] >> face_verts[i][2];
    }

    vertex_struct_ids.assign(nv, {});

    int num_structs = 0;
    if (!(f >> num_structs)) {
        std::cerr << "[load_matstruct] no struct section in: " << fname << "\n";
        return true; // geometry loaded, no struct data — not an error
    }

    for (int s = 0; s < num_structs; ++s) {
        int struct_id, type_id, count;
        f >> struct_id >> type_id >> count;

        for (int j = 0; j < count; ++j) {
            int elem_id;
            f >> elem_id;

            if (type_id == 3) {
                // JUNCTION — elem_id is a vertex index
                if (elem_id >= 0 && elem_id < nv)
                    vertex_struct_ids[elem_id].insert(struct_id);
            }
            else if (type_id == 1) {
                // SEAM — elem_id is an edge index; propagate to both endpoints
                if (elem_id >= 0 && elem_id < ne) {
                    vertex_struct_ids[edge_verts[elem_id][0]].insert(struct_id);
                    vertex_struct_ids[edge_verts[elem_id][1]].insert(struct_id);
                }
            }
            else if (type_id == 0) {
                // SHEET — elem_id is a face index; propagate to all 3 corners
                if (elem_id >= 0 && elem_id < nf) {
                    vertex_struct_ids[face_verts[elem_id][0]].insert(struct_id);
                    vertex_struct_ids[face_verts[elem_id][1]].insert(struct_id);
                    vertex_struct_ids[face_verts[elem_id][2]].insert(struct_id);
                }
            }
            else if (type_id == 2) {
                // BOUNDARY — elem_id is an edge index; propagate to both endpoints
                if (elem_id >= 0 && elem_id < ne) {
                    vertex_struct_ids[edge_verts[elem_id][0]].insert(struct_id);
                    vertex_struct_ids[edge_verts[elem_id][1]].insert(struct_id);
                }
            }
        }
    }

    return true;
}
