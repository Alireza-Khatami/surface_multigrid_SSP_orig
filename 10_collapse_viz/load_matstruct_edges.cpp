#include "load_matstruct_edges.h"

#include <fstream>
#include <iostream>
#include <array>

bool load_matstruct_edges(const std::string& fname,
                          std::vector<MatStructEdge>& boundary_edges,
                          std::vector<MatStructEdge>& seam_edges)
{
    boundary_edges.clear();
    seam_edges.clear();

    std::ifstream f(fname);
    if (!f) {
        std::cerr << "[load_matstruct_edges] cannot open: " << fname << "\n";
        return false;
    }

    int nv, ne, nf;
    if (!(f >> nv >> ne >> nf) || nv <= 0) {
        std::cerr << "[load_matstruct_edges] bad header in: " << fname << "\n";
        return false;
    }

    // Skip vertex lines ("v x y z r")
    for (int i = 0; i < nv; ++i) {
        char ch; double x, y, z, r;
        f >> ch >> x >> y >> z >> r;
    }

    // Read edges — need endpoint vertex indices.
    std::vector<std::array<int,2>> edge_verts(ne);
    for (int i = 0; i < ne; ++i) {
        char ch;
        f >> ch >> edge_verts[i][0] >> edge_verts[i][1];
    }

    // Skip face lines ("f v0 v1 v2")
    for (int i = 0; i < nf; ++i) {
        char ch; int a, b, c;
        f >> ch >> a >> b >> c;
    }

    int num_structs = 0;
    if (!(f >> num_structs)) {
        std::cerr << "[load_matstruct_edges] no struct section in: " << fname << "\n";
        return true; // geometry loaded, no struct data — not an error
    }

    for (int s = 0; s < num_structs; ++s) {
        int struct_id, type_id, count;
        f >> struct_id >> type_id >> count;

        for (int j = 0; j < count; ++j) {
            int elem_id;
            f >> elem_id;

            if (type_id == 1) {
                // SEAM — elem_id is an edge index
                if (elem_id >= 0 && elem_id < ne)
                    seam_edges.push_back({ edge_verts[elem_id][0], edge_verts[elem_id][1], struct_id });
            }
            else if (type_id == 2) {
                // BOUNDARY — elem_id is an edge index
                if (elem_id >= 0 && elem_id < ne)
                    boundary_edges.push_back({ edge_verts[elem_id][0], edge_verts[elem_id][1], struct_id });
            }
            // type_id == 0 (SHEET, face-based) and type_id == 3 (JUNCTION,
            // vertex-based) carry no edge information — skip.
        }
    }

    return true;
}
