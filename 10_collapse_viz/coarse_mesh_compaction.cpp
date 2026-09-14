#include "coarse_mesh_compaction.h"
#include "face_dead.h"

#include <igl/remove_unreferenced.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace Eigen;

CoarseMeshCompaction build_compact_coarse_mesh(const MatrixXd & gV, const MatrixXi & gF)
{
    // Collect live faces: not dead, not incident to the infinity cap vertex.
    // Walking the full gF.rows() (rather than stopping at gFaceSheetID.size())
    // is safe: any row beyond the original face count is a boundary-to-infinity
    // cap face, which always has an infinity vertex and is filtered out below.
    std::vector<std::array<int,3>> face_rows;
    std::vector<int> face_orig_idx;
    face_rows.reserve(gF.rows());
    face_orig_idx.reserve(gF.rows());
    for (int f = 0; f < gF.rows(); f++) {
        if (!is_face_live(gF, gV, f)) continue;
        face_rows.push_back({gF(f,0), gF(f,1), gF(f,2)});
        face_orig_idx.push_back(f);
    }
    MatrixXi Flive((int)face_rows.size(), 3);
    for (int i = 0; i < (int)face_rows.size(); i++)
        Flive.row(i) << face_rows[i][0], face_rows[i][1], face_rows[i][2];

    // igl::remove_unreferenced(V, F, NV, NF, I, J): I is #V-sized (old->new,
    // -1 if dropped), J is #NV-sized (new->old). Bind them to the matching
    // struct fields — swapping these silently hands every caller a
    // wrong-sized/wrong-direction map.
    CoarseMeshCompaction out;
    VectorXi I, J;
    igl::remove_unreferenced(gV.leftCols(3), Flive, out.Vbase, out.Fout, I, J);
    out.oldToNew = I;
    out.newToOld = J;

    // remove_unreferenced doesn't reorder faces, so Fout row i still corresponds
    // to face_rows[i] / face_orig_idx[i].
    out.faceOrigIdx = Map<VectorXi>(face_orig_idx.data(), (int)face_orig_idx.size());
    return out;
}
