#include "coarse_mesh_compaction.h"
#include "face_dead.h"

#include <igl/remove_unreferenced.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <utility>
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
    out.NC = (int)out.Vbase.rows();
    return out;
}

std::vector<std::vector<int>> extend_with_stale_chains(
    CoarseMeshCompaction & cmc,
    const MatrixXd & gV,
    const std::vector<std::vector<int>> & gStaleChains)
{
    // Assign a new compact index to every chain vertex not already covered
    // by the coarse mesh, in a fixed order (walk gStaleChains in order,
    // first-seen-gets-next-index) — this is the ONE place that ordering is
    // decided, so every caller that runs this against the same gStaleChains
    // gets identical indices.
    std::vector<int> new_old_ids; // old (gV) ids of the newly-appended vertices, in assignment order
    for (const auto & chain : gStaleChains) {
        for (int vid : chain) {
            if (vid < 0 || vid >= cmc.oldToNew.size()) continue; // out-of-range guard, shouldn't happen
            if (cmc.oldToNew(vid) >= 0) continue;                // already covered by the coarse mesh
            cmc.oldToNew(vid) = cmc.NC + (int)new_old_ids.size();
            new_old_ids.push_back(vid);
        }
    }

    if (!new_old_ids.empty()) {
        const int oldRows = (int)cmc.Vbase.rows();
        const int addRows = (int)new_old_ids.size();
        MatrixXd Vext(oldRows + addRows, 3);
        Vext.topRows(oldRows) = cmc.Vbase;
        for (int i = 0; i < addRows; i++)
            Vext.row(oldRows + i) = gV.row(new_old_ids[i]).leftCols(3);
        cmc.Vbase = std::move(Vext);

        VectorXi newToOldExt(oldRows + addRows);
        newToOldExt.head(oldRows) = cmc.newToOld;
        for (int i = 0; i < addRows; i++)
            newToOldExt(oldRows + i) = new_old_ids[i];
        cmc.newToOld = std::move(newToOldExt);
    }

    // Re-express each chain as compact indices via the now-extended oldToNew.
    std::vector<std::vector<int>> chainsCompact;
    chainsCompact.reserve(gStaleChains.size());
    for (const auto & chain : gStaleChains) {
        std::vector<int> compact;
        compact.reserve(chain.size());
        bool ok = true;
        for (int vid : chain) {
            if (vid < 0 || vid >= cmc.oldToNew.size() || cmc.oldToNew(vid) < 0) { ok = false; break; }
            compact.push_back(cmc.oldToNew(vid));
        }
        if (ok && !compact.empty())
            chainsCompact.push_back(std::move(compact));
        else
            fprintf(stderr, "[cmc] WARNING: dropped a stale chain that could not be fully remapped\n");
    }
    return chainsCompact;
}
