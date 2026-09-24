#pragma once
#include <Eigen/Core>
#include <vector>

// The single shared compaction of the live decimated mesh (gV/gF) into a
// coarse vertex/face list. Every exporter that writes "the simplified/coarse
// mesh" (simplified_*.obj, the .c2f bundle's coarseV, *_simp_visualize_info.json)
// must build its vertex numbering from this — building an independent
// remap makes vertex index i refer to a different physical point in each file.
//
// A face is kept iff it is not dead (igl::collapse_edge tombstone) and has
// no vertex at infinity (the boundary-to-infinity cap). Vbase/Fout come from
// igl::remove_unreferenced over exactly those kept faces.
//
// After build_compact_coarse_mesh(), indices 0..NC-1 are this face-referenced
// coarse mesh. extend_with_stale_chains() (below) may append further, naked
// (non-face-referenced) vertices past NC — Vbase/newToOld/oldToNew grow, but
// NC itself stays fixed at the original face-referenced count so callers can
// always tell the two regions apart.
struct CoarseMeshCompaction {
    Eigen::MatrixXd Vbase;    // Vbase.rows() x 3 compacted vertex positions (NC rows until extended)
    Eigen::MatrixXi Fout;     // FC x 3 compacted coarse faces (indices into Vbase, always < NC)
    Eigen::VectorXi newToOld; // size Vbase.rows(); newToOld(i) = original (gV) vertex id
    Eigen::VectorXi oldToNew; // size gV.rows(); oldToNew(v) = compact index, or -1 if not live/extended
    Eigen::VectorXi faceOrigIdx; // size FC; faceOrigIdx(f) = original (gF) row this compact face came from
    int NC = 0;               // face-referenced coarse vertex count, fixed at build time
};

CoarseMeshCompaction build_compact_coarse_mesh(const Eigen::MatrixXd & gV, const Eigen::MatrixXi & gF);

// Appends stale-chain vertices (naked, not referenced by any live face) to
// cmc.Vbase/newToOld/oldToNew, past the existing NC entries. A chain vertex
// already covered by the coarse mesh (oldToNew(v) >= 0) keeps its existing
// index — it is not duplicated. New vertices are assigned indices in a fixed
// deterministic order: gStaleChains walked in order, first-seen-gets-next-index.
//
// Returns the chains re-expressed as compact indices (0..Vbase.rows()-1).
// Throws std::runtime_error on an empty chain or a vertex with no valid gV row.
//
// Call this AFTER build_compact_coarse_mesh() and BEFORE reading cmc.Vbase.rows()
// as the final vertex count — cmc.NC still reports the pre-extension count.
std::vector<std::vector<int>> extend_with_stale_chains(
    CoarseMeshCompaction & cmc,
    const Eigen::MatrixXd & gV,
    const std::vector<std::vector<int>> & gStaleChains);
