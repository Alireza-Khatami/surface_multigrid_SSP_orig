#pragma once
#include <Eigen/Core>

// The single shared compaction of the live decimated mesh (gV/gF) into a
// coarse vertex/face list. Every exporter that writes "the simplified/coarse
// mesh" (simplified_*.obj, the .c2f bundle's coarseV, *_simp_visualize_info.json)
// must build its vertex numbering from this — building an independent
// remap makes vertex index i refer to a different physical point in each file.
//
// A face is kept iff it is not dead (igl::collapse_edge tombstone) and has
// no vertex at infinity (the boundary-to-infinity cap). Vbase/Fout come from
// igl::remove_unreferenced over exactly those kept faces.
struct CoarseMeshCompaction {
    Eigen::MatrixXd Vbase;    // NC x 3 compacted coarse vertex positions
    Eigen::MatrixXi Fout;     // FC x 3 compacted coarse faces (indices into Vbase)
    Eigen::VectorXi newToOld; // size NC; newToOld(i) = original (gV) vertex id
    Eigen::VectorXi oldToNew; // size gV.rows(); oldToNew(v) = compact index, or -1 if not live
    Eigen::VectorXi faceOrigIdx; // size FC; faceOrigIdx(f) = original (gF) row this compact face came from
};

CoarseMeshCompaction build_compact_coarse_mesh(const Eigen::MatrixXd & gV, const Eigen::MatrixXi & gF);
