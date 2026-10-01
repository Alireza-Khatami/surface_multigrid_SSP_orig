#pragma once
// After decimation: subdivide the compact coarse mesh with the subdivided-sample
// tracker's subdivision (build_subdiv_mesh, uniform midpoint 1->4) and map every
// subdivided vertex back to the fine mesh with query_coarse_to_fine.
#include "coarse_mesh_compaction.h"
#include "subdiv_sample_tracker/subdiv_mesh.h"

#include <Eigen/Core>
#include <cstdint>
#include <string>
#include <vector>

struct CoarseSubdivC2F {
    SubdivMesh S;                   // subdivided (cmc.Vbase, cmc.Fout); carriers on the coarse mesh
    Eigen::MatrixXd P;              // Vs x 3 fine correspondences (coarse position when not mapped)
    std::vector<int32_t> fineFace;  // gFO row per vertex, -1 when not mapped
    Eigen::MatrixXd fineBary;       // Vs x 3, in gFO.row(fineFace) corner order
    // Clamp-and-renormalize statistics of each vertex's backward walk
    // (query_coarse_to_fine's C2FQueryStats); 0 for vertices not queried.
    Eigen::VectorXi walkSteps, clampedSteps, farSteps;
    Eigen::VectorXd maxNegBary, sumNegBary, maxSnapRel;
};

// Subdivides (cmc.Vbase, cmc.Fout) until it has >= nTarget vertices and queries
// each vertex (a coarse face + barycentric) on the fine mesh. Vertices on no
// coarse face (naked stale-chain vertices) are not mapped. Throws on invalid
// input (from build_subdiv_mesh).
CoarseSubdivC2F coarse_subdiv_c2f_build(const CoarseMeshCompaction & cmc, int64_t nTarget);

// Writes the subdivided coarse connectivity twice, same vertex order and faces:
//   coarseObjPath  at the coarse positions
//   fineObjPath    at the fine correspondences
// Skipped above maxObjVerts vertices.
void coarse_subdiv_c2f_write(const CoarseSubdivC2F & C, int64_t maxObjVerts,
                             const std::string & coarseObjPath, const std::string & fineObjPath);

// Per-vertex clamp statistics of the coarse -> fine walk, one CSV row per
// subdivided vertex (same order as the OBJs): vid, mapped, steps, clamped_steps,
// far_steps, max_neg_bary, sum_neg_bary, max_snap_rel.
void coarse_subdiv_c2f_write_clamp_csv(const CoarseSubdivC2F & C, const std::string & csvPath);
