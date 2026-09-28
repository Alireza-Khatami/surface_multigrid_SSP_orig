#pragma once
// After decimation: subdivide the compact coarse mesh with the subdivided-sample
// tracker's subdivision (build_subdiv_mesh, uniform midpoint 1->4) and map every
// subdivided vertex back to the fine mesh with query_coarse_to_fine.
#include "coarse_mesh_compaction.h"

#include <cstdint>
#include <string>

// Subdivides (cmc.Vbase, cmc.Fout) until it has >= nTarget vertices, queries
// each vertex (a coarse face + barycentric) on the fine mesh, and writes the
// subdivided coarse connectivity twice, same vertex order and faces:
//   coarseObjPath  at the coarse positions
//   fineObjPath    at the fine correspondences
// Vertices on no coarse face (naked stale-chain vertices) keep their coarse
// position in both. OBJs are skipped above maxObjVerts vertices.
// Throws on invalid input (from build_subdiv_mesh).
void coarse_subdiv_c2f_export(const CoarseMeshCompaction & cmc, int64_t nTarget, int64_t maxObjVerts,
                              const std::string & coarseObjPath, const std::string & fineObjPath);
