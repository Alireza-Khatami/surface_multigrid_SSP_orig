#pragma once
#include "coarse_mesh_compaction.h"
#include <string>

// Mesh writers for the compact coarse mesh. All use cmc's numbering, so vertex
// i is the same point in every file. Each throws std::runtime_error on I/O failure.

// v + f lines, then one 1-based l line per stale chain.
void write_coarse_obj(const std::string & path, const CoarseMeshCompaction & cmc);

// ASCII PLY: vertices, faces, and stale-chain edges as an "edge" element.
void write_coarse_ply(const std::string & path, const CoarseMeshCompaction & cmc);

// All compact vertices + l lines only (MeshLab doesn't draw PLY edges).
void write_stale_chains_obj(const std::string & path, const CoarseMeshCompaction & cmc);
