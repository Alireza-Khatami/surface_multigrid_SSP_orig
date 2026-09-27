#pragma once
// Tracks every vertex of the subdivided fine mesh (subdiv_mesh.h) through the
// decimation, as a (gF face, barycentric) point on the decimating mesh.
// The subdivided mesh itself is never decimated: its vertices, faces and
// fine-mesh coordinates are fixed; only each vertex's position on gV/gF moves.
//
// Per collapse, samples on the collapse's pre-ring faces are cast
// UV_pre -> UV_post per sheet (the same cast the old sample trackers used).
// Corner vertex ids stored per sample may reference vertices absorbed by later
// collapses; they are resolved through a survivor redirect (d -> s) instead of
// rewriting every sample after every collapse.

#include "subdiv_mesh.h"
#include "subdiv_struct_ids.h"
#include "subdiv_relax.h"
#include "../coarse_mesh_compaction.h"

#include <Eigen/Core>
#include <cstdint>
#include <string>

// Build the subdivided mesh (>= nTarget vertices), its struct IDs (from ms,
// the parsed .ma_struct; nullptr = none), relax it on its own structures until
// it stops changing (relax = true; subdiv_relax.h) and seed every vertex on its
// fine face. Call once after init_ssp() and load_matstruct(). Throws on invalid
// input or if the relaxation does not converge.
void subdiv_tracker_init(int64_t nTarget, const MatStruct * ms, bool relax = true);

bool subdiv_tracker_enabled();

// Remap samples through gDecInfo.back(). Call right after every successful
// SSP_collapse_edge; s / d are the survivor / absorbed vertex of that collapse.
void subdiv_tracker_update(int s, int d);

// Write subdiv_<stem>.sdt (raw binary, layout in subdiv_tracker.cpp) and run the
// end-of-run consistency checks. Throws if a sample cannot be resolved on the
// compact coarse mesh.
void subdiv_tracker_save(const CoarseFaceLookup & lookup, const std::string & path);

// Subdivided connectivity with every vertex at its current coarse position.
// Skipped (with a message) above maxVerts vertices, since OBJ text gets huge.
void subdiv_tracker_export_deformed_obj(const std::string & path, int64_t maxVerts = 2000000);

// The subdivided mesh at its fine-mesh positions (before any collapse).
// Same vertex order and faces as the deformed OBJ. Same size limit.
void subdiv_tracker_export_fine_obj(const std::string & path, int64_t maxVerts = 2000000);

// The subdivided mesh before relaxation (exact midpoint positions). Same
// connectivity and vertex order as the fine OBJ. Only after a relaxed init.
void subdiv_tracker_export_seed_obj(const std::string & path, int64_t maxVerts = 2000000);

// The relaxation graph (.slg, layout in subdiv_relax.cpp). Only after a relaxed init.
void subdiv_tracker_export_graph(const std::string & path);

// ---- read access (debug trackers, viewer) ----
bool subdiv_tracker_relaxed();
const Eigen::MatrixXd & subdiv_tracker_seed_positions();  // empty unless relaxed
const RelaxGraph &      subdiv_tracker_graph();           // empty unless relaxed
const SubdivMesh &     subdiv_tracker_mesh();
const StructPalette &  subdiv_tracker_palette();
const std::vector<int32_t> & subdiv_tracker_set_ids();
int  subdiv_tracker_collapses();      // collapses seen by subdiv_tracker_update
bool subdiv_tracker_is_tracked(int i);  // false for an original vertex on no face
int  subdiv_tracker_cur_face(int i);    // gF row
Eigen::Vector3d subdiv_tracker_cur_bary(int i);
Eigen::Vector3i subdiv_tracker_cur_corners(int i);  // live gV ids (redirect applied)
Eigen::Vector3d subdiv_tracker_cur_pos(int i);      // position on the decimating mesh
void subdiv_tracker_cur_positions(Eigen::MatrixXd & P); // all vertices, Vs x 3
