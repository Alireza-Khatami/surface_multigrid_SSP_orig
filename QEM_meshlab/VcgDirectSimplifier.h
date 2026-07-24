// VcgDirectSimplifier — drive vcg's REAL TriEdgeCollapseQuadric on the medial
// mesh, bypassing QMAT's port (VcgQuadricSimplifier).  The point is to compare
// against / fall back to the canonical MeshLab implementation without copying
// any code: vcglib is pulled via FetchContent and we use exactly the recipe
// from vcglib/apps/tridecimator/tridecimator.cpp.
//
// Only enabled when CMake configures with -DQMAT_WITH_VCGLIB=ON (default).
// Selected at runtime via main_cli's --simplifier vcg-direct flag.
//
// IMPORTANT: this file MUST NOT include any vcg headers (those drag Eigen
// through with macros that conflict with our Wm4 / CGAL world).  All vcg types
// stay hidden inside VcgDirectSimplifier.cpp.

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

class SlabMesh;

// Snapshot of the simplified mesh + per-primitive attributes carried in shadow
// tables across collapses.  All arrays use compact 0..N-1 indices.  Indices in
// `faces` and `edges` reference `vertices`.
//
// Per-primitive attributes mirror QMAT's slab-side data: cluster_type and
// struct_ids on vertices, topo_type and struct_ids on edges, struct_id on
// faces.  Merge rules (applied inside VDEdgeCollapse::Execute):
//   - vertex struct_ids: union of both endpoints
//   - vertex cluster_type: survivor wins (matches MergeVertices)
//   - vertex topo flags: OR for seam/junction/boundary; sheet = both-sheets &&
//     no seam && no boundary (matches MergeVertices)
//   - edge struct_ids: union when two parent edges fold into one survivor edge
//   - edge topo_type: SlabEdge::CombineTopoType (flag-union)
//   - face struct_id: persists — face entity survives the collapse with the
//     same struct_id; vcg just re-points its vertex refs.
struct VcgDirectSnapshot
{
	std::vector<std::array<double, 3>> vertices;
	std::vector<std::array<int,    3>> faces;
	std::vector<std::array<int,    2>> edges;   // derived from face vertex pairs

	// Per-vertex (size == vertices.size()).
	// vertex_cluster_type values follow SlabVertex::ClusterType uint8_t encoding.
	// vertex_first_struct_id = -1 when the slab vertex's struct_ids set is empty;
	//   otherwise the smallest id in the set (sufficient for colouring).
	// vertex_struct_ids: full set per vertex (ascending), exposed for panel text
	//   + 3.3 struct_ids equality check; empty == not part of any struct.
	// vertex_topo_flags bits: 0=sheet, 1=seam, 2=junction, 3=boundary.
	// vertex_radius: SlabVertex::sphere.radius * bb_diagonal_length at extract
	//   time.  Seeded from sphere.radius in BuildFromSlab and merged with
	//   survivor-wins on each collapse.  Needed by .ma / .mat_typed exports.
	std::vector<uint8_t> vertex_cluster_type;
	std::vector<int>     vertex_first_struct_id;
	std::vector<std::vector<int>> vertex_struct_ids;
	std::vector<uint8_t> vertex_topo_flags;
	std::vector<double>  vertex_radius;

	// Per-face (size == faces.size()).
	std::vector<int>     face_struct_id;          // -1 if not part of any named struct

	// Per-edge (size == edges.size()).
	// edge_topo_type values follow SlabEdge::TopoType uint8_t encoding.
	// edge_struct_match = 1 when the edge has non-empty struct_ids and both
	//   endpoints carry exactly that set (i.e. struct-collapsible by QMAT's
	//   rule), 0 otherwise.
	// edge_struct_ids: full set per edge (ascending); empty == not a struct edge.
	std::vector<uint8_t> edge_topo_type;
	std::vector<int>     edge_first_struct_id;
	std::vector<std::vector<int>> edge_struct_ids;
	std::vector<uint8_t> edge_struct_match;

	// Per-vertex initial-MAT ancestor ids (indexes into original_positions).
	// Mirrors QMAT's SlabVertex::original_ancestors + SlabMesh::original_positions.
	std::vector<std::vector<unsigned>> vertex_original_ancestors;
	// Initial-MAT vertex positions (scaled), captured once at BuildFromSlab.
	std::vector<std::array<double, 3>> original_positions;

	// Per-edge last rejection reason (size == edges.size()).  255 = never
	// attempted / no rejection on record.  Otherwise a SlabMesh::RejectionReason
	// enum value (cast to uint8_t).
	std::vector<uint8_t> edge_last_rejection;
	// Per-edge survivor candidate position recorded at rejection time (scaled).
	// Only meaningful when edge_last_rejection[i] != 255.
	std::vector<std::array<double, 3>> edge_rejection_target;
};

// Fires immediately after each collapse executes, with the current surviving
// mesh state (compact 0..N-1 indices, same layout as VcgDirectResult).
using LiveUpdateCallback = std::function<void(const VcgDirectSnapshot& snap)>;

struct VcgDirectParams
{
	// One of these stops the run.  TargetFaceNum wins when both are set.
	// -1 means "unset"; if both are -1, target = FN()/2 (half the faces).
	int    TargetFaceNum         = -1;
	int    TargetVertexNum       = -1;

	// MeshLab QEM defaults — see meshlabplugins/filter_meshing/meshfilter.cpp
	// for the GUI defaults and tri_edge_collapse_quadric.h for the internal
	// struct defaults (which DIFFER from the GUI: see project_meshlab_qem_defaults).
	double QualityThr               = 0.3;   // sliver veto threshold
	double BoundaryQuadricWeight    = 1.0;   // MeshLab UI slider; we multiply by 0.5
	                                         // before assigning to vcg (matches
	                                         // meshfilter.cpp:14 effective weight).
	bool   OptimalPlacement         = true;  // "Optimal position" checkbox
	bool   PreserveTopology         = true;  // "Preserve topology" checkbox
	                                         // (vcg LinkConditions gate)
	bool   NormalCheck              = true;  // "Preserve normal" checkbox

	// VDE-specific gates layered on top of vcg's IsFeasible.  Each can be
	// toggled independently — turning one off does NOT disable the others.
	// EulerCheck:      Δχ = S - F_inc ≠ 0     →  EulerCharacteristicChange
	// SameTopoCheck:   endpoint cluster_type mismatch  →  DifferentClusterType
	// StructIdsCheck:  edge.struct_ids set != both endpoints' sets
	//                                                  →  struct_ids_sets_different
	bool   EulerCheck               = true;
	bool   SameTopoCheck            = true;
	bool   StructIdsCheck           = true;
};

struct VcgDirectResult
{
	// Final simplified mesh + per-primitive attributes.  Same layout as the
	// LiveUpdateCallback's snapshot, so callers can use a single rendering
	// codepath for both per-collapse updates and the post-run final state.
	VcgDirectSnapshot snapshot;

	int initial_vertex_count = 0;
	int initial_face_count   = 0;
	int final_vertex_count   = 0;
	int final_face_count     = 0;
	int collapses_performed  = 0;
};

// Build a vcg::TriMesh from `sm` (active vertices/faces only, scaled the same
// way as _mat_initial.off), run vcg's TriEdgeCollapseQuadric driven by `params`,
// and write the result into `out`.  Returns false if the slab mesh has no
// triangular faces (nothing to do).
//
// If `live_callback` is non-empty it fires once per executed collapse with the
// current surviving mesh state (compact 0..N-1 indices).  Default is no-op.
bool RunVcgDirectSimplify(const SlabMesh& sm,
                          const VcgDirectParams& params,
                          VcgDirectResult& out,
                          LiveUpdateCallback live_callback = {});
