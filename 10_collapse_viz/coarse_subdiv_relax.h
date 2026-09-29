#pragma once
// Structure-aware relaxation of the subdivided coarse mesh on the FINE MAT.
//
// The subdivided coarse mesh (coarse_subdiv_c2f.h) gives the connectivity; its
// vertices start at their coarse -> fine correspondences. The existing
// relaxation (subdiv_relax_solve_project) evens out their spacing, projecting
// every vertex onto its own structure of the fine mesh: sheet vertices onto
// their sheet's faces, seam/boundary vertices onto their curve's edges,
// junctions fixed.
//
// Structure IDs come from the coarse mesh the subdivision was built on:
//   coarse vertex : its struct IDs as tracked by simp_viz_tracker
//   coarse face   : sheet IDs of its gF row (= the fine face it started as)
//   coarse edge   : seam/boundary IDs of every fine curve edge joining an
//                   ancestor of one endpoint to an ancestor of the other
//                   (ancestors from simp_viz_tracker)
// and are spread to the subdivided vertices by build_struct_sets.
#include "coarse_mesh_compaction.h"
#include "coarse_subdiv_c2f.h"
#include "load_matstruct.h"

#include <cstdint>
#include <string>

// Relaxes C's vertices on the fine mesh and writes the result to objPath (same
// vertices and faces as the coarse_subdiv OBJs; skipped above maxObjVerts).
// Needs the fine .ma_struct and the struct-ID collapse gate (--mat_struct_check),
// which keeps the coarse vertices' struct IDs exact. method: "newton" (subdiv_relax)
// or "solve_project" (subdiv_relax_solve_project). curveAnchorTol: adaptive seam
// anchors, solve_project only, as --subdiv_relax_anchor_tol. maxIter: newton
// iterations per class (-1: until converged). Throws if a
// consistency check fails.
void coarse_subdiv_relax_export(const CoarseMeshCompaction & cmc, const CoarseSubdivC2F & C,
                                const MatStruct & ms, const std::string & method, double curveAnchorTol,
                                int64_t maxIter,
                                int64_t maxObjVerts, const std::string & objPath);
