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

// What coarse_subdiv_relax_export runs (all from the command line, main.cpp).
struct CoarseSubdivRelaxConfig {
    // "newton" (subdiv_relax), "solve_project" (subdiv_relax_solve_project) or
    // "explicit" (subdiv_relax_explicit: small Laplacian steps, curves and sheets
    // together, no linear solve).
    std::string method = "newton";
    double  curveAnchorTol = 3e-3;  // solve_project: adaptive seam anchors (x diag)
    int64_t maxIter = -1;           // newton: iterations per class (-1: until converged)
    bool    perCoarseFace = false;  // hold vertices on coarse vertices / edges; relax each coarse face's interior
    bool    noNewFolds = false;     // newton / explicit: no step may fold an unfolded triangle
    bool    localProjection = false;// newton: closest point reachable from the current location
    bool    jointPass = false;      // newton: curves + sheets in one pass on the symmetric graph
    bool    jointSolve = false;     // solve_project: one LU solve of curves + sheets (directed graph)
    // explicit
    double  explicitLambda = 0.5;
    int64_t explicitMaxIter = 20000;
    double  explicitTol = 1e-7;
    bool    explicitGlobalProj = false;  // global closest point instead of local
    // explicit: "uniform" (w_ij = 1) or "cotan" (cotangent weights of the
    // subdivided coarse mesh at its coarse positions, negative ones clamped to 0)
    std::string weights = "uniform";
};

// Relaxes C's vertices on the fine mesh and writes the result to objPath (same
// vertices and faces as the coarse_subdiv OBJs; skipped above maxObjVerts).
// Needs the fine .ma_struct and the struct-ID collapse gate (--mat_struct_check),
// which keeps the coarse vertices' struct IDs exact. Folded (noNewFolds, explicit
// logging): a subdivided triangle whose orientation disagrees with the majority of
// its coarse face's seed triangles. graphDir (empty: skip): the relaxation graph at
// the seed positions as sheets_/curves_/junctions_<graphStem>.ply (coloured by
// structure id). Explicit snapshots: <objPath without .obj>_it<N>.obj. Also checks
// the projector's BVHs against brute force. Throws if a consistency check fails.
void coarse_subdiv_relax_export(const CoarseMeshCompaction & cmc, const CoarseSubdivC2F & C,
                                const MatStruct & ms, const CoarseSubdivRelaxConfig & cfg,
                                int64_t maxObjVerts, const std::string & objPath,
                                const std::string & graphDir, const std::string & graphStem);
