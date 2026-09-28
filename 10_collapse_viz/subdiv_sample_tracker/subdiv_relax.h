#pragma once
// Structure-aware relaxation of the subdivided vertices (md_files/subdiv_relax_plan.md).
//
// Graph (row i = the vertices that pull on i, uniform weights 1/deg):
//   JUNCTION row: empty (never moves)
//   CURVE row   : CURVE/JUNCTION neighbours sharing a seam/boundary id
//   SHEET row   : any neighbour sharing a sheet id
// Roles by priority: JUNCTION > CURVE (seam or boundary) > SHEET.
//
// Relaxation runs until the positions stop changing: the largest move of one
// iteration <= tol * bbox diagonal. After every step each vertex is projected
// back onto its own structure of the original MAT (a curve vertex onto the
// edges of its seam/boundary ids, a sheet vertex onto the faces of its sheet
// ids). The fixed point is x = Pi(x + lambda L x): the Laplacian has no
// component along the vertex's own structure.

#include "subdiv_mesh.h"
#include "subdiv_struct_ids.h"
#include "../load_matstruct.h"

#include <Eigen/Core>
#include <cstdint>
#include <string>
#include <vector>

enum RelaxRole : uint8_t {
    RELAX_SHEET    = 0,
    RELAX_CURVE    = 1,
    RELAX_JUNCTION = 2,
};

struct RelaxGraph {
    std::vector<uint8_t> role;     // Vs
    std::vector<int64_t> rowOffs;  // Vs+1; row i = cols[rowOffs[i] .. rowOffs[i+1]), ascending
    std::vector<int32_t> cols;

    // Build statistics.
    int64_t nUndirected = 0;          // undirected edges of M.F
    int64_t kind[5] = {};             // kept i<-j: S<-S, S<-C, S<-J, C<-C, C<-J
    int64_t droppedSheetSheet = 0;    // sheet-sheet pairs sharing no sheet id (expected 0)
    int64_t sheetWithoutSheetId = 0;  // SHEET vertices whose set has no sheet id
    int64_t isolatedSheet = 0;        // SHEET rows with no entry
    int64_t isolatedCurve = 0;        // CURVE rows with no entry
    int64_t nRole[3] = {};
};

// haveStruct = false: no .ma_struct, every vertex is SHEET and the graph is the
// plain two-way mesh adjacency.
RelaxGraph build_relax_graph(const SubdivMesh & M,
                             const StructPalette & pal,
                             const std::vector<int32_t> & setId,
                             const MatStruct * ms);

// Raw binary graph file (.slg), layout in subdiv_relax.cpp. V = seed positions.
void save_relax_graph(const std::string & path,
                      const Eigen::MatrixXd & V,
                      const RelaxGraph & G,
                      const StructPalette & pal,
                      const std::vector<int32_t> & setId);

enum class RelaxSolver {
    // Per iteration: one tangent-plane linear solve of T_i^T (L x)_i = 0 (curves
    // first, then sheets with the curves fixed), then projection. Converges in
    // few iterations; the default.
    Newton,
    // x <- Pi(x + lambda L x). The literal scheme; very slow on large graphs,
    // kept as the reference the tests compare Newton against.
    Jacobi,
};

struct RelaxOptions {
    RelaxSolver solver = RelaxSolver::Newton;
    double  tol     = 1e-12;   // relative to the bbox diagonal of VO
    int64_t maxIter = -1;      // -1: 100000 for Newton (per class), 1000000 for Jacobi
    double  lambda  = 0.5;     // Jacobi step
    bool    verbose = true;
    int     curveAnchors = 0;  // solve_project experiment only: fixed vertices per curve group
    // solve_project experiment only, used when curveAnchors == 0: adaptive anchors,
    // Douglas-Peucker on each seam chain with this tolerance (x bbox diagonal).
    double  curveAnchorTol = 0.0;
};

struct RelaxReport {
    int64_t itersCurve = 0, itersSheet = 0;  // Newton: per class; Jacobi: itersSheet = total
    double  deltaCurve = 0.0, deltaSheet = 0.0;  // last max move / diagonal
    bool    converged = false;
    int64_t halvings = 0;          // Newton: total line-search step halvings
    int64_t lineSearchStalls = 0;  // Newton: passes stopped because no step decreased E
    int64_t nFree = 0, nFixed = 0;
    int64_t nPinned = 0, nPinnedCurve = 0;
    int64_t nCurveAnchors = 0;    // solve_project experiment: extra fixed curve vertices
    std::vector<int64_t> anchors; // solve_project experiment: every anchor vertex chosen  // one vertex per group nothing fixed pulls on
    int64_t plainSteps = 0;       // Newton: plain projected steps taken between Newton rounds
    int64_t seedOffStructure = 0; // seed not on its own structure (kept fixed; expected 0)
    double  maxMove = 0.0, meanMove = 0.0;       // |relaxed - seed| / diagonal
    // Fixed-point check: max |tangential (L x)_i| / mean edge length, over free
    // vertices strictly inside a face (sheet) or an edge (curve).
    double  residualSheet = 0.0, residualCurve = 0.0;
    int64_t nResidualSheet = 0, nResidualCurve = 0;
    // The definition itself: max move of one step x <- Pi(x + 0.5 L x) from the
    // result, / diagonal. Must be <= tol for "stopped changing".
    double  jacobiStepMove = 0.0;
    int64_t jacobiStepVertex = -1;
    // Consistency checks, all expected 0.
    int64_t fixedMoved = 0;       // junction / fixed vertex not bit-identical to seed
    int64_t posMismatch = 0;      // V != interpolation of (fineFace, fineBary)
    int64_t badBary = 0;          // negative or not summing to 1 (1e-12)
    int64_t offStructure = 0;     // fine face / edge not in the vertex's own structure
};

// Moves M.V and rewrites M.fineFace / M.fineBary. Carriers are untouched (they
// describe the seed). Vertices on no face are fixed.
RelaxReport subdiv_relax(SubdivMesh & M,
                         const Eigen::MatrixXd & VO,
                         const Eigen::MatrixXi & FO,
                         const MatStruct * ms,
                         const StructPalette & pal,
                         const std::vector<int32_t> & setId,
                         const RelaxGraph & G,
                         const RelaxOptions & opt = RelaxOptions());

// EXPERIMENT (subdiv_relax_solve_project.cpp): one 3D solve of L x = 0 per pass
// (curves, then sheets with curves fixed), then projection onto each vertex's
// own structure. No iteration; opt.solver / maxIter / lambda are ignored. The
// report's jacobiStepMove says how far the result is from a resting state.
RelaxReport subdiv_relax_solve_project(SubdivMesh & M,
                                       const Eigen::MatrixXd & VO,
                                       const Eigen::MatrixXi & FO,
                                       const MatStruct * ms,
                                       const StructPalette & pal,
                                       const std::vector<int32_t> & setId,
                                       const RelaxGraph & G,
                                       const RelaxOptions & opt = RelaxOptions());

// Path for opening a file for writing. On Windows, returns the absolute path in
// long-path form (prefix backslash backslash ? backslash), so output paths over 260 characters (long run folders
// + long mesh stems) do not fail. Elsewhere, returns path unchanged.
std::string subdiv_long_path(const std::string & path);

struct MeshQuality {
    double  edgeCV = 0.0;               // std / mean of edge lengths
    double  minAngle = 0.0, p1 = 0.0, p5 = 0.0, median = 0.0;  // per-triangle min angle, degrees
    int64_t degenerate = 0;             // area <= 1e-14 * diag^2
    int64_t flippedVsRef = 0;           // normal . reference normal < 0 (ref given)
};

MeshQuality subdiv_mesh_quality(const Eigen::MatrixXd & V, const Eigen::MatrixXi & F,
                                const Eigen::MatrixXd * Vref = nullptr);
