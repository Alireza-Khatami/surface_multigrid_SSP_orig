#ifndef QUERY_COARSE_TO_FINE_H
#define QUERY_COARSE_TO_FINE_H

#include <Eigen/Core>
#include <vector>
#include <single_collapse_data.h>
#include <vector_mod.h>
#include <compute_barycentric.h>
#include <fstream>

#include <igl/min.h>
#include <igl/find.h>
#include <igl/setunion.h>
#include <igl/unique.h>
#include <igl/parallel_for.h>

// Optional per-query statistics of the clamp-and-renormalize step of the walk.
// At each undone collapse the query point is mapped into the pre-collapse UV
// chart; if it lies outside every pre-collapse triangle, the least-outside
// triangle is chosen, its negative barycentric coordinates are set to 0 and the
// rest renormalized (the point is snapped onto that triangle's border).
// All vectors have one entry per query (row of BC).
struct C2FQueryStats {
  Eigen::VectorXi steps;       // collapses undone by the walk
  Eigen::VectorXi clamped;     // steps that had to clamp (most negative coordinate < -1e-12)
  Eigen::VectorXi farOutside;  // steps outside every triangle by >= 1 in barycentric terms (row 0 is taken)
  Eigen::VectorXd maxNegBary;  // largest negative coordinate clamped away (0: never clamped)
  Eigen::VectorXd sumNegBary;  // sum of those over the walk
  Eigen::VectorXd maxSnapRel;  // largest UV snap distance / longest edge of the chosen UV triangle
};

// stats (optional): filled with the per-query statistics above.
void query_coarse_to_fine(
  const std::vector<single_collapse_data> & decInfo,
  const Eigen::VectorXi & IM,
  const std::vector<std::vector<int>> & decIM,
  const Eigen::VectorXi & IMF,
  const Eigen::VectorXi & faceSheetID,
  Eigen::MatrixXd & BC,
  Eigen::MatrixXi & BF,
  Eigen::VectorXi & FIdx,
  C2FQueryStats * stats = nullptr);
#endif