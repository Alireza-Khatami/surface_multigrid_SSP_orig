#ifndef PARTITION_INTO_SHEETS_H
#define PARTITION_INTO_SHEETS_H

#include <Eigen/Core>

// BFS partition of original mesh F into manifold sheets.
// Crosses only edges shared by exactly 2 faces (manifold interior edges).
// Stops at non-manifold edges (3+ faces) and boundary edges (1 face).
//
// Call this on the ORIGINAL mesh BEFORE connect_boundary_to_infinity.
//
// Inputs:
//   F  #F×3 original mesh faces
// Outputs:
//   faceSheetID  #F VectorXi, faceSheetID(f) = 0-indexed sheet of face f
//   numSheets    total number of sheets
void partition_into_sheets(
    const Eigen::MatrixXi & F,
    Eigen::VectorXi & faceSheetID,
    int & numSheets);

#endif
