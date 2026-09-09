#pragma once
// SSP_exhaustion_full_diagnostic.h
//
// After queue exhaustion, runs the full SSP_collapse_edge pipeline (including
// joint_lscm UV validity checks) on a DEEP COPY of the mesh, so that UV
// face-flip and angle-sum rejection reasons are captured.  The original mesh
// is NOT modified.
//
// Call once from the queue-exhaustion branch in main.cpp.

#include <Eigen/Core>
#include <vector>
#include <tuple>
#include <string>

// Quadric = (A, b, c) as used by qslim: cost = p'Ap + 2b'p + c.
using SSP_Quadric = std::tuple<Eigen::MatrixXd, Eigen::RowVectorXd, double>;

// Runs a full collapse loop (qslim cost + joint_lscm UV checks) on a copy of
// the exhausted mesh state.  All rejection events are logged to:
//   <out_dir>/exhausted_queue_full_rejections.log
//
// Parameters:
//   V, F, E, EMAP, EF, EI, C, EQ  — exhausted mesh state (read-only; deep copied internally)
//   quadrics                       — per-vertex QEM quadrics at exhaustion time
//   faceSheetID                    — face→sheet mapping (read-only; shared with original)
//   out_dir                        — output directory (with trailing slash or separator)
//   collapse_count                 — gCollapseCount at exhaustion (for log header)
void SSP_exhaustion_full_diagnostic(
    const Eigen::MatrixXd &              V,
    const Eigen::MatrixXi &              F,
    const Eigen::MatrixXi &              E,
    const Eigen::VectorXi &              EMAP,
    const Eigen::MatrixXi &              EF,
    const Eigen::MatrixXi &              EI,
    const Eigen::MatrixXd &              C,
    const Eigen::VectorXi &              EQ,
    const std::vector<SSP_Quadric> &     quadrics,
    const Eigen::VectorXi &              faceSheetID,
    const std::string &                  out_dir,
    int                                  collapse_count);
