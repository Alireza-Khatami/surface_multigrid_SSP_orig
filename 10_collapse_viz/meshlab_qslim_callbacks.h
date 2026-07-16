#pragma once
#include "meshlab_qslim_config.h"
#include <decimate_func_types.h>
#include <Eigen/Core>
#include <vector>

// Initialize per-vertex MeshLab quadrics and fill all three SSP callbacks.
// Implements vcglib's QEM (area-weighted quadrics + boundary reinforcement +
// ScaleIndependent normalisation + SVD/fullPivLu placement + soft/hard quality
// and normal checks), adapted for Eigen matrices.
//
// Parameters:
//   V, F, E, EF  — current mesh arrays (after connect_boundary_to_infinity)
//   VF           — per-vertex face list (maintained live by SSP_collapse_edge)
//   v1, v2       — working ints, written by pre_fn and read by post_fn;
//                  must outlive the returned callbacks
//   cfg          — all algorithm settings; read from meshlab_qem.ini
//
// Source reference: QEM_meshlab/VcgQuadricSimplifier.{h,cpp}
// Call once after init_ssp() completes to enable per-edge rejection logging.
// Suppressed during initialization to avoid flooding from the initial full-heap build.
void meshlab_enable_cost_logging();

void meshlab_setup_callbacks(
    const Eigen::MatrixXd                        & V,
    const Eigen::MatrixXi                        & F,
    const Eigen::MatrixXi                        & E,
    const Eigen::MatrixXi                        & EF,
    const std::vector<std::vector<int>>          & VF,
    int                                          & v1,
    int                                          & v2,
    const MeshlabQEMConfig                       & cfg,
    decimate_cost_and_placement_func             & cost_fn,
    decimate_pre_collapse_func                   & pre_fn,
    decimate_post_collapse_func                  & post_fn);
