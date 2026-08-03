#pragma once
#include <string>
#include <Eigen/Dense>

// Seed vertex samples (one per fine mesh vertex) plus n_per_face interior
// barycentric samples per fine mesh triangle.  Call once after init_ssp().
void sample_tracker_init(int n_per_face = 2);

// Remap all samples through the latest entry in gDecInfo.
// Call immediately after every successful SSP_collapse_edge step.
void sample_tracker_update();

// Write fine-mesh seeds and coarse-mesh correspondences to text files.
void sample_tracker_save(const std::string& fine_path,
                         const std::string& coarse_path);

// Register polyscope point clouds in world space (normal view).
// No-op when C2F_VIZ_DIAGNOSTIC is not defined.
void sample_tracker_show();

// Register per-collapse ring sample points on their UV surfaces in the canonical frame.
// uv_pre_3d / uv_post_3d are the already-rotated UV mesh vertex positions (from
// compute_ring_geometry + rotation); FUV_pre / FUV_post are the face connectivity
// matrices that index into those vertex arrays.
// No-op when C2F_VIZ_DIAGNOSTIC is not defined.
void sample_tracker_show_canonical(const Eigen::MatrixXd& uv_pre_3d,
                                   const Eigen::MatrixXd& uv_post_3d,
                                   const Eigen::MatrixXi& FUV_pre,
                                   const Eigen::MatrixXi& FUV_post);
