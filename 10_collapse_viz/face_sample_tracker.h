#pragma once
#include <string>
#include <Eigen/Dense>

// Seed vertex samples (one per fine mesh vertex) plus n_total interior
// barycentric samples distributed across fine mesh triangles proportional
// to face area (larger faces receive more samples).  Call once after init_ssp().
void sample_tracker_init(int n_total = 2000);

// Remap all samples through the latest entry in gDecInfo.
// Call immediately after every successful SSP_collapse_edge step.
void sample_tracker_update();

// Write fine-mesh seeds and coarse-mesh correspondences to text files.
// vertices_path receives one row per original fine-mesh vertex:
//   fine_vertex_id  coarse_face_id  b0 b1 b2  bv0 bv1 bv2
void sample_tracker_save(const std::string& fine_path,
                         const std::string& coarse_path,
                         const std::string& vertices_path);

// Register polyscope point clouds in world space (normal view).
// No-op when C2F_VIZ_DIAGNOSTIC is not defined.
void sample_tracker_show();

// Register fine-mesh vertex positions and their coarse correspondences as
// separate point clouds with a fine→coarse vector quantity.
// No-op when C2F_VIZ_DIAGNOSTIC is not defined.
void sample_tracker_show_vertices();

// Register per-collapse ring sample points on their UV surfaces in the canonical frame.
// uv_pre_3d / uv_post_3d are the already-rotated UV mesh vertex positions (from
// compute_ring_geometry + rotation); FUV_pre / FUV_post are the face connectivity
// matrices that index into those vertex arrays.
// No-op when C2F_VIZ_DIAGNOSTIC is not defined.
void sample_tracker_show_canonical(const Eigen::MatrixXd& uv_pre_3d,
                                   const Eigen::MatrixXd& uv_post_3d,
                                   const Eigen::MatrixXi& FUV_pre,
                                   const Eigen::MatrixXi& FUV_post);
