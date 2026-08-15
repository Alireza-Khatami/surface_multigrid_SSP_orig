#pragma once
#include <string>
#include <Eigen/Dense>

// Configure a subset of fine-mesh vertices to track and a file to receive the
// per-collapse-step walk trace.  Call BEFORE sample_tracker_init().
// vertex_list_path: text file with one fine_vertex_id (int) per line.
// trace_output_path: receives one line per tracked sample per update step.
void sample_tracker_set_trace(const std::string& vertex_list_path,
                               const std::string& trace_output_path);

// Seed vertex samples (one per fine mesh vertex, or only the subset configured
// via sample_tracker_set_trace) plus n_total interior barycentric samples
// distributed across fine mesh triangles proportional to face area.
// Call once after init_ssp().
// NOTE: interior samples are currently disabled — see INTERIOR_SAMPLES comment.
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

// Export a deformed fine mesh: original topology (gFO), but each vertex position is
// replaced by its coarse correspondence (interpolated via cur_BC / cur_BF).
// Vertices with no vertex-sample entry keep their original gVO position.
void sample_tracker_export_deformed_mesh(const std::string& path);

// Register per-collapse ring sample points on their UV surfaces in the canonical frame.
// uv_pre_3d / uv_post_3d are the already-rotated UV mesh vertex positions (from
// compute_ring_geometry + rotation); FUV_pre / FUV_post are the face connectivity
// matrices that index into those vertex arrays.
// No-op when C2F_VIZ_DIAGNOSTIC is not defined.
void sample_tracker_show_canonical(const Eigen::MatrixXd& uv_pre_3d,
                                   const Eigen::MatrixXd& uv_post_3d,
                                   const Eigen::MatrixXi& FUV_pre,
                                   const Eigen::MatrixXi& FUV_post);
