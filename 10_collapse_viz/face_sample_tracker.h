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

// ---- Face flip tracker ----
// Track one original face's 3 vertices through the decimation via the sample
// tracker.  After every successful collapse, checks whether any tracked vertex
// moved in 3D and whether the tracked triangle flipped vs its original normal.
//
// Usage:
//   face_flip_tracker_init(face_idx);          // after sample_tracker_init()
//   // in do_next_step(), around sample_tracker_update():
//   face_flip_tracker_pre_update();            // before SSP_collapse_edge loop
//   sample_tracker_update();
//   face_flip_tracker_post_update();           // after update; sets flip flag if detected
//
// Visualisation (polyscope, called from visualizer.cpp):
//   face_flip_tracker_show_viz();

// Call after sample_tracker_init(). face_idx is a gFO row index.
void face_flip_tracker_init(int face_idx);

// Snapshot current 3D positions of the 3 tracked vertices.
// Call once per do_next_step() BEFORE the SSP_collapse_edge attempt loop.
void face_flip_tracker_pre_update();

// Compare positions after the collapse + sample_tracker_update().
// Records trajectory points for any vertex that moved and checks for a flip.
// Call AFTER sample_tracker_update() on a successful collapse.
void face_flip_tracker_post_update();

// Query functions used by visualizer.cpp.
bool face_flip_tracker_enabled();
bool face_flip_tracker_flip_detected();
int  face_flip_tracker_flip_at_collapse();
int  face_flip_tracker_face_idx();

// Current 3D position of tracked vertex i (0/1/2) via the sample tracker.
Eigen::Vector3d face_flip_tracker_cur_pos(int i);

// Full position trajectory of tracked vertex i (one entry per move event).
const std::vector<Eigen::Vector3d>& face_flip_tracker_traj(int i);

// ---- Vertex watch tracker ----
// Given a fine-mesh vertex ID, tracks which coarse triangle currently contains
// it (cur_BF).  Before each collapse the cur_BF is snapshotted; if either
// collapse endpoint (survivor s or absorbed d) appears in that snapshot the
// watch triggers, decimation is stopped, and a message is shown in the UI.
//
// Usage:
//   vertex_watch_set(fine_vtx_id);          // activate (can call at any time)
//   vertex_watch_clear();                    // deactivate
//   // in do_next_step(), before the tries loop:
//   vertex_watch_pre_step();
//   // after SSP_collapse_edge succeeds, before sample_tracker_update():
//   vertex_watch_check_collapse(s, d);
//
// Query (used by visualizer.cpp):
//   vertex_watch_active()
//   vertex_watch_triggered()
//   vertex_watch_trigger_at_collapse()
//   vertex_watch_fine_vtx()
//   vertex_watch_cur_BF()

void vertex_watch_set(int fine_vtx_id);
void vertex_watch_clear();
void vertex_watch_pre_step();
void vertex_watch_check_collapse(int s, int d);
bool vertex_watch_active();
bool vertex_watch_triggered();
int  vertex_watch_trigger_at_collapse();
int  vertex_watch_fine_vtx();
Eigen::Vector3i vertex_watch_cur_BF();
