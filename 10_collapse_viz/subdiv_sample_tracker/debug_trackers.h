#pragma once
// Face-flip and vertex-watch debug trackers, reading positions from the
// subdivided-mesh tracker (subdiv_tracker.h). Same API and behavior as the
// versions that lived in the old face_sample_tracker.
//
// Both take fine-mesh (gVO) vertex / gFO face ids: an original vertex keeps
// its id in the subdivided mesh, so vertex v is tracked as subdivided vertex v.
// They need the subdivided tracker to be enabled (--n_subdiv_samples).

#include <Eigen/Core>
#include <vector>

// ---- Face flip tracker ----
// Tracks one original face's 3 vertices. After every successful collapse,
// checks whether any tracked vertex moved and whether the triangle they span
// flipped against the face's original normal.
//
// Usage:
//   face_flip_tracker_init(face_idx);     // after subdiv_tracker_init()
//   face_flip_tracker_pre_update();       // in do_next_step(), before the collapse attempts
//   subdiv_tracker_update(s, d);
//   face_flip_tracker_post_update();      // after the update, on a successful collapse

void face_flip_tracker_init(int face_idx);  // face_idx: gFO row
void face_flip_tracker_pre_update();
void face_flip_tracker_post_update();

bool face_flip_tracker_enabled();
bool face_flip_tracker_flip_detected();
int  face_flip_tracker_flip_at_collapse();
int  face_flip_tracker_face_idx();
Eigen::Vector3d face_flip_tracker_cur_pos(int i);                  // i = 0, 1, 2
const std::vector<Eigen::Vector3d> & face_flip_tracker_traj(int i); // one entry per move

// ---- Vertex watch tracker ----
// Watches which coarse triangle currently holds a fine vertex. Before each
// collapse its corners are snapshotted; if the collapse's survivor or absorbed
// vertex shares a live face with one of those corners, the watch triggers.
//
// Usage:
//   vertex_watch_set(fine_vtx_id) / vertex_watch_clear()
//   vertex_watch_pre_step();              // in do_next_step(), before the collapse attempts
//   vertex_watch_check_collapse(s, d);    // after a successful collapse, before the tracker update

void vertex_watch_set(int fine_vtx_id);
void vertex_watch_clear();
void vertex_watch_pre_step();
void vertex_watch_check_collapse(int s, int d);
bool vertex_watch_active();
bool vertex_watch_triggered();
int  vertex_watch_trigger_at_collapse();
int  vertex_watch_fine_vtx();
Eigen::Vector3i vertex_watch_cur_BF();
