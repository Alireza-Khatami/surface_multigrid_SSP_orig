#pragma once
#include "coarse_mesh_compaction.h"
#include <string>

// Tracks samples seeded on the fine mesh's boundary/seam edges (from a
// .ma_struct file) through the decimation, the same way face_sample_tracker
// tracks generic fine-mesh point samples — but as a fully separate parallel
// tracker (own state, own output, no shared code with face_sample_tracker).
//
// Seeding: for each struct_id whose type is SEAM (1) or BOUNDARY (2) in the
// .ma_struct file, place `samples_per_struct` points along all the edges
// that make up that struct's curve. For each point, seed one sample on the
// first gFO face incident to that edge. (Seam edges can have 2+ incident
// faces — one per sheet touching them, 3+ at a non-manifold junction — but
// we no longer seed one sample per sheet: joint_lscm_seam_pinned already
// forces vi/vj/vk to a shared UV target across sheets for every seam
// collapse (joint_lscm_pinned.cpp:274-298), so a single sample per point is
// trusted to represent all sheets. This also keeps sample counts
// manageable, since edge_sample_tracker_update()'s per-collapse vertex
// fixup is O(total samples).)
//
// Two placement modes, both weighting each edge by its own length (an edge
// twice as long gets twice the point budget) — the difference is only in
// HOW points are placed within that per-edge budget:
//   deterministic (default) — evenly spaced by walking the struct's total
//     arc length in samples_per_struct equal steps. No randomness, so
//     display density is uniform along the curve with no clumping/gaps.
//   Monte Carlo (deterministic=false) — each point is an independent random
//     draw (edge picked via arc-length-weighted CDF, then uniform t within
//     it). Kept in the code (not deleted) but excluded from the default
//     pipeline: independent random draws produce visible Poisson
//     clumping/gaps once edge_sample_tracker_viz.py thins samples for
//     display (see this session's "discontinuities" investigation).
//
// Tracking: identical per-sheet UV_pre -> UV_post barycentric cast as
// sample_tracker_update() in face_sample_tracker.cpp; call
// edge_sample_tracker_update() right after every successful
// SSP_collapse_edge, next to sample_tracker_update().

// Call once after init_ssp() and after load_matstruct()/gFO,gVO are ready.
// matstruct_path: same .ma_struct file used for struct IDs.
// samples_per_struct: number of points placed per seam/boundary struct_id.
// deterministic: true (default) = evenly-spaced by arc length; false = the
//   original Monte Carlo random-draw sampler (kept for reference/opt-in,
//   not used by default — see the mode comment above).
void edge_sample_tracker_init(const std::string& matstruct_path,
                              int samples_per_struct = 1000,
                              bool deterministic = true);

// Remap all edge samples through the latest entry in gDecInfo.
// Call immediately after every successful SSP_collapse_edge step, next to
// sample_tracker_update().
void edge_sample_tracker_update();

// Write all edge samples to one file:
//   id  type_id  struct_id  src_v0  src_v1  t  seed_face_id  cur_FIdx
//   b0 b1 b2  bv0 bv1 bv2  cfi cb0 cb1 cb2
// type_id follows the .ma_struct convention: 1 = seam, 2 = boundary.
// cfi/cb*: compact face into lookup.cmc.Fout, bary in that face's corner order.
// Throws if any sample cannot be resolved on the compact mesh.
void edge_sample_tracker_save(const CoarseFaceLookup& lookup, const std::string& path);

// Register polyscope point clouds (boundary / seam, colored distinctly).
// No-op when C2F_VIZ_DIAGNOSTIC is not defined.
void edge_sample_tracker_show();
