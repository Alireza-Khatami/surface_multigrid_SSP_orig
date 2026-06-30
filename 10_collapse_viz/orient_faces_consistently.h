#pragma once
#include <Eigen/Core>

// orient_faces_consistently
//
// Re-winds the face matrix so that every pair of faces sharing a manifold
// interior edge (exactly 2 incident faces) traverses that edge in OPPOSITE
// directions — consistent CCW / outward-normal convention.
//
// Non-manifold edges (3+ incident faces) are treated as seams: the BFS stops
// there, so each manifold-connected patch is oriented independently.  After
// per-patch consistency is established, each patch is tested against the
// outward direction (component centroid relative to the overall mesh centroid)
// and flipped if its average normal points inward.
//
// Key difference from igl::bfs_orient: face flips are only committed in a
// single final pass, after the BFS has finished assigning orientations.
// igl::bfs_orient modifies faces during the BFS, which can double-flip a face
// if two already-visited neighbours both push it to the queue before it is
// popped.
//
// Inputs:
//   V   #V x 3   vertex positions  (used for orient-outward heuristic)
//   F   #F x 3   triangle face indices
// Outputs:
//   FF  #F x 3   consistently oriented faces  (may alias F)
//   C   #F        0-based component id per face
// Returns:
//   number of faces that were re-wound
int orient_faces_consistently(
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    Eigen::MatrixXi       & FF,
    Eigen::VectorXi       & C);
