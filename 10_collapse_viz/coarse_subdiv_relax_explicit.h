#pragma once
// Explicit (small-step) Laplacian relaxation of the subdivided coarse mesh on the
// fine MAT, everything at once.
//
// Unlike the Newton solver (subdiv_relax), nothing is solved: every iteration
// moves all free vertices together by a small step towards the mean of their
// neighbours and projects them back onto their own structure,
//
//     x_i <- Pi_i( x_i + lambda * (mean_j x_j - x_i) ),
//
// with the symmetric mesh graph: every vertex is pulled by all its mesh
// neighbours, so curves and sheets relax together (curve vertices are also
// pulled by the sheet vertices next to them, and only slide along their curve
// through the projection). Junctions are fixed. Stops when no vertex moves more
// than tol * bbox diagonal, or after maxIter iterations.
#include "subdiv_sample_tracker/subdiv_relax.h"

#include <Eigen/Core>
#include <cstdint>
#include <string>
#include <vector>

struct ExplicitRelaxOptions {
    double  lambda   = 0.5;     // step: fraction of the way to the neighbour mean
    int64_t maxIter  = 20000;
    double  tol      = 1e-7;    // stop when the largest move <= tol * bbox diagonal
    // Project each step to the closest point reachable from the vertex's current
    // location (Projector::project_local; a small step slides), or the global
    // closest point of its structure (Projector::project).
    bool    localProjection = true;
    // Optional, one entry per vertex: nonzero = held at its seed.
    std::vector<uint8_t> holdFixed;
    // Optional, one row per face of M.F: reference normal; folded = n . ref <= 0.
    // Logged when given; with noNewFolds, no step may fold an unfolded triangle
    // (the vertices of such a triangle keep their position for that step).
    Eigen::MatrixXd foldRef;
    bool    noNewFolds = false;
    int64_t logEvery = 100;     // progress line every logEvery iterations
    // OBJ snapshots of the positions at these iterations (and none if the prefix
    // is empty): <snapshotPrefix>it<N>.obj
    std::vector<int64_t> snapshotIters { 1, 10, 100, 1000, 10000 };
    std::string snapshotPrefix;
};

// Moves M.V and rewrites M.fineFace / M.fineBary. G must be the symmetric graph
// (build_relax_graph with ms = nullptr) with the structure roles. Fills the
// RelaxReport fields that apply (itersSheet = iterations, deltaSheet = last
// largest move, converged, moves, projection and fold counters, checks).
RelaxReport subdiv_relax_explicit(SubdivMesh & M,
                                  const Eigen::MatrixXd & VO,
                                  const Eigen::MatrixXi & FO,
                                  const MatStruct * ms,
                                  const StructPalette & pal,
                                  const std::vector<int32_t> & setId,
                                  const RelaxGraph & G,
                                  const ExplicitRelaxOptions & opt);
