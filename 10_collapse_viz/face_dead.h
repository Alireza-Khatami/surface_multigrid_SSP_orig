#pragma once
#include <igl/collapse_edge.h>  // IGL_COLLAPSE_EDGE_NULL
#include <Eigen/Core>
#include <cmath>

// Returns true when face f in F has been marked dead by igl::collapse_edge.
// igl sets ALL THREE vertex slots to IGL_COLLAPSE_EDGE_NULL (= 0 in this build),
// so checking only F(f,0) gives false positives for any live face whose first
// vertex happens to be vertex 0.
static inline bool is_face_dead(const Eigen::MatrixXi & F, int f)
{
    return F(f,0) == IGL_COLLAPSE_EDGE_NULL
        && F(f,1) == IGL_COLLAPSE_EDGE_NULL
        && F(f,2) == IGL_COLLAPSE_EDGE_NULL;
}

// Returns true when face f is "live" for coarse-mesh export purposes: not
// dead, and none of its three vertices sit at the boundary-to-infinity cap
// vertex. Checking only F(f,0) for infinity (as some call sites used to)
// gives false positives whenever the infinity vertex isn't in slot 0.
static inline bool is_face_live(const Eigen::MatrixXi & F, const Eigen::MatrixXd & V, int f)
{
    if (is_face_dead(F, f)) return false;
    return !std::isinf(V(F(f,0), 0))
        && !std::isinf(V(F(f,1), 0))
        && !std::isinf(V(F(f,2), 0));
}
