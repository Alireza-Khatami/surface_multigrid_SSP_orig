#pragma once
#include <igl/collapse_edge.h>  // IGL_COLLAPSE_EDGE_NULL
#include <Eigen/Core>

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
