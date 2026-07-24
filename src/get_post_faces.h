#ifndef GET_POST_FACES_H
#define GET_POST_FACES_H

#include <vector>
#include <iostream>
#include <igl/slice.h>

#include <Eigen/Core>
#include <Eigen/Dense>

// Returns false when 3+ flap faces are found (non-manifold-within-sheet topology
// produced by VF-merge after prior collapses).  Caller should treat this as an
// invalid collapse attempt rather than asserting.
bool get_post_faces(
	const Eigen::MatrixXi & F_pre,
	const int & vi,
	const int & vj,
  Eigen::VectorXi & f_keep,
	Eigen::MatrixXi & F_post);

#endif