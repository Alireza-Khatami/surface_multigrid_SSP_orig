#ifndef JOINT_LSCM_H
#define JOINT_LSCM_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <igl/writeOBJ.h>
#include <igl/boundary_loop.h>
#include <igl/vector_area_matrix.h>
#include <igl/cotmatrix.h>
#include <igl/repdiag.h>
#include <igl/slice_into.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/slice.h>
#include <igl/setdiff.h>
#include <igl/internal_angles.h>

#include <get_post_faces.h>
#include <vector_area_matrix_size.h>
#include <mqwf_dense.h>
#include <mqwf_dense_data.h>
#include <remove_vector_element.h>
#include <cotmatrix_dense.h>
#include <quasi_conformal_error.h>

#include <profc.h>
#include <vector>
#include <optional>
#include <iostream>
#include <stdlib.h>
#include <limits>
#include <cmath>
#include <set>
#include <unordered_map>

// Double cover visualization data — populated for boundary LSCM cases (lscm_case >= 1).
// FUV_dc_* has 2*nF rows (top sheet original + bottom sheet reversed winding).
// UV_dc_* has nVjoint_dc rows (top-sheet slots + bottom-sheet B_reflected copies).
struct DCVizData {
    bool has_data = false;
    Eigen::MatrixXi FUV_dc_pre, FUV_dc_post;
    Eigen::MatrixXd UV_dc_pre, UV_dc_post;
    // B vertex classification (local one-ring indices, same space as FUV_pre/UV_pre).
    // B_glued: arc endpoints shared between both DC sheets.
    // B_reflected: middle arc vertices — top-sheet copies keep original index;
    //   bottom-sheet copies are at UV_dc rows nVjoint+k = (V_pre.rows()+1)+k.
    std::vector<int> B_glued;
    std::vector<int> B_reflected;
    // Post-UV symmetry across y=0 (the seam line defined by the (-1,0)↔(+1,0) pins).
    //   +1 = all B_reflected top/bottom pairs are mirror-symmetric within tolerance.
    //    0 = DC solved OK, but at least one pair violates symmetry; max_err has the worst deviation.
    //   -1 = DC solve failed (NaN UV) — symmetry could not be measured.
    // TODO: investigate why DC fails produce NaN UV and whether the arc topology is the root cause.
    int    dc_uv_symmetric    = 1;
    double dc_uv_asym_max_err = 0.0;
};

// Build double cover: F_dc = [F; F_reversed] where F_reversed swaps columns 1 and 2.
// The vertex set is unchanged — boundary vertices are shared between the two sheets.
void build_double_cover_faces(
    const Eigen::MatrixXi & F,
    Eigen::MatrixXi & F_dc);

// Joint LSCM using double cover for boundary cases (replaces cases 1 & 2).
//
// Strategy (B-vertex seam):
//   mesh_bd_verts  — actual mesh boundary vertices in the one-ring (local indices).
//                    Caller passes at least {vi, vj}; ideally the full global boundary
//                    set (TODO: wire up global boundary data from the caller).
//   B vertices     — all one-ring vertices NOT in mesh_bd_verts.
//                    They form the seam: shared between DC top and bottom sheets AND
//                    shared between pre/post joint meshes (the joint-LSCM coupling).
//   vi / vj / nV   — get separate UV slots on the bottom DC sheet (free variables).
//   Pinning        — all B vertices are pinned to a unit semicircle arc in UV space
//                    (evenly distributed between (0,0) and (1,0) via the arc top).
//                    vi/vj are free → LSCM finds their conformal positions.
//
// If fewer than 2 B vertices exist the function sets UV_pre/UV_post to NaN and
// emits a [DC-SKIP] log so the caller's fallback runs.
//
// UV_pre, UV_post: UV coords for original (top-sheet) faces only.
// FUV_dc_pre/post, UV_dc_pre/post: full double cover for visualization.
void joint_lscm_double_cover(
    const Eigen::MatrixXd & V_pre,
    const Eigen::MatrixXi & FUV_pre,
    const Eigen::MatrixXd & V_post,
    const Eigen::MatrixXi & FUV_post,
    const int & vi,
    const int & vj,
    const std::set<int> & mesh_bd_verts,
    const bool isDebug,
    Eigen::MatrixXd & UV_pre,
    Eigen::MatrixXd & UV_post,
    Eigen::MatrixXi & FUV_dc_pre,
    Eigen::MatrixXi & FUV_dc_post,
    Eigen::MatrixXd & UV_dc_pre,
    Eigen::MatrixXd & UV_dc_post,
    std::vector<int> & out_B_glued,
    std::vector<int> & out_B_reflected);

// Compute a joinit Least-squares conformal map parametrization for the edge onering mesh before an edge collapse (V_pre, FUV_pre)and the vertex onering mesh after the collapse (V_post, FUV_post). We constrain both meshes to have the same boundary curve for bijectivity
//
// Inputs:
// V_pre    #V_pre by 3 list of mesh vertex positions
// FUV_pre  #F_pre by 3 list of mesh faces
// V_post   #V_post by 3 list of mesh vertex positions
// FUV_post #F_post by 3 list of mesh faces
// vi       one of the vertex indices of the edge to be collapsed
// vj       the other vertex index of the edge to be collapsed
// Nsv      ordered neighboring vertex indices of vi
// Ndv      ordered neighboring vertex indices of vj
//
// Outputs:
// UV_pre    #V_pre by 2 list of UV mesh vertex positions of V_pre
// UV_post   #V_post by 2 list of UV mesh vertex positions of V_post
// dc_viz    if non-null and a boundary case is detected, filled with double cover data
bool joint_lscm(
  const Eigen::MatrixXd & V_pre,
  const Eigen::MatrixXi & FUV_pre,
  const Eigen::MatrixXd & V_post,
  const Eigen::MatrixXi & FUV_post,
  const int & vi,
  const int & vj,
  const std::vector<int> & Nsv,
  const std::vector<int> & Ndv,
  Eigen::MatrixXd & UV_pre,
  Eigen::MatrixXd & UV_post,
  std::optional<int> * out_case = nullptr,
  DCVizData * dc_viz = nullptr,
  int collapse_idx = -1);

// Open / close the DC log file.  Must be called from main() with the per-shape
// output path (e.g. out_dir + "dc_log_" + stem + ".txt") before the first collapse.
// Falls back to stderr if dc_log_open() was never called.
void dc_log_open(const char * path);
void dc_log_close();

// Write a per-sheet header line to dc_log.txt so sheet boundaries are visible
// in the log even when no DC is attempted (Case 0/1 collapses).
void dc_log_sheet_header(int collapse_idx, int sid,
                         int vi_global, int vj_global,
                         int nFpre, int nV, bool is_seam);

// Check whether the double-cover post-UV is symmetric across y=0 (the seam line).
//
// The DC pins B_glued[0] at (-1,0) and B_glued[1] at (+1,0), so the expected
// symmetry axis is y=0.  For each B_reflected[k], its top-sheet UV slot
// (index B_reflected[k]) and bottom-sheet UV slot (index nVjoint+k) must satisfy:
//   - same x:  |UV_dc(top,0) − UV_dc(bot,0)| ≤ tol
//   - mirror y: |UV_dc(top,1) + UV_dc(bot,1)| ≤ tol
// vi and vj themselves are checked for |y| ≤ tol (they must lie on the seam line).
// max_err is set to the worst observed deviation across all checks.
bool check_dc_symmetry(
    const Eigen::MatrixXd & UV_dc,
    const std::vector<int> & B_reflected,
    int nVjoint,
    int vi, int vj,
    double tol,
    double & max_err);

bool check_valid_UV_lscm(
  const Eigen::MatrixXd & V_pre,
  const Eigen::MatrixXd & UV_pre,
  const Eigen::MatrixXi & FUV_pre,
  const Eigen::MatrixXd & V_post,
  const Eigen::MatrixXd & UV_post,
  const Eigen::MatrixXi & FUV_post,
  const int & vi,
  const int & vj,
  const std::vector<int> & Nsv,
  const std::vector<int> & Ndv);

void flatten(
	const Eigen::MatrixXd & Vjoint_pre,
  const Eigen::MatrixXi & Fjoint_pre,
  const Eigen::MatrixXd & Vjoint_post,
  const Eigen::MatrixXi & Fjoint_post,
	const Eigen::VectorXi & b_UV,
	const Eigen::VectorXd & bc_UV,
	const int & nVjoint,
	const bool & isDebug,
	Eigen::VectorXd & UVjoint_flat);

void joint_lscm_case0(
	const Eigen::MatrixXd & V_pre,
  const Eigen::MatrixXi & FUV_pre,
  const Eigen::MatrixXd & V_post,
  const Eigen::MatrixXi & FUV_post,
  const int & vi,
  const int & vj,
  const Eigen::VectorXi & bdLoop,
  const bool & verbose,
	const bool & isDebug, 
  const Eigen::VectorXi & onBd,
  Eigen::MatrixXd & UV_pre,
  Eigen::MatrixXd & UV_post);

void joint_lscm_case1(
	const Eigen::MatrixXd & V_pre,
  const Eigen::MatrixXi & FUV_pre,
  const Eigen::MatrixXd & V_post,
  const Eigen::MatrixXi & FUV_post,
  const int & vi,
  const int & vj,
  const Eigen::VectorXi & bdLoop,
  const bool & verbose,
	const bool & isDebug,
  const Eigen::VectorXi & onBd,
  Eigen::MatrixXd & UV_pre,
  Eigen::MatrixXd & UV_post);

// Double-cover solver for Case 1 (one boundary endpoint).
// Reflects the open fan across the outer arc to form a closed disc.
// Pins the two outer ring vertices adjacent to v_bd at (-1,0) and (+1,0).
// UV_pre/UV_post: top-sheet UV for original faces (same index space as FUV_pre/FUV_post).
// FUV_dc_*/UV_dc_*: full double cover for visualization.
void joint_lscm_case1_dc(
    const Eigen::MatrixXd & V_pre,
    const Eigen::MatrixXi & FUV_pre,
    const Eigen::MatrixXd & V_post,
    const Eigen::MatrixXi & FUV_post,
    const int & vi,
    const int & vj,
    const Eigen::VectorXi & onBd,
    const Eigen::VectorXi & bdLoop,   // outer ring from joint_lscm() (seam-aware)
    const std::vector<int> & Nsv,     // vi's (E(e,0)'s) ordered neighbor list
    const std::vector<int> & Ndv,     // vj's (E(e,1)'s) ordered neighbor list
    const bool isDebug,
    Eigen::MatrixXd & UV_pre,
    Eigen::MatrixXd & UV_post,
    Eigen::MatrixXi & FUV_dc_pre,
    Eigen::MatrixXi & FUV_dc_post,
    Eigen::MatrixXd & UV_dc_pre,
    Eigen::MatrixXd & UV_dc_post,
    std::vector<int> & out_B_glued,
    std::vector<int> & out_B_reflected);

void joint_lscm_case2(
	const Eigen::MatrixXd & V_pre,
  const Eigen::MatrixXi & FUV_pre,
  const Eigen::MatrixXd & V_post,
  const Eigen::MatrixXi & FUV_post,
  const int & vi,
  const int & vj,
  const Eigen::VectorXi & bdLoop,
  const bool & verbose,
	const bool & isDebug,
  const Eigen::VectorXi & onBd, 
  Eigen::MatrixXd & UV_pre,
  Eigen::MatrixXd & UV_post);

void case2_constraint3_snap1(
  const Eigen::MatrixXd & V_pre,
  const Eigen::MatrixXi & FUV_pre,
  const Eigen::MatrixXd & V_post,
  const Eigen::MatrixXi & FUV_post,
  const int & vi,
  const int & vj,
  const Eigen::VectorXi & bdLoop,
  const bool & verbose,
	const bool & isDebug,
  const Eigen::VectorXi & onBd, 
  const int & snapIdx,
  Eigen::MatrixXd & UV_pre,
  Eigen::MatrixXd & UV_post
);

void case2_constraint4(
  const Eigen::MatrixXd & V_pre,
  const Eigen::MatrixXi & FUV_pre,
  const Eigen::MatrixXd & V_post,
  const Eigen::MatrixXi & FUV_post,
  const int & vi,
  const int & vj,
  const Eigen::VectorXi & bdLoop,
  const bool & verbose,
	const bool & isDebug,
  const Eigen::VectorXi & onBd, 
  Eigen::MatrixXd & UV_pre,
  Eigen::MatrixXd & UV_post
);

// void joint_lscm_case3(
// 	const Eigen::MatrixXd & V_pre,
//   const Eigen::MatrixXi & FUV_pre,
//   const Eigen::MatrixXd & V_post,
//   const Eigen::MatrixXi & FUV_post,
//   const int & vi,
//   const int & vj,
//   const Eigen::VectorXi & bdLoop,
//   const bool & verbose,
// 	const bool & isDebug,
//   const Eigen::VectorXi & onBd, 
//   const int & v_flapCorner,
//   Eigen::MatrixXd & UV_pre,
//   Eigen::MatrixXd & UV_post);


#endif