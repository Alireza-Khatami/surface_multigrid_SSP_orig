#ifndef SSP_COLLAPSE_EDGE_H
#define SSP_COLLAPSE_EDGE_H
// #include <igl/decimate_func_types.h>

// what Derek included
#include <igl/writeOBJ.h>
#include <igl/unique.h>
#include <igl/remove_unreferenced.h>
#include <igl/lscm.h>
#include <igl/internal_angles.h>
#include <igl/find.h>
#include <igl/intersect.h>
#include <igl/setdiff.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>
#include <set>
#include <unordered_map>
#include <cmath>
#include <algorithm> 
#include <map>

#include <single_collapse_data.h>
#include <remove_row.h>
#include <get_collapse_onering_faces.h>
#include <vector_mod.h>
#include <get_post_faces.h>
#include <joint_lscm.h>
// #include <joint_arap2D.h>
#include <profc.h>
#include <remove_unreferenced_lessF.h>
#include <decimate_func_types.h>
#include <min_heap.h>


// namespace igl
// {
  // Assumes (V,F) is a closed manifold mesh (except for previously collapsed
  // faces which should be set to: 
  // [IGL_COLLAPSE_EDGE_NULL IGL_COLLAPSE_EDGE_NULL IGL_COLLAPSE_EDGE_NULL].
  // Collapses exactly two faces and exactly 3 edges from E (e and one side of
  // each face gets collapsed to the other). This is implemented in a way that
  // it can be repeatedly called until satisfaction and then the garbage in F
  // can be collected by removing NULL faces.
  //
  // Inputs:
  //   e  index into E of edge to try to collapse. E(e,:) = [s d] or [d s] so
  //     that s<d, then d is collapsed to s.
  ///  p  dim list of vertex position where to place merged vertex
  // Inputs/Outputs:
  //   V  #V by dim list of vertex positions, lesser index of E(e,:) will be set
  //     to midpoint of edge.
  //   F  #F by 3 list of face indices into V.
  //   E  #E by 2 list of edge indices into V.
  //   EMAP #F*3 list of indices into E, mapping each directed edge to unique
  //     unique edge in E
  //   EF  #E by 2 list of edge flaps, EF(e,0)=f means e=(i-->j) is the edge of
  //     F(f,:) opposite the vth corner, where EI(e,0)=v. Similarly EF(e,1) "
  //     e=(j->i)
  //   EI  #E by 2 list of edge flap corners (see above).
  //   e1  index into E of edge collpased on left
  //   e2  index into E of edge collpased on right
  //   f1  index into F of face collpased on left
  //   f2  index into F of face collpased on right
  // Returns true if edge was collapsed
  #define IGL_COLLAPSE_EDGE_NULL 0

// Snapshot of the sheet geometry that caused the most recent DC failure.
// Populated whenever joint_lscm returns false with dc_viz.has_data == true.
// Cleared by calling SSP_clear_dc_fail_snap().
struct DCFailSnap {
    bool              valid          = false;
    Eigen::MatrixXd   V;          // compact 3D vertices for the failing sheet (V_pre_si)
    Eigen::MatrixXi   F;          // face indices into V          (FUV_pre_si)
    int               global_sheet_id = -1;
    int               vi = -1, vj = -1;  // global edge endpoints
};
const DCFailSnap & SSP_get_dc_fail_snap();
void              SSP_clear_dc_fail_snap();

// Snapshot captured at the -1 injection / case-selection point for each sheet.
// Lets the visualizer show exactly why a particular LSCM case was chosen.
struct BDSnap {
    bool valid = false;
    int  collapse_idx   = -1;
    int  sid            = -1;   // local BFS sheet id
    int  vi_global      = -1;   // E(e,0) — survivor
    int  vj_global      = -1;   // E(e,1) — absorbed
    int  active_sheets  =  0;

    // VF adjacency for each global endpoint (face indices, is_inf flag)
    struct FaceEntry { int f; bool is_inf; };
    std::vector<FaceEntry> vi_vf;   // VF[vi_global]
    std::vector<FaceEntry> vj_vf;   // VF[vj_global]

    bool vi_on_boundary = false;    // vtx_on_boundary(vi_global)
    bool vj_on_boundary = false;    // vtx_on_boundary(vj_global)
    bool injected_ndv   = false;    // -1 was prepended to Ndv_local
    bool injected_nsv   = false;    // -1 was prepended to Nsv_local

    std::vector<int> Nsv_local;     // walk around vj (after injection)
    std::vector<int> Ndv_local;     // walk around vi (after injection)
    std::vector<int> subsetVIdx;    // local → global vertex mapping

    int  lscm_case      = -1;       // 0 / 1 / 2 (filled after joint_lscm returns)
};
const BDSnap & SSP_get_bd_snap();
void           SSP_clear_bd_snap();
// bool SSP_collapse_edge(
//     const int e,
//     const Eigen::RowVectorXd & p,
//     Eigen::MatrixXd & V,
//     Eigen::MatrixXi & F,
//     Eigen::MatrixXi & E,
//     Eigen::VectorXi & EMAP,
//     Eigen::MatrixXi & EF,
//     Eigen::MatrixXi & EI,
//     int & e1,
//     int & e2,
//     int & f1,
//     int & f2);
  // Inputs:
bool SSP_collapse_edge(
    const int e,
    const Eigen::RowVectorXd & p,
    /*const*/ std::vector<int> & Nsv,
    const std::vector<int> & Nsf,
    /*const*/ std::vector<int> & Ndv,
    const std::vector<int> & Ndf,
    Eigen::MatrixXd & V,
    Eigen::MatrixXi & F,
    Eigen::MatrixXi & E,
    Eigen::VectorXi & EMAP,
    Eigen::MatrixXi & EF,
    Eigen::MatrixXi & EI,
    int & e1,
    int & e2,
    int & f1,
    int & f2,
    std::vector<single_collapse_data> & decInfo,
    std::vector<std::vector<int>> & decIM,
    single_collapse_data & data,
    Eigen::VectorXi & FIdx_onering_pre,
    const Eigen::VectorXi & faceSheetID,
    const std::vector<std::vector<int>> & VF,
    Eigen::VectorXi & EQ);
// bool SSP_collapse_edge(
//     const int e,
//     const Eigen::RowVectorXd & p,
//     Eigen::MatrixXd & V,
//     Eigen::MatrixXi & F,
//     Eigen::MatrixXi & E,
//     Eigen::VectorXi & EMAP,
//     Eigen::MatrixXi & EF,
//     Eigen::MatrixXi & EI);
//   // Collapse least-cost edge from a priority queue and update queue 
//   //
//   // Inputs/Outputs:
//   //   cost_and_placement  function computing cost of collapsing an edge and 3d
//   //     position where it should be placed:
//   //     cost_and_placement(V,F,E,EMAP,EF,EI,cost,placement);
//   //     **If the edges is collapsed** then this function will be called on all
//   //     edges of all faces previously incident on the endpoints of the
//   //     collapsed edge.
//   //   Q  queue containing pairs of costs and edge indices and insertion "time"
//   //   EQ  #E list of "time" of last time pushed into Q
//   //   C  #E by dim list of stored placements
// bool SSP_collapse_edge(
//     const decimate_cost_and_placement_func & cost_and_placement,
//     Eigen::MatrixXd & V,
//     Eigen::MatrixXi & F,
//     Eigen::MatrixXi & E,
//     Eigen::VectorXi & EMAP,
//     Eigen::MatrixXi & EF,
//     Eigen::MatrixXi & EI,
//     igl::min_heap< std::tuple<double,int,int> > & Q,
//     Eigen::VectorXi & EQ,
//     Eigen::MatrixXd & C);
//   // Inputs:
//   //   pre_collapse  callback called with index of edge whose collapse is about
//   //     to be attempted. This function should return whether to **proceed**
//   //     with the collapse: returning true means "yes, try to collapse",
//   //     returning false means "No, consider this edge 'uncollapsable', behave
//   //     as if collapse_edge(e) returned false.
//   //   post_collapse  callback called with index of edge whose collapse was
//   //     just attempted and a flag revealing whether this was successful.
// bool SSP_collapse_edge(
//     const decimate_cost_and_placement_func & cost_and_placement,
//     const decimate_pre_collapse_func       & pre_collapse,
//     const decimate_post_collapse_func      & post_collapse,
//     Eigen::MatrixXd & V,
//     Eigen::MatrixXi & F,
//     Eigen::MatrixXi & E,
//     Eigen::VectorXi & EMAP,
//     Eigen::MatrixXi & EF,
//     Eigen::MatrixXi & EI,
//     igl::min_heap< std::tuple<double,int,int> > & Q,
//     Eigen::VectorXi & EQ,
//     Eigen::MatrixXd & C);

void SSP_seam_log_open(const char * path);
void SSP_seam_log_close();

void SSP_rej_log_open(const char * path);
void SSP_rej_log_close();
FILE * SSP_rej_log_file();  // shared by SSP_collapse_edge and qslim callbacks

// Enable/disable all paper validity checks (UV face flip, UV angle sum,
// Euclidean face flip, skinny triangle).  Default: enabled.
// Call SSP_validity_checks_enable(false) from main to turn off via --no-validity-checks.
void SSP_validity_checks_enable(bool enable);
bool SSP_validity_checks_enabled();

bool SSP_collapse_edge(
    const decimate_cost_and_placement_func & cost_and_placement,
    const decimate_pre_collapse_func       & pre_collapse,
    const decimate_post_collapse_func      & post_collapse,
    Eigen::MatrixXd & V,
    Eigen::MatrixXi & F,
    Eigen::MatrixXi & E,
    Eigen::VectorXi & EMAP,
    Eigen::MatrixXi & EF,
    Eigen::MatrixXi & EI,
    min_heap< std::tuple<double,int,int> > & Q,
    Eigen::VectorXi & EQ,
    Eigen::MatrixXd & C,
    int & e,
    int & e1,
    int & e2,
    int & f1,
    int & f2,
    std::vector<single_collapse_data> & decInfo,
    std::vector<std::vector<int>> & decIM,
    std::vector<std::vector<int>> * VF,
    const Eigen::VectorXi & faceSheetID);
// }

#endif
