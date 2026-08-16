#ifndef SINGEL_COLLAPSE_DATA_H
#define SINGEL_COLLAPSE_DATA_H

#include <vector>
#include <optional>
#include <Eigen/Core>

// A face incident to the absorbed vertex d that belongs to a sheet NOT active
// for this collapse.  gF gets d→s remapped for it, but no decIM entry is written.
struct NonActiveSheetFace
{
    int      face_idx        = -1;
    int      sheet_id        = -1;
    bool     is_infinity_face = false;  // true if any vertex of this face is the ∞ vertex
    Eigen::Vector3d p0, p1, p2;        // pre-collapse 3D positions (safe to display)
};

// Per-sheet UV data for one edge collapse.
// For manifold meshes, single_collapse_data.sheets has exactly one entry.
// For non-manifold meshes, one SheetData per topological sheet touched by the collapse.
struct SheetData
{
    int global_sheet_id = -1;      // sheet index from partition_into_sheets
    Eigen::VectorXi b;              // b(0)=local vi, b(1)=local vj in this sheet
    Eigen::VectorXi subsetVIdx;     // sorted ascending global vertex indices
    Eigen::MatrixXd UV_pre, UV_post;
    Eigen::MatrixXi FUV_pre, FUV_post;
    Eigen::VectorXi FIdx_pre, FIdx_post;
    // 3D ring geometry in local index space (same indexing as FUV_pre/FUV_post).
    // V_post has b(0) (survivor) moved to the collapse placement p; all other rows == V_pre.
    Eigen::MatrixXd V_pre, V_post;
};

struct single_collapse_data
{
    std::vector<SheetData> sheets;  // one entry per sheet (size 1 for manifold/regular edges)
    Eigen::MatrixXd V_pre, V_post;  // 3D geometry from first successful sheet (for display)
    std::vector<int> Nsv, Ndv;     // local winding-order neighbour lists (first sheet)
    int numFlapFaces = 0;           // faces killed in topology pass (2=manifold, >2=seam edge)
    std::optional<int> lscm_case;   // joint_lscm case: 0=both interior, 1=one on bd, 2=both on bd
    std::vector<NonActiveSheetFace> non_active_faces; // faces remapped but not in active sheet
};

#endif
