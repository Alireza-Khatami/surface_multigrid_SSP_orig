#ifndef SINGEL_COLLAPSE_DATA_H
#define SINGEL_COLLAPSE_DATA_H

#include <vector>
#include <Eigen/Core>

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
};

struct single_collapse_data
{
    std::vector<SheetData> sheets;  // one entry per sheet (size 1 for manifold/regular edges)
    Eigen::MatrixXd V_pre, V_post;  // 3D geometry from first successful sheet (for display)
    std::vector<int> Nsv, Ndv;     // local winding-order neighbour lists (first sheet)
    int numFlapFaces = 0;           // faces killed in topology pass (2=manifold, >2=seam edge)
};

#endif
