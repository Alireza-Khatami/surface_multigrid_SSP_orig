#pragma once
#include <Eigen/Dense>
#include <single_collapse_data.h>
#include <vector>

struct Bundle {
    Eigen::MatrixXd coarseV;   // NC × 3
    Eigen::MatrixXi coarseF;   // FC × 3
    Eigen::MatrixXd fineV;     // NF × 3
    Eigen::MatrixXi fineF;     // FF × 3
    Eigen::MatrixXd corrVec;   // NC × 3  (fine_pos - coarse_pos)

    // v2 bundle: SSP query data
    bool hasSspData = false;
    int nV_total = 0;
    int nF_total = 0;
    std::vector<int> vtxMap;                   // compact→global vertex (NC entries)
    std::vector<int> faceMap;                  // compact→global face   (FC entries)
    Eigen::VectorXi faceSheetID;
    std::vector<std::vector<int>> decIM;
    std::vector<single_collapse_data> decInfo;
};
