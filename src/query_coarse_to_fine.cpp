#include "query_coarse_to_fine.h"

void query_coarse_to_fine(
  const std::vector<single_collapse_data> & decInfo,
  const Eigen::VectorXi & IM,
  const std::vector<std::vector<int>> & decIM,
  const Eigen::VectorXi & IMF,
  const Eigen::VectorXi & faceSheetID,
  Eigen::MatrixXd & BC,
  Eigen::MatrixXi & BF,
  Eigen::VectorXi & FIdx)
{
  using namespace std;
  using namespace Eigen;
  bool verbose = false;

  int numQuery = BF.rows();
  int decLength = decInfo.size();

  VectorXi queryCounts(BC.rows());
  queryCounts.setZero();

  // convert vertex indices from V to VO
  {
    // change BF to be on fine mesh indices
    for (int r=0; r<BF.rows(); r++){
      for (int c=0; c<BF.cols(); c++){
        BF(r,c) = IM(BF(r,c));
      }
    }
  }

  // convert vertex indices from V to VO
  {
    for (int ii=0; ii<FIdx.size(); ii++)
      FIdx(ii) = IMF(FIdx(ii));
  }

  // for (int qIdx=0; qIdx<numQuery; qIdx++)
  igl::parallel_for(
    numQuery,
    [&verbose, &FIdx, &BC, &BF, &decIM, &decInfo, &queryCounts, &faceSheetID](const int qIdx)
  {
    // // print progress
    // if (qIdx % 1 == 0 && verbose) 
    //   cout << "coarse to fine : " << qIdx << "/" << BF.rows() << endl;

    // go through the decInfo 
    int dIdx = decInfo.size();

    while (true)
    {
      // get query FIdx 
      int queryFIdx = FIdx(qIdx);

      // find the dIdx
      vector<int> dIdxList = decIM[queryFIdx];
      bool if_find_dIdx = false;
      for (int ii=dIdxList.size()-1; ii>-1; ii--)
      {
        if (dIdx > dIdxList[ii])
        {
          dIdx = dIdxList[ii];
          if_find_dIdx = true;
          break;
        }
      }

      // if not dIdx left, break
      if (!if_find_dIdx)
        break;

      ///////// start query /////////
      if (verbose)
        cout << "qIdx: " << qIdx << ", dIdx: " << dIdx << endl; 

      // Route to the correct sheet for this face
      const SheetData * sd_ptr = nullptr;
      {
        int sid = (FIdx(qIdx) < faceSheetID.size()) ? faceSheetID(FIdx(qIdx)) : 0;
        for (auto & sd : decInfo[dIdx].sheets)
          if (sd.global_sheet_id == sid) { sd_ptr = &sd; break; }
        if (!sd_ptr && !decInfo[dIdx].sheets.empty())
          sd_ptr = &decInfo[dIdx].sheets[0];  // fallback for manifold meshes
      }
      if (!sd_ptr) continue;
      const SheetData & sd = *sd_ptr;

      // get vi and vj
      int vi = sd.subsetVIdx(sd.b(0));
      int vj = sd.subsetVIdx(sd.b(1));

      VectorXi f = BF.row(qIdx);

      // find f in subsetVIdx
      int v0, v1, v2; // such that subsetVIdx(v0) == f(0)
      {
        VectorXi v0_vec, v1_vec, v2_vec;
        igl::find((sd.subsetVIdx.array() == f(0)).eval(), v0_vec);
        igl::find((sd.subsetVIdx.array() == f(1)).eval(), v1_vec);
        igl::find((sd.subsetVIdx.array() == f(2)).eval(), v2_vec);

        assert(v0_vec.size() == 1);
        assert(v1_vec.size() == 1);
        assert(v2_vec.size() == 1);

        v0 = v0_vec(0);
        v1 = v1_vec(0);
        v2 = v2_vec(0);
      }

      // get query UV (coarse → fine: uses UV_post as input space, UV_pre as output)
      VectorXd queryUV =
          BC(qIdx,0) * sd.UV_post.row(v0)
        + BC(qIdx,1) * sd.UV_post.row(v1)
        + BC(qIdx,2) * sd.UV_post.row(v2);

      Eigen::MatrixXd B;
      {
        compute_barycentric(queryUV, sd.UV_pre, sd.FUV_pre, B);
        if (verbose)
          cout << "B: \n" << B << endl;
      }

      // snap to the closest one
      VectorXd distToValid = -B.rowwise().minCoeff();
      double minD = 1.0;
      int idxToFUV;
      for (int bb=0;bb<distToValid.size(); bb++)
      {
        if (distToValid(bb) < minD)
        {
          minD = distToValid(bb);
          idxToFUV = bb;
        }
      }

      // avoid numerical error of barycentric coordinate
      B(idxToFUV, 0) = max(0.0, B(idxToFUV, 0));
      B(idxToFUV, 1) = max(0.0, B(idxToFUV, 1));
      B(idxToFUV, 2) = max(0.0, B(idxToFUV, 2));
      B.row(idxToFUV) = B.row(idxToFUV).array() / B.row(idxToFUV).sum();

      BC.row(qIdx) = B.row(idxToFUV);

      BF(qIdx, 0) = sd.subsetVIdx(sd.FUV_pre(idxToFUV,0));
      BF(qIdx, 1) = sd.subsetVIdx(sd.FUV_pre(idxToFUV,1));
      BF(qIdx, 2) = sd.subsetVIdx(sd.FUV_pre(idxToFUV,2));
      FIdx(qIdx) = sd.FIdx_pre(idxToFUV);
    }
  // };
  },1000);

  if (verbose)
    cout << "finish query points\n";

}