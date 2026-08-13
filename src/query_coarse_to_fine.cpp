#include "query_coarse_to_fine.h"
#include <iostream>
#include <atomic>
#include <sstream>

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

  std::atomic<int> cnt_early_exit{0};
  std::atomic<int> cnt_sheet_fallback{0};
  std::atomic<int> cnt_sheet_miss{0};
  std::atomic<int> cnt_pre_row_miss{0};

#ifdef SSP_LSCM_LOG
  fprintf(stderr,
    "[query_coarse_to_fine START]  queries=%d  decInfo.size=%zu\n",
    numQuery, decInfo.size());
#endif

  // convert BF vertex indices from V to VO (fine mesh indices)
  {
    for (int r=0; r<BF.rows(); r++)
      for (int c=0; c<BF.cols(); c++)
        BF(r,c) = IM(BF(r,c));
  }

  // convert FIdx face indices from V to VO
  {
    for (int ii=0; ii<FIdx.size(); ii++)
      FIdx(ii) = IMF(FIdx(ii));
  }

  igl::parallel_for(
    numQuery,
    [&verbose, &FIdx, &BC, &BF, &decIM, &decInfo, &queryCounts, &faceSheetID,
     &cnt_early_exit, &cnt_sheet_fallback, &cnt_sheet_miss, &cnt_pre_row_miss](const int qIdx)
  {
#ifdef SSP_LSCM_LOG
    // Per-query log buffer — flushed atomically so parallel output doesn't interleave.
    std::ostringstream log;
#endif

    int dIdx = decInfo.size();
    int walk_steps = 0;
    int initial_FIdx = FIdx(qIdx);

#ifdef SSP_LSCM_LOG
    log << "[WALK-START] qIdx=" << qIdx
        << "  initial_FIdx=" << initial_FIdx
        << "  initial_BF=(" << BF(qIdx,0) << "," << BF(qIdx,1) << "," << BF(qIdx,2) << ")"
        << "  initial_BC=(" << BC(qIdx,0) << "," << BC(qIdx,1) << "," << BC(qIdx,2) << ")\n";
#endif

    while (true)
    {
      int queryFIdx = FIdx(qIdx);

      // find the next collapse index to process (newest one older than dIdx)
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

      if (!if_find_dIdx)
      {
        ++cnt_early_exit;
#ifdef SSP_LSCM_LOG
        log << "[WALK-EXIT] qIdx=" << qIdx
            << "  steps=" << walk_steps
            << "  exit_FIdx=" << (int)FIdx(qIdx)
            << "  dIdx_at_exit=" << dIdx
            << "  final_BF=(" << BF(qIdx,0) << "," << BF(qIdx,1) << "," << BF(qIdx,2) << ")"
            << "  final_BC=(" << BC(qIdx,0) << "," << BC(qIdx,1) << "," << BC(qIdx,2) << ")"
            << "  decIM=[";
        for (int e : decIM[(int)FIdx(qIdx)]) log << e << " ";
        log << "]\n";
#endif
        break;
      }

      if (verbose)
        cout << "qIdx: " << qIdx << ", dIdx: " << dIdx << endl;

      // Route to the correct sheet for this face
      const SheetData * sd_ptr = nullptr;
      int sid = (FIdx(qIdx) < faceSheetID.size()) ? faceSheetID(FIdx(qIdx)) : 0;
      {
        for (auto & sd : decInfo[dIdx].sheets)
          if (sd.global_sheet_id == sid) { sd_ptr = &sd; break; }
        if (!sd_ptr && !decInfo[dIdx].sheets.empty()) {
          if (decInfo[dIdx].sheets.size() == 1) {
            // Manifold collapse: single sheet, [0] is always the right one.
            ++cnt_sheet_fallback;
#ifdef SSP_LSCM_LOG
            log << "[SHEET-FALLBACK] qIdx=" << qIdx
                << "  FIdx=" << (int)FIdx(qIdx)
                << "  face_sheet=" << sid
                << "  dIdx=" << dIdx
                << "  available_sheets=[";
            for (auto & s : decInfo[dIdx].sheets) log << s.global_sheet_id << " ";
            log << "]\n";
#endif
            sd_ptr = &decInfo[dIdx].sheets[0];
          } else {
            // Seam collapse: no SheetData matched this face's sheet.
            // Using a different sheet's UV would corrupt the sample — skip.
            ++cnt_sheet_miss;
#ifdef SSP_LSCM_LOG
            log << "[SHEET-MISS] qIdx=" << qIdx
                << "  FIdx=" << (int)FIdx(qIdx)
                << "  face_sheet=" << sid
                << "  dIdx=" << dIdx
                << "  available_sheets=[";
            for (auto & s : decInfo[dIdx].sheets) log << s.global_sheet_id << " ";
            log << "]\n";
#endif
          }
        }
      }
      if (!sd_ptr) continue;
      const SheetData & sd = *sd_ptr;

      // Find the row in FIdx_pre that matches the current face
      int pre_row = -1;
      {
        for (int r = 0; r < (int)sd.FIdx_pre.size(); r++)
          if (sd.FIdx_pre(r) == queryFIdx) { pre_row = r; break; }
        if (pre_row < 0) {
          ++cnt_pre_row_miss;
#ifdef SSP_LSCM_LOG
          log << "[PRE-ROW-MISS] qIdx=" << qIdx
              << "  FIdx=" << (int)FIdx(qIdx)
              << "  dIdx=" << dIdx
              << "  sd.sheet=" << sd.global_sheet_id
              << "  FIdx_pre.size=" << (int)sd.FIdx_pre.size()
              << "\n";
#endif
          continue;
        }
      }

      int v0 = sd.FUV_pre(pre_row, 0);
      int v1 = sd.FUV_pre(pre_row, 1);
      int v2 = sd.FUV_pre(pre_row, 2);

      // Project current BC into UV_post to get the query point
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

      // Pick the face in UV_pre whose barycentric coords are most "inside"
      VectorXd distToValid = -B.rowwise().minCoeff();
      double minD = 1.0;
      int idxToFUV = 0;
      for (int bb=0; bb<distToValid.size(); bb++)
      {
        if (distToValid(bb) < minD)
        {
          minD = distToValid(bb);
          idxToFUV = bb;
        }
      }

      // Clamp small negatives from numerical error
      B(idxToFUV, 0) = max(0.0, B(idxToFUV, 0));
      B(idxToFUV, 1) = max(0.0, B(idxToFUV, 1));
      B(idxToFUV, 2) = max(0.0, B(idxToFUV, 2));
      B.row(idxToFUV) = B.row(idxToFUV).array() / B.row(idxToFUV).sum();

      int new_FIdx = sd.FIdx_pre(idxToFUV);
      int new_v0   = sd.subsetVIdx(sd.FUV_pre(idxToFUV,0));
      int new_v1   = sd.subsetVIdx(sd.FUV_pre(idxToFUV,1));
      int new_v2   = sd.subsetVIdx(sd.FUV_pre(idxToFUV,2));

#ifdef SSP_LSCM_LOG
      log << "[WALK-STEP] qIdx=" << qIdx
          << "  step=" << walk_steps
          << "  from_FIdx=" << queryFIdx
          << "  dIdx=" << dIdx
          << "  sheet=" << sd.global_sheet_id
          << "  uv_post_corner=(" << sd.UV_post.row(v0) << " | " << sd.UV_post.row(v1) << " | " << sd.UV_post.row(v2) << ")"
          << "  queryUV=(" << queryUV(0) << "," << queryUV(1) << ")"
          << "  best_row=" << idxToFUV
          << "  new_FIdx=" << new_FIdx
          << "  new_BF=(" << new_v0 << "," << new_v1 << "," << new_v2 << ")"
          << "  new_BC=(" << B(idxToFUV,0) << "," << B(idxToFUV,1) << "," << B(idxToFUV,2) << ")"
          << "\n";
#endif

      walk_steps++;
      BC.row(qIdx)  = B.row(idxToFUV);
      BF(qIdx, 0)   = new_v0;
      BF(qIdx, 1)   = new_v1;
      BF(qIdx, 2)   = new_v2;
      FIdx(qIdx)    = new_FIdx;
    }

#ifdef SSP_LSCM_LOG
    // Flush the entire per-query trace atomically
    fprintf(stderr, "%s", log.str().c_str());
#endif
  }, 1000);

#ifdef SSP_LSCM_LOG
  fprintf(stderr,
    "[query_coarse_to_fine SUMMARY]  queries=%d  early_exits=%d  "
    "sheet_fallbacks=%d  sheet_misses=%d  pre_row_misses=%d\n",
    numQuery, cnt_early_exit.load(), cnt_sheet_fallback.load(),
    cnt_sheet_miss.load(), cnt_pre_row_miss.load());
#endif

  if (verbose)
    cout << "finish query points\n";
}
