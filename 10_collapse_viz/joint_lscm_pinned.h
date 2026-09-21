#pragma once

// Seam UV pinning fix.
//
// For a seam collapse (2+ active sheets), joint_lscm_double_cover
// (src/joint_lscm.cpp) pins only 2 points per sheet — B_glued[0]->(-1,0)
// and B_glued[1]->(+1,0). vi, vj, and the post-collapse merged point
// ("vk") are left as FREE variables of the LSCM solve, so each sheet
// places them wherever is locally optimal for that sheet's own geometry.
// Confirmed empirically (md_files/seam_uv_consistency_findings.md): 368 of
// 545 seam collapses in a test run had vi/vj/vk disagree across sheets by
// more than 1e-9, up to 0.28 on a UV domain spanning roughly [-1,1].
//
// This file adds vi/vj/vk as ADDITIONAL hard pins, to fixed literal UV
// targets shared by every sheet of every seam collapse, so all sheets are
// forced to agree by construction instead of just being checked for
// agreement after the fact. See md_files/seam_uv_pinning_fix.md for the
// full investigation and design writeup.
//
// Deliberately NOT added to src/joint_lscm.cpp (left completely untouched)
// — lives here in 10_collapse_viz/ instead, reusing joint_lscm.h's already-
// exported helpers (flatten, check_valid_UV_lscm, DCVizData,
// check_dc_symmetry, quasi_conformal_error) via #include <joint_lscm.h>.
// Gated behind the SSP_SEAM_UV_PINNING compile definition (set only by
// 10_collapse_viz/CMakeLists.txt) so SSP_collapse_edge.cpp's #include of
// this header never reaches 08_subdiv_remesh / 11_correspond_viz, which
// don't compile this file and don't have it on their include path.
//
// Case 1 (joint_lscm_case1_dc) is DEFERRED — not mirrored here. It never
// fires for a real seam collapse under the current code (seam collapses
// always force onBd.sum()==2, i.e. Case 2 — see SSP_collapse_edge.cpp
// lines ~887-892), so it has no effect on the problem this file solves.

#include <joint_lscm.h>

#include <Eigen/Dense>
#include <optional>
#include <vector>

// Pinned double-cover solve for the seam (Case 2) case: identical in
// structure to joint_lscm_double_cover, except vi, vj, and the internal
// post-collapse "nV" slot (the merged point, "vk") are ALSO hard-pinned —
// to fixed targets shared by every sheet — on top of the original
// B_glued[0]->(-1,0), B_glued[1]->(+1,0) pins. Fully self-contained
// (computes its own boundary loop via igl::boundary_loop), exactly like
// the original it mirrors.
void joint_lscm_double_cover_pinned(
    const Eigen::MatrixXd & V_pre,
    const Eigen::MatrixXi & FUV_pre,
    const Eigen::MatrixXd & V_post,
    const Eigen::MatrixXi & FUV_post,
    const int & vi,
    const int & vj,
    const bool isDebug,
    Eigen::MatrixXd & UV_pre,
    Eigen::MatrixXd & UV_post,
    Eigen::MatrixXi & FUV_dc_pre,
    Eigen::MatrixXi & FUV_dc_post,
    Eigen::MatrixXd & UV_dc_pre,
    Eigen::MatrixXd & UV_dc_post,
    std::vector<int> & out_B_glued,
    std::vector<int> & out_B_reflected);

// Top-level entry point mirroring joint_lscm()'s signature, for a drop-in
// call-site swap in SSP_collapse_edge.cpp. Only valid to call for a genuine
// seam collapse (2+ active sheets) — where the caller has already forced
// onBd.sum()==2 via -1 injection into both Nsv/Ndv, so Case 2 is the only
// possibility and no case-dispatch logic is needed here. *out_case is
// always set to 2. Nsv/Ndv are accepted (used only by check_valid_UV_lscm's
// debug-dump path, not by the solve itself) purely for call-site parity
// with joint_lscm().
bool joint_lscm_seam_pinned(
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
    std::optional<int> * out_case,
    DCVizData * dc_viz,
    int collapse_idx);

// Open/close this file's own diagnostic log ([SEAM-PIN-*] lines). Falls
// back to stderr if never opened. Separate from joint_lscm.cpp's dc_log()
// because that accessor is `static` (not exported).
void seam_uv_pinned_log_open(const char * path);
void seam_uv_pinned_log_close();
