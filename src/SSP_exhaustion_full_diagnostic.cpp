// SSP_exhaustion_full_diagnostic.cpp
//
// Runs a second, isolated collapse loop on a deep copy of the exhausted mesh.
// Captures UV face-flip and angle-sum rejection reasons that the lightweight
// gCostFn-only diagnostic cannot reach (because joint_lscm only runs inside
// the full SSP_collapse_edge pipeline).
//
// CRITICAL: this file does NOT touch qslim or joint_lscm logic/calculations.
// It only wraps the existing SSP_collapse_edge with a fresh copy of the data.

#include "SSP_exhaustion_full_diagnostic.h"
#include "SSP_collapse_edge.h"
#include "SSP_rejection_detail.h"
#include "SSP_qslim_optimal_collapse_edge_callbacks.h"

#include <igl/vertex_triangle_adjacency.h>
#include <igl/per_vertex_point_to_plane_quadrics.h>
#include <igl/edge_flaps.h>
#include <igl/collapse_edge.h>   // IGL_COLLAPSE_EDGE_NULL
#include <min_heap.h>
#include <single_collapse_data.h>
#include <decimate_func_types.h>

#include <Eigen/Core>
#include <cstdio>
#include <cmath>
#include <limits>
#include <vector>
#include <tuple>
#include <string>

void SSP_exhaustion_full_diagnostic(
    const Eigen::MatrixXd &          V_orig,
    const Eigen::MatrixXi &          F_orig,
    const Eigen::MatrixXi &          E_orig,
    const Eigen::VectorXi &          EMAP_orig,
    const Eigen::MatrixXi &          EF_orig,
    const Eigen::MatrixXi &          EI_orig,
    const Eigen::MatrixXd &          C_orig,
    const Eigen::VectorXi &          EQ_orig,
    const std::vector<SSP_Quadric> & quadrics_orig,
    const Eigen::VectorXi &          faceSheetID,
    const std::string &              out_dir,
    int                              collapse_count)
{
    const std::string log_path = out_dir + "exhausted_queue_full_rejections.log";
    FILE* log = fopen(log_path.c_str(), "w");
    if (!log) {
        fprintf(stderr, "[EXHFULL] could not open %s\n", log_path.c_str());
        return;
    }

    const bool was_validity = SSP_validity_checks_enabled();
    fprintf(stderr, "[EXHFULL] starting full-collapse diagnostic on mesh copy"
            "  (main_validity_checks=%s) -> %s\n",
            was_validity ? "ON" : "OFF", log_path.c_str());
    fprintf(log,
        "# Exhaustion full-collapse diagnostic\n"
        "# Runs the complete SSP_collapse_edge pipeline (including joint_lscm)\n"
        "# on a DEEP COPY of the mesh -- original is unchanged.\n"
        "# Struct-gate (--mat_struct_check) is NOT applied here; all live edges are tested.\n"
        "# Validity checks are forced ON for this pass (regardless of --validity-checks flag)\n"
        "#   so that UV and Euclidean rejection reasons are always captured.\n"
        "# main_run_validity_checks=%s  collapses_in_main_run=%d\n",
        was_validity ? "ON" : "OFF", collapse_count);

    // ── 1. Deep copy mesh state ───────────────────────────────────────────────
    Eigen::MatrixXd              V2    = V_orig;
    Eigen::MatrixXi              F2    = F_orig;
    Eigen::MatrixXi              E2    = E_orig;
    Eigen::VectorXi              EMAP2 = EMAP_orig;
    Eigen::MatrixXi              EF2   = EF_orig;
    Eigen::MatrixXi              EI2   = EI_orig;
    Eigen::MatrixXd              C2    = C_orig;
    Eigen::VectorXi              EQ2   = EQ_orig;
    std::vector<SSP_Quadric>     quadrics2 = quadrics_orig;

    // ── 2. Rebuild VF from copied F ──────────────────────────────────────────
    std::vector<std::vector<int>> VF2;
    std::vector<std::vector<int>> VFi2_unused;
    igl::vertex_triangle_adjacency((int)V2.rows(), F2, VF2, VFi2_unused);

    // ── 3. Build fresh qslim callbacks bound to the copy ─────────────────────
    // These capture E2, quadrics2, VF2 by reference — all live in this scope.
    // The struct gate is intentionally omitted: we want to probe all live edges.
    int v1_diag = -1, v2_diag = -1;
    decimate_cost_and_placement_func costFn2;
    decimate_pre_collapse_func       preFn2;
    decimate_post_collapse_func      postFn2;
    SSP_qslim_optimal_collapse_edge_callbacks(
        E2, quadrics2, v1_diag, v2_diag, VF2,
        costFn2, preFn2, postFn2);

    // ── 4. Build fresh priority queue from costs on copy ─────────────────────
    min_heap<std::tuple<double,int,int>> Q2;
    {
        // Recompute each edge's cost on the copy; rebuild C2 and Q2.
        // Sequential (not parallel) to avoid the thread-interleaving issue,
        // and because the diagnostic is a one-shot pass.
        for (int e = 0; e < E2.rows(); ++e) {
            double cost;
            Eigen::RowVectorXd p;
            costFn2(e, V2, F2, E2, EMAP2, EF2, EI2, cost, p);
            C2.row(e) = p;
            Q2.emplace(cost, e, 0);
        }
    }
    EQ2 = Eigen::VectorXi::Zero(E2.rows());

    // ── 5. Force validity checks ON for this pass ────────────────────────────
    // The main run may have been launched without --validity-checks, which would
    // make the UV and Euclidean gates globally disabled.  The diagnostic always
    // needs them enabled so joint_lscm failures produce [UV-REJECT] log entries.
    // was_validity was captured above (before the log header) for the restore.
    SSP_validity_checks_enable(true);

    // ── 6. Reset rejection caps so diagnostic output isn't silenced ──────────
    SSP_reset_uv_rej_caps();
    SSP_qslim_reset_counters();

    // ── 7. Redirect SSP_rej_log to the diagnostic log file ───────────────────
    FILE* orig_log = SSP_rej_log_swap(log);

    // ── 8. Run full collapse loop on the copy ─────────────────────────────────
    std::vector<single_collapse_data> decInfo2;
    std::vector<std::vector<int>>     decIM2(F2.rows());

    int n_attempted  = 0;
    int n_collapsed  = 0;
    int n_rejected   = 0;

    static constexpr double kInf = std::numeric_limits<double>::infinity();

    while (!Q2.empty() && std::get<0>(Q2.top()) != kInf) {
        int e_out, e1, e2, f1, f2;
        bool ok = SSP_collapse_edge(
            costFn2, preFn2, postFn2,
            V2, F2, E2, EMAP2, EF2, EI2,
            Q2, EQ2, C2,
            e_out, e1, e2, f1, f2,
            decInfo2, decIM2,
            &VF2, faceSheetID);
        ++n_attempted;
        if (ok) {
            ++n_collapsed;
            SSP_rej_set_collapse_num(collapse_count + n_collapsed);
        } else {
            ++n_rejected;
        }
    }

    // Count remaining live edges that had ∞ cost (struct-gate or degenerate)
    int n_inf_top = 0;
    {
        // Drain the rest of the queue to count ∞-cost entries
        min_heap<std::tuple<double,int,int>> tmp = Q2;
        while (!tmp.empty()) {
            if (std::get<0>(tmp.top()) == kInf) ++n_inf_top;
            tmp.pop();
        }
    }

    fprintf(log,
        "[EXHFULL-SUMMARY] attempted=%d  collapsed=%d  rejected=%d  inf_cost_remaining=%d\n",
        n_attempted, n_collapsed, n_rejected, n_inf_top);
    fprintf(stderr,
        "[EXHFULL] done: attempted=%d  collapsed=%d  rejected=%d  inf_remaining=%d  -> %s\n",
        n_attempted, n_collapsed, n_rejected, n_inf_top, log_path.c_str());

    // ── 9. Restore original state ─────────────────────────────────────────────
    SSP_validity_checks_enable(was_validity);
    SSP_rej_log_swap(orig_log);
    fclose(log);
}
