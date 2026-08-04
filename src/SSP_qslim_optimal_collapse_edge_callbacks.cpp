// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2016 Alec Jacobson <alecjacobson@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#include "SSP_qslim_optimal_collapse_edge_callbacks.h"
#include <igl/collapse_edge.h>  // IGL_COLLAPSE_EDGE_NULL
#include <Eigen/LU>
#include <cstdio>
#include <cmath>
#include <vector>

static bool s_qslim_log_enabled = false;
void SSP_qslim_enable_log(bool enable) { s_qslim_log_enabled = enable; }

// ── Rejection diagnostics (always active, no macro gate) ─────────────────
static FILE* rej_log() {
    static FILE* fp = fopen("collapse_rejections_log.txt", "a");
    return fp;
}
static int s_cost_n      = 0;  // total cost_fn calls
static int s_finite_n    = 0;  // calls where cost was finite before validity checks
static int s_flip_rej_n  = 0;  // rejected by Euclidean face flip
static int s_qual_rej_n  = 0;  // rejected by skinny triangle quality
static int s_passed_n    = 0;  // passed all checks (finite cost kept)

void SSP_qslim_optimal_collapse_edge_callbacks(
  Eigen::MatrixXi & E,
  std::vector<std::tuple<Eigen::MatrixXd,Eigen::RowVectorXd,double> > &
    quadrics,
  int & v1,
  int & v2,
  const std::vector<std::vector<int>> & VF,
  decimate_cost_and_placement_func & cost_and_placement,
  decimate_pre_collapse_func       & pre_collapse,
  decimate_post_collapse_func      & post_collapse)
{
  using namespace igl;
  typedef std::tuple<Eigen::MatrixXd,Eigen::RowVectorXd,double> Quadric;

  cost_and_placement = [&quadrics,&v1,&v2,&VF](
    const int e,
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const Eigen::MatrixXi & E,
    const Eigen::VectorXi & /*EMAP*/,
    const Eigen::MatrixXi & /*EF*/,
    const Eigen::MatrixXi & /*EI*/,
    double & cost,
    Eigen::RowVectorXd & p)
  {
    // Combined quadric
    Quadric quadric_p = quadrics[E(e,0)] + quadrics[E(e,1)];
    // Quadric: p'Ap + 2b'p + c  ->  optimal: pA = -b
    const auto & A = std::get<0>(quadric_p);
    const auto & b = std::get<1>(quadric_p);
    const auto & c = std::get<2>(quadric_p);
    p    = -b * A.inverse();
    cost = p.dot(p*A) + 2*p.dot(b) + c;

    // Fall back to midpoint when the quadric is degenerate (NaN/Inf) or when
    // the cost is so small the bowl is too flat to trust the unconstrained minimum.
    static const double kFlatCostThreshold = 1e-5;
    bool used_midpoint = false;
    if (std::isinf(cost) || cost != cost || cost < kFlatCostThreshold)
    {
      used_midpoint = true;
      p    = 0.5 * (V.row(E(e,0)) + V.row(E(e,1)));
      cost = p.dot(p*A) + 2*p.dot(b) + c;
      if (std::isinf(cost) || cost != cost)
      {
        cost = std::numeric_limits<double>::infinity();
        p.setConstant(0);
      }
    }

// #ifdef ML_QEM_LOG
//     if (s_qslim_log_enabled)
//     {
//       Eigen::RowVectorXd va = V.row(E(e,0));
//       Eigen::RowVectorXd vb = V.row(E(e,1));

//       // [QSLIM-INF] — even midpoint gave NaN/Inf; edge permanently stuck.
//       // Skip when either endpoint is a boundary-to-infinity vertex (expected inf cost).
//       if (std::isinf(cost) && !std::isinf(va(0)) && !std::isinf(vb(0)))
//       {
//         fprintf(stderr,
//           "[QSLIM-INF] e=%d  v=(%d,%d)"
//           "  va=(%.4g,%.4g,%.4g)  vb=(%.4g,%.4g,%.4g)\n",
//           e, E(e,0), E(e,1),
//           va(0), va(1), va(2),
//           vb(0), vb(1), vb(2));
//       }

//       // [QSLIM-MIDPOINT-FALLBACK] — A.inverse() was degenerate; used midpoint instead.
//       if (used_midpoint && !std::isinf(cost))
//       {
//         fprintf(stderr,
//           "[QSLIM-MIDPOINT-FALLBACK] e=%d  v=(%d,%d)"
//           "  va=(%.4g,%.4g,%.4g)  vb=(%.4g,%.4g,%.4g)"
//           "  p=(%.4g,%.4g,%.4g)  cost=%.6g\n",
//           e, E(e,0), E(e,1),
//           va(0), va(1), va(2),
//           vb(0), vb(1), vb(2),
//           p(0),  p(1),  p(2), cost);
//       }

//       // [QSLIM-OOB] — optimal p escaped the original mesh bounding box.
//       if (!std::isinf(cost))
//       {
//         // Lazily build bbox from non-infinity vertices on the first call.
//         static Eigen::RowVectorXd bbox_min, bbox_max;
//         static bool bbox_ready = false;
//         if (!bbox_ready)
//         {
//           const double inf = std::numeric_limits<double>::infinity();
//           bbox_min = Eigen::RowVectorXd::Constant(V.cols(),  inf);
//           bbox_max = Eigen::RowVectorXd::Constant(V.cols(), -inf);
//           for (int i = 0; i < V.rows(); i++) {
//             if (std::isinf(V(i, 0))) continue;
//             bbox_min = bbox_min.cwiseMin(V.row(i));
//             bbox_max = bbox_max.cwiseMax(V.row(i));
//           }
//           bbox_ready = true;
//         }
//         bool oob = false;
//         for (int d = 0; d < (int)p.size(); d++)
//           if (p(d) < bbox_min(d) || p(d) > bbox_max(d)) { oob = true; break; }
//         if (oob)
//         {
//           fprintf(stderr,
//             "[QSLIM-OOB] e=%d  v=(%d,%d)"
//             "  va=(%.4g,%.4g,%.4g)  vb=(%.4g,%.4g,%.4g)"
//             "  p=(%.4g,%.4g,%.4g)  cost=%.6g"
//             "  bbox=(%.4g,%.4g,%.4g)-(%.4g,%.4g,%.4g)\n",
//             e, E(e,0), E(e,1),
//             va(0), va(1), va(2),
//             vb(0), vb(1), vb(2),
//             p(0),  p(1),  p(2), cost,
//             bbox_min(0), bbox_min(1), bbox_min(2),
//             bbox_max(0), bbox_max(1), bbox_max(2));
//         }
//       }
//     }
// #endif

    // ── Paper validity checks (Neural Subdivision, Hsueh-Ti et al. 2020) ────────
    // §4: Euclidean Face Flips  — dot(n_pre, n_post) > δ=0.2
    // §4: Skinny Triangles       — Q > 0.2 for all neighboring faces (Euclidean)
    // UV-domain checks (UV face flips, UV angle sum) are enforced in
    // SSP_collapse_edge.cpp after joint_lscm, where UV_post is available.
    ++s_cost_n;
    if (!std::isinf(cost))
    {
      ++s_finite_n;
      const Eigen::Vector3d optPos(p(0), p(1), p(2));
      const int va = E(e,0), vb = E(e,1);
      const double kFlipDelta = 0.2;
      const double kQualThr   = 0.2;
      bool reject = false;
      bool reject_was_flip = false;
      int  reject_fi = -1;
      double reject_dot = 0.0, reject_q = 0.0;

      for (int side = 0; side < 2 && !reject; ++side)
      {
        const int mv  = (side == 0) ? va : vb;
        const int opp = (side == 0) ? vb : va;
        if (mv >= (int)VF.size()) continue;
        for (int fi : VF[mv])
        {
          if (fi < 0 || fi >= F.rows()) continue;
          if (F(fi,0)==IGL_COLLAPSE_EDGE_NULL &&
              F(fi,1)==IGL_COLLAPSE_EDGE_NULL &&
              F(fi,2)==IGL_COLLAPSE_EDGE_NULL) continue;
          const int fv0=F(fi,0), fv1=F(fi,1), fv2=F(fi,2);
          if (fv0==opp || fv1==opp || fv2==opp) continue;
          if (std::isinf(V(fv0,0))||std::isinf(V(fv1,0))||std::isinf(V(fv2,0))) continue;

          auto pt = [&](int v) -> Eigen::Vector3d {
            return (v == mv) ? optPos : Eigen::Vector3d(V(v,0), V(v,1), V(v,2));
          };
          Eigen::Vector3d q0=pt(fv0), q1=pt(fv1), q2=pt(fv2);
          Eigen::Vector3d o0(V(fv0,0),V(fv0,1),V(fv0,2));
          Eigen::Vector3d o1(V(fv1,0),V(fv1,1),V(fv1,2));
          Eigen::Vector3d o2(V(fv2,0),V(fv2,1),V(fv2,2));

          // Euclidean face flip
          Eigen::Vector3d nPre  = (o1-o0).cross(o2-o0);
          Eigen::Vector3d nPost = (q1-q0).cross(q2-q0);
          double lPre = nPre.norm(), lPost = nPost.norm();
          double dot_val = (lPre > 1e-30 && lPost > 1e-30)
                           ? nPre.dot(nPost) / (lPre * lPost) : 1.0;
          if (dot_val <= kFlipDelta)
          { reject = true; reject_was_flip = true; reject_fi = fi; reject_dot = dot_val; break; }

          // Skinny triangle
          double sl0=(q1-q0).squaredNorm(), sl1=(q2-q1).squaredNorm(), sl2=(q0-q2).squaredNorm();
          double denom = sl0+sl1+sl2;
          double qPost = (denom > 1e-30) ?
              2.0*std::sqrt(3.0)*(q1-q0).cross(q2-q0).norm()/denom : 0.0;
          if (qPost < kQualThr)
          { reject = true; reject_fi = fi; reject_q = qPost; break; }
        }
      }

      if (reject)
      {
        if (reject_was_flip) ++s_flip_rej_n; else ++s_qual_rej_n;
        FILE* lf = rej_log();
        const int total_rej = s_flip_rej_n + s_qual_rej_n;
        if (lf && total_rej <= 200)  // cap file entries
          fprintf(lf, "[QSLIM-REJECT] reason=%-10s  e=%d  va=%d  vb=%d  fi=%d"
                  "  dot=%.4f  q=%.4f  cost_before=%.6g\n",
                  reject_was_flip ? "face_flip" : "quality",
                  e, va, vb, reject_fi, reject_dot, reject_q, cost);
        if (total_rej <= 10)  // first 10 also to stderr
          fprintf(stderr, "[QSLIM-REJECT] reason=%-10s  e=%d  va=%d  vb=%d  fi=%d"
                  "  dot=%.4f  q=%.4f\n",
                  reject_was_flip ? "face_flip" : "quality",
                  e, va, vb, reject_fi, reject_dot, reject_q);
        cost = std::numeric_limits<double>::infinity();
        p.setConstant(0);
      }
      else
      {
        ++s_passed_n;
      }
    }
    // periodic summary to stderr
    if (s_cost_n % 500 == 0)
      fprintf(stderr, "[QSLIM-STATS] calls=%d  finite=%d  flip_rej=%d  qual_rej=%d  passed=%d\n",
              s_cost_n, s_finite_n, s_flip_rej_n, s_qual_rej_n, s_passed_n);
  };

  // Remember endpoints.
  pre_collapse = [&v1,&v2](
    const Eigen::MatrixXd &                             ,/*V*/
    const Eigen::MatrixXi &                             ,/*F*/
    const Eigen::MatrixXi & E                           ,
    const Eigen::VectorXi &                             ,/*EMAP*/
    const Eigen::MatrixXi &                             ,/*EF*/
    const Eigen::MatrixXi &                             ,/*EI*/
    const igl::min_heap< std::tuple<double,int,int> > &  ,/*Q*/
    const Eigen::VectorXi &                             ,/*EQ*/
    const Eigen::MatrixXd &                             ,/*C*/
    const int e)->bool
  {
    v1 = E(e,0);
    v2 = E(e,1);
    return true;
  };

  // Update quadric after collapse.
  post_collapse = [&v1,&v2,&quadrics](
      const Eigen::MatrixXd &                             ,/*V*/
      const Eigen::MatrixXi &                             ,/*F*/
      const Eigen::MatrixXi &                             ,/*E*/
      const Eigen::VectorXi &                             ,/*EMAP*/
      const Eigen::MatrixXi &                             ,/*EF*/
      const Eigen::MatrixXi &                             ,/*EI*/
      const igl::min_heap< std::tuple<double,int,int> > &  ,/*Q*/
      const Eigen::VectorXi &                             ,/*EQ*/
      const Eigen::MatrixXd &                             ,/*C*/
      const int                                           ,/*e*/
      const int                                           ,/*e1*/
      const int                                           ,/*e2*/
      const int                                           ,/*f1*/
      const int                                           ,/*f2*/
      const bool collapsed)->void
  {
    if (collapsed)
      quadrics[v1<v2 ? v1 : v2] = quadrics[v1] + quadrics[v2];
  };
}
