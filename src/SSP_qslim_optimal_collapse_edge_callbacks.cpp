// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2016 Alec Jacobson <alecjacobson@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#include "SSP_qslim_optimal_collapse_edge_callbacks.h"
#include <Eigen/LU>
#include <cstdio>
#include <cmath>

static bool s_qslim_log_enabled = false;
void SSP_qslim_enable_log(bool enable) { s_qslim_log_enabled = enable; }

void SSP_qslim_optimal_collapse_edge_callbacks(
  Eigen::MatrixXi & E,
  std::vector<std::tuple<Eigen::MatrixXd,Eigen::RowVectorXd,double> > &
    quadrics,
  int & v1,
  int & v2,
  decimate_cost_and_placement_func & cost_and_placement,
  decimate_pre_collapse_func       & pre_collapse,
  decimate_post_collapse_func      & post_collapse)
{
  using namespace igl;
  typedef std::tuple<Eigen::MatrixXd,Eigen::RowVectorXd,double> Quadric;

  cost_and_placement = [&quadrics,&v1,&v2](
    const int e,
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & /*F*/,
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

#ifdef ML_QEM_LOG
    if (s_qslim_log_enabled)
    {
      Eigen::RowVectorXd va = V.row(E(e,0));
      Eigen::RowVectorXd vb = V.row(E(e,1));

      // [QSLIM-INF] — even midpoint gave NaN/Inf; edge permanently stuck.
      // Skip when either endpoint is a boundary-to-infinity vertex (expected inf cost).
      if (std::isinf(cost) && !std::isinf(va(0)) && !std::isinf(vb(0)))
      {
        fprintf(stderr,
          "[QSLIM-INF] e=%d  v=(%d,%d)"
          "  va=(%.4g,%.4g,%.4g)  vb=(%.4g,%.4g,%.4g)\n",
          e, E(e,0), E(e,1),
          va(0), va(1), va(2),
          vb(0), vb(1), vb(2));
      }

      // [QSLIM-MIDPOINT-FALLBACK] — A.inverse() was degenerate; used midpoint instead.
      if (used_midpoint && !std::isinf(cost))
      {
        fprintf(stderr,
          "[QSLIM-MIDPOINT-FALLBACK] e=%d  v=(%d,%d)"
          "  va=(%.4g,%.4g,%.4g)  vb=(%.4g,%.4g,%.4g)"
          "  p=(%.4g,%.4g,%.4g)  cost=%.6g\n",
          e, E(e,0), E(e,1),
          va(0), va(1), va(2),
          vb(0), vb(1), vb(2),
          p(0),  p(1),  p(2), cost);
      }

      // [QSLIM-OOB] — optimal p escaped the original mesh bounding box.
      if (!std::isinf(cost))
      {
        // Lazily build bbox from non-infinity vertices on the first call.
        static Eigen::RowVectorXd bbox_min, bbox_max;
        static bool bbox_ready = false;
        if (!bbox_ready)
        {
          const double inf = std::numeric_limits<double>::infinity();
          bbox_min = Eigen::RowVectorXd::Constant(V.cols(),  inf);
          bbox_max = Eigen::RowVectorXd::Constant(V.cols(), -inf);
          for (int i = 0; i < V.rows(); i++) {
            if (std::isinf(V(i, 0))) continue;
            bbox_min = bbox_min.cwiseMin(V.row(i));
            bbox_max = bbox_max.cwiseMax(V.row(i));
          }
          bbox_ready = true;
        }
        bool oob = false;
        for (int d = 0; d < (int)p.size(); d++)
          if (p(d) < bbox_min(d) || p(d) > bbox_max(d)) { oob = true; break; }
        if (oob)
        {
          fprintf(stderr,
            "[QSLIM-OOB] e=%d  v=(%d,%d)"
            "  va=(%.4g,%.4g,%.4g)  vb=(%.4g,%.4g,%.4g)"
            "  p=(%.4g,%.4g,%.4g)  cost=%.6g"
            "  bbox=(%.4g,%.4g,%.4g)-(%.4g,%.4g,%.4g)\n",
            e, E(e,0), E(e,1),
            va(0), va(1), va(2),
            vb(0), vb(1), vb(2),
            p(0),  p(1),  p(2), cost,
            bbox_min(0), bbox_min(1), bbox_min(2),
            bbox_max(0), bbox_max(1), bbox_max(2));
        }
      }
    }
#endif
  };

  // Remember endpoints.
  pre_collapse = [&v1,&v2](
    const Eigen::MatrixXd &                             ,/*V*/
    const Eigen::MatrixXi &                             ,/*F*/
    const Eigen::MatrixXi & E                           ,
    const Eigen::VectorXi &                             ,/*EMAP*/
    const Eigen::MatrixXi &                             ,/*EF*/
    const Eigen::MatrixXi &                             ,/*EI*/
    const min_heap< std::tuple<double,int,int> > &  ,/*Q*/
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
      const min_heap< std::tuple<double,int,int> > &  ,/*Q*/
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
