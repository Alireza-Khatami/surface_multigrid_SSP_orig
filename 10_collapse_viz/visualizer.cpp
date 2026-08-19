#include "visualizer.h"
#include "sheet_seam_viz.h"
#include "coarse_fine_viz.h"
#include "face_sample_tracker.h"

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>
#include <polyscope/point_cloud.h>

#include <igl/remove_unreferenced.h>
#include <igl/collapse_edge.h>  // IGL_COLLAPSE_EDGE_NULL
#include "face_dead.h"
#include <igl/writeOBJ.h>

#include <single_collapse_data.h>
#include <compute_barycentric.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <set>
#include <limits>
#include <cmath>
#include <thread>
#include <chrono>
#include <string>

using namespace Eigen;

// ---- SSP state defined in main.cpp ----
extern MatrixXd gV;
extern MatrixXi gF;
extern MatrixXd gVO;          // original mesh vertices
extern MatrixXi gFO;          // original mesh faces (post orient_faces_consistently)
extern VectorXi gFaceFlipped; // 1 = face was CW and got re-wound, 0 = already CCW
extern int      gCollapseCount;
extern bool     gFinished;
extern int      gDecType;   // 0=midpoint, 1=qslim, 2=meshlab
extern std::vector<single_collapse_data> gDecInfo;
extern std::vector<std::set<int>> gVertexStructIDs;

bool do_next_step();  // defined in main.cpp

// ---- display-only state ----
static float gUVOffset          = 1.5f;
static float gRingScale         = 1.0f;
static float gStepDelayMs       = 0.0f;
static bool  gRunning           = false;
static bool  gRunToSeam         = false;
static bool  gRunToBdCase       = false;  // stop at LSCM Case 1 or 2 (one/both endpoints on boundary)
static bool  gStopAtDC         = false;  // persistent: stop + enter canonical view on every double cover case
static int   gBreakAtCollapse   = -1;     // stop when gCollapseCount reaches this; -1 = disabled
static char  gBreakAtBuf[16]    = "";
static bool  gCanonicalView     = false;
static bool  gShowRingPre       = true;
static bool  gShowRingPost      = true;
static bool  gShowUVPre         = false;
static bool  gShowUVPost        = false;
// Canonical-view two-group toggles
static bool  gShowCanonRing     = true;   // one-ring + non-active sheets
static bool  gShowCanonUV       = true;   // UV meshes + UV collapsed edge
// DC vertex group toggles
static bool  gShowDCVertVi      = true;   // vi and vj (collapse endpoints)
static bool  gShowDCVertBglued  = true;   // B_glued: arc endpoints shared between sheets
static bool  gShowDCBrefTop     = true;   // B_reflected top-sheet copies
static bool  gShowDCBrefBot     = true;   // B_reflected bottom-sheet copies
static bool  gShowArrowPre      = false;
static bool  gShowArrowPost     = false;
static bool  gShowCorrArrows    = false; // pre → post correspondence arrows (query_coarse_to_fine)
static bool  gShowCorrPts      = false;
static float gPostHeight        = 1.0f;   // elevation of ring_post above ring_pre (× ring_span, along avg_normal)
static bool  gShowCollapsedEdge = true;
static bool  gShowMeshPre       = false;
static bool  gShowOrigOrient    = false; // original fine mesh with CW/CCW face colors
static float edge_radius        = 0.00006f;
static float pts_radius         = 0.00008f;

// Pre-collapse mesh snapshot (compacted, taken just before each step)
static MatrixXd gPreV;
static MatrixXi gPreF;
static bool     gHasPreMesh = false;

// Export directory: editable via the UI, written as snapshot_at_<N>.obj
static char gExportDir[512] = ".";

struct DisplaySnap {
    bool valid = false;
    int vi = -1, vj = -1;
    MatrixXd V_pre;
    MatrixXd V_post;   // post-collapse 3D geometry (s moved to p, flap faces gone)
    MatrixXi FUV_pre, FUV_post;
    MatrixXd UV_pre,  UV_post;
    VectorXi b;
    // Double cover (boundary cases only)
    bool has_dc = false;
    MatrixXi FUV_dc_pre, FUV_dc_post;
    MatrixXd UV_dc_pre, UV_dc_post;
    std::vector<int> dc_B_glued;      // arc endpoint indices (local one-ring space)
    std::vector<int> dc_B_reflected;  // middle B indices (top-sheet local indices)
} gSnap;

// ---- helpers ----

static MatrixXi live_faces()
{
    std::vector<std::array<int,3>> rows;
    rows.reserve(gF.rows());
    for (int f = 0; f < gF.rows(); f++) {
        int v0 = gF(f,0), v1 = gF(f,1), v2 = gF(f,2);
        if (is_face_dead(gF, f)) continue;
        if (std::isinf(gV(v0,0)) || std::isinf(gV(v1,0)) || std::isinf(gV(v2,0))) continue;
        rows.push_back({v0, v1, v2});
    }
    MatrixXi F(rows.size(), 3);
    for (int i = 0; i < (int)rows.size(); i++)
        F.row(i) << rows[i][0], rows[i][1], rows[i][2];
    return F;
}

static MatrixXd safe_V()
{
    MatrixXd V = gV.leftCols(3);
    for (int v = 0; v < V.rows(); v++)
        if (std::isinf(V(v,0)) || std::isinf(V(v,1)) || std::isinf(V(v,2)))
            V.row(v).setZero();
    return V;
}

// Export the current live mesh to <gExportDir>/latest_snapshot.obj,
// overwriting the previous file each time.  Called automatically before
// every collapse so the file always holds the last good state — if the
// program crashes, reload latest_snapshot.obj to resume from there.
static void export_current_mesh()
{
    MatrixXi Fl = live_faces();
    MatrixXd Vs = safe_V();
    MatrixXd Vc; MatrixXi Fc; VectorXi I1, I2;
    igl::remove_unreferenced(Vs, Fl, Vc, Fc, I1, I2);

    std::string path = std::string(gExportDir) + "/latest_snapshot.obj";
    bool ok = igl::writeOBJ(path, Vc, Fc);
    if (!ok)
        fprintf(stderr, "[export] FAILED to write %s\n", path.c_str());
}

static void snapshot_pre_mesh()
{
    MatrixXi Fl = live_faces();
    MatrixXd Vs = safe_V();
    VectorXi I1, I2;
    igl::remove_unreferenced(Vs, Fl, gPreV, gPreF, I1, I2);
    gHasPreMesh = true;
    export_current_mesh();   // always overwrite latest_snapshot.obj
}

static void refresh_snap()
{
    if (gDecInfo.empty()) return;
    const single_collapse_data & d = gDecInfo.back();
    gSnap.V_pre  = d.V_pre;
    gSnap.V_post = d.V_post;
    if (!d.sheets.empty()) {
        const SheetData & sd = d.sheets[0];
        gSnap.valid    = true;
        gSnap.b        = sd.b;
        gSnap.vi       = sd.subsetVIdx(sd.b(0));
        gSnap.vj       = sd.subsetVIdx(sd.b(1));
        gSnap.FUV_pre  = sd.FUV_pre;
        gSnap.FUV_post = sd.FUV_post;
        gSnap.UV_pre   = sd.UV_pre;
        gSnap.UV_post  = sd.UV_post;
        gSnap.has_dc          = sd.has_double_cover;
        gSnap.FUV_dc_pre      = sd.FUV_dc_pre;
        gSnap.FUV_dc_post     = sd.FUV_dc_post;
        gSnap.UV_dc_pre       = sd.UV_dc_pre;
        gSnap.UV_dc_post      = sd.UV_dc_post;
        gSnap.dc_B_glued      = sd.dc_B_glued;
        gSnap.dc_B_reflected  = sd.dc_B_reflected;
    }
}

// ---- shared geometry ----

struct DisplayGeometry {
    MatrixXd V_ring;       // one-ring vertices in 3D (scaled, from V_pre)
    MatrixXd V_ring_post;  // post one-ring vertices in 3D (scaled, from V_post)
    MatrixXd uv_pre_3d;    // UV pre panel lifted into 3D
    MatrixXd uv_post_3d;   // UV post panel lifted into 3D
    MatrixXd arrows_pre;   // ring → UV pre vectors
    MatrixXd arrows_post;  // ring → UV post vectors
    MatrixXd corr_arrows;  // pre → post correspondence vectors (query_coarse_to_fine logic)
    MatrixXd corr_pts;    // tips of corr_arrows: 3D positions on post one-ring
    Vector3d avg_normal;   // one-ring average face normal
    Vector3d centroid;     // one-ring centroid
    double   ring_span;
    // Double cover UV panels (boundary cases only)
    bool     has_dc = false;
    MatrixXd dc_uv_pre_3d;
    MatrixXd dc_uv_post_3d;
};

static DisplayGeometry compute_ring_geometry()
{
    DisplayGeometry g;

    // centroid and ring_span
    g.centroid   = Vector3d::Zero();
    g.ring_span  = 0.0;
    g.avg_normal = Vector3d::Zero();
    {
        int cnt = 0;
        for (int v = 0; v < gSnap.V_pre.rows(); v++) {
            if (std::isfinite(gSnap.V_pre(v,0))) {
                g.centroid += gSnap.V_pre.row(v).transpose();
                cnt++;
            }
        }
        if (cnt > 0) g.centroid /= cnt;
        for (int ax = 0; ax < 3; ax++) {
            double lo =  std::numeric_limits<double>::infinity();
            double hi = -std::numeric_limits<double>::infinity();
            for (int v = 0; v < gSnap.V_pre.rows(); v++) {
                double val = gSnap.V_pre(v, ax);
                if (std::isfinite(val)) { lo = std::min(lo, val); hi = std::max(hi, val); }
            }
            if (hi > lo) g.ring_span = std::max(g.ring_span, hi - lo);
        }
        if (g.ring_span < 1e-10) g.ring_span = 1.0;
    }

    // pre one-ring scaled around centroid
    g.V_ring.resize(gSnap.V_pre.rows(), 3);
    for (int v = 0; v < gSnap.V_pre.rows(); v++) {
        if (std::isfinite(gSnap.V_pre(v,0)))
            g.V_ring.row(v) = g.centroid.transpose() +
                              (gSnap.V_pre.row(v) - g.centroid.transpose()) * (double)gRingScale;
        else
            g.V_ring.row(v) = gSnap.V_pre.row(v);
    }

    // post one-ring at V_post positions (same centering/scale as V_ring),
    // then elevated above ring_pre along the average face normal.
    g.V_ring_post.resize(gSnap.V_post.rows(), 3);
    for (int v = 0; v < gSnap.V_post.rows(); v++) {
        if (std::isfinite(gSnap.V_post(v,0)))
            g.V_ring_post.row(v) = g.centroid.transpose() +
                                   (gSnap.V_post.row(v) - g.centroid.transpose()) * (double)gRingScale;
        else
            g.V_ring_post.row(v).setZero();
    }
    {
        // Elevate every valid post vertex along avg_normal (computed below; precompute here
        // by using a temporary pass so we can apply it before building corr_arrows).
        // avg_normal is filled in the loop further down, so compute a quick version now.
        Vector3d tmp_normal = Vector3d::Zero();
        for (int f = 0; f < gSnap.FUV_pre.rows(); f++) {
            Vector3d p0 = gSnap.V_pre.row(gSnap.FUV_pre(f,0)).transpose();
            Vector3d p1 = gSnap.V_pre.row(gSnap.FUV_pre(f,1)).transpose();
            Vector3d p2 = gSnap.V_pre.row(gSnap.FUV_pre(f,2)).transpose();
            if (p0.allFinite() && p1.allFinite() && p2.allFinite())
                tmp_normal += (p1 - p0).cross(p2 - p0);
        }
        if (tmp_normal.squaredNorm() < 1e-20) tmp_normal = Vector3d::UnitY();
        else tmp_normal.normalize();
        Vector3d elev = tmp_normal * (double)gPostHeight * g.ring_span;
        for (int v = 0; v < g.V_ring_post.rows(); v++)
            if (std::isfinite(g.V_ring_post(v, 0)))
                g.V_ring_post.row(v) += elev.transpose();
    }

    // Pre → Post correspondence: for each pre vertex i, project UV_pre.row(i) into
    // UV_post/FUV_post using the exact same compute_barycentric + face-selection logic
    // as each single step of query_coarse_to_fine. Arrow = post_pos − pre_pos.
    {
        int nV = (int)gSnap.UV_pre.rows();
        g.corr_arrows.resize(nV, 3);
        g.corr_arrows.setZero();
        g.corr_pts.resize(nV, 3);
        g.corr_pts.setConstant(std::numeric_limits<double>::quiet_NaN());
        for (int i = 0; i < nV; i++) {
            if (!std::isfinite(g.V_ring(i, 0))) continue;
            VectorXd queryUV = gSnap.UV_pre.row(i).transpose();
            MatrixXd B;
            compute_barycentric(queryUV, gSnap.UV_post, gSnap.FUV_post, B);
            // Same face-selection as query_coarse_to_fine
            VectorXd distToValid = -B.rowwise().minCoeff();
            double minD = 1.0;
            int idxToFUV = 0;
            for (int bb = 0; bb < (int)distToValid.size(); bb++)
                if (distToValid(bb) < minD) { minD = distToValid(bb); idxToFUV = bb; }
            for (int c = 0; c < 3; c++) B(idxToFUV, c) = std::max(0.0, B(idxToFUV, c));
            double bsum = B.row(idxToFUV).sum();
            if (bsum > 1e-12) B.row(idxToFUV) /= bsum;
            // 3D position on elevated post one-ring
            Vector3d post_pos =
                B(idxToFUV, 0) * g.V_ring_post.row(gSnap.FUV_post(idxToFUV, 0)).transpose() +
                B(idxToFUV, 1) * g.V_ring_post.row(gSnap.FUV_post(idxToFUV, 1)).transpose() +
                B(idxToFUV, 2) * g.V_ring_post.row(gSnap.FUV_post(idxToFUV, 2)).transpose();
            // Arrow from pre_pos → post_pos; tip as separate point cloud
            g.corr_arrows.row(i) = (post_pos - g.V_ring.row(i).transpose()).transpose();
            g.corr_pts.row(i)    = post_pos.transpose();
        }
    }

    // average face normal
    for (int f = 0; f < gSnap.FUV_pre.rows(); f++) {
        Vector3d p0 = gSnap.V_pre.row(gSnap.FUV_pre(f,0)).transpose();
        Vector3d p1 = gSnap.V_pre.row(gSnap.FUV_pre(f,1)).transpose();
        Vector3d p2 = gSnap.V_pre.row(gSnap.FUV_pre(f,2)).transpose();
        if (!p0.allFinite() || !p1.allFinite() || !p2.allFinite()) continue;
        g.avg_normal += (p1 - p0).cross(p2 - p0);
    }
    if (g.avg_normal.squaredNorm() < 1e-20) g.avg_normal = Vector3d::UnitZ();
    else g.avg_normal.normalize();

    // tangent frame for UV panel — initial arbitrary choice, corrected below
    Vector3d arb = (std::abs(g.avg_normal.dot(Vector3d::UnitX())) < 0.9)
                   ? Vector3d::UnitX() : Vector3d::UnitY();
    Vector3d t1 = (arb - g.avg_normal * g.avg_normal.dot(arb)).normalized();
    Vector3d t2 = g.avg_normal.cross(t1).normalized();

    double uv_span_u = gSnap.UV_pre.col(0).maxCoeff() - gSnap.UV_pre.col(0).minCoeff();
    double uv_span_v = gSnap.UV_pre.col(1).maxCoeff() - gSnap.UV_pre.col(1).minCoeff();
    double uv_span   = std::max(uv_span_u, uv_span_v);
    double uv_scale  = (uv_span > 1e-10) ? g.ring_span / uv_span : 1.0;
    uv_scale *= (double)gRingScale;

    double u_center = (gSnap.UV_pre.col(0).maxCoeff() + gSnap.UV_pre.col(0).minCoeff()) * 0.5;
    double v_center = (gSnap.UV_pre.col(1).maxCoeff() + gSnap.UV_pre.col(1).minCoeff()) * 0.5;
    Vector3d panel_center = g.centroid + g.avg_normal * (double)gUVOffset * g.ring_span;

    // Rotate (t1, t2) by β so the UV panel's in-plane orientation matches the
    // one-ring's tangent-plane projection (2D Procrustes):
    //   β = atan2( Σ(y·u − x·v),  Σ(x·u + y·v) )
    // where (x,y) = V_pre vertex projected onto (t1,t2),  (u,v) = centered UV_pre vertex.
    {
        double A = 0.0, B = 0.0;
        for (int i = 0; i < gSnap.V_pre.rows(); i++) {
            if (!std::isfinite(gSnap.V_pre(i, 0))) continue;
            Vector3d p = gSnap.V_pre.row(i).transpose() - g.centroid;
            double x = t1.dot(p), y = t2.dot(p);
            double u = gSnap.UV_pre(i, 0) - u_center;
            double v = gSnap.UV_pre(i, 1) - v_center;
            A += x * u + y * v;
            B += y * u - x * v;
        }
        double beta = std::atan2(-B, A);  // maps UV→3D: R*(u,v)=(x,y), β = atan2(xv-yu, xu+yv)
        double cb = std::cos(beta), sb = std::sin(beta);
        Vector3d t1_new =  cb * t1 + sb * t2;
        Vector3d t2_new = -sb * t1 + cb * t2;
        t1 = t1_new;
        t2 = t2_new;
    }

    auto make3d = [&](const MatrixXd & UV) {
        MatrixXd P(UV.rows(), 3);
        for (int i = 0; i < UV.rows(); i++) {
            double u = (UV(i,0) - u_center) * uv_scale ;
            double v = (UV(i,1) - v_center) * uv_scale ;
            P.row(i) = (panel_center + t1*u + t2*v).transpose();
        }
        return P;
    };

    g.uv_pre_3d  = make3d(gSnap.UV_pre);
    g.uv_post_3d = make3d(gSnap.UV_post);

    if (gSnap.has_dc && gSnap.UV_dc_pre.rows() > 0) {
        g.has_dc = true;
        // Recompute center/scale from the full DC UV (includes bottom-sheet B rows)
        // so the DC panel is centered and scaled correctly.
        const Eigen::MatrixXd & UVdc = gSnap.UV_dc_pre;
        double dc_u_center = (UVdc.col(0).maxCoeff() + UVdc.col(0).minCoeff()) * 0.5;
        double dc_v_center = (UVdc.col(1).maxCoeff() + UVdc.col(1).minCoeff()) * 0.5;
        double dc_span = std::max(UVdc.col(0).maxCoeff() - UVdc.col(0).minCoeff(),
                                  UVdc.col(1).maxCoeff() - UVdc.col(1).minCoeff());
        double dc_scale = (dc_span > 1e-10) ? g.ring_span / dc_span : 1.0;
        dc_scale *= (double)gRingScale;
        auto make3d_dc = [&](const Eigen::MatrixXd & UV) {
            Eigen::MatrixXd P(UV.rows(), 3);
            for (int i = 0; i < UV.rows(); i++) {
                double u = (UV(i,0) - dc_u_center) * dc_scale;
                double v = (UV(i,1) - dc_v_center) * dc_scale;
                P.row(i) = (panel_center + t1*u + t2*v).transpose();
            }
            return P;
        };
        g.dc_uv_pre_3d  = make3d_dc(gSnap.UV_dc_pre);
        g.dc_uv_post_3d = make3d_dc(gSnap.UV_dc_post);
    }

    // arrow vectors
    int nV = gSnap.V_pre.rows();
    g.arrows_pre.resize(nV, 3);  g.arrows_pre.setZero();
    g.arrows_post.resize(nV, 3); g.arrows_post.setZero();
    for (int i = 0; i < nV; i++) {
        if (std::isfinite(g.V_ring(i,0)))
            g.arrows_pre.row(i)  = g.uv_pre_3d.row(i)  - g.V_ring.row(i);
        if (std::isfinite(g.V_ring_post(i,0)))
            g.arrows_post.row(i) = g.uv_post_3d.row(i) - g.V_ring_post.row(i);
    }

    return g;
}

static void register_ring_geometry(const DisplayGeometry & g)
{
    // one-ring meshes + matching point clouds
    {
        auto * rp = polyscope::registerSurfaceMesh("one_ring_pre", g.V_ring, gSnap.FUV_pre);
        rp->setSurfaceColor({0.3f, 0.55f, 1.0f})->setEdgeWidth(1.5)->setSmoothShade(false);
        rp->setEnabled(gShowRingPre);
        rp->addVertexVectorQuantity("to_uv_pre", g.arrows_pre, polyscope::VectorType::AMBIENT)->setEnabled(gShowArrowPre);
        {
            auto * cq = rp->addVertexVectorQuantity("pre_to_post_corr", g.corr_arrows,
                                                    polyscope::VectorType::AMBIENT);
            cq->setEnabled(gShowCorrArrows);
            cq->setVectorColor({0.1f, 0.9f, 0.3f});
        }

        polyscope::registerPointCloud("ring_pre_pts", g.V_ring)
            ->setPointColor({0.3f, 0.55f, 1.0f})->setPointRadius(pts_radius, true)->setEnabled(gShowRingPre);

        auto * ro = polyscope::registerSurfaceMesh("one_ring_post", g.V_ring_post, gSnap.FUV_post);
        ro->setSurfaceColor({1.0f, 0.5f, 0.15f})->setEdgeWidth(1.5)->setSmoothShade(false)->setTransparency(1.0f);
        ro->setEnabled(gShowRingPost);
        ro->addVertexVectorQuantity("to_uv_post", g.arrows_post, polyscope::VectorType::AMBIENT)->setEnabled(gShowArrowPost);

        polyscope::registerPointCloud("ring_post_pts", g.V_ring_post)
            ->setPointColor({1.0f, 0.5f, 0.15f})->setPointRadius(pts_radius, true)->setEnabled(gShowRingPost);

        polyscope::registerPointCloud("corr_pts", g.corr_pts)
            ->setPointColor({0.1f, 0.9f, 0.3f})->setPointRadius(pts_radius * 50.0f, true)->setEnabled(gShowCorrPts);
    }

    // collapsed edge in 3-D: s at pre position → d at pre position
    {
        MatrixXd eV(2, 3);
        eV.row(0) = g.V_ring.row(gSnap.b(0));
        eV.row(1) = g.V_ring.row(gSnap.b(1));
        MatrixXi eE(1, 2); eE << 0, 1;
        polyscope::registerCurveNetwork("collapsed_edge", eV, eE)
            ->setRadius(edge_radius)->setColor({1.0f, 0.05f, 0.05f})
            ->setEnabled(gShowCollapsedEdge);
    }

    // UV meshes + matching point clouds
    {
        auto * up = polyscope::registerSurfaceMesh("uv_pre", g.uv_pre_3d, gSnap.FUV_pre);
        up->setSurfaceColor({0.3f, 0.55f, 1.0f})->setEdgeWidth(1.5)->setSmoothShade(false);
        up->setEnabled(gShowUVPre);

        polyscope::registerPointCloud("uv_pre_pts", g.uv_pre_3d)
            ->setPointColor({0.3f, 0.55f, 1.0f})->setPointRadius(pts_radius, true)->setEnabled(gShowUVPre);

        auto * uo = polyscope::registerSurfaceMesh("uv_post", g.uv_post_3d, gSnap.FUV_post);
        uo->setSurfaceColor({1.0f, 0.5f, 0.15f})->setEdgeWidth(1.5)->setSmoothShade(false)->setTransparency(1.0f);
        uo->setEnabled(gShowUVPost);

        polyscope::registerPointCloud("uv_post_pts", g.uv_post_3d)
            ->setPointColor({1.0f, 0.5f, 0.15f})->setPointRadius(pts_radius, true)->setEnabled(gShowUVPost);
    }

    // collapsed edge in UV space
    {
        MatrixXd euV(2, 3);
        euV.row(0) = g.uv_pre_3d.row(gSnap.b(0));
        euV.row(1) = g.uv_pre_3d.row(gSnap.b(1));
        MatrixXi euE(1, 2); euE << 0, 1;
        polyscope::registerCurveNetwork("uv_collapsed_edge", euV, euE)
            ->setRadius(edge_radius)->setColor({1.0f, 0.05f, 0.05f})
            ->setEnabled(gShowCollapsedEdge);
    }

}

// ---- canonical view ----
// Print a full diagnostic dump of the current snap's DC/joint-LSCM data to stdout.
static void print_dc_snap()
{
    const auto& s = gSnap;
    if (!s.valid) { printf("[PRINT-DC] no valid snap\n"); return; }

    printf("\n========== DC / Joint-LSCM Diagnostic Dump ==========\n");
    printf("  vi=%d  vj=%d  local_vi=%d  local_vj=%d\n",
           s.vi, s.vj, s.b.size() > 0 ? s.b(0) : -1, s.b.size() > 1 ? s.b(1) : -1);
    printf("  has_dc=%d\n", (int)s.has_dc);
    printf("  V_pre: %dx3   V_post: %dx3\n",
           (int)s.V_pre.rows(), (int)s.V_post.rows());

    // FUV_pre
    printf("\n--- FUV_pre (%dx3) ---\n", (int)s.FUV_pre.rows());
    for (int f = 0; f < s.FUV_pre.rows(); f++)
        printf("  f%d: [%d %d %d]\n", f, s.FUV_pre(f,0), s.FUV_pre(f,1), s.FUV_pre(f,2));

    // FUV_post
    printf("\n--- FUV_post (%dx3) ---\n", (int)s.FUV_post.rows());
    for (int f = 0; f < s.FUV_post.rows(); f++)
        printf("  f%d: [%d %d %d]\n", f, s.FUV_post(f,0), s.FUV_post(f,1), s.FUV_post(f,2));

    // UV_pre
    printf("\n--- UV_pre (%dx2) ---\n", (int)s.UV_pre.rows());
    for (int v = 0; v < s.UV_pre.rows(); v++)
        printf("  v%d: (%.6f, %.6f)\n", v, s.UV_pre(v,0), s.UV_pre(v,1));

    // UV_post
    printf("\n--- UV_post (%dx2) ---\n", (int)s.UV_post.rows());
    for (int v = 0; v < s.UV_post.rows(); v++)
        printf("  v%d: (%.6f, %.6f)\n", v, s.UV_post(v,0), s.UV_post(v,1));

    if (s.has_dc) {
        printf("\n--- DC B classification ---\n");
        printf("  B_glued (%d):", (int)s.dc_B_glued.size());
        for (int b : s.dc_B_glued)   printf(" %d", b);
        printf("\n  B_reflected (%d):", (int)s.dc_B_reflected.size());
        for (int b : s.dc_B_reflected) printf(" %d", b);
        printf("\n");

        printf("\n--- FUV_dc_pre (%dx3) ---\n", (int)s.FUV_dc_pre.rows());
        for (int f = 0; f < s.FUV_dc_pre.rows(); f++)
            printf("  f%d: [%d %d %d]%s\n", f,
                   s.FUV_dc_pre(f,0), s.FUV_dc_pre(f,1), s.FUV_dc_pre(f,2),
                   f == s.FUV_pre.rows() - 1 ? "  <- top/bot split" : "");

        printf("\n--- FUV_dc_post (%dx3) ---\n", (int)s.FUV_dc_post.rows());
        for (int f = 0; f < s.FUV_dc_post.rows(); f++)
            printf("  f%d: [%d %d %d]%s\n", f,
                   s.FUV_dc_post(f,0), s.FUV_dc_post(f,1), s.FUV_dc_post(f,2),
                   f == s.FUV_post.rows() - 1 ? "  <- top/bot split" : "");

        printf("\n--- UV_dc_pre (%dx2) ---\n", (int)s.UV_dc_pre.rows());
        int nVjoint = (int)s.V_pre.rows() + 1;
        for (int v = 0; v < s.UV_dc_pre.rows(); v++) {
            const char* tag = "";
            if (v == s.b(0))    tag = "  <- local_vi (top)";
            else if (v == s.b(1)) tag = "  <- local_vj (top)";
            else if (v == (int)s.V_pre.rows()) tag = "  <- vi post-collapse slot";
            else {
                for (int k = 0; k < (int)s.dc_B_reflected.size(); k++)
                    if (v == nVjoint + k) { tag = "  <- B_ref_bot"; break; }
            }
            printf("  v%d: (%.6f, %.6f)%s\n", v, s.UV_dc_pre(v,0), s.UV_dc_pre(v,1), tag);
        }

        printf("\n--- UV_dc_post (%dx2) ---\n", (int)s.UV_dc_post.rows());
        for (int v = 0; v < s.UV_dc_post.rows(); v++) {
            const char* tag = "";
            if (v == s.b(0))    tag = "  <- local_vi (top)";
            else if (v == s.b(1)) tag = "  <- local_vj (top)";
            else if (v == (int)s.V_pre.rows()) tag = "  <- vi post-collapse slot";
            else {
                for (int k = 0; k < (int)s.dc_B_reflected.size(); k++)
                    if (v == nVjoint + k) { tag = "  <- B_ref_bot"; break; }
            }
            printf("  v%d: (%.6f, %.6f)%s\n", v, s.UV_dc_post(v,0), s.UV_dc_post(v,1), tag);
        }
    }
    printf("======================================================\n\n");
    fflush(stdout);
}

// Clears all structures and re-registers the one-ring/UV geometry rotated so that
// avg_normal aligns with world Y-up: one-ring lies flat on the XZ floor, UV panels
// float directly above. The main decimated mesh is hidden in this mode.
static void show_canonical_view()
{
    if (!gSnap.valid) return;

    DisplayGeometry g = compute_ring_geometry();

    // Rotation: avg_normal → world Y-up (0,1,0)
    Vector3d world_up(0.0, -1.0, 0.0);
    Matrix3d R;
    Vector3d axis = g.avg_normal.cross(world_up);
    double   sinA = axis.norm();
    double   cosA = g.avg_normal.dot(world_up);
    if (sinA < 1e-6)
        R = (cosA > 0.0) ? Matrix3d::Identity()
                         : (Matrix3d() << -1,0,0, 0,-1,0, 0,0,1).finished();
    else
        R = AngleAxisd(std::atan2(sinA, cosA), axis / sinA).toRotationMatrix();

    // Apply rotation + centering to every vertex buffer
    auto rot = [&](const MatrixXd & P) -> MatrixXd {
        MatrixXd out(P.rows(), 3);
        for (int i = 0; i < P.rows(); i++) {
            if (!std::isfinite(P(i,0))) { out.row(i).setZero(); continue; }
            out.row(i) = (R * (P.row(i).transpose() - g.centroid)).transpose();
        }
        return out;
    };

    DisplayGeometry gc;
    gc.V_ring      = rot(g.V_ring);
    gc.V_ring_post = rot(g.V_ring_post);
    gc.uv_pre_3d   = rot(g.uv_pre_3d);
    gc.uv_post_3d  = rot(g.uv_post_3d);
    gc.corr_pts    = rot(g.corr_pts);

    // After R, avg_normal maps to world_up=(0,-1,0), so all elevations are in -Y.
    // Move UV_pre to the floor (Y=0, same level as ring_pre) and
    // UV_post to ring_post height (Y = -gPostHeight * ring_span).
    // This must be done before the arrow vectors below are computed.
    double y_panel_canon, y_post_canon;
    {
        double y_panel = -(double)gUVOffset   * g.ring_span; // current Y of both UV panels
        double y_post  = -(double)gPostHeight * g.ring_span; // desired Y for UV_post
        y_panel_canon = y_panel;
        y_post_canon  = y_post;
        for (int i = 0; i < gc.uv_pre_3d.rows(); i++)
            gc.uv_pre_3d(i, 1) -= y_panel;          // cancel panel offset → Y = 0
        for (int i = 0; i < gc.uv_post_3d.rows(); i++)
            gc.uv_post_3d(i, 1) += y_post - y_panel; // bring to ring_post Y
    }

    // Rotate + Y-adjust double cover panels (same offsets as regular UV panels).
    if (g.has_dc) {
        gc.has_dc = true;
        gc.dc_uv_pre_3d  = rot(g.dc_uv_pre_3d);
        gc.dc_uv_post_3d = rot(g.dc_uv_post_3d);
        for (int i = 0; i < gc.dc_uv_pre_3d.rows(); i++)
            gc.dc_uv_pre_3d(i, 1)  -= y_panel_canon;
        for (int i = 0; i < gc.dc_uv_post_3d.rows(); i++)
            gc.dc_uv_post_3d(i, 1) += y_post_canon - y_panel_canon;
    }

    int nV = gc.V_ring.rows();
    gc.arrows_pre.resize(nV, 3);  gc.arrows_pre.setZero();
    gc.arrows_post.resize(nV, 3); gc.arrows_post.setZero();
    gc.corr_arrows.resize(nV, 3); gc.corr_arrows.setZero();
    for (int i = 0; i < nV; i++) {
        if (std::isfinite(g.V_ring(i,0))) {
            gc.arrows_pre.row(i)  = gc.uv_pre_3d.row(i)  - gc.V_ring.row(i);
            gc.arrows_post.row(i) = gc.uv_post_3d.row(i) - gc.V_ring_post.row(i);
        }
        if (g.corr_arrows.row(i).squaredNorm() > 0)
            gc.corr_arrows.row(i) = (R * g.corr_arrows.row(i).transpose()).transpose();
    }

    polyscope::removeAllStructures();
    clear_seam_onering();  // structures are gone; keep name list consistent
    register_ring_geometry(gc);

    // Apply two-group canonical visibility (overrides register_ring_geometry defaults).
    // Ring group: one-ring meshes, corr pts, 3-D collapsed edge, non-active sheets
    polyscope::getSurfaceMesh("one_ring_pre")    ->setEnabled(gShowCanonRing);
    polyscope::getSurfaceMesh("one_ring_post")   ->setEnabled(gShowCanonRing);
    polyscope::getPointCloud ("corr_pts")        ->setEnabled(gShowCanonRing);
    polyscope::getCurveNetwork("collapsed_edge") ->setEnabled(gShowCanonRing);
    // UV group: UV meshes, UV points, ring sample points (pre+post), UV collapsed edge.
    // When DC data is present the DC overlay replaces the regular UV panels.
    bool showRegularUV = gShowCanonUV && !gc.has_dc;
    polyscope::getSurfaceMesh("uv_pre")              ->setEnabled(showRegularUV);
    polyscope::getSurfaceMesh("uv_post")             ->setEnabled(showRegularUV);
    polyscope::getPointCloud ("uv_pre_pts")          ->setEnabled(showRegularUV);
    polyscope::getPointCloud ("uv_post_pts")         ->setEnabled(showRegularUV);
    polyscope::getPointCloud ("ring_pre_pts")        ->setEnabled(gShowCanonUV);
    polyscope::getPointCloud ("ring_post_pts")       ->setEnabled(gShowCanonUV);
    polyscope::getCurveNetwork("uv_collapsed_edge")  ->setEnabled(showRegularUV);

    // Non-active sheet faces: faces incident to d that belong to sheets NOT
    // processed by this collapse, PLUS all extra active sheets (sheets[1...]).
    //
    // Why include extra active sheets here: for a seam collapse every sheet that
    // contains the seam edge has BOTH endpoints and becomes "active" (LSCM runs for
    // each).  Their faces all go into FIdx_combined inside SSP_collapse_edge, so they
    // are excluded from data.non_active_faces even though they are NOT the primary
    // sheet shown in the one-ring/UV panels.  We fold them in explicitly so all
    // non-primary-sheet faces appear together in this purple overlay.
    if (!gDecInfo.empty()) {
        const auto & collapse = gDecInfo.back();
        std::vector<std::array<double,3>> verts;
        std::vector<std::array<int,3>>    faces;

        // (a) d's faces in truly inactive sheets (classic non-active faces)
        for (const auto & naf : collapse.non_active_faces) {
            if (naf.is_infinity_face) continue;
            if (!naf.p0.allFinite() || !naf.p1.allFinite() || !naf.p2.allFinite()) continue;
            int base = (int)verts.size();
            for (const Vector3d & pt : {naf.p0, naf.p1, naf.p2}) {
                Vector3d rp = R * (pt - g.centroid);
                verts.push_back({rp.x(), rp.y(), rp.z()});
            }
            faces.push_back({base, base+1, base+2});
        }

        // (b) extra active sheets (sheets[1...]) — seam collapses only.
        // Their pre-collapse one-ring faces, transformed into the canonical frame.
        for (int si = 1; si < (int)collapse.sheets.size(); si++) {
            const SheetData & es = collapse.sheets[si];
            for (int fi = 0; fi < es.FUV_pre.rows(); fi++) {
                int i0 = es.FUV_pre(fi,0), i1 = es.FUV_pre(fi,1), i2 = es.FUV_pre(fi,2);
                if (i0 >= es.V_pre.rows() || i1 >= es.V_pre.rows() || i2 >= es.V_pre.rows()) continue;
                Vector3d p0 = es.V_pre.row(i0).transpose();
                Vector3d p1 = es.V_pre.row(i1).transpose();
                Vector3d p2 = es.V_pre.row(i2).transpose();
                if (!p0.allFinite() || !p1.allFinite() || !p2.allFinite()) continue;
                int base = (int)verts.size();
                for (const Vector3d & pt : {p0, p1, p2}) {
                    Vector3d rp = R * (pt - g.centroid);
                    verts.push_back({rp.x(), rp.y(), rp.z()});
                }
                faces.push_back({base, base+1, base+2});
            }
        }

        if (!faces.empty()) {
            MatrixXd nafV((int)verts.size(), 3);
            MatrixXi nafF((int)faces.size(), 3);
            for (int i = 0; i < (int)verts.size(); i++)
                nafV.row(i) << verts[i][0], verts[i][1], verts[i][2];
            for (int i = 0; i < (int)faces.size(); i++)
                nafF.row(i) << faces[i][0], faces[i][1], faces[i][2];
            polyscope::registerSurfaceMesh("non_active_sheet_faces", nafV, nafF)
                ->setSurfaceColor({0.55f, 0.3f, 0.85f})
                ->setEdgeWidth(1.5)
                ->setSmoothShade(false)
                ->setTransparency(0.45f)
                ->setEnabled(gShowCanonRing);
        }
    }

    // Double cover visualization (boundary cases): flat overlay on UV panels.
    if (gc.has_dc) {
        polyscope::registerSurfaceMesh("dc_uv_pre", gc.dc_uv_pre_3d, gSnap.FUV_dc_pre)
            ->setSurfaceColor({0.2f, 0.85f, 0.85f})   // teal — top+bottom sheet pre
            ->setEdgeWidth(1.0)->setSmoothShade(false)->setTransparency(0.45f)
            ->setEnabled(gShowCanonUV);
        polyscope::registerSurfaceMesh("dc_uv_post", gc.dc_uv_post_3d, gSnap.FUV_dc_post)
            ->setSurfaceColor({0.85f, 0.65f, 0.2f})   // amber — top+bottom sheet post
            ->setEdgeWidth(1.0)->setSmoothShade(false)->setTransparency(0.45f)
            ->setEnabled(gShowCanonUV);

        // DC vertex group point clouds — two sets:
        //   *_uv  : positioned in canonical UV space (dc_uv_pre_3d), enabled with gShowCanonUV
        //   *_ring: positioned in canonical 3D ring space (gc.V_ring),  enabled with gShowCanonRing
        if (gc.dc_uv_pre_3d.rows() > 0 && !gSnap.dc_B_glued.empty()) {
            int nVjoint = (int)gSnap.V_pre.rows() + 1;
            const auto & Bg  = gSnap.dc_B_glued;
            const auto & Brt = gSnap.dc_B_reflected;
            int nBref = (int)Brt.size();
            int local_vi = gSnap.b(0), local_vj = gSnap.b(1);

            // vi / vj — red
            {
                MatrixXd uv_pts(2, 3), ring_pts(2, 3);
                uv_pts.row(0)   = gc.dc_uv_pre_3d.row(local_vi);
                uv_pts.row(1)   = gc.dc_uv_pre_3d.row(local_vj);
                ring_pts.row(0) = gc.V_ring.row(local_vi);
                ring_pts.row(1) = gc.V_ring.row(local_vj);
                polyscope::registerPointCloud("dc_vi_vj_uv", uv_pts)
                    ->setPointColor({1.0f, 0.3f, 0.3f})->setPointRadius(0.016f, true)
                    ->setEnabled(gShowDCVertVi && gShowCanonUV);
                polyscope::registerPointCloud("dc_vi_vj_ring", ring_pts)
                    ->setPointColor({1.0f, 0.3f, 0.3f})->setPointRadius(0.016f, true)
                    ->setEnabled(gShowDCVertVi && gShowCanonRing);
            }

            // B_glued — green
            {
                MatrixXd uv_pts((int)Bg.size(), 3), ring_pts((int)Bg.size(), 3);
                for (int k = 0; k < (int)Bg.size(); k++) {
                    uv_pts.row(k)   = gc.dc_uv_pre_3d.row(Bg[k]);
                    ring_pts.row(k) = gc.V_ring.row(Bg[k]);
                }
                polyscope::registerPointCloud("dc_B_glued_uv", uv_pts)
                    ->setPointColor({0.2f, 1.0f, 0.3f})->setPointRadius(0.016f, true)
                    ->setEnabled(gShowDCVertBglued && gShowCanonUV);
                polyscope::registerPointCloud("dc_B_glued_ring", ring_pts)
                    ->setPointColor({0.2f, 1.0f, 0.3f})->setPointRadius(0.016f, true)
                    ->setEnabled(gShowDCVertBglued && gShowCanonRing);
            }

            if (nBref > 0) {
                // B_reflected top sheet — blue
                {
                    MatrixXd uv_pts(nBref, 3), ring_pts(nBref, 3);
                    for (int k = 0; k < nBref; k++) {
                        uv_pts.row(k)   = gc.dc_uv_pre_3d.row(Brt[k]);
                        ring_pts.row(k) = gc.V_ring.row(Brt[k]);
                    }
                    polyscope::registerPointCloud("dc_B_ref_top_uv", uv_pts)
                        ->setPointColor({0.3f, 0.5f, 1.0f})->setPointRadius(0.016f, true)
                        ->setEnabled(gShowDCBrefTop && gShowCanonUV);
                    polyscope::registerPointCloud("dc_B_ref_top_ring", ring_pts)
                        ->setPointColor({0.3f, 0.5f, 1.0f})->setPointRadius(0.016f, true)
                        ->setEnabled(gShowDCBrefTop && gShowCanonRing);
                }

                // B_reflected bottom sheet — gold (UV only; ring shares same 3D pos as top)
                {
                    MatrixXd brb_pts(nBref, 3);
                    for (int k = 0; k < nBref; k++) {
                        int bot_idx = nVjoint + k;
                        if (bot_idx < (int)gc.dc_uv_pre_3d.rows())
                            brb_pts.row(k) = gc.dc_uv_pre_3d.row(bot_idx);
                        else
                            brb_pts.row(k).setZero();
                    }
                    polyscope::registerPointCloud("dc_B_ref_bot_uv", brb_pts)
                        ->setPointColor({1.0f, 0.8f, 0.1f})->setPointRadius(0.016f, true)
                        ->setEnabled(gShowDCBrefBot && gShowCanonUV);
                }
            }
        }
    }

    sample_tracker_show_canonical(gc.uv_pre_3d, gc.uv_post_3d, gSnap.FUV_pre, gSnap.FUV_post);
}

// ---- face flip tracker visualization ----
static void face_flip_tracker_show_viz()
{
    if (!face_flip_tracker_enabled()) return;

    // Colors per vertex: red, green, blue
    static const float kColors[3][3] = {
        {0.9f, 0.2f, 0.2f},
        {0.2f, 0.85f, 0.3f},
        {0.3f, 0.5f, 1.0f}
    };
    static const char* kNames[3] = {
        "fft_traj_0", "fft_traj_1", "fft_traj_2"
    };

    // Current triangle — green if ok, red if flipped
    {
        MatrixXd triV(3, 3);
        for (int i = 0; i < 3; i++)
            triV.row(i) = face_flip_tracker_cur_pos(i).transpose();
        MatrixXi triF(1, 3); triF << 0, 1, 2;
        bool flipped = face_flip_tracker_flip_detected();
        auto* triMesh = polyscope::registerSurfaceMesh("fft_triangle", triV, triF);
        if (flipped) triMesh->setSurfaceColor({0.95f, 0.2f, 0.15f});
        else         triMesh->setSurfaceColor({0.2f,  0.9f, 0.35f});
        triMesh
            ->setEdgeWidth(2.0)
            ->setSmoothShade(false)
            ->setTransparency(0.35f);
    }

    // Trajectory curve + current position for each of the 3 tracked vertices
    for (int i = 0; i < 3; i++) {
        const auto& traj = face_flip_tracker_traj(i);
        if (traj.size() < 2) {
            // Just a point cloud for the seed position
            if (!traj.empty()) {
                MatrixXd pt(1, 3); pt.row(0) = traj[0].transpose();
                polyscope::registerPointCloud(kNames[i], pt)
                    ->setPointColor({kColors[i][0], kColors[i][1], kColors[i][2]})
                    ->setPointRadius(0.006, true);
            }
            continue;
        }

        int nPts = (int)traj.size();
        MatrixXd nodes(nPts, 3);
        for (int k = 0; k < nPts; k++) nodes.row(k) = traj[k].transpose();

        MatrixXi edges(nPts - 1, 2);
        for (int k = 0; k < nPts - 1; k++) edges.row(k) << k, k + 1;

        polyscope::registerCurveNetwork(kNames[i], nodes, edges)
            ->setColor({kColors[i][0], kColors[i][1], kColors[i][2]})
            ->setRadius(0.003, true);
    }
}

// ---- update polyscope display ----
void update_display()
{
    // Main mesh (only in normal view)
    if (!gCanonicalView) {
        MatrixXi Flive = live_faces();
        MatrixXd Vd, Vc; MatrixXi Fc; VectorXi I1, I2;
        Vd = safe_V();
        igl::remove_unreferenced(Vd, Flive, Vc, Fc, I1, I2);

        // Update mat_struct_ids point cloud to reflect current live vertices.
        if (!gVertexStructIDs.empty()) {
            const int nS = (int)gVertexStructIDs.size();
            const int nLive = (int)Vc.rows();
            Eigen::VectorXd structScalar(nLive);
            for (int i = 0; i < nLive; ++i) {
                int orig = I2(i);
                uint32_t h = 2166136261u;
                if (orig < nS) {
                    for (int id : gVertexStructIDs[orig]) {
                        h ^= (uint32_t)id;
                        h *= 16777619u;
                    }
                }
                structScalar(i) = (double)h;
            }
            auto* pc = polyscope::registerPointCloud("mat_struct_ids", Vc);
            pc->setPointRadius(0.004, true);
            pc->addScalarQuantity("struct_hash", structScalar)->setEnabled(true);
        }

        auto * m = polyscope::registerSurfaceMesh("mesh", Vc, Fc);
        m->setSurfaceColor({0.75f, 0.75f, 0.75f});
        m->setEdgeWidth(0.5);

        // Pre-collapse ghost mesh
        if (gHasPreMesh) {
            auto * mp = polyscope::registerSurfaceMesh("mesh_pre", gPreV, gPreF);
            mp->setSurfaceColor({0.4f, 0.9f, 0.4f})->setEdgeWidth(0.5)
              ->setTransparency(0.5f)->setEnabled(gShowMeshPre);
        }

        // Original fine mesh with face orientation colors:
        //   green = face was already CCW (kept), red = face was CW (re-wound)
        if (gFO.rows() > 0 && gFaceFlipped.size() == gFO.rows()) {
            int nF = gFO.rows();
            MatrixXd fc(nF, 3);
            for (int f = 0; f < nF; f++)
                fc.row(f) = gFaceFlipped(f)
                    ? RowVector3d(1.0, 0.2, 0.2)   // red  = was CW
                    : RowVector3d(0.2, 0.85, 0.3);  // green = already CCW
            auto * om = polyscope::registerSurfaceMesh("orig_mesh_orient", gVO, gFO);
            om->setEdgeWidth(0.5)->setEnabled(gShowOrigOrient);
            om->addFaceColorQuantity("cw_ccw", fc)->setEnabled(true);
        }

        update_sheet_display();
        update_seam_onering_display();
    }

    sample_tracker_show();
    sample_tracker_show_vertices();
    face_flip_tracker_show_viz();

    if (!gSnap.valid) return;

    if (gCanonicalView) {
        show_canonical_view();
    } else {
        DisplayGeometry g = compute_ring_geometry();
        register_ring_geometry(g);
    }
}

// ---- ImGui callback ----
void ui_callback()
{
    sheet_seam_pick_check();
    coarse_fine_pick_check();

    // Drive continuous decimation
    if (gRunning && !gFinished) {
        snapshot_pre_mesh();
        if (do_next_step()) {
            refresh_snap();
            update_display();
            ring_post_c2f_diagnostic();

            bool hitSeam = !gDecInfo.empty() && gDecInfo.back().sheets.size() > 1;
            if (gRunToSeam && hitSeam) {
                gRunToSeam = false;
                gRunning   = false;
            }

            if (!gDecInfo.empty()) {
                const auto & lc = gDecInfo.back().lscm_case;
                bool is_bd_case = lc.has_value() && lc.value() >= 1;
                if (is_bd_case) {
#ifdef SSP_LSCM_LOG
                    fprintf(stderr, "[joint_lscm] case %d triggered (%s)\n",
                        lc.value() + 1,
                        lc.value() == 1 ? "one endpoint on boundary"
                                        : "both endpoints on boundary");
#endif
                    if (gRunToBdCase) {
                        gRunToBdCase = false;
                        gRunning     = false;
                    }
                }
            }
            // Stop at DC: fires only when the successful collapse actually used the
            // double cover solve (true geometric boundary, not seam-injected).
            if (gStopAtDC && gSnap.has_dc) {
                gRunning       = false;
                gCanonicalView = true;
            }

            if (gBreakAtCollapse > 0 && gCollapseCount >= gBreakAtCollapse) {
                gRunning         = false;
                gBreakAtCollapse = -1;
            }

            if (!gRunToSeam && !gRunToBdCase && gRunning)
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(static_cast<int>(gStepDelayMs)));
        }
        if (gFinished)
            gRunning = false;

        // Stop immediately if the face flip tracker detected a flip this step
        if (face_flip_tracker_flip_detected())
            gRunning = false;
        // Stop if the vertex watch tracker triggered this step
        if (vertex_watch_triggered())
            gRunning = false;
    }

    ImGui::SetNextWindowSize({340, 420}, ImGuiCond_FirstUseEver);
    ImGui::Begin("SSP Collapse Visualizer");

    ImGui::Text("Collapses: %d  [%s]", gCollapseCount,
        gDecType == 0 ? "midpoint" : gDecType == 1 ? "qslim" : "meshlab");
    if (gSnap.valid)
        ImGui::Text("Edge: vi=%d  vj=%d", gSnap.vi, gSnap.vj);
    else
        ImGui::Text("(no collapse yet)");

    if (gFinished) {
        ImGui::TextColored({0.4f,1.0f,0.4f,1.0f}, "Reached target faces.");
        gRunning = false;
    } else if (gRunning) {
        if (ImGui::Button("Stop  [Space]") ||
            ImGui::IsKeyPressed(ImGuiKey_Space, /*repeat=*/false))
        {
            gRunning = false;
        }
    } else {
        if (ImGui::Button("Next  [Space]") ||
            ImGui::IsKeyPressed(ImGuiKey_Space, /*repeat=*/false))
        {
            snapshot_pre_mesh();
            if (do_next_step()) { refresh_snap(); update_display(); ring_post_c2f_diagnostic(); }
        }
        ImGui::SameLine();
        if (ImGui::Button("Run"))            { gRunning = true; }
        ImGui::SameLine();
        if (ImGui::Button("Next seam"))      { gRunToSeam   = true; gRunning = true; }
        ImGui::SameLine();
        if (ImGui::Button("Next bd case"))   { gRunToBdCase = true; gRunning = true; }
        ImGui::SameLine();
        ImGui::Checkbox("Stop at DC", &gStopAtDC);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Halt + switch to canonical view on every double cover (boundary) case");

        ImGui::SetNextItemWidth(80);
        ImGui::InputText("##breakat", gBreakAtBuf, sizeof(gBreakAtBuf),
                         ImGuiInputTextFlags_CharsDecimal);
        ImGui::SameLine();
        if (ImGui::Button("Run to #")) {
            int n = std::atoi(gBreakAtBuf);
            if (n > gCollapseCount) { gBreakAtCollapse = n; gRunning = true; }
        }
        if (gBreakAtCollapse > 0)
            ImGui::SameLine(), ImGui::TextColored({1.0f,0.8f,0.2f,1.0f},
                "(waiting for #%d)", gBreakAtCollapse);
    }

    // Canonical / normal view toggle
    if (gSnap.valid) {
        ImGui::Separator();
        if (!gCanonicalView) {
            if (ImGui::Button("Canonical View")) {
                gCanonicalView = true;
                show_canonical_view();
                polyscope::view::resetCameraToHomeView();
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(one-ring on floor, UV above)");
        } else {
            if (ImGui::Button("Back to Main View")) {
                gCanonicalView = false;
                polyscope::removeAllStructures();
                update_display();
            }
        }
    }

    ImGui::Separator();
    ImGui::TextDisabled("Blue   = pre-collapse");
    ImGui::TextDisabled("Orange = post-collapse");
    ImGui::TextDisabled("Red line = collapsed edge");

    ImGui::Separator();
    ImGui::Text("Step delay (ms):");
    ImGui::SliderFloat("##delay", &gStepDelayMs, 0.0f, 2000.0f);

    ImGui::Separator();
    bool redraw = false;
    ImGui::Text("UV offset (x ring size):");
    redraw |= ImGui::SliderFloat("##uvoffset", &gUVOffset, -2.0f, 2.0f);
    ImGui::Text("Post ring height (x ring size):");
    redraw |= ImGui::SliderFloat("##postheight", &gPostHeight, 0.0f, 3.0f);
    ImGui::Text("Scale:");
    redraw |= ImGui::SliderFloat("##scale", &gRingScale, 0.1f, 5.0f);
    if (redraw && gSnap.valid) update_display();

    ImGui::Separator();
    ImGui::Text("Visibility:");
    bool vis = false;
    if (gCanonicalView) {
        vis |= ImGui::Checkbox("UV", &gShowCanonUV);
        vis |= ImGui::Checkbox("One ring (+ non-active sheets)", &gShowCanonRing);
        if (gSnap.has_dc) {
            ImGui::Separator();
            ImGui::Text("DC vertex groups:");
            // vi / vj row
            vis |= ImGui::Checkbox("vi / vj", &gShowDCVertVi);
            ImGui::SameLine(); ImGui::TextColored({1.0f, 0.3f, 0.3f, 1.0f}, "[red]");
            // B_glued row
            vis |= ImGui::Checkbox("B0 / Bn  (arc endpoints, glued)", &gShowDCVertBglued);
            ImGui::SameLine(); ImGui::TextColored({0.2f, 1.0f, 0.3f, 1.0f}, "[green]");
            // B_reflected top row
            vis |= ImGui::Checkbox("B_mid  top sheet", &gShowDCBrefTop);
            ImGui::SameLine(); ImGui::TextColored({0.3f, 0.5f, 1.0f, 1.0f}, "[blue]");
            // B_reflected bot row
            vis |= ImGui::Checkbox("B_mid  bottom sheet", &gShowDCBrefBot);
            ImGui::SameLine(); ImGui::TextColored({1.0f, 0.8f, 0.1f, 1.0f}, "[gold]");
        }
        ImGui::Separator();
        if (ImGui::Button("Print DC / LSCM info")) print_dc_snap();
    } else {
        vis |= ImGui::Checkbox("Ring pre",        &gShowRingPre);   ImGui::SameLine();
        vis |= ImGui::Checkbox("Ring post",       &gShowRingPost);
        vis |= ImGui::Checkbox("UV pre",          &gShowUVPre);     ImGui::SameLine();
        vis |= ImGui::Checkbox("UV post",         &gShowUVPost);
        vis |= ImGui::Checkbox("Arrows pre",      &gShowArrowPre);  ImGui::SameLine();
        vis |= ImGui::Checkbox("Arrows post",     &gShowArrowPost);
        vis |= ImGui::Checkbox("Pre→Post corr",   &gShowCorrArrows); ImGui::SameLine();
        vis |= ImGui::Checkbox("Corr pts",        &gShowCorrPts);
        vis |= ImGui::Checkbox("Collapsed edges", &gShowCollapsedEdge);
        if (gHasPreMesh)
            vis |= ImGui::Checkbox("Mesh before collapse", &gShowMeshPre);
        vis |= ImGui::Checkbox("Fine mesh orient (red=CW, green=CCW)", &gShowOrigOrient);
    }
    if (vis && gSnap.valid) update_display();

    ImGui::Separator();
    sheet_seam_imgui_section();

    ImGui::Separator();
    coarse_fine_imgui_section();

    ImGui::Separator();
    ImGui::Text("Export (latest_snapshot.obj):");
    ImGui::SetNextItemWidth(220);
    ImGui::InputText("##exportdir", gExportDir, sizeof(gExportDir));
    ImGui::SameLine();
    if (ImGui::Button("Save now")) export_current_mesh();
    ImGui::TextDisabled("Auto-saved before every collapse.");

    if (face_flip_tracker_enabled()) {
        ImGui::Separator();
        ImGui::Text("Face Flip Tracker  (face %d)", face_flip_tracker_face_idx());
        if (face_flip_tracker_flip_detected()) {
            ImGui::TextColored({1.0f, 0.2f, 0.2f, 1.0f},
                "*** FLIP at collapse #%d ***", face_flip_tracker_flip_at_collapse());
        } else {
            ImGui::TextColored({0.3f, 1.0f, 0.4f, 1.0f}, "OK — no flip yet");
        }
        ImGui::TextDisabled("v0 cur: (%.3f, %.3f, %.3f)",
            face_flip_tracker_cur_pos(0).x(),
            face_flip_tracker_cur_pos(0).y(),
            face_flip_tracker_cur_pos(0).z());
        ImGui::TextDisabled("v1 cur: (%.3f, %.3f, %.3f)",
            face_flip_tracker_cur_pos(1).x(),
            face_flip_tracker_cur_pos(1).y(),
            face_flip_tracker_cur_pos(1).z());
        ImGui::TextDisabled("v2 cur: (%.3f, %.3f, %.3f)",
            face_flip_tracker_cur_pos(2).x(),
            face_flip_tracker_cur_pos(2).y(),
            face_flip_tracker_cur_pos(2).z());
    }

    // ---- Vertex Watch Tracker UI ----
    {
        ImGui::Separator();
        ImGui::Text("Vertex Watch Tracker");

        static char gVWTInputBuf[16] = "";
        ImGui::SetNextItemWidth(100);
        ImGui::InputText("Fine vtx ID", gVWTInputBuf, sizeof(gVWTInputBuf),
                         ImGuiInputTextFlags_CharsDecimal);
        ImGui::SameLine();
        if (ImGui::Button("Watch")) {
            int vid = atoi(gVWTInputBuf);
            vertex_watch_set(vid);
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            vertex_watch_clear();
            gVWTInputBuf[0] = '\0';
        }

        if (vertex_watch_active()) {
            Eigen::Vector3i bf = vertex_watch_cur_BF();
            ImGui::TextDisabled("vtx %d  coarse face: (%d, %d, %d)",
                vertex_watch_fine_vtx(), bf(0), bf(1), bf(2));
            if (vertex_watch_triggered()) {
                ImGui::TextColored({1.0f, 0.3f, 0.1f, 1.0f},
                    "*** HIT: fine vtx %d in one-ring of collapse #%d ***",
                    vertex_watch_fine_vtx(), vertex_watch_trigger_at_collapse());
            } else {
                ImGui::TextColored({0.3f, 1.0f, 0.4f, 1.0f}, "Watching — no hit yet");
            }
        } else {
            ImGui::TextDisabled("(inactive)");
        }
    }

    ImGui::End();
}
