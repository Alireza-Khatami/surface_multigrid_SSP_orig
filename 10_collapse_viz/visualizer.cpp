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
#include <igl/writePLY.h>
#include <ctime>

#include <single_collapse_data.h>
#include <SSP_collapse_edge.h>
#include <compute_barycentric.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <set>
#include <map>
#include <limits>
#include <cmath>
#include <thread>
#include <chrono>
#include <string>
#include <cstdarg>

using namespace Eigen;

// ---- output path globals defined in main.cpp ----
extern std::string gStem;
extern std::string gOutDir;

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
extern std::vector<std::vector<int>> gStaleChains;

bool do_next_step();  // defined in main.cpp

// ---- display-only state ----
static float gUVOffset          = 1.5f;
static float gRingScale         = 1.0f;
static float gStepDelayMs       = 0.0f;
static bool  gRunning           = false;
static bool  gRunToSeam         = false;
static bool  gRunToBdCase       = false;  // stop at LSCM Case 1 or 2 (one/both endpoints on boundary)
static bool  gStopAtDC         = false;  // persistent: stop + enter canonical view on every double cover case
static bool  gStopAtDCAsym    = false;  // persistent: stop when a DC Post UV is not symmetric (sym=0)
static bool  gStopAtDCFail   = false;  // persistent: stop when DC solve itself failed (sym=-1, NaN UV)
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
static bool  gShowCase1PinViz   = true;   // Case 1: vertex role colors, arc curve, pin markers
// DC vertex group toggles
static bool  gShowBDVtx         = true;   // boundary vertex highlight (Case 1/2, magenta)
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

// Canonical-mesh export panel: 4 checkboxes + cached geometry
static bool gExportUVPre    = true;
static bool gExportUVPost   = true;
static bool gExportRingPre  = true;
static bool gExportRingPost = true;

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
    std::optional<int> lscm_case;     // 0=both interior, 1=one on bd, 2=both on bd
    // Post-UV symmetry result for this sheet's DC solve (only valid when has_dc==true).
    // +1 = symmetric,  0 = asymmetric (DC OK but UV not mirrored),  -1 = DC failed (NaN UV).
    int    dc_uv_symmetric    = 1;
    double dc_uv_asym_max_err = 0.0;
} gSnap;

// Per-sheet browsing: all sheets from the most recent collapse, and which is active.
static std::vector<SheetData> gAllSheets;
static int gActiveSheetIdx = 0;

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

static void apply_sheet_to_snap(const SheetData & sd)
{
    gSnap.valid    = true;
    gSnap.b        = sd.b;
    gSnap.vi       = sd.subsetVIdx(sd.b(0));
    gSnap.vj       = sd.subsetVIdx(sd.b(1));
    // Use this sheet's own 3D geometry — each sheet has its own local vertex set.
    // Without this, V_pre and UV_pre row counts diverge → OOB crash.
    gSnap.V_pre    = sd.V_pre;
    gSnap.V_post   = sd.V_post;
    gSnap.FUV_pre  = sd.FUV_pre;
    gSnap.FUV_post = sd.FUV_post;
    gSnap.UV_pre   = sd.UV_pre;
    gSnap.UV_post  = sd.UV_post;
    gSnap.has_dc          = sd.has_double_cover;
    gSnap.FUV_dc_pre      = sd.FUV_dc_pre;
    gSnap.FUV_dc_post     = sd.FUV_dc_post;
    gSnap.UV_dc_pre       = sd.UV_dc_pre;
    gSnap.UV_dc_post      = sd.UV_dc_post;
    gSnap.dc_B_glued          = sd.dc_B_glued;
    gSnap.dc_B_reflected      = sd.dc_B_reflected;
    gSnap.dc_uv_symmetric     = sd.dc_uv_symmetric;
    gSnap.dc_uv_asym_max_err  = sd.dc_uv_asym_max_err;
}

static void refresh_snap()
{
    if (gDecInfo.empty()) return;
    const single_collapse_data & d = gDecInfo.back();
    gSnap.V_pre  = d.V_pre;
    gSnap.V_post = d.V_post;
    // Only keep active-sheet entries.  Non-active-sheet face stubs are single
    // triangles (FUV_pre.rows() == 1, b(0) == 3 → OOB into 3-row V_pre → crash).
    // Active sheets always have ≥ 3 faces (the collapse loop rejects ≤ 2).
    gAllSheets.clear();
    for (const SheetData & sd : d.sheets)
        if (sd.FUV_pre.rows() >= 3)
            gAllSheets.push_back(sd);
    gActiveSheetIdx = 0;
    if (!gAllSheets.empty())
        apply_sheet_to_snap(gAllSheets[0]);
    gSnap.lscm_case = d.lscm_case;
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

static DisplayGeometry gLastCanonGeom;   // saved from last show_canonical_view()
static bool            gHasCanonGeom = false;

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

    // ---- Case 1 UV pin visualization ----------------------------------------
    // Only meaningful when lscm_case==1 and BDSnap is valid.
    // (1) Vertex role colors on uv_pre surface mesh
    // (2) Boundary arc curve network (one-ring boundary edges in UV)
    // (3) Pin marker point clouds: vi pin (green) and vj pin (blue)
    // -------------------------------------------------------------------------
    if (gSnap.lscm_case.has_value() && gSnap.lscm_case.value() == 1 &&
        gSnap.UV_pre.rows() > 0 && gSnap.b.size() >= 2 &&
        gSnap.FUV_pre.rows() > 0)
    {
        const BDSnap & bds = SSP_get_bd_snap();
        bool bds_ok  = bds.valid;
        int  bd_local = -1;
        bool bd_is_vi = false;
        if (bds_ok) {
            bd_is_vi = bds.injected_ndv;
            bd_local = bd_is_vi ? (int)gSnap.b(0) : (int)gSnap.b(1);
        }
        int local_vi = (int)gSnap.b(0);
        int local_vj = (int)gSnap.b(1);
        int nV = (int)gSnap.UV_pre.rows();

        // -- edge valence for ring_bd detection (same logic as build_dc_snap_str) --
        std::map<std::pair<int,int>, int> edge_cnt;
        for (int f = 0; f < gSnap.FUV_pre.rows(); f++)
            for (int e = 0; e < 3; e++) {
                int a = gSnap.FUV_pre(f,e), b2 = gSnap.FUV_pre(f,(e+1)%3);
                if (a > b2) std::swap(a, b2);
                edge_cnt[{a,b2}]++;
            }
        std::set<int> ring_bd;
        for (auto& kv : edge_cnt)
            if (kv.second == 1) { ring_bd.insert(kv.first.first); ring_bd.insert(kv.first.second); }

        // U value of bd_vtx for HIGHER/LOWER comparison
        double bd_U = (bds_ok && bd_local >= 0 && bd_local < nV)
                      ? gSnap.UV_pre(bd_local, 1) : 0.0;  // col1 = U

        // (1) Vertex role colors: build per-vertex RGB for uv_pre
        {
            // Colors: bd_vtx=red, other_pin=green, ring_bd HIGHER=orange, LOWER=purple, SAME=yellow, interior=gray
            Eigen::MatrixXd vColors(nV, 3);
            for (int v = 0; v < nV; v++) {
                if (bds_ok && v == bd_local) {
                    vColors.row(v) << 1.0, 0.08, 0.08;  // red — boundary vertex (pinned)
                } else if (v == local_vi || v == local_vj) {
                    // other collapse endpoint (non-boundary pin)
                    vColors.row(v) << 0.1, 0.85, 0.2;   // green — other pin
                } else if (ring_bd.count(v)) {
                    double v_U  = gSnap.UV_pre(v, 1);
                    double diff = v_U - bd_U;
                    if (diff >  1e-7)
                        vColors.row(v) << 1.0, 0.55, 0.1;   // orange — HIGHER
                    else if (diff < -1e-7)
                        vColors.row(v) << 0.65, 0.1, 0.9;   // purple — LOWER
                    else
                        vColors.row(v) << 0.95, 0.9, 0.1;   // yellow — SAME
                } else {
                    vColors.row(v) << 0.55, 0.55, 0.55;     // gray — interior
                }
            }
            auto * up = polyscope::getSurfaceMesh("uv_pre");
            auto * cq = up->addVertexColorQuantity("case1_pin_roles", vColors);
            cq->setEnabled(gShowCase1PinViz);
        }

        // (2) Boundary arc curve network — boundary edges of FUV_pre in UV space
        {
            std::vector<std::array<double,3>> arcV;
            std::vector<std::array<int,2>>    arcE;
            std::map<int,int> vmap;  // local UV vertex → arcV index
            auto get_or_add = [&](int v) -> int {
                auto it = vmap.find(v);
                if (it != vmap.end()) return it->second;
                int idx = (int)arcV.size();
                vmap[v] = idx;
                Eigen::Vector3d p = g.uv_pre_3d.row(v);
                arcV.push_back({p.x(), p.y(), p.z()});
                return idx;
            };
            for (auto& kv : edge_cnt) {
                if (kv.second != 1) continue;  // boundary edges only
                int a = kv.first.first, b2 = kv.first.second;
                arcE.push_back({get_or_add(a), get_or_add(b2)});
            }
            if (!arcV.empty()) {
                Eigen::MatrixXd aV(arcV.size(), 3);
                Eigen::MatrixXi aE(arcE.size(), 2);
                for (int i = 0; i < (int)arcV.size(); i++) aV.row(i) << arcV[i][0], arcV[i][1], arcV[i][2];
                for (int i = 0; i < (int)arcE.size(); i++) aE.row(i) << arcE[i][0], arcE[i][1];
                polyscope::registerCurveNetwork("uv_case1_arc", aV, aE)
                    ->setRadius(edge_radius * 1.8)->setColor({0.0f, 0.85f, 1.0f})
                    ->setEnabled(false);
            } else {
                // Register a degenerate network so get calls in update_display don't crash
                Eigen::MatrixXd dV(1,3); dV.setZero();
                Eigen::MatrixXi dE(0,2);
                polyscope::registerCurveNetwork("uv_case1_arc", dV, dE)->setEnabled(false);
            }
        }

        // (3) Pin marker point clouds: vi_pin and vj_pin in UV space
        {
            // vi pin point
            if (local_vi >= 0 && local_vi < nV) {
                Eigen::MatrixXd viPt(1, 3);
                viPt.row(0) = g.uv_pre_3d.row(local_vi);
                polyscope::registerPointCloud("uv_case1_vi_pin", viPt)
                    ->setPointColor({0.1f, 0.85f, 0.2f})
                    ->setPointRadius(0.03, true)
                    ->setEnabled(false);
            }
            // vj pin point
            if (local_vj >= 0 && local_vj < nV) {
                Eigen::MatrixXd vjPt(1, 3);
                vjPt.row(0) = g.uv_pre_3d.row(local_vj);
                polyscope::registerPointCloud("uv_case1_vj_pin", vjPt)
                    ->setPointColor({0.2f, 0.4f, 1.0f})
                    ->setPointRadius(0.03, true)
                    ->setEnabled(false);
            }
            // bd_vtx pin point (extra large, red)
            if (bds_ok && bd_local >= 0 && bd_local < nV) {
                Eigen::MatrixXd bdPt(1, 3);
                bdPt.row(0) = g.uv_pre_3d.row(bd_local);
                polyscope::registerPointCloud("uv_case1_bd_pin", bdPt)
                    ->setPointColor({1.0f, 0.08f, 0.08f})
                    ->setPointRadius(0.03, true)
                    ->setEnabled(false);
            } else {
                Eigen::MatrixXd dummy(0,3);
                polyscope::registerPointCloud("uv_case1_bd_pin", dummy)->setEnabled(false);
            }
        }
    } else {
        // Not Case 1 or missing data — register empty objects so update_display can always find them
        Eigen::MatrixXd dV(1,3); dV.setZero();
        Eigen::MatrixXi dE(0,2);
        polyscope::registerCurveNetwork("uv_case1_arc", dV, dE)->setEnabled(false);
        Eigen::MatrixXd dummy(0,3);
        polyscope::registerPointCloud("uv_case1_vi_pin", dummy)->setEnabled(false);
        polyscope::registerPointCloud("uv_case1_vj_pin", dummy)->setEnabled(false);
        polyscope::registerPointCloud("uv_case1_bd_pin", dummy)->setEnabled(false);
    }

}

// ---- canonical mesh export ----
static void export_canonical_meshes()
{
    if (!gHasCanonGeom || !gSnap.valid) {
        fprintf(stderr, "[EXPORT] no canonical geometry — enter canonical view first\n");
        return;
    }
    time_t t = time(nullptr);
    char dt[32];
    strftime(dt, sizeof(dt), "%Y%m%d_%H%M%S", localtime(&t));

    std::string base = gOutDir + "collapse_"
                     + std::to_string(gCollapseCount) + "_";

    auto save_ply = [&](const char * tag, const MatrixXd & V, const MatrixXi & F) {
        std::string path = base + tag + "_" + dt + ".ply";
        if (!igl::writePLY(path, V, F))
            fprintf(stderr, "[EXPORT] FAILED: %s\n", path.c_str());
        else
            fprintf(stderr, "[EXPORT] wrote %s  (%d verts, %d faces)\n",
                    path.c_str(), (int)V.rows(), (int)F.rows());
    };

    if (gExportUVPre)    save_ply("UV_Pre",       gLastCanonGeom.uv_pre_3d,   gSnap.FUV_pre);
    if (gExportUVPost)   save_ply("UV_Post",      gLastCanonGeom.uv_post_3d,  gSnap.FUV_post);
    if (gExportRingPre)  save_ply("OneRing_Pre",  gLastCanonGeom.V_ring,      gSnap.FUV_pre);
    if (gExportRingPost) save_ply("OneRing_Post", gLastCanonGeom.V_ring_post, gSnap.FUV_post);
}

// ---- canonical view ----

// Helper: append a formatted line to a std::string.
static void snap_appendf(std::string& out, const char* fmt, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    out += buf;
}

// Build the DC / Joint-LSCM diagnostic as a string (shown inline and printed to stdout).
static std::string build_dc_snap_str()
{
    const auto& s = gSnap;
    if (!s.valid) return "[no valid snap]\n";

    std::string out;
    snap_appendf(out, "vi=%d  vj=%d  local_vi=%d  local_vj=%d\n",
                 s.vi, s.vj,
                 s.b.size() > 0 ? s.b(0) : -1,
                 s.b.size() > 1 ? s.b(1) : -1);
    if (s.lscm_case.has_value()) {
        const char* case_desc[] = {"both interior", "one on boundary", "both on boundary"};
        int c = s.lscm_case.value();
        snap_appendf(out, "lscm_case=%d  (%s)\n", c,
                     (c >= 0 && c <= 2) ? case_desc[c] : "unknown");
    } else {
        out += "lscm_case=n/a\n";
    }

    // Case 1 UV pin confirmation: the boundary vertex is always pinned to an inner UV
    // position (vi->0,0 or vj->1,0) regardless of which is topologically on the boundary.
    if (s.lscm_case.has_value() && s.lscm_case.value() == 1 &&
        s.UV_pre.rows() > 0 && s.b.size() >= 2)
    {
        const BDSnap & bds = SSP_get_bd_snap();
        if (bds.valid) {
            bool bd_is_vi = bds.injected_ndv;
            int  bd_local = bd_is_vi ? (int)s.b(0) : (int)s.b(1);
            // UV stored as col(0)=V, col(1)=U due to column-swap in reshape.
            // vi pin: (0,0).  vj pin: (0,1).
            double exp_col0 = 0.0;
            double exp_col1 = bd_is_vi ? 0.0 : 1.0;
            if (bd_local >= 0 && bd_local < (int)s.UV_pre.rows() &&
                s.FUV_pre.rows() > 0)
            {
                // col(0)=V, col(1)=U due to column-swap in reshape.
                // Read as U=col1, V=col0 and display in (U,V) order throughout.
                double act_U = s.UV_pre(bd_local, 1);
                double act_V = s.UV_pre(bd_local, 0);
                double exp_U = exp_col1;   // vi->0  vj->1
                double exp_V = exp_col0;   // always 0
                bool ok = std::fabs(act_U - exp_U) < 1e-4 &&
                          std::fabs(act_V - exp_V) < 1e-4;

                out += "\n[CASE1_BD_PINNED_TO_CENTER]\n";
                snap_appendf(out, "  boundary_vertex=%s  local=%d\n",
                             bd_is_vi ? "vi" : "vj", bd_local);
                snap_appendf(out, "  expected_UV=(U=%.4f, V=%.4f)  actual_UV=(U=%.6f, V=%.6f)  -> %s\n",
                             exp_U, exp_V, act_U, act_V,
                             ok ? "CONFIRMED" : "MISMATCH!");
                out += "  raw storage: col(0)=V  col(1)=U  (columns swapped in reshape)\n";
                out += "  pin order: vi->(U=0,V=0)  vj->(U=1,V=0)  by local index, not topology\n";
                out += "  root cause of UV orientation swap between Case 1 collapses\n";

                // Find all vertices on the actual 3D boundary of the one-ring
                // by counting edge valence in FUV_pre (valence==1 → boundary edge).
                std::map<std::pair<int,int>, int> edge_cnt;
                for (int f = 0; f < s.FUV_pre.rows(); f++) {
                    for (int e = 0; e < 3; e++) {
                        int a = s.FUV_pre(f, e), b = s.FUV_pre(f, (e+1)%3);
                        if (a > b) std::swap(a, b);
                        edge_cnt[{a, b}]++;
                    }
                }
                std::set<int> ring_bd;
                for (auto& kv : edge_cnt)
                    if (kv.second == 1) {
                        ring_bd.insert(kv.first.first);
                        ring_bd.insert(kv.first.second);
                    }

                // Compare other one-ring boundary vertices' UV to detected bd vertex.
                // All values shown as (U, V) = (col1, col0) after swap.
                out += "\n  -- one-ring boundary verts (FUV_pre edge valence==1) vs detected bd UV --\n";
                out += "     format:  local=N  UV=(U, V)  U_vs_bd_vtx: HIGHER/LOWER/SAME\n";
                bool any_other = false;
                bool has_higher = false, has_lower = false;
                for (int v : ring_bd) {
                    if (v == bd_local) continue;
                    any_other = true;
                    if (v < 0 || v >= (int)s.UV_pre.rows()) {
                        snap_appendf(out, "     local=%d  UV=<out of range>\n", v);
                        continue;
                    }
                    double v_U   = s.UV_pre(v, 1);   // col1=U
                    double v_V   = s.UV_pre(v, 0);   // col0=V
                    double diffU = v_U - act_U;
                    if (diffU >  1e-7) has_higher = true;
                    if (diffU < -1e-7) has_lower  = true;
                    const char* rel = std::fabs(diffU) < 1e-7 ? "SAME"
                                    : diffU > 0.0             ? "HIGHER"
                                                               : "LOWER";
                    snap_appendf(out,
                        "     local=%d  UV=(U=%.6f, V=%.6f)  U_vs_bd_vtx: %s (%+.6f)\n",
                        v, v_U, v_V, rel, diffU);
                }
                if (!any_other)
                    out += "     (no other boundary vertices found in one-ring)\n";

                // Arc position conclusion — this is the actual inner-pin check.
                if (any_other) {
                    if (has_higher && has_lower)
                        out += "\n  [ARC_POSITION] MIXED — pin is INTERIOR of boundary arc"
                               " (arc verts on both sides → genuinely inner pin)\n";
                    else if (has_higher)
                        out += "\n  [ARC_POSITION] ALL_HIGHER — pin is at LOW end of arc"
                               " (arc extends above pin only)\n";
                    else if (has_lower)
                        out += "\n  [ARC_POSITION] ALL_LOWER — pin is at HIGH end of arc"
                               " (arc extends below pin only)\n";
                    else
                        out += "\n  [ARC_POSITION] ALL_SAME — all arc verts at same U as pin\n";
                }

                // Also note where the two collapse endpoints sit relative to ring_bd.
                bool vi_in_ring = ring_bd.count((int)s.b(0)) > 0;
                bool vj_in_ring = ring_bd.count((int)s.b(1)) > 0;
                snap_appendf(out,
                    "\n  vi(local=%d) in ring_bd=%d  vj(local=%d) in ring_bd=%d\n",
                    (int)s.b(0), vi_in_ring,
                    (int)s.b(1), vj_in_ring);
            }
        }
    }

    snap_appendf(out, "has_dc=%d\n", (int)s.has_dc);
    snap_appendf(out, "V_pre: %dx3   V_post: %dx3\n",
                 (int)s.V_pre.rows(), (int)s.V_post.rows());

    snap_appendf(out, "\n-- FUV_pre (%dx3) --\n", (int)s.FUV_pre.rows());
    for (int f = 0; f < s.FUV_pre.rows(); f++)
        snap_appendf(out, "  f%d:[%d %d %d]\n", f,
                     s.FUV_pre(f,0), s.FUV_pre(f,1), s.FUV_pre(f,2));

    snap_appendf(out, "\n-- FUV_post (%dx3) --\n", (int)s.FUV_post.rows());
    for (int f = 0; f < s.FUV_post.rows(); f++)
        snap_appendf(out, "  f%d:[%d %d %d]\n", f,
                     s.FUV_post(f,0), s.FUV_post(f,1), s.FUV_post(f,2));

    snap_appendf(out, "\n-- UV_pre (%dx2) --\n", (int)s.UV_pre.rows());
    for (int v = 0; v < s.UV_pre.rows(); v++)
        snap_appendf(out, "  v%d:(%.6f, %.6f)\n", v, s.UV_pre(v,0), s.UV_pre(v,1));

    snap_appendf(out, "\n-- UV_post (%dx2) --\n", (int)s.UV_post.rows());
    for (int v = 0; v < s.UV_post.rows(); v++)
        snap_appendf(out, "  v%d:(%.6f, %.6f)\n", v, s.UV_post(v,0), s.UV_post(v,1));

    if (s.has_dc) {
        out += "\n-- DC B classification --\n";
        snap_appendf(out, "  B_glued (%d):", (int)s.dc_B_glued.size());
        for (int b : s.dc_B_glued)    snap_appendf(out, " %d", b);
        out += "\n";
        snap_appendf(out, "  B_reflected (%d):", (int)s.dc_B_reflected.size());
        for (int b : s.dc_B_reflected) snap_appendf(out, " %d", b);
        out += "\n";

        snap_appendf(out, "\n-- FUV_dc_pre (%dx3) --\n", (int)s.FUV_dc_pre.rows());
        for (int f = 0; f < s.FUV_dc_pre.rows(); f++)
            snap_appendf(out, "  f%d:[%d %d %d]%s\n", f,
                         s.FUV_dc_pre(f,0), s.FUV_dc_pre(f,1), s.FUV_dc_pre(f,2),
                         f == s.FUV_pre.rows() - 1 ? " <- top/bot split" : "");

        snap_appendf(out, "\n-- FUV_dc_post (%dx3) --\n", (int)s.FUV_dc_post.rows());
        for (int f = 0; f < s.FUV_dc_post.rows(); f++)
            snap_appendf(out, "  f%d:[%d %d %d]%s\n", f,
                         s.FUV_dc_post(f,0), s.FUV_dc_post(f,1), s.FUV_dc_post(f,2),
                         f == s.FUV_post.rows() - 1 ? " <- top/bot split" : "");

        snap_appendf(out, "\n-- UV_dc_pre (%dx2) --\n", (int)s.UV_dc_pre.rows());
        int nVjoint = (int)s.V_pre.rows() + 1;
        for (int v = 0; v < s.UV_dc_pre.rows(); v++) {
            const char* tag = "";
            if      (v == s.b(0))               tag = " <- vi(top)";
            else if (v == s.b(1))               tag = " <- vj(top)";
            else if (v == (int)s.V_pre.rows())  tag = " <- vi_post";
            else for (int k = 0; k < (int)s.dc_B_reflected.size(); k++)
                if (v == nVjoint + k) { tag = " <- B_ref_bot"; break; }
            snap_appendf(out, "  v%d:(%.6f, %.6f)%s\n",
                         v, s.UV_dc_pre(v,0), s.UV_dc_pre(v,1), tag);
        }

        snap_appendf(out, "\n-- UV_dc_post (%dx2) --\n", (int)s.UV_dc_post.rows());
        for (int v = 0; v < s.UV_dc_post.rows(); v++) {
            const char* tag = "";
            if      (v == s.b(0))               tag = " <- vi(top)";
            else if (v == s.b(1))               tag = " <- vj(top)";
            else if (v == (int)s.V_post.rows()) tag = " <- vi_post";
            else for (int k = 0; k < (int)s.dc_B_reflected.size(); k++)
                if (v == nVjoint + k) { tag = " <- B_ref_bot"; break; }
            snap_appendf(out, "  v%d:(%.6f, %.6f)%s\n",
                         v, s.UV_dc_post(v,0), s.UV_dc_post(v,1), tag);
        }
    }
    return out;
}

// Print a full diagnostic dump of the current snap's DC/joint-LSCM data to stdout.
static void print_dc_snap()
{
    if (!gSnap.valid) { printf("[PRINT-DC] no valid snap\n"); return; }
    printf("\n========== DC / Joint-LSCM Diagnostic Dump ==========\n");
    printf("%s", build_dc_snap_str().c_str());
    printf("======================================================\n\n");
    fflush(stdout);
}

static void print_bd_snap()
{
    const BDSnap & b = SSP_get_bd_snap();
    if (!b.valid) { printf("[BD-SNAP] no data recorded yet\n"); return; }

    printf("\n========== Boundary Detection / Case Selection Dump ==========\n");
    printf("  collapse_idx=%d  sid=%d\n", b.collapse_idx, b.sid);
    printf("  vi_global=%d  vj_global=%d  active_sheets=%d\n",
           b.vi_global, b.vj_global, b.active_sheets);
    printf("  vi_on_boundary=%d  vj_on_boundary=%d\n",
           (int)b.vi_on_boundary, (int)b.vj_on_boundary);
    printf("  injected_ndv(around vi)=%d  injected_nsv(around vj)=%d\n",
           (int)b.injected_ndv, (int)b.injected_nsv);
    if (b.lscm_case >= 0) {
        const char* desc[] = {"both interior", "one on boundary", "both on boundary"};
        printf("  lscm_case=%d  (%s)\n", b.lscm_case,
               b.lscm_case <= 2 ? desc[b.lscm_case] : "unknown");
    } else {
        printf("  lscm_case=not yet set\n");
    }

    // local→global vertex map
    printf("\n--- subsetVIdx (local → global) ---\n");
    for (int i = 0; i < (int)b.subsetVIdx.size(); i++)
        printf("  local%d → global%d\n", i, b.subsetVIdx[i]);

    // Nsv_local walk (around vj)
    printf("\n--- Nsv_local (walk around vj=%d, local indices) ---\n", b.vj_global);
    printf(" [");
    for (int v : b.Nsv_local) { if (v == -1) printf(" -1(inf)"); else printf(" %d(g%d)", v, (v>=0&&v<(int)b.subsetVIdx.size())?b.subsetVIdx[v]:-1); }
    printf(" ]\n");

    // Ndv_local walk (around vi)
    printf("\n--- Ndv_local (walk around vi=%d, local indices) ---\n", b.vi_global);
    printf(" [");
    for (int v : b.Ndv_local) { if (v == -1) printf(" -1(inf)"); else printf(" %d(g%d)", v, (v>=0&&v<(int)b.subsetVIdx.size())?b.subsetVIdx[v]:-1); }
    printf(" ]\n");

    // VF for vi
    printf("\n--- VF[vi=%d] (%zu faces) ---\n", b.vi_global, b.vi_vf.size());
    for (auto & fe : b.vi_vf)
        printf("  f=%d  %s\n", fe.f, fe.is_inf ? "<< INFINITY FACE" : "");

    // VF for vj
    printf("\n--- VF[vj=%d] (%zu faces) ---\n", b.vj_global, b.vj_vf.size());
    for (auto & fe : b.vj_vf)
        printf("  f=%d  %s\n", fe.f, fe.is_inf ? "<< INFINITY FACE" : "");

    printf("==============================================================\n\n");
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

    // Case 1 pin visualization group (vertex colors live on uv_pre, managed by quantity enable)
    bool showC1 = gShowCase1PinViz && showRegularUV &&
                  gSnap.lscm_case.has_value() && gSnap.lscm_case.value() == 1;
    if (polyscope::hasCurveNetwork("uv_case1_arc"))
        polyscope::getCurveNetwork("uv_case1_arc")->setEnabled(showC1);
    if (polyscope::hasPointCloud("uv_case1_vi_pin"))
        polyscope::getPointCloud("uv_case1_vi_pin")->setEnabled(showC1);
    if (polyscope::hasPointCloud("uv_case1_vj_pin"))
        polyscope::getPointCloud("uv_case1_vj_pin")->setEnabled(showC1);
    if (polyscope::hasPointCloud("uv_case1_bd_pin"))
        polyscope::getPointCloud("uv_case1_bd_pin")->setEnabled(showC1);

    // Non-active sheet geometry: two kinds rendered separately.
    //
    // (a) NAF faces — faces of the absorbed vertex in sheets that don't contain the
    //     collapsed edge at all.  Rendered as one neutral gray mesh.
    //
    // (b) Non-active gAllSheets entries — sheets that DO contain the edge (seam
    //     collapses) but are not the currently selected active sheet.  Each gets its
    //     own palette color so the user can visually separate them.
    if (!gDecInfo.empty()) {
        const auto & collapse = gDecInfo.back();

        // Shared HSV helper used by both (a) and (b).
        auto hsv_to_rgb = [](float h, float s, float v) -> std::array<float,3> {
            float c = v * s, x = c * (1.f - std::fabs(std::fmod(h * 6.f, 2.f) - 1.f));
            float m = v - c;
            float r,g,b;
            int hi = (int)(h * 6.f);
            switch (hi % 6) {
                case 0: r=c; g=x; b=0; break;
                case 1: r=x; g=c; b=0; break;
                case 2: r=0; g=c; b=x; break;
                case 3: r=0; g=x; b=c; break;
                case 4: r=x; g=0; b=c; break;
                default:r=c; g=0; b=x; break;
            }
            return {r+m, g+m, b+m};
        };

        // (a) NAF faces — grouped by sheet_id, one mesh per sheet with its own hue.
        {
            // Group by sheet_id.
            std::map<int, std::vector<const NonActiveSheetFace*>> by_sheet;
            for (const auto & naf : collapse.non_active_faces) {
                if (naf.is_infinity_face) continue;
                if (!naf.p0.allFinite() || !naf.p1.allFinite() || !naf.p2.allFinite()) continue;
                by_sheet[naf.sheet_id].push_back(&naf);
            }
            int nNafSheets = (int)by_sheet.size();
            if (nNafSheets < 1) nNafSheets = 1;
            int slot = 0;
            for (auto & kv : by_sheet) {
                std::vector<std::array<double,3>> verts;
                std::vector<std::array<int,3>>    faces;
                for (const auto * naf : kv.second) {
                    int base = (int)verts.size();
                    for (const Vector3d & pt : {naf->p0, naf->p1, naf->p2}) {
                        Vector3d rp = R * (pt - g.centroid);
                        verts.push_back({rp.x(), rp.y(), rp.z()});
                    }
                    faces.push_back({base, base+1, base+2});
                }
                MatrixXd nafV((int)verts.size(), 3);
                MatrixXi nafF((int)faces.size(), 3);
                for (int i = 0; i < (int)verts.size(); i++)
                    nafV.row(i) << verts[i][0], verts[i][1], verts[i][2];
                for (int i = 0; i < (int)faces.size(); i++)
                    nafF.row(i) << faces[i][0], faces[i][1], faces[i][2];
                float hue = (float)slot / (float)nNafSheets;
                // Offset NAF hues by 0.5 so they don't collide with the (b) block hues.
                hue = std::fmod(hue + 0.5f, 1.0f);
                auto c = hsv_to_rgb(hue, 0.55f, 0.80f);  // desaturated vs active sheets
                char name[64];
                snprintf(name, sizeof(name), "naf_sheet_sid%d", kv.first);
                polyscope::registerSurfaceMesh(name, nafV, nafF)
                    ->setSurfaceColor({c[0], c[1], c[2]})
                    ->setEdgeWidth(1.0)
                    ->setSmoothShade(false)
                    ->setTransparency(1.0f)
                    ->setEnabled(false);
                slot++;
            }
        }

        // (b) Non-active gAllSheets entries — one mesh per sheet, hues evenly distributed
        //     across the full color wheel so every sheet gets a unique color regardless
        //     of how many sheets there are.

        // Count non-active sheets first so we can space hues evenly.
        int nNonActive = (int)gAllSheets.size() - 1;  // one is the active sheet
        if (nNonActive < 1) nNonActive = 1;            // avoid div-by-zero
        int colorSlot = 0;
        for (int si = 0; si < (int)gAllSheets.size(); si++) {
            if (si == gActiveSheetIdx) continue;
            const SheetData & es = gAllSheets[si];
            std::vector<std::array<double,3>> verts;
            std::vector<std::array<int,3>>    faces;
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
            if (!faces.empty()) {
                char name[64];
                snprintf(name, sizeof(name), "sheet_inactive_%d_sid%d", si, es.global_sheet_id);
                MatrixXd shV((int)verts.size(), 3);
                MatrixXi shF((int)faces.size(), 3);
                for (int i = 0; i < (int)verts.size(); i++)
                    shV.row(i) << verts[i][0], verts[i][1], verts[i][2];
                for (int i = 0; i < (int)faces.size(); i++)
                    shF.row(i) << faces[i][0], faces[i][1], faces[i][2];
                float hue = (float)colorSlot / (float)nNonActive;
                auto c = hsv_to_rgb(hue, 0.75f, 0.9f);
                polyscope::registerSurfaceMesh(name, shV, shF)
                    ->setSurfaceColor({c[0], c[1], c[2]})
                    ->setEdgeWidth(1.5)
                    ->setSmoothShade(false)
                    ->setTransparency(0.45f)
                    ->setEnabled(gShowCanonRing);
            }
            colorSlot++;
        }
    }

    // DC-fail sheet: render as opaque red mesh so the failing geometry is obvious.
    {
        const DCFailSnap & dcf = SSP_get_dc_fail_snap();
        if (dcf.valid && dcf.F.rows() > 0) {
            // Lift into canonical 3D ring space using the same transform as one_ring_pre.
            Eigen::MatrixXd V_fail = dcf.V;
            // Center and scale to match gc.V_ring framing (same affine as register_ring_geometry).
            if (gc.V_ring.rows() > 0) {
                Eigen::RowVector3d ctr = gc.V_ring.colwise().mean();
                double scale = (gc.V_ring.rowwise() - ctr).rowwise().norm().maxCoeff();
                if (scale < 1e-12) scale = 1.0;
                Eigen::RowVector3d fail_ctr = V_fail.colwise().mean();
                double fail_scale = (V_fail.rowwise() - fail_ctr).rowwise().norm().maxCoeff();
                if (fail_scale < 1e-12) fail_scale = 1.0;
                V_fail = ((V_fail.rowwise() - fail_ctr) / fail_scale) * scale;
                V_fail = V_fail.rowwise() + ctr;
            }
            char label[64];
            snprintf(label, sizeof(label), "dc_fail_sheet_sid%d", dcf.global_sheet_id);
            polyscope::registerSurfaceMesh(label, V_fail, dcf.F)
                ->setSurfaceColor({1.0f, 0.10f, 0.10f})
                ->setEdgeWidth(2.0)
                ->setSmoothShade(false)
                ->setTransparency(1.0f)
                ->setEnabled(false);
        }
    }

    // Double cover visualization (boundary cases): flat overlay on UV panels.
    if (gc.has_dc) {
        polyscope::registerSurfaceMesh("dc_uv_pre", gc.dc_uv_pre_3d, gSnap.FUV_dc_pre)
            ->setSurfaceColor({0.2f, 0.85f, 0.85f})   // teal — top+bottom sheet pre
            ->setEdgeWidth(1.0)->setSmoothShade(false)->setTransparency(1.0f)
            ->setEnabled(gShowCanonUV);
        polyscope::registerSurfaceMesh("dc_uv_post", gc.dc_uv_post_3d, gSnap.FUV_dc_post)
            ->setSurfaceColor({0.85f, 0.65f, 0.2f})   // amber — top+bottom sheet post
            ->setEdgeWidth(1.0)->setSmoothShade(false)->setTransparency(1.0f)
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

    // Boundary-vertex highlight: magenta point cloud on UV_pre showing which
    // vertex(es) were detected as on-boundary and drove the Case 1 / Case 2 selection.
    //   Case 1 — one vertex: determined by BDSnap injection flags.
    //   Case 2 — both vi and vj are boundary (seam collapse or both on open boundary).
    // For DC (Case 2) positions come from dc_uv_pre_3d; otherwise from uv_pre_3d.
    if (gSnap.lscm_case.has_value() && gSnap.b.size() >= 2) {
        int lscm_case   = gSnap.lscm_case.value();
        int local_vi    = gSnap.b(0);
        int local_vj    = gSnap.b(1);

        std::vector<int> bd_locals;
        if (lscm_case == 2) {
            bd_locals = {local_vi, local_vj};
        } else if (lscm_case == 1) {
            const BDSnap & bds = SSP_get_bd_snap();
            if (bds.valid) {
                if (bds.injected_ndv) bd_locals.push_back(local_vi);  // vi was boundary
                if (bds.injected_nsv) bd_locals.push_back(local_vj);  // vj was boundary
            }
            if (bd_locals.empty()) {
                // BDSnap not recorded for this collapse — fall back to showing both
                bd_locals = {local_vi, local_vj};
            }
        }

        if (!bd_locals.empty()) {
            // Pick the right UV position buffer: DC overlay or regular UV panel.
            const MatrixXd & uv_src = gc.has_dc ? gc.dc_uv_pre_3d : gc.uv_pre_3d;
            if (uv_src.rows() > 0) {
                MatrixXd bd_pts((int)bd_locals.size(), 3);
                bool any_valid = false;
                for (int k = 0; k < (int)bd_locals.size(); k++) {
                    int idx = bd_locals[k];
                    if (idx >= 0 && idx < (int)uv_src.rows()) {
                        bd_pts.row(k) = uv_src.row(idx);
                        any_valid = true;
                    } else {
                        bd_pts.row(k).setZero();
                    }
                }
                if (any_valid) {
                    polyscope::registerPointCloud("bd_vtx_uv_pre", bd_pts)
                        ->setPointColor({1.0f, 0.15f, 0.85f})  // magenta
                        ->setPointRadius(0.024f, true)
                        ->setEnabled(gShowBDVtx && gShowCanonUV);
                }
            }
        }
    }

    sample_tracker_show_canonical(gc.uv_pre_3d, gc.uv_post_3d, gSnap.FUV_pre, gSnap.FUV_post);

    // Cache for the export panel.
    gLastCanonGeom = gc;
    gHasCanonGeom  = true;
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

// ---- stale chain visualization ----
static std::vector<uint8_t> gStaleChainVisible;  // uint8_t avoids vector<bool> proxy issues
static bool                 gStaleChainShowAll = true;

static std::array<float,3> stale_hsv_rgb(float h, float s, float v)
{
    float c = v*s, x = c*(1.f - std::fabs(std::fmod(h*6.f, 2.f) - 1.f)), m = v-c;
    float r,g,b;
    switch ((int)(h*6.f) % 6) {
        case 0: r=c;g=x;b=0;break; case 1:r=x;g=c;b=0;break;
        case 2: r=0;g=c;b=x;break; case 3:r=0;g=x;b=c;break;
        case 4: r=x;g=0;b=c;break; default:r=c;g=0;b=x;break;
    }
    return {r+m, g+m, b+m};
}

static void update_stale_chains_display()
{
    if (gStaleChains.empty()) return;
    int nC = (int)gStaleChains.size();
    if ((int)gStaleChainVisible.size() != nC)
        gStaleChainVisible.assign(nC, 1);

    for (int ci = 0; ci < nC; ci++) {
        const auto & chain = gStaleChains[ci];
        int nV = (int)chain.size();
        if (nV < 2) continue;

        MatrixXd Vc(nV, 3);
        for (int k = 0; k < nV; k++) {
            int vid = chain[k];
            if (vid < (int)gVO.rows()) Vc.row(k) = gVO.row(vid);
            else                       Vc.row(k).setZero();
        }
        int nEdges = nV - 1;
        MatrixXi Ec(nEdges, 2);
        for (int k = 0; k < nEdges; k++) Ec.row(k) << k, k+1;

        auto col = stale_hsv_rgb((float)ci / (float)std::max(1, nC), 0.85f, 0.95f);
        char nm[64]; snprintf(nm, sizeof(nm), "stale_chain_%d", ci);
        polyscope::registerCurveNetwork(nm, Vc, Ec)
            ->setRadius(0.003, true)
            ->setColor({col[0], col[1], col[2]})
            ->setEnabled(gStaleChainShowAll && gStaleChainVisible[ci]);
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
            pc->addScalarQuantity("struct_hash", structScalar)->setEnabled(false);
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
        update_stale_chains_display();
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
            // show_canonical_view() must be called here because update_display() already
            // ran above with gCanonicalView=false; setting the flag alone leaves the
            // normal-view geometry on screen and won't re-render on the next frame
            // (gRunning is false so the running block won't execute).
            if (gStopAtDC && gSnap.has_dc) {
                gRunning       = false;
                gCanonicalView = true;
                show_canonical_view();
                polyscope::view::resetCameraToHomeView();
            }
            // Stop when DC Post UV is not symmetric across y=0 (sym=0: DC solved but UV asymmetric).
            if (gStopAtDCAsym && gSnap.has_dc && gSnap.dc_uv_symmetric == 0) {
                gRunning       = false;
                gCanonicalView = true;
                show_canonical_view();
                polyscope::view::resetCameraToHomeView();
            }
            // Stop when the DC solve itself failed (sym=-1: NaN UV, symmetry unmeasurable).
            // TODO: these cases need root-cause investigation — see [DC-FATAL] + [DC-SYM] in dc_log.
            if (gStopAtDCFail && gSnap.has_dc && gSnap.dc_uv_symmetric == -1) {
                gRunning       = false;
                gCanonicalView = true;
                show_canonical_view();
                polyscope::view::resetCameraToHomeView();
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
    if (gSnap.valid) {
        ImGui::Text("Edge: vi=%d  vj=%d", gSnap.vi, gSnap.vj);
        if (gSnap.lscm_case.has_value()) {
            int c = gSnap.lscm_case.value();
            static const char* case_label[] = {
                "Case 0: both interior",
                "Case 1: one on boundary",
                "Case 2: both on boundary (DC)"
            };
            ImVec4 col = (c == 0) ? ImVec4{0.7f,0.7f,0.7f,1.0f}
                       : (c == 1) ? ImVec4{1.0f,0.75f,0.2f,1.0f}
                                  : ImVec4{1.0f,0.35f,0.35f,1.0f};
            ImGui::TextColored(col, "LSCM: %s",
                               (c >= 0 && c <= 2) ? case_label[c] : "unknown");

            // Case 1 inline UV pin confirmation
            if (c == 1 && gSnap.UV_pre.rows() > 0 && gSnap.b.size() >= 2) {
                const BDSnap & bds = SSP_get_bd_snap();
                if (bds.valid) {
                    bool bd_is_vi = bds.injected_ndv;
                    int  bd_local = bd_is_vi ? (int)gSnap.b(0) : (int)gSnap.b(1);
                    // UV stored as col(0)=V, col(1)=U (columns swapped in reshape).
                    // vi pin: (0,0).  vj pin: (0,1).
                    double exp_col0 = 0.0;
                    double exp_col1 = bd_is_vi ? 0.0 : 1.0;
                    if (bd_local >= 0 && bd_local < (int)gSnap.UV_pre.rows()) {
                        double act_U = gSnap.UV_pre(bd_local, 1);  // col1=U
                        double act_V = gSnap.UV_pre(bd_local, 0);  // col0=V
                        bool ok = std::fabs(act_U - exp_col1) < 1e-4 &&
                                  std::fabs(act_V - exp_col0) < 1e-4;
                        ImGui::Indent();
                        ImGui::TextColored({0.95f, 0.85f, 0.2f, 1.0f},
                                           "[CASE1_BD_PINNED_TO_CENTER]");
                        ImGui::Text("  bd_vtx=%s (local %d)  UV=(U=%.4f, V=%.4f)",
                                    bd_is_vi ? "vi" : "vj", bd_local, act_U, act_V);
                        ImGui::Unindent();
                    }
                }
            }
        }
    } else {
        ImGui::Text("(no collapse yet)");
    }

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
        ImGui::NewLine();
        ImGui::Checkbox("Stop at DC", &gStopAtDC);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Halt + switch to canonical view on every double cover (boundary) case");
        ImGui::SameLine();
        ImGui::Checkbox("Stop on DC asym (sym=0)", &gStopAtDCAsym);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Halt when DC solved OK but Post UV is NOT symmetric across y=0 (sym=0).\n"
                "Symmetry axis: y=0 (seam line between the (-1,0) and (+1,0) pins).\n"
                "Check: each B_reflected top/bottom pair must have same x and opposite y.\n"
                "vi and vj must have |y| < 1e-4.\n"
                "Does NOT fire on DC-fail cases (sym=-1) — use the next checkbox for those.");
        ImGui::NewLine();
        ImGui::Checkbox("Stop on DC fail (sym=-1)", &gStopAtDCFail);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Halt when the DC solve itself failed — UV contains NaN (sym=-1).\n"
                "Distinct from sym=0 (DC OK but asymmetric).\n"
                "TODO: investigate root cause via [DC-FATAL] + [DC-SYM] entries in dc_log.");
        // Show symmetry status for the current DC snap
        if (gSnap.has_dc) {
            int s = gSnap.dc_uv_symmetric;
            if (s == 1)
                ImGui::TextColored({0.4f,1.0f,0.4f,1.0f},
                    "DC UV sym=+1 (symmetric, err=%.5f)", gSnap.dc_uv_asym_max_err);
            else if (s == 0)
                ImGui::TextColored({1.0f,0.6f,0.1f,1.0f},
                    "DC UV sym=0 (asymmetric, max_err=%.5f)", gSnap.dc_uv_asym_max_err);
            else
                ImGui::TextColored({1.0f,0.2f,0.2f,1.0f},
                    "DC UV sym=-1 (DC FAILED — NaN UV, unmeasurable)");
        }

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
        // Per-sheet selector — only shown when there are multiple sheets.
        if ((int)gAllSheets.size() > 1) {
            ImGui::Text("Sheet:");
            for (int i = 0; i < (int)gAllSheets.size(); i++) {
                bool active = (i == gActiveSheetIdx);
                char label[32];
                snprintf(label, sizeof(label), "Sheet %d (sid=%d)##sh%d",
                         i, gAllSheets[i].global_sheet_id, i);
                if (ImGui::Checkbox(label, &active) && active) {
                    gActiveSheetIdx = i;
                    apply_sheet_to_snap(gAllSheets[i]);
                    vis = true;   // trigger redraw
                }
            }
            ImGui::Separator();
        }
        // DC-fail snapshot controls
        {
            const DCFailSnap & dcf = SSP_get_dc_fail_snap();
            if (dcf.valid) {
                ImGui::TextColored({1.0f, 0.3f, 0.3f, 1.0f},
                    "DC FAIL: sid=%d  vi=%d vj=%d  nF=%d",
                    dcf.global_sheet_id, dcf.vi, dcf.vj, dcf.F.rows());
                ImGui::SameLine();
                if (ImGui::Button("Clear##dcfail")) {
                    char label[64];
                    snprintf(label, sizeof(label), "dc_fail_sheet_sid%d", dcf.global_sheet_id);
                    polyscope::removeStructure(label, /*errorIfAbsent=*/false);
                    SSP_clear_dc_fail_snap();
                }
            } else {
                ImGui::TextDisabled("No DC fail recorded");
            }
        }
        ImGui::Separator();

        // Seam diagnostic — inline collapsible + print button.
        {
            const std::string& seam_log =
                (!gDecInfo.empty()) ? gDecInfo.back().onering_seam_log : std::string{};
            bool has_seam = !seam_log.empty();
            if (ImGui::CollapsingHeader(has_seam ? "Seam Log" : "Seam Log (none)")) {
                if (has_seam) {
                    ImGui::InputTextMultiline(
                        "##seamlog",
                        const_cast<char*>(seam_log.c_str()),
                        seam_log.size() + 1,
                        ImVec2(-1.0f, 150.0f),
                        ImGuiInputTextFlags_ReadOnly);
                } else {
                    ImGui::TextDisabled("no seam log for this collapse");
                }
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Print##seam")) {
                if (has_seam) { printf("\n%s\n", seam_log.c_str()); fflush(stdout); }
                else          { printf("[Log Seam Info] no data\n"); fflush(stdout); }
            }
        }
        ImGui::Separator();

        vis |= ImGui::Checkbox("UV", &gShowCanonUV);
        if (gSnap.lscm_case.has_value() && gSnap.lscm_case.value() >= 1) {
            ImGui::SameLine();
            vis |= ImGui::Checkbox("Boundary vtx [magenta]", &gShowBDVtx);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Case 1: the vertex detected as on-boundary (drove case selection).\n"
                    "Case 2: both vi and vj (both boundary / seam collapse).");
        }
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
        // DC / Joint-LSCM diagnostic — inline collapsible panel.
        {
            static std::string s_dc_str;
            static int s_dc_last_collapse = -2;
            if (gCollapseCount != s_dc_last_collapse) {
                s_dc_str = build_dc_snap_str();
                s_dc_last_collapse = gCollapseCount;
            }
            if (ImGui::CollapsingHeader("DC / LSCM Info")) {
                ImGui::InputTextMultiline(
                    "##dcinfo",
                    const_cast<char*>(s_dc_str.c_str()),
                    s_dc_str.size() + 1,
                    ImVec2(-1.0f, 200.0f),
                    ImGuiInputTextFlags_ReadOnly);
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Print##dc")) print_dc_snap();
        }
        ImGui::SameLine();
        if (ImGui::Button("Print BD detection")) print_bd_snap();

        // ---- Export canonical meshes panel ----
        ImGui::Separator();
        if (ImGui::CollapsingHeader("Export Canonical Meshes (.ply)")) {
            ImGui::TextDisabled("%scollapse_%d_<type>_<datetime>.ply",
                gOutDir.c_str(), gCollapseCount);
            ImGui::Checkbox("UV Pre##expuvpre",           &gExportUVPre);
            ImGui::SameLine();
            ImGui::Checkbox("UV Post##expuvpost",         &gExportUVPost);
            ImGui::Checkbox("One Ring Pre##expringpre",   &gExportRingPre);
            ImGui::SameLine();
            ImGui::Checkbox("One Ring Post##expringpost", &gExportRingPost);
            if (ImGui::Button("Export selected##expbtn"))
                export_canonical_meshes();
        }
    } else {
        vis |= ImGui::Checkbox("Ring pre",        &gShowRingPre);   ImGui::SameLine();
        vis |= ImGui::Checkbox("Ring post",       &gShowRingPost);
        vis |= ImGui::Checkbox("UV pre",          &gShowUVPre);     ImGui::SameLine();
        vis |= ImGui::Checkbox("UV post",         &gShowUVPost);
        if (gSnap.lscm_case.has_value() && gSnap.lscm_case.value() == 1)
            vis |= ImGui::Checkbox("Case1 pin viz (role colors / arc / pins)", &gShowCase1PinViz);
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

    // ---- Stale Chains panel ----
    if (!gStaleChains.empty()) {
        ImGui::Separator();
        int nC = (int)gStaleChains.size();
        if ((int)gStaleChainVisible.size() != nC)
            gStaleChainVisible.assign(nC, 1);

        if (ImGui::CollapsingHeader(
                (std::string("Stale Chains (") + std::to_string(nC) + ")").c_str()))
        {
            bool allChanged = ImGui::Checkbox("Show all##sc_all", &gStaleChainShowAll);
            if (allChanged) {
                for (int ci = 0; ci < nC; ci++) {
                    char nm[64]; snprintf(nm, sizeof(nm), "stale_chain_%d", ci);
                    if (polyscope::hasCurveNetwork(nm))
                        polyscope::getCurveNetwork(nm)->setEnabled(gStaleChainShowAll && gStaleChainVisible[ci]);
                }
            }
            for (int ci = 0; ci < nC; ci++) {
                if (ci % 3 != 0) ImGui::SameLine();
                char label[48], nm[64];
                snprintf(label, sizeof(label), "C%d(%dv)##sc%d",
                         ci, (int)gStaleChains[ci].size(), ci);
                snprintf(nm, sizeof(nm), "stale_chain_%d", ci);
                auto col = stale_hsv_rgb((float)ci / (float)std::max(1, nC), 0.85f, 0.95f);
                ImGui::PushStyleColor(ImGuiCol_CheckMark, ImVec4(col[0], col[1], col[2], 1.f));
                bool vis = gStaleChainVisible[ci] != 0;
                if (ImGui::Checkbox(label, &vis)) {
                    gStaleChainVisible[ci] = vis ? 1 : 0;
                    if (polyscope::hasCurveNetwork(nm))
                        polyscope::getCurveNetwork(nm)->setEnabled(gStaleChainShowAll && vis);
                }
                ImGui::PopStyleColor();
            }
        }
    }

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
