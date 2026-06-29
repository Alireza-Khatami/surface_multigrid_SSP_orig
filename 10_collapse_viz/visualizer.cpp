#include "visualizer.h"

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>

#include <igl/remove_unreferenced.h>
#include <igl/collapse_edge.h>  // IGL_COLLAPSE_EDGE_NULL

#include <single_collapse_data.h>

#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <cmath>
#include <thread>
#include <chrono>

using namespace Eigen;

// ---- SSP state defined in main.cpp ----
extern MatrixXd gV;
extern MatrixXi gF;
extern int      gCollapseCount;
extern bool     gFinished;
extern std::vector<single_collapse_data> gDecInfo;

bool do_next_step();  // defined in main.cpp

// ---- display-only state ----
static float gUVOffset        = 1.5f;
static float gRingScale       = 1.0f;
static float gStepDelayMs     = 100.0f;   // milliseconds to sleep after each step
static bool  gRunning         = false;
static bool  gRunToSeam       = false;    // skip delay until next seam-edge collapse
static bool  gShowRingPre     = true;
static bool  gShowRingPost    = true;
static bool  gShowUVPre       = true;
static bool  gShowUVPost      = true;
static bool  gShowArrowPre    = true;
static bool  gShowArrowPost   = true;
static bool  gShowCollapsedEdge = true;
static float edge_radius      = 0.00006f;

struct DisplaySnap {
    bool valid = false;
    int vi = -1, vj = -1;
    MatrixXd V_pre;
    MatrixXi FUV_pre, FUV_post;
    MatrixXd UV_pre,  UV_post;
    VectorXi b;
} gSnap;

// ---- helpers ----

static MatrixXi live_faces()
{
    std::vector<std::array<int,3>> rows;
    rows.reserve(gF.rows());
    for (int f = 0; f < gF.rows(); f++) {
        int v0 = gF(f,0), v1 = gF(f,1), v2 = gF(f,2);
        if (v0 == IGL_COLLAPSE_EDGE_NULL) continue;
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

// Populate gSnap from the most recent collapse record.
static void refresh_snap()
{
    if (gDecInfo.empty()) return;
    const single_collapse_data & d = gDecInfo.back();
    gSnap.V_pre = d.V_pre;
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
    }
}

// ---- update polyscope display ----
void update_display()
{
    // Main mesh
    {
        MatrixXi Flive = live_faces();
        MatrixXd Vd, Vc; MatrixXi Fc; VectorXi I1, I2;
        Vd = safe_V();
        igl::remove_unreferenced(Vd, Flive, Vc, Fc, I1, I2);
        auto * m = polyscope::registerSurfaceMesh("mesh", Vc, Fc);
        m->setSurfaceColor({0.75f, 0.75f, 0.75f});
        m->setEdgeWidth(0.5);
    }

    if (!gSnap.valid) return;

    // ---- centroid and ring_span (finite vertices only) ----
    Vector3d centroid = Vector3d::Zero();
    double ring_span = 0.0;
    {
        int cnt = 0;
        for (int v = 0; v < gSnap.V_pre.rows(); v++) {
            if (std::isfinite(gSnap.V_pre(v,0))) {
                centroid += gSnap.V_pre.row(v).transpose();
                cnt++;
            }
        }
        if (cnt > 0) centroid /= cnt;
        for (int ax = 0; ax < 3; ax++) {
            double lo =  std::numeric_limits<double>::infinity();
            double hi = -std::numeric_limits<double>::infinity();
            for (int v = 0; v < gSnap.V_pre.rows(); v++) {
                double val = gSnap.V_pre(v, ax);
                if (std::isfinite(val)) { lo = std::min(lo, val); hi = std::max(hi, val); }
            }
            if (hi > lo) ring_span = std::max(ring_span, hi - lo);
        }
        if (ring_span < 1e-10) ring_span = 1.0;
    }

    // ---- one-ring vertices scaled around centroid ----
    MatrixXd V_scaled(gSnap.V_pre.rows(), 3);
    for (int v = 0; v < gSnap.V_pre.rows(); v++) {
        if (std::isfinite(gSnap.V_pre(v,0)))
            V_scaled.row(v) = centroid.transpose() +
                              (gSnap.V_pre.row(v) - centroid.transpose()) * (double)gRingScale;
        else
            V_scaled.row(v) = gSnap.V_pre.row(v);
    }

    // ---- UV panel normal and tangent frame ----
    Vector3d avg_normal = Vector3d::Zero();
    for (int f = 0; f < gSnap.FUV_pre.rows(); f++) {
        Vector3d p0 = gSnap.V_pre.row(gSnap.FUV_pre(f,0)).transpose();
        Vector3d p1 = gSnap.V_pre.row(gSnap.FUV_pre(f,1)).transpose();
        Vector3d p2 = gSnap.V_pre.row(gSnap.FUV_pre(f,2)).transpose();
        if (!p0.allFinite() || !p1.allFinite() || !p2.allFinite()) continue;
        avg_normal += (p1 - p0).cross(p2 - p0);
    }
    if (avg_normal.squaredNorm() < 1e-20) avg_normal = Vector3d::UnitZ();
    else avg_normal.normalize();

    Vector3d arb = (std::abs(avg_normal.dot(Vector3d::UnitX())) < 0.9)
                   ? Vector3d::UnitX() : Vector3d::UnitY();
    Vector3d t1 = (arb - avg_normal * avg_normal.dot(arb)).normalized();
    Vector3d t2 = avg_normal.cross(t1).normalized();

    double uv_span_u = gSnap.UV_pre.col(0).maxCoeff() - gSnap.UV_pre.col(0).minCoeff();
    double uv_span_v = gSnap.UV_pre.col(1).maxCoeff() - gSnap.UV_pre.col(1).minCoeff();
    double uv_span   = std::max(uv_span_u, uv_span_v);
    double uv_scale  = (uv_span > 1e-10) ? ring_span / uv_span : 1.0;
    uv_scale *= (double)gRingScale;

    double u_center = (gSnap.UV_pre.col(0).maxCoeff() + gSnap.UV_pre.col(0).minCoeff()) * 0.5;
    double v_center = (gSnap.UV_pre.col(1).maxCoeff() + gSnap.UV_pre.col(1).minCoeff()) * 0.5;
    Vector3d panel_center = centroid + avg_normal * (double)gUVOffset * ring_span;

    auto make3d = [&](const MatrixXd & UV) {
        MatrixXd P(UV.rows(), 3);
        for (int i = 0; i < UV.rows(); i++) {
            double u = (UV(i,0) - u_center) * uv_scale * -1.0;
            double v = (UV(i,1) - v_center) * uv_scale * -1.0;
            P.row(i) = (panel_center + t1*u + t2*v).transpose();
        }
        return P;
    };

    MatrixXd uv_pre_3d  = make3d(gSnap.UV_pre);
    MatrixXd uv_post_3d = make3d(gSnap.UV_post);

    // ---- arrow vectors: one-ring → UV ----
    int nV = gSnap.V_pre.rows();
    MatrixXd arrows_pre(nV, 3), arrows_post(nV, 3);
    for (int i = 0; i < nV; i++) {
        if (std::isfinite(V_scaled(i,0))) {
            arrows_pre.row(i)  = uv_pre_3d.row(i)  - V_scaled.row(i);
            arrows_post.row(i) = uv_post_3d.row(i) - V_scaled.row(i);
        } else {
            arrows_pre.row(i).setZero();
            arrows_post.row(i).setZero();
        }
    }

    // ---- one-ring 3-D meshes ----
    {
        auto * rp = polyscope::registerSurfaceMesh("one_ring_pre", V_scaled, gSnap.FUV_pre);
        rp->setSurfaceColor({0.3f, 0.55f, 1.0f})->setEdgeWidth(1.5)->setSmoothShade(false);
        rp->setEnabled(gShowRingPre);
        rp->addVertexVectorQuantity("to_uv_pre", arrows_pre, polyscope::VectorType::AMBIENT)->setEnabled(gShowArrowPre);

        auto * ro = polyscope::registerSurfaceMesh("one_ring_post", V_scaled, gSnap.FUV_post);
        ro->setSurfaceColor({1.0f, 0.5f, 0.15f})->setEdgeWidth(1.5)->setSmoothShade(false)->setTransparency(0.4f);
        ro->setEnabled(gShowRingPost);
        ro->addVertexVectorQuantity("to_uv_post", arrows_post, polyscope::VectorType::AMBIENT)->setEnabled(gShowArrowPost);
    }

    // ---- collapsed edge in 3-D ----
    {
        MatrixXd eV(2, 3);
        eV.row(0) = V_scaled.row(gSnap.b(0));
        eV.row(1) = V_scaled.row(gSnap.b(1));
        MatrixXi eE(1, 2); eE << 0, 1;
        polyscope::registerCurveNetwork("collapsed_edge", eV, eE)
            ->setRadius(edge_radius)->setColor({1.0f, 0.05f, 0.05f})
            ->setEnabled(gShowCollapsedEdge);
    }

    // ---- UV meshes ----
    {
        auto * up = polyscope::registerSurfaceMesh("uv_pre", uv_pre_3d, gSnap.FUV_pre);
        up->setSurfaceColor({0.3f, 0.55f, 1.0f})->setEdgeWidth(1.5)->setSmoothShade(false);
        up->setEnabled(gShowUVPre);

        auto * uo = polyscope::registerSurfaceMesh("uv_post", uv_post_3d, gSnap.FUV_post);
        uo->setSurfaceColor({1.0f, 0.5f, 0.15f})->setEdgeWidth(1.5)->setSmoothShade(false)->setTransparency(0.4f);
        uo->setEnabled(gShowUVPost);
    }

    // ---- collapsed edge in UV space ----
    {
        MatrixXd euV(2, 3);
        euV.row(0) = uv_pre_3d.row(gSnap.b(0));
        euV.row(1) = uv_pre_3d.row(gSnap.b(1));
        MatrixXi euE(1, 2); euE << 0, 1;
        polyscope::registerCurveNetwork("uv_collapsed_edge", euV, euE)
            ->setRadius(edge_radius)->setColor({1.0f, 0.05f, 0.05f})
            ->setEnabled(gShowCollapsedEdge);
    }
}

// ---- ImGui callback ----
void ui_callback()
{
    // Drive continuous decimation — sleep between steps so each frame is visible
    if (gRunning && !gFinished) {
        if (do_next_step()) {
            refresh_snap();
            update_display();

            bool hitSeam = !gDecInfo.empty() && gDecInfo.back().numFlapFaces > 2;
            if (gRunToSeam && hitSeam)
                gRunToSeam = false;  // stop at the seam edge, resume normal delay

            if (!gRunToSeam) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(static_cast<int>(gStepDelayMs)));
            }
        }
        if (gFinished)
            gRunning = false;
    }

    ImGui::SetNextWindowSize({320, 360}, ImGuiCond_FirstUseEver);
    ImGui::Begin("SSP Collapse Visualizer");

    ImGui::Text("Collapses: %d", gCollapseCount);
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
            if (do_next_step()) { refresh_snap(); update_display(); }
        }
        ImGui::SameLine();
        if (ImGui::Button("Run")) {
            gRunning = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Next seam")) {
            gRunToSeam = true;
            gRunning   = true;
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
    ImGui::Text("Scale:");
    redraw |= ImGui::SliderFloat("##scale", &gRingScale, 0.1f, 5.0f);
    if (redraw && gSnap.valid) update_display();

    ImGui::Separator();
    ImGui::Text("Visibility:");
    bool vis = false;
    vis |= ImGui::Checkbox("Ring pre",        &gShowRingPre);   ImGui::SameLine();
    vis |= ImGui::Checkbox("Ring post",       &gShowRingPost);
    vis |= ImGui::Checkbox("UV pre",          &gShowUVPre);     ImGui::SameLine();
    vis |= ImGui::Checkbox("UV post",         &gShowUVPost);
    vis |= ImGui::Checkbox("Arrows pre",      &gShowArrowPre);  ImGui::SameLine();
    vis |= ImGui::Checkbox("Arrows post",     &gShowArrowPost);
    vis |= ImGui::Checkbox("Collapsed edges", &gShowCollapsedEdge);
    if (vis && gSnap.valid) update_display();

    ImGui::End();
}
