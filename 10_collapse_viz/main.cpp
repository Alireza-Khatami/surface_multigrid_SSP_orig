#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>

#include <igl/read_triangle_mesh.h>
#include <igl/connect_boundary_to_infinity.h>
#include <igl/edge_flaps.h>
#include <igl/is_edge_manifold.h>
#include <igl/shortest_edge_and_midpoint.h>
#include <igl/parallel_for.h>
#include <igl/remove_unreferenced.h>
#include <igl/collapse_edge.h>  // IGL_COLLAPSE_EDGE_NULL

#include <SSP_collapse_edge.h>
#include <single_collapse_data.h>
#include <always_try_never_care.h>
#include <min_heap.h>

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <tuple>
#include <limits>
#include <string>
#include <cmath>

using namespace Eigen;

// ---- SSP loop state ----
static MatrixXd gV;
static MatrixXi gF, gE;
static VectorXi gEMAP;
static MatrixXi gEF, gEI;
static igl::min_heap<std::tuple<double,int,int>> gQ;
static VectorXi gEQ;
static MatrixXd gC;
static std::vector<single_collapse_data> gDecInfo;
static std::vector<std::vector<int>> gDecIM;
static int gTargetFaces  = 100;
static int gCollapseCount = 0;
static bool gFinished    = false;
static float gUVOffset   = 1.5f;  // UV panel offset along one-ring normal (× ring span)
static float gRingScale  = 1.0f;  // uniform scale for both one-ring and UV panel
static bool  gShowRingPre       = true;
static bool  gShowRingPost      = true;
static bool  gShowUVPre         = true;
static bool  gShowUVPost        = true;
static bool  gShowArrowPre      = true;   // arrows: pre one-ring → pre UV
static bool  gShowArrowPost     = true;   // arrows: post one-ring → post UV
static bool  gShowCollapsedEdge = true;   // both collapsed-edge curve networks
static float edge_radius = 0.00006f;

// ---- display snapshot from the last accepted collapse ----
struct DisplaySnap {
    bool valid = false;
    int vi = -1, vj = -1;          // global vertex indices
    MatrixXd V_pre;                 // 3-D one-ring vertices
    MatrixXi FUV_pre, FUV_post;     // one-ring faces pre / post
    MatrixXd UV_pre,  UV_post;      // 2-D UV layouts
    VectorXi b;                     // local indices [vi_local, vj_local]
} gSnap;

// ---- helpers ----

// Returns live, non-infinity faces from the decimated mesh
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

// Returns V with any inf entries replaced by zero (for safe upload to polyscope)
static MatrixXd safe_V()
{
    MatrixXd V = gV.leftCols(3);
    for (int v = 0; v < V.rows(); v++)
        if (std::isinf(V(v,0)) || std::isinf(V(v,1)) || std::isinf(V(v,2)))
            V.row(v).setZero();
    return V;
}

// ---- init ----
static void init_ssp(const std::string & mesh_path, int tarF)
{
    MatrixXd VO; MatrixXi FO;
    igl::read_triangle_mesh(mesh_path, VO, FO);
    std::cout << "Loaded: |V|=" << VO.rows() << "  |F|=" << FO.rows() << "\n";

    gTargetFaces = tarF;

    igl::connect_boundary_to_infinity(VO, FO, gV, gF);
    igl::edge_flaps(gF, gE, gEMAP, gEF, gEI);

    {
        Array<bool,Dynamic,Dynamic> BF;
        Array<bool,Dynamic,1> BE;
        if (!igl::is_edge_manifold(gF, gE.rows(), gEMAP, BF, BE)) {
            std::cerr << "Input mesh is not edge-manifold – aborting.\n";
            std::exit(1);
        }
    }

    gEQ = VectorXi::Zero(gE.rows());
    gC.resize(gE.rows(), gV.cols());
    {
        VectorXd costs(gE.rows());
        igl::parallel_for(gE.rows(), [&](const int e) {
            double cost = e;
            RowVectorXd p(1, 3);
            igl::shortest_edge_and_midpoint(e, gV, gF, gE, gEMAP, gEF, gEI, cost, p);
            gC.row(e) = p;
            costs(e) = cost;
        }, 10000);
        for (int e = 0; e < gE.rows(); e++)
            gQ.emplace(costs(e), e, 0);
    }

    gDecIM.resize(gF.rows());
    gDecInfo.reserve(FO.rows() - tarF + 1);
}

// ---- one step ----
// Advances until the next ACCEPTED collapse and captures its data.
static bool do_next_step()
{
    if (gFinished) return false;

    decimate_pre_collapse_func  always_try;
    decimate_post_collapse_func never_care;
    always_try_never_care(always_try, never_care);

    for (int tries = 0; tries < 1'000'000; tries++) {
        if (gQ.empty() || std::get<0>(gQ.top()) == std::numeric_limits<double>::infinity()) {
            gFinished = true;
            return false;
        }
        int e, e1, e2, f1, f2;
        bool ok = SSP_collapse_edge(
            igl::shortest_edge_and_midpoint, always_try, never_care,
            gV, gF, gE, gEMAP, gEF, gEI,
            gQ, gEQ, gC, e, e1, e2, f1, f2,
            gDecInfo, gDecIM);

        if (ok) {
            gCollapseCount++;

            const single_collapse_data & d = gDecInfo.back();
            gSnap.valid   = true;
            gSnap.b       = d.b;
            gSnap.vi      = d.subsetVIdx(d.b(0));
            gSnap.vj      = d.subsetVIdx(d.b(1));
            gSnap.V_pre   = d.V_pre;
            gSnap.FUV_pre  = d.FUV_pre;
            gSnap.FUV_post = d.FUV_post;
            gSnap.UV_pre   = d.UV_pre;
            gSnap.UV_post  = d.UV_post;

            // count live (non-inf) faces for stopping condition
            int live = (int)live_faces().rows();
            if (live <= gTargetFaces) gFinished = true;
            return true;
        }
    }
    gFinished = true;
    return false;
}

// ---- update polyscope display ----
static void update_display()
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

    // ---- shared: centroid and ring_span (finite vertices only) ----
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

    // ---- UV 3-D positions (computed early so arrows can reference them) ----
    // Area-weighted average normal (skip faces touching the infinity vertex)
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
            double u = (UV(i,0) - u_center) * uv_scale * -1.0f ;
            double v = (UV(i,1) - v_center) * uv_scale * -1.0f;
            P.row(i) = (panel_center + t1*u + t2*v).transpose();
        }
        return P;
    };

    MatrixXd uv_pre_3d  = make3d(gSnap.UV_pre);
    MatrixXd uv_post_3d = make3d(gSnap.UV_post);

    // ---- arrow vectors: one-ring → UV (zero out rows with inf origin) ----
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

    // ---- one-ring 3-D meshes + vector quantities ----
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
static void ui_callback()
{
    ImGui::SetNextWindowSize({320, 360}, ImGuiCond_FirstUseEver);
    ImGui::Begin("SSP Collapse Visualizer");

    ImGui::Text("Collapses: %d", gCollapseCount);
    if (gSnap.valid)
        ImGui::Text("Edge: vi=%d  vj=%d", gSnap.vi, gSnap.vj);
    else
        ImGui::Text("(no collapse yet)");

    if (gFinished) {
        ImGui::TextColored({0.4f,1.0f,0.4f,1.0f}, "Reached target faces.");
    } else {
        if (ImGui::Button("Next collapse  [Space]") ||
            ImGui::IsKeyPressed(ImGuiKey_Space, /*repeat=*/false))
        {
            if (do_next_step())
                update_display();
        }
    }

    ImGui::Separator();
    ImGui::TextDisabled("Blue  = pre-collapse");
    ImGui::TextDisabled("Orange = post-collapse");
    ImGui::TextDisabled("Red line = collapsed edge");

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
    vis |= ImGui::Checkbox("Ring pre",      &gShowRingPre);   ImGui::SameLine();
    vis |= ImGui::Checkbox("Ring post",     &gShowRingPost);
    vis |= ImGui::Checkbox("UV pre",        &gShowUVPre);     ImGui::SameLine();
    vis |= ImGui::Checkbox("UV post",       &gShowUVPost);
    vis |= ImGui::Checkbox("Arrows pre",    &gShowArrowPre);  ImGui::SameLine();
    vis |= ImGui::Checkbox("Arrows post",   &gShowArrowPost);
    vis |= ImGui::Checkbox("Collapsed edges", &gShowCollapsedEdge);
    if (vis && gSnap.valid) update_display();

    ImGui::End();
}

// ---- main ----
int main(int argc, char * argv[])
{
    if (argc < 3) {
        std::cerr << "usage: collapse_viz_bin  <mesh_path>  <target_faces>\n";
        return 1;
    }
    std::string mesh_path = argv[1];
    int tarF = std::stoi(argv[2]);

    init_ssp(mesh_path, tarF);

    polyscope::init();
    polyscope::state::userCallback = ui_callback;

    // Show initial mesh
    update_display();

    polyscope::show();
    return 0;
}
