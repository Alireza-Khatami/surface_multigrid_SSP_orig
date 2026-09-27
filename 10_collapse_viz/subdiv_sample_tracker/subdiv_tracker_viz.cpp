#include "subdiv_tracker_viz.h"
#include "subdiv_tracker.h"

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>
#include <imgui.h>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>

using namespace Eigen;

namespace {

int  gLevel          = -1;       // display level; -1 = pick from gMaxDisplayVerts
int  gMaxDisplayVerts = 500000;
int  gRefreshEvery   = 1;        // re-upload deformed positions every K collapses
bool gShowFine       = false;   // surface meshes
bool gShowDeformed   = false;
bool gShowFinePts    = true;    // point clouds colored per struct set
bool gShowDefPts     = false;
bool gShowDisp       = false;
bool gDirty          = true;     // re-register everything on next update
int  gLastRefresh    = -1;       // collapse count at the last position upload
int  gQuery          = -1;       // highlighted subdivided vertex

const char * kFine = "subdiv_fine";
const char * kDef  = "subdiv_deformed";
const char * kQry  = "subdiv_query";
std::array<double, 3> type_color(uint8_t mask);
const char * kFinePts = "subdiv_pts_fine";
const char * kDefPts  = "subdiv_pts_deformed";

// One distinct color per struct set: hues spaced by the golden ratio, with
// saturation/value cycling so neighbouring set ids stay distinguishable.
std::array<double, 3> set_color(int k)
{
    const double h = std::fmod(0.1 + 0.6180339887498949 * k, 1.0) * 6.0;
    const double sat = (k % 3 == 0) ? 0.85 : (k % 3 == 1) ? 0.65 : 0.95;
    const double val = (k % 2 == 0) ? 0.95 : 0.75;
    const int i = (int)h;
    const double f = h - i, p = val * (1 - sat), q = val * (1 - sat * f), t = val * (1 - sat * (1 - f));
    switch (i % 6) {
    case 0:  return { val, t, p };
    case 1:  return { q, val, p };
    case 2:  return { p, val, t };
    case 3:  return { p, q, val };
    case 4:  return { t, p, val };
    default: return { val, p, q };
    }
}

MatrixXd set_colors(int nv)
{
    const std::vector<int32_t> & S = subdiv_tracker_set_ids();
    MatrixXd c(nv, 3);
    for (int i = 0; i < nv; ++i) {
        const auto a = set_color(S[i]);
        c.row(i) << a[0], a[1], a[2];
    }
    return c;
}

void add_point_quantities(polyscope::PointCloud * pc, int nv)
{
    const StructPalette & P = subdiv_tracker_palette();
    const std::vector<int32_t> & S = subdiv_tracker_set_ids();
    MatrixXd tc(nv, 3);
    VectorXd sid(nv), rad(nv);
    for (int i = 0; i < nv; ++i) {
        const uint8_t m = P.typeMask[S[i]];
        const auto a = type_color(m);
        tc.row(i) << a[0], a[1], a[2];
        sid(i) = S[i];
        // Radius factor by structure: junction > seam / boundary > sheet.
        rad(i) = (m & STRUCT_MASK_JUNCTION) ? 1.0
               : (m & (STRUCT_MASK_SEAM | STRUCT_MASK_BOUNDARY)) ? 0.7
               : 0.35;
    }
    pc->addColorQuantity("struct set", set_colors(nv))->setEnabled(true);
    pc->addColorQuantity("struct type", tc);
    pc->addScalarQuantity("struct set id", sid);
    pc->addScalarQuantity("radius by struct type", rad);
    // Autoscaled: radius = point radius * factor / max factor, so junctions get the full radius.
    pc->setPointRadiusQuantity("radius by struct type", true);
    pc->setPointRadius(0.004, true);
}

std::array<double, 3> type_color(uint8_t mask)
{
    if (mask & STRUCT_MASK_JUNCTION) return { 0.58, 0.20, 0.80 };  // purple
    if (mask & STRUCT_MASK_SEAM)     return { 1.00, 0.65, 0.00 };  // orange (simp_viz MS_Seam)
    if (mask & STRUCT_MASK_BOUNDARY) return { 0.86, 0.08, 0.24 };  // crimson (simp_viz MS_Boundary)
    if (mask & STRUCT_MASK_SHEET)    return { 0.70, 0.75, 0.85 };
    return { 0.30, 0.30, 0.30 };                                    // no struct id
}

std::array<double, 3> carrier_color(uint8_t t)
{
    if (t == SUBDIV_CARRIER_VERTEX) return { 0.90, 0.20, 0.20 };
    if (t == SUBDIV_CARRIER_EDGE)   return { 0.95, 0.80, 0.10 };
    return { 0.70, 0.70, 0.70 };
}

int pick_level(const SubdivMesh & M)
{
    int l = 0;
    while (l < M.nLevels && M.levelVerts[l + 1] <= gMaxDisplayVerts) ++l;
    return l;
}

void add_quantities(polyscope::SurfaceMesh * sm, int nv)
{
    const SubdivMesh & M = subdiv_tracker_mesh();
    const StructPalette & P = subdiv_tracker_palette();
    const std::vector<int32_t> & S = subdiv_tracker_set_ids();
    MatrixXd tc(nv, 3), cc(nv, 3);
    VectorXd sid(nv);
    for (int i = 0; i < nv; ++i) {
        const auto a = type_color(P.typeMask[S[i]]);
        const auto b = carrier_color(M.carrierType[i]);
        tc.row(i) << a[0], a[1], a[2];
        cc.row(i) << b[0], b[1], b[2];
        sid(i) = S[i];
    }
    sm->addVertexColorQuantity("struct set", set_colors(nv))->setEnabled(true);
    sm->addVertexColorQuantity("struct type", tc);
    sm->addVertexColorQuantity("carrier (vertex/edge/face)", cc);
    sm->addVertexScalarQuantity("struct set id", sid);
}

} // namespace

void subdiv_tracker_viz_update()
{
    if (!subdiv_tracker_enabled()) return;
    const SubdivMesh & M = subdiv_tracker_mesh();
    if (gLevel < 0 || gLevel > M.nLevels) { gLevel = pick_level(M); gDirty = true; }
    const int nv = (int)M.levelVerts[gLevel];
    const int nC = subdiv_tracker_collapses();

    const bool full = gDirty;
    const bool pos  = full || nC < gLastRefresh || nC - gLastRefresh >= gRefreshEvery;

    MatrixXd Pd;
    if (pos) {
        Pd.resize(nv, 3);
        for (int i = 0; i < nv; ++i) Pd.row(i) = subdiv_tracker_cur_pos(i).transpose();
    }

    if (full) {
        const MatrixXi Fl = subdiv_level_faces(M, gLevel);
        auto * fine = polyscope::registerSurfaceMesh(kFine, M.V.topRows(nv), Fl);
        add_quantities(fine, nv);
        fine->setEnabled(gShowFine);

        auto * def = polyscope::registerSurfaceMesh(kDef, Pd, Fl);
        add_quantities(def, nv);
        def->setEnabled(gShowDeformed);

        auto * fp = polyscope::registerPointCloud(kFinePts, M.V.topRows(nv));
        add_point_quantities(fp, nv);
        fp->setEnabled(gShowFinePts);

        auto * dp = polyscope::registerPointCloud(kDefPts, Pd);
        add_point_quantities(dp, nv);
        dp->setEnabled(gShowDefPts);
        gDirty = false;
    } else if (pos) {
        polyscope::getSurfaceMesh(kDef)->updateVertexPositions(Pd);
        polyscope::getPointCloud(kDefPts)->updatePointPositions(Pd);
    }

    if (pos) {
        auto * def = polyscope::getSurfaceMesh(kDef);
        // Drawn at the tracked position, pointing back to the fine position.
        def->addVertexVectorQuantity("to fine position", (M.V.topRows(nv) - Pd).eval(),
                                     polyscope::VectorType::AMBIENT)
            ->setEnabled(gShowDisp);
        gLastRefresh = nC;
    }

    if (gQuery >= 0 && gQuery < (int)M.V.rows()) {
        MatrixXd q(2, 3);
        q.row(0) = M.V.row(gQuery);
        q.row(1) = subdiv_tracker_cur_pos(gQuery).transpose();
        polyscope::registerPointCloud(kQry, q)
            ->setPointColor({ 0.1f, 0.9f, 1.0f })
            ->setPointRadius(0.006, true);
    } else if (polyscope::hasPointCloud(kQry)) {
        polyscope::removePointCloud(kQry);
    }
}

void subdiv_tracker_viz_ui()
{
    ImGui::Separator();
    ImGui::Text("Subdivided sample tracker");
    if (!subdiv_tracker_enabled()) {
        ImGui::TextDisabled("(off: run with --n_subdiv_samples N)");
        return;
    }
    const SubdivMesh & M = subdiv_tracker_mesh();
    const StructPalette & P = subdiv_tracker_palette();
    ImGui::TextDisabled("%d levels  %lld verts  %lld faces  %d struct sets",
                        M.nLevels, (long long)M.V.rows(), (long long)M.F.rows(), P.size());

    int lvl = gLevel < 0 ? 0 : gLevel;
    char lbl[64];
    snprintf(lbl, sizeof(lbl), "level %%d (%lld verts)", (long long)M.levelVerts[lvl]);
    if (ImGui::SliderInt("display level", &lvl, 0, M.nLevels, lbl)) { gLevel = lvl; gDirty = true; }
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputInt("refresh every K collapses", &gRefreshEvery)) gRefreshEvery = std::max(1, gRefreshEvery);

    bool redraw = false;
    if (ImGui::Checkbox("fine##subdiv", &gShowFine) && polyscope::hasSurfaceMesh(kFine))
        polyscope::getSurfaceMesh(kFine)->setEnabled(gShowFine);
    ImGui::SameLine();
    if (ImGui::Checkbox("deformed##subdiv", &gShowDeformed) && polyscope::hasSurfaceMesh(kDef))
        polyscope::getSurfaceMesh(kDef)->setEnabled(gShowDeformed);
    ImGui::SameLine();
    if (ImGui::Checkbox("displacement##subdiv", &gShowDisp)) redraw = true;
    if (ImGui::Checkbox("fine points##subdiv", &gShowFinePts) && polyscope::hasPointCloud(kFinePts))
        polyscope::getPointCloud(kFinePts)->setEnabled(gShowFinePts);
    ImGui::SameLine();
    if (ImGui::Checkbox("deformed points##subdiv", &gShowDefPts) && polyscope::hasPointCloud(kDefPts))
        polyscope::getPointCloud(kDefPts)->setEnabled(gShowDefPts);
    if (ImGui::Button("refresh now##subdiv")) { gLastRefresh = -1; redraw = true; }

    static int q = 0;
    ImGui::SetNextItemWidth(120);
    ImGui::InputInt("sub vertex##subdiv", &q);
    ImGui::SameLine();
    if (ImGui::Button("show##subdiv")) { gQuery = q; redraw = true; }
    ImGui::SameLine();
    if (ImGui::Button("clear##subdiv")) { gQuery = -1; redraw = true; }

    if (gQuery >= 0 && gQuery < (int)M.V.rows()) {
        static const char * kCarrier[3] = { "vertex", "edge", "face" };
        const int i = gQuery;
        ImGui::TextDisabled("carrier %s %d   fine face %d  bary (%.4f %.4f %.4f)",
                            kCarrier[M.carrierType[i]], M.carrierIndex[i], M.fineFace[i],
                            M.fineBary(i, 0), M.fineBary(i, 1), M.fineBary(i, 2));
        const int k = subdiv_tracker_set_ids()[i];
        std::string ids;
        for (int j = P.offsets[k]; j < P.offsets[k + 1]; ++j) ids += std::to_string(P.ids[j]) + " ";
        ImGui::TextDisabled("struct set %d: { %s}", k, ids.c_str());
        if (subdiv_tracker_is_tracked(i)) {
            const Vector3i c = subdiv_tracker_cur_corners(i);
            const Vector3d b = subdiv_tracker_cur_bary(i);
            ImGui::TextDisabled("now: gF face %d  corners (%d %d %d)  bary (%.4f %.4f %.4f)",
                                subdiv_tracker_cur_face(i), c(0), c(1), c(2), b(0), b(1), b(2));
        } else {
            ImGui::TextDisabled("not tracked (vertex on no face)");
        }
    }

    if (redraw) subdiv_tracker_viz_update();
}
