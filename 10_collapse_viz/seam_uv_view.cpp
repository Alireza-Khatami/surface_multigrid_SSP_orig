#include "seam_uv_view.h"

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace Eigen;

namespace {

std::array<float,3> hsv_to_rgb(float h, float s, float v) {
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
}

} // namespace

namespace {

// Lifts a sheet's raw UV matrix into 3D as (u, v, 0) — no rotation, centroid
// shift, or scale of any kind. The z=0 column is only there because Polyscope
// surface meshes need 3D vertices; it does not alter u/v.
MatrixXd lift_uv(const MatrixXd & UV) {
    MatrixXd uv3(UV.rows(), 3);
    uv3.setZero();
    uv3.col(0) = UV.col(0);
    uv3.col(1) = UV.col(1);
    return uv3;
}

// (Re)creates a fresh, empty polyscope group with the given name. Using
// removeGroup(...,/*errorIfAbsent=*/false) first — rather than createGroup +
// catch — keeps this callable every redraw without throwing/logging spurious
// exceptions when the group already exists from a previous frame.
polyscope::Group* fresh_group(const char * name) {
    polyscope::removeGroup(name, /*errorIfAbsent=*/false);
    return polyscope::createGroup(name);
}

// True iff idx appears in F anywhere. get_post_faces (src/get_post_faces.cpp)
// already remaps every occurrence of the absorbed vertex onto the survivor
// before FUV_post is built, so FUV itself is the ground truth for "is this
// vertex actually part of this mesh" — no need to separately re-derive or
// hardcode which of vi/vj survives; just ask the face list.
bool referenced_in(const MatrixXi & F, int idx) {
    for (int r = 0; r < F.rows(); r++)
        for (int c = 0; c < F.cols(); c++)
            if (F(r, c) == idx) return true;
    return false;
}

// Registers one sheet's overlay mesh + vi/vj marker for either UV_pre/FUV_pre
// or UV_post/FUV_post — same logic, just parameterized on which raw solver
// output to draw and which group (Pre or Post) it belongs to. UV/FUV are
// used exactly as joint_lscm produced them — nothing here alters a UV value.
void show_one_sheet(int si, int nSheets, const SheetData & es,
                     const MatrixXd & UV, const MatrixXi & FUV,
                     const char * suffix, float sat, float val,
                     float edge_width, float transparency, float pt_radius,
                     bool enabled, polyscope::Group & group)
{
    if (UV.rows() == 0 || FUV.rows() == 0) return;

    float hue = (float)si / (float)nSheets;
    auto c = hsv_to_rgb(hue, sat, val);

    MatrixXd uv3 = lift_uv(UV);

    char name[80];
    snprintf(name, sizeof(name), "uvview_sheet_%d_sid%d_%s", si, es.global_sheet_id, suffix);
    polyscope::registerSurfaceMesh(name, uv3, FUV)
        ->setSurfaceColor({c[0], c[1], c[2]})
        ->setEdgeWidth(edge_width)
        ->setSmoothShade(false)
        ->setTransparency(transparency)
        ->setEnabled(enabled)
        ->addToGroup(group);

    // vi / vj markers, same color as this sheet's mesh. A candidate is
    // plotted only if FUV actually references it — for FUV_post that means
    // whichever of vi/vj got folded away by get_post_faces is naturally
    // dropped, with no special-case knowledge of collapse semantics needed
    // here. If sheets are not aligned in UV, same-global-vertex markers from
    // different sheets will NOT coincide even though they represent the
    // same vi/vj.
    if (es.b.size() >= 2) {
        std::vector<int> marker_idx;
        for (int lv : {(int)es.b(0), (int)es.b(1)})
            if (lv >= 0 && lv < uv3.rows() && referenced_in(FUV, lv))
                marker_idx.push_back(lv);

        if (!marker_idx.empty()) {
            MatrixXd pts((int)marker_idx.size(), 3);
            for (int k = 0; k < (int)marker_idx.size(); k++)
                pts.row(k) = uv3.row(marker_idx[k]);
            char pname[80];
            snprintf(pname, sizeof(pname), "uvview_sheet_%d_sid%d_%s_vivj", si, es.global_sheet_id, suffix);
            polyscope::registerPointCloud(pname, pts)
                ->setPointColor({c[0], c[1], c[2]})
                ->setPointRadius(pt_radius, true)
                ->setEnabled(enabled)
                ->addToGroup(group);
        }
    }
}

} // namespace

void show_seam_uv_view(const std::vector<SheetData> & sheets, bool show_pre, bool show_post)
{
    int nSheets = (int)sheets.size();
    if (nSheets == 0) return;

    // One group per pre/post, each holding every sheet's mesh + vi/vj point
    // cloud for that side — keeps them from being interleaved/mixed in the
    // structure list and lets the whole side be toggled/collapsed as a unit.
    polyscope::Group* preGrp  = fresh_group("UV View: Pre");
    polyscope::Group* postGrp = fresh_group("UV View: Post");

    for (int si = 0; si < nSheets; si++) {
        const SheetData & es = sheets[si];

        // Pre-collapse — direct joint_lscm UV_pre/FUV_pre, bright/opaque.
        show_one_sheet(si, nSheets, es, es.UV_pre, es.FUV_pre, "pre",
                       /*sat=*/0.8f, /*val=*/0.95f,
                       /*edge_width=*/1.5f, /*transparency=*/0.55f,
                       /*pt_radius=*/0.02f, show_pre, *preGrp);

        // Post-collapse — direct joint_lscm UV_post/FUV_post, desaturated so
        // pre/post are visually distinguishable while sharing the sheet's hue.
        show_one_sheet(si, nSheets, es, es.UV_post, es.FUV_post, "post",
                       /*sat=*/0.35f, /*val=*/0.75f,
                       /*edge_width=*/1.5f, /*transparency=*/0.55f,
                       /*pt_radius=*/0.016f, show_post, *postGrp);
    }
}
