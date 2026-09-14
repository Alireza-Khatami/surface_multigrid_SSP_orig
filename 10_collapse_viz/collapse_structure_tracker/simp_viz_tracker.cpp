#include "simp_viz_tracker.h"
#include "../face_dead.h"

#ifdef C2F_VIZ_DIAGNOSTIC
#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>
#include <polyscope/pick.h>
#include <imgui.h>
#endif

#include <Eigen/Dense>

#include <fstream>
#include <iostream>
#include <sstream>
#include <array>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace Eigen;

// ---- topo type constants (must match Python TOPO_TYPE_MAP) ----
static constexpr int MS_Sheet             = 0;
static constexpr int MS_Seam              = 1;
static constexpr int MS_Boundary          = 2;
static constexpr int MS_Junction          = 3;
static constexpr int MS_Sheet_Boundary    = 4;
static constexpr int MS_Seam_Boundary     = 5;
static constexpr int MS_Junction_Boundary = 6;
static constexpr int MS_Unknown           = 7;  // vertex not covered by any struct element
static constexpr int TOPO_COUNT           = 8;

static constexpr std::array<std::array<uint8_t,3>,8> TOPO_RGB = {{
    {100,149,237}, // MS_Sheet
    {255,165,  0}, // MS_Seam
    {220, 20, 60}, // MS_Boundary
    {148,  0,211}, // MS_Junction
    {  0,200,100}, // MS_Sheet_Boundary
    {255,215,  0}, // MS_Seam_Boundary
    {255, 69,  0}, // MS_Junction_Boundary
    {160,160,160}, // MS_Unknown  (grey)
}};

static const char* TOPO_NAME[8] = {
    "MS_Sheet","MS_Seam","MS_Boundary","MS_Junction",
    "MS_Sheet_Boundary","MS_Seam_Boundary","MS_Junction_Boundary",
    "MS_Unknown"
};

// ---- externs from main.cpp ----
extern MatrixXd gV;
extern MatrixXd gVO;
extern MatrixXi gF;
extern std::vector<std::set<int>> gVertexStructIDs;

// ---- module state ----
static int gNumInitial = 0;

// Per gV-vertex-id (size grows with gV if needed in on_collapse).
static std::vector<int>                     gTopoType;   // MS_* id, -1 = unknown
static std::vector<std::set<int>>           gStructIds;
static std::vector<std::unordered_set<int>> gAncestors;

#ifdef C2F_VIZ_DIAGNOSTIC
// point-cloud index → gV vertex id (rebuilt by simp_viz_tracker_update_display)
static std::vector<int> gLiveVertMap;

static bool        gShowAncestors   = false;
static int         gPickedGVIndex   = -1;  // gV index of last picked simplified vert
static std::string gPickedInfo;
using PickSel = std::pair<polyscope::Structure*, size_t>;
static PickSel     gPickLastSel     = {nullptr, 0};
#endif

// ---- helpers ----

static void ensure_size(int n)
{
    if (n > (int)gTopoType.size()) {
        gTopoType.resize(n, MS_Unknown);
        gStructIds.resize(n);
        gAncestors.resize(n);
    }
}

// Derive per-vertex topo type by re-reading the .ma_struct file and counting
// seam/boundary edge degrees and junction membership.
static bool derive_topo_types(const std::string& fname, int nv)
{
    std::ifstream f(fname);
    if (!f) return false;

    int fnv, fne, fnf;
    if (!(f >> fnv >> fne >> fnf) || fnv <= 0) return false;

    // Skip vertex lines ("v x y z r")
    for (int i = 0; i < fnv; ++i) {
        char ch; double x, y, z, r;
        f >> ch >> x >> y >> z >> r;
    }

    // Read edge endpoints
    std::vector<std::array<int,2>> edge_ep(fne);
    for (int i = 0; i < fne; ++i) {
        char ch;
        f >> ch >> edge_ep[i][0] >> edge_ep[i][1];
    }

    // Skip face lines
    for (int i = 0; i < fnf; ++i) {
        char ch; int a, b, c;
        f >> ch >> a >> b >> c;
    }

    int num_structs = 0;
    if (!(f >> num_structs)) return true;  // no struct section — topo stays -1

    std::vector<bool> is_junction(nv, false);
    std::vector<int>  seam_deg(nv, 0);
    std::vector<int>  bnd_deg(nv, 0);

    for (int s = 0; s < num_structs; ++s) {
        int struct_id, type_id, count;
        if (!(f >> struct_id >> type_id >> count)) break;

        for (int j = 0; j < count; ++j) {
            int elem_id;
            if (!(f >> elem_id)) break;

            if (type_id == 3) {
                // JUNCTION — elem_id is a vertex index
                if (elem_id >= 0 && elem_id < nv)
                    is_junction[elem_id] = true;
            } else if (type_id == 1) {
                // SEAM — elem_id is an edge index
                if (elem_id >= 0 && elem_id < fne) {
                    int u = edge_ep[elem_id][0], v = edge_ep[elem_id][1];
                    if (u >= 0 && u < nv) seam_deg[u]++;
                    if (v >= 0 && v < nv) seam_deg[v]++;
                }
            } else if (type_id == 2) {
                // BOUNDARY — elem_id is an edge index
                if (elem_id >= 0 && elem_id < fne) {
                    int u = edge_ep[elem_id][0], v = edge_ep[elem_id][1];
                    if (u >= 0 && u < nv) bnd_deg[u]++;
                    if (v >= 0 && v < nv) bnd_deg[v]++;
                }
            }
            // type_id == 0 (SHEET) — purely a face-element struct; vertices
            // that belong only to sheets are MS_Sheet, handled by the else branch below.
        }
    }

    for (int i = 0; i < nv; ++i) {
        bool has_seam = seam_deg[i] > 0;
        bool has_bnd  = bnd_deg[i] > 0;
        if (is_junction[i]) {
            // A junction vertex that also touches a seam/boundary is at the
            // end of a junction curve → Junction_Boundary; otherwise interior.
            gTopoType[i] = (has_seam || has_bnd) ? MS_Junction_Boundary : MS_Junction;
        } else if (has_seam) {
            // degree-1 seam endpoint → Seam_Boundary; interior → Seam
            gTopoType[i] = (seam_deg[i] == 1) ? MS_Seam_Boundary : MS_Seam;
        } else if (has_bnd) {
            // degree-1 boundary endpoint → Sheet_Boundary; interior → Boundary
            gTopoType[i] = (bnd_deg[i] == 1) ? MS_Sheet_Boundary : MS_Boundary;
        } else {
            gTopoType[i] = MS_Sheet;
        }
    }
    return true;
}

// ---- public API ----

void simp_viz_tracker_init(const std::string& matstruct_path, int n_initial)
{
    gNumInitial = n_initial;
    int total   = (int)gV.rows();  // n_initial + 1 (infinity cap)

    ensure_size(total);

    // Seed struct IDs from the global table already loaded by main.cpp
    const int nS = (int)gVertexStructIDs.size();
    for (int i = 0; i < total && i < nS; ++i)
        gStructIds[i] = gVertexStructIDs[i];

    // Each original vertex is its own ancestor; infinity cap vertex has none
    for (int i = 0; i < n_initial; ++i)
        gAncestors[i].insert(i);

    // Derive topo types
    if (!matstruct_path.empty()) {
        if (derive_topo_types(matstruct_path, n_initial))
            fprintf(stderr, "[simp_viz] topo types derived  (%d vertices)  %s\n",
                    n_initial, matstruct_path.c_str());
        else
            fprintf(stderr, "[simp_viz] could not derive topo types from %s — all -1\n",
                    matstruct_path.c_str());
    }

#ifdef C2F_VIZ_DIAGNOSTIC
    gLiveVertMap.clear();
    gPickedGVIndex = -1;
    gPickedInfo.clear();
    gPickLastSel   = {nullptr, 0};
#endif
}

void simp_viz_tracker_on_collapse(int s, int d)
{
    int needed = std::max(s, d) + 1;
    if (needed > (int)gTopoType.size()) ensure_size(needed);

    // Union ancestors: d's set folds into s's
    gAncestors[s].insert(gAncestors[d].begin(), gAncestors[d].end());

    // Union struct IDs (same-struct collapses only, but be safe)
    for (int id : gStructIds[d]) gStructIds[s].insert(id);

    // Topo type: same-struct collapse preserves type — keep s's value.
    // Do NOT inherit d's type when s is MS_Unknown — that would hide a real gap
    // in .ma_struct coverage.  Warn once per unique unknown vertex encountered
    // (the struct gate should have already blocked unknown-struct-set collapses,
    // but topo type is derived independently so it can be unknown even when the
    // struct ID set is non-empty, e.g. if derive_topo_types failed to open the file).
    static std::unordered_set<int> sWarnedUnknown;
    auto warn_unknown = [&](int v) {
        if (gTopoType[v] == MS_Unknown && sWarnedUnknown.insert(v).second)
            fprintf(stderr, "[simp_viz] WARNING: vertex %d has MS_Unknown topo type"
                    " — check .ma_struct coverage\n", v);
    };
    warn_unknown(s);
    warn_unknown(d);
}

void simp_viz_tracker_write_json(const std::string& path)
{
    // Collect live vertices (appear in live, finite faces) in a stable order
    std::vector<int> live_verts;
    {
        std::unordered_set<int> seen;
        for (int f = 0; f < gF.rows(); ++f) {
            if (is_face_dead(gF, f)) continue;
            for (int c = 0; c < 3; ++c) {
                int v = gF(f, c);
                if (std::isinf(gV(v, 0))) continue;
                if (seen.insert(v).second) live_verts.push_back(v);
            }
        }
    }

    std::ofstream out(path);
    if (!out) {
        fprintf(stderr, "[simp_viz] cannot open for writing: %s\n", path.c_str());
        return;
    }

    // ---- legends ----
    out << "{\n  \"legends\": {\n    \"topo_types\": [\n";
    for (int i = 0; i < TOPO_COUNT; ++i) {
        out << "      { \"id\": " << i
            << ", \"name\": \"" << TOPO_NAME[i] << "\""
            << ", \"rgb\": [" << (int)TOPO_RGB[i][0]
            << ", " << (int)TOPO_RGB[i][1]
            << ", " << (int)TOPO_RGB[i][2] << "] }";
        if (i < TOPO_COUNT - 1) out << ",";
        out << "\n";
    }
    out << "    ]\n  },\n";

    // ---- vertices ----
    out << "  \"vertices\": [\n";
    int nL = (int)live_verts.size();
    for (int i = 0; i < nL; ++i) {
        int gv = live_verts[i];
        int tt = (gv < (int)gTopoType.size()) ? gTopoType[gv] : MS_Unknown;
        if (tt < 0) tt = MS_Unknown;

        // position (current gV position after collapse placement)
        out << "    { \"position\": ["
            << gV(gv, 0) << ", " << gV(gv, 1) << ", " << gV(gv, 2)
            << "], \"topo_type\": " << tt << ", \"struct_ids\": [";
        if (gv < (int)gStructIds.size()) {
            bool first = true;
            for (int id : gStructIds[gv]) {
                if (!first) out << ", ";
                out << id;
                first = false;
            }
        }
        out << "], \"original_ancestors\": [";
        if (gv < (int)gAncestors.size()) {
            bool first = true;
            for (int a : gAncestors[gv]) {
                if (!first) out << ", ";
                out << a;
                first = false;
            }
        }
        out << "] }";
        if (i < nL - 1) out << ",";
        out << "\n";
    }
    out << "  ]\n}\n";

    fprintf(stderr, "[simp_viz] wrote %d simplified vertices  ->  %s\n", nL, path.c_str());
}

// ============================================================
// Polyscope visualization — compiled only with C2F_VIZ_DIAGNOSTIC
// ============================================================
#ifdef C2F_VIZ_DIAGNOSTIC

static void rebuild_live_vert_map()
{
    gLiveVertMap.clear();
    std::unordered_set<int> seen;
    for (int f = 0; f < gF.rows(); ++f) {
        if (is_face_dead(gF, f)) continue;
        for (int c = 0; c < 3; ++c) {
            int v = gF(f, c);
            if (std::isinf(gV(v, 0))) continue;
            if (seen.insert(v).second) gLiveVertMap.push_back(v);
        }
    }
}

void simp_viz_tracker_update_display()
{
    rebuild_live_vert_map();
    int nL = (int)gLiveVertMap.size();
    if (nL == 0) return;

    MatrixXd pts(nL, 3);
    MatrixXd colors(nL, 3);

    for (int i = 0; i < nL; ++i) {
        int gv = gLiveVertMap[i];
        pts.row(i) = gV.row(gv).leftCols(3);

        int tt = (gv < (int)gTopoType.size()) ? gTopoType[gv] : MS_Unknown;
        if (tt < 0 || tt >= TOPO_COUNT) tt = MS_Unknown;
        colors(i, 0) = TOPO_RGB[tt][0] / 255.0;
        colors(i, 1) = TOPO_RGB[tt][1] / 255.0;
        colors(i, 2) = TOPO_RGB[tt][2] / 255.0;
    }

    auto* pc = polyscope::registerPointCloud("simp_viz_verts", pts);
    pc->setPointRadius(0.006, true);
    pc->addColorQuantity("topo_type", colors)->setEnabled(true);
    pc->setEnabled(true);
}

void simp_viz_tracker_pick_check()
{
    if (!gShowAncestors) return;
    if (!polyscope::pick::haveSelection()) return;

    auto sel = polyscope::pick::getSelection();
    if (sel == gPickLastSel) return;
    gPickLastSel = sel;

    polyscope::Structure* str = sel.first;
    if (!str || str->name != "simp_viz_verts") return;

    size_t localIdx = sel.second;
    if (localIdx >= gLiveVertMap.size()) return;

    int gv = gLiveVertMap[localIdx];
    if (gv == gPickedGVIndex) return;
    gPickedGVIndex = gv;

    // Build info string
    char buf[256];
    gPickedInfo.clear();
    snprintf(buf, sizeof(buf), "Simplified vertex: gV[%d]", gv);
    gPickedInfo += buf;

    int tt = (gv < (int)gTopoType.size()) ? gTopoType[gv] : MS_Unknown;
    if (tt < 0 || tt >= TOPO_COUNT) tt = MS_Unknown;
    if (tt != MS_Unknown)
        snprintf(buf, sizeof(buf), "\nTopo: %s (%d)", TOPO_NAME[tt], tt);
    else
        snprintf(buf, sizeof(buf), "\nTopo: unknown");
    gPickedInfo += buf;

    // struct IDs
    gPickedInfo += "\nStruct IDs: {";
    if (gv < (int)gStructIds.size()) {
        bool first = true;
        for (int id : gStructIds[gv]) {
            if (!first) gPickedInfo += ", ";
            gPickedInfo += std::to_string(id);
            first = false;
        }
    }
    gPickedInfo += "}";

    // ancestors
    const auto& anc = (gv < (int)gAncestors.size())
                      ? gAncestors[gv]
                      : std::unordered_set<int>{};
    snprintf(buf, sizeof(buf), "\nAncestors: %d original vertices", (int)anc.size());
    gPickedInfo += buf;

    // Show ancestor positions as a point cloud on gVO
    int nA = (int)anc.size();
    MatrixXd ancPts(nA, 3);
    int idx = 0;
    for (int a : anc) {
        if (a >= 0 && a < (int)gVO.rows())
            ancPts.row(idx) = gVO.row(a);
        else
            ancPts.row(idx).setZero();
        ++idx;
    }

    auto* apc = polyscope::registerPointCloud("simp_viz_ancestors", ancPts);
    apc->setPointColor({1.0f, 0.85f, 0.1f});  // gold
    apc->setPointRadius(0.008, true);
    apc->setEnabled(true);
}

void simp_viz_tracker_imgui_section()
{
    ImGui::Separator();
    if (!ImGui::CollapsingHeader("Simp Viz Info")) return;

    ImGui::TextDisabled("Tracks topo type, struct IDs, and collapse ancestors.");

    if (ImGui::Checkbox("Show Ancestors on Click##simp_anc", &gShowAncestors)) {
        // When turned off, hide the ancestor cloud
        if (!gShowAncestors && polyscope::hasPointCloud("simp_viz_ancestors"))
            polyscope::getPointCloud("simp_viz_ancestors")->setEnabled(false);
        // Reset last pick so re-enabling re-queries immediately
        if (gShowAncestors) {
            gPickLastSel   = {nullptr, 0};
            gPickedGVIndex = -1;
        }
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Click any point in 'simp_viz_verts' to highlight\n"
            "its original pre-collapse ancestors (gold) on gVO.");

    if (gPickedGVIndex >= 0) {
        ImGui::Separator();
        ImGui::TextUnformatted(gPickedInfo.c_str());

        if (ImGui::Button("Clear##simp_clear")) {
            gPickedGVIndex = -1;
            gPickedInfo.clear();
            gPickLastSel   = {nullptr, 0};
            if (polyscope::hasPointCloud("simp_viz_ancestors"))
                polyscope::getPointCloud("simp_viz_ancestors")->setEnabled(false);
        }
    }
}

#endif  // C2F_VIZ_DIAGNOSTIC
