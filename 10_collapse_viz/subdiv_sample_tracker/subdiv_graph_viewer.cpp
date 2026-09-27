// Viewer for the relaxation graph file (subdiv_graph_<stem>.slg, written by
// subdiv_relax.cpp). It only parses the file and draws what is in it, so what
// you see is exactly the graph the relaxation used.
//
// Usage: subdiv_graph_viewer <subdiv_graph_*.slg>
//
//   - points colored by role (sheet / curve = seam or boundary / junction),
//     radius junction > curve > sheet;
//   - one curve network per edge kind, each toggleable:
//       S<->S and C<->C two-way pairs: full segment;
//       S<-C, S<-J, C<-J one-way: half segment from the pulled vertex toward
//       the vertex that pulls it (so the direction is visible without arrows);
//       anything else would be a bug and is drawn in red as "UNEXPECTED";
//   - vertex query (type an id or click a point): role, struct ids, the row
//     (who pulls it) and the column (whom it pulls), highlighted.

#include <polyscope/curve_network.h>
#include <polyscope/pick.h>
#include <polyscope/point_cloud.h>
#include <polyscope/polyscope.h>

#include "imgui.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

struct Slg {
    uint64_t Vs = 0, nnz = 0, P = 0, nPI = 0;
    std::vector<double>  V;        // Vs*3
    std::vector<uint8_t> role;
    std::vector<int32_t> set;
    std::vector<int64_t> row;
    std::vector<int32_t> cols;
    std::vector<int32_t> po, pid;
    std::vector<uint8_t> pm;
};

bool read_slg(const std::string & path, Slg & g, std::string & err)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { err = "cannot open " + path; return false; }
    const size_t size = (size_t)f.tellg();
    std::vector<char> buf(size);
    f.seekg(0);
    f.read(buf.data(), (std::streamsize)size);
    if (size < 56 || std::memcmp(buf.data(), "SUBDIVG\0", 8) != 0) { err = "bad magic"; return false; }
    uint32_t version, hbytes;
    std::memcpy(&version, buf.data() + 8, 4);
    std::memcpy(&hbytes, buf.data() + 12, 4);
    uint64_t h[5];
    std::memcpy(h, buf.data() + 16, 40);
    g.Vs = h[0]; g.nnz = h[1]; g.P = h[2]; g.nPI = h[3];
    const uint64_t nArr = h[4];
    if (version != 1 || nArr != 8 || hbytes != 56 + 8 * nArr) { err = "unsupported header"; return false; }
    uint64_t off[8];
    std::memcpy(off, buf.data() + 56, 64);
    auto grab = [&](int k, auto & vec, size_t n) {
        using T = typename std::decay_t<decltype(vec)>::value_type;
        if (off[k] + n * sizeof(T) > size) return false;
        vec.resize(n);
        std::memcpy(vec.data(), buf.data() + off[k], n * sizeof(T));
        return true;
    };
    if (!grab(0, g.V, g.Vs * 3) || !grab(1, g.role, g.Vs) || !grab(2, g.set, g.Vs) || !grab(3, g.row, g.Vs + 1)
        || !grab(4, g.cols, g.nnz) || !grab(5, g.po, g.P + 1) || !grab(6, g.pid, g.nPI) || !grab(7, g.pm, g.P)) {
        err = "truncated file";
        return false;
    }
    if ((uint64_t)g.row[g.Vs] != g.nnz) { err = "row offsets do not end at nnz"; return false; }
    return true;
}

Slg G;
bool has(int64_t i, int32_t j)
{
    return std::binary_search(G.cols.begin() + G.row[i], G.cols.begin() + G.row[i + 1], j);
}

const char * kRoleName[3] = { "sheet", "curve (seam/boundary)", "junction" };
const glm::vec3 kRoleColor[3] = { { 0.55f, 0.75f, 0.95f }, { 0.95f, 0.55f, 0.15f }, { 0.85f, 0.10f, 0.10f } };
const float kRoleRadius[3] = { 0.35f, 0.7f, 1.0f };

enum Kind { K_SS, K_CC, K_SC, K_SJ, K_CJ, K_BAD, K_N };
const char * kKindName[K_N] = { "S<->S (two-way)", "C<->C (two-way)", "S<-C (one-way)", "S<-J (one-way)",
                                "C<-J (one-way)", "UNEXPECTED" };
const glm::vec3 kKindColor[K_N] = { { 0.45f, 0.60f, 0.85f }, { 0.95f, 0.55f, 0.10f }, { 0.20f, 0.70f, 0.20f },
                                    { 0.60f, 0.20f, 0.70f }, { 0.85f, 0.10f, 0.40f }, { 1.0f, 0.0f, 0.0f } };
// Segments per kind: pairs of 3D points.
std::array<std::vector<glm::vec3>, K_N> segs;
std::array<int64_t, K_N> kindCount{};
std::array<bool, K_N> registered{};
std::array<bool, K_N> wantShown = { false, true, true, true, true, true };

glm::vec3 P(int64_t i) { return { (float)G.V[3 * i], (float)G.V[3 * i + 1], (float)G.V[3 * i + 2] }; }

void build_segments()
{
    for (auto & s : segs) s.clear();
    kindCount.fill(0);
    for (int64_t i = 0; i < (int64_t)G.Vs; ++i) {
        for (int64_t q = G.row[i]; q < G.row[i + 1]; ++q) {
            const int32_t j = G.cols[q];
            const uint8_t ri = G.role[i], rj = G.role[j];
            const bool back = has(j, (int32_t)i);
            int k;
            if (back) {
                if (i > j) continue;  // two-way pair: once
                k = (ri == 0 && rj == 0) ? K_SS : (ri == 1 && rj == 1) ? K_CC : K_BAD;
            } else {
                k = (ri == 0 && rj == 1) ? K_SC : (ri == 0 && rj == 2) ? K_SJ : (ri == 1 && rj == 2) ? K_CJ : K_BAD;
            }
            ++kindCount[k];
            const glm::vec3 a = P(i), b = P(j);
            segs[k].push_back(a);
            segs[k].push_back(back ? b : 0.5f * (a + b));
        }
    }
}

void register_kind(int k)
{
    if (registered[k] || segs[k].empty()) return;
    std::vector<std::array<size_t, 2>> e(segs[k].size() / 2);
    for (size_t s = 0; s < e.size(); ++s) e[s] = { 2 * s, 2 * s + 1 };
    auto * cn = polyscope::registerCurveNetwork(std::string("edges ") + kKindName[k], segs[k], e);
    cn->setColor(kKindColor[k]);
    cn->setRadius(k == K_SS ? 0.0004f : 0.0008f, true);
    registered[k] = true;
}

int64_t gQuery = -1;
char gQueryBuf[32] = "";
std::vector<int64_t> gPullers, gPulled;

void set_query(int64_t v)
{
    gQuery = v;
    gPullers.clear();
    gPulled.clear();
    if (polyscope::hasPointCloud("query")) polyscope::removePointCloud("query");
    if (polyscope::hasPointCloud("query pullers")) polyscope::removePointCloud("query pullers");
    if (polyscope::hasPointCloud("query pulled")) polyscope::removePointCloud("query pulled");
    if (v < 0 || v >= (int64_t)G.Vs) return;
    for (int64_t q = G.row[v]; q < G.row[v + 1]; ++q) gPullers.push_back(G.cols[q]);
    for (int64_t i = 0; i < (int64_t)G.Vs; ++i)
        if (i != v && has(i, (int32_t)v)) gPulled.push_back(i);
    auto pc = [&](const char * name, const std::vector<int64_t> & ids, glm::vec3 col, float r) {
        if (ids.empty()) return;
        std::vector<glm::vec3> p;
        for (int64_t i : ids) p.push_back(P(i));
        auto * c = polyscope::registerPointCloud(name, p);
        c->setPointColor(col);
        c->setPointRadius(r, true);
    };
    pc("query", { v }, { 1, 1, 0 }, 0.009f);
    pc("query pullers", gPullers, { 0, 0.9f, 0.2f }, 0.006f);
    pc("query pulled", gPulled, { 0.9f, 0, 0.9f }, 0.005f);
}

void ui()
{
    ImGui::Text("%llu vertices, %llu directed entries", (unsigned long long)G.Vs, (unsigned long long)G.nnz);
    int64_t nRole[3] = {};
    for (uint8_t r : G.role) ++nRole[r];
    ImGui::Text("sheet %lld | curve %lld | junction %lld", (long long)nRole[0], (long long)nRole[1], (long long)nRole[2]);
    ImGui::Separator();
    for (int k = 0; k < K_N; ++k) {
        if (k == K_BAD && kindCount[k] == 0) { ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1), "UNEXPECTED kinds: 0"); continue; }
        bool show = registered[k] && polyscope::hasCurveNetwork(std::string("edges ") + kKindName[k])
                    && polyscope::getCurveNetwork(std::string("edges ") + kKindName[k])->isEnabled();
        char label[96];
        std::snprintf(label, sizeof(label), "%s: %lld", kKindName[k], (long long)kindCount[k]);
        if (ImGui::Checkbox(label, &show)) {
            if (show) register_kind(k);
            if (registered[k]) polyscope::getCurveNetwork(std::string("edges ") + kKindName[k])->setEnabled(show);
        }
    }
    ImGui::Separator();
    ImGui::InputText("vertex id", gQueryBuf, sizeof(gQueryBuf));
    ImGui::SameLine();
    if (ImGui::Button("query")) set_query(std::atoll(gQueryBuf));
    if (polyscope::pick::haveSelection()) {
        auto sel = polyscope::pick::getSelection();
        if (sel.first && sel.first->name == "vertices" && (int64_t)sel.second != gQuery
            && sel.second < G.Vs) {
            std::snprintf(gQueryBuf, sizeof(gQueryBuf), "%lld", (long long)sel.second);
            set_query((int64_t)sel.second);
        }
    }
    if (gQuery >= 0 && gQuery < (int64_t)G.Vs) {
        const int k = G.set[gQuery];
        ImGui::Text("v %lld: %s", (long long)gQuery, kRoleName[G.role[gQuery]]);
        std::string ids;
        for (int32_t a = G.po[k]; a < G.po[k + 1]; ++a) ids += std::to_string(G.pid[a]) + " ";
        ImGui::TextWrapped("struct set %d (mask 0x%x): %s", k, G.pm[k], ids.c_str());
        auto list = [&](const char * title, const std::vector<int64_t> & v) {
            std::string s;
            for (size_t t = 0; t < v.size() && t < 24; ++t)
                s += std::to_string(v[t]) + "(" + "SCJ"[G.role[v[t]]] + ") ";
            if (v.size() > 24) s += "...";
            ImGui::TextWrapped("%s [%zu]: %s", title, v.size(), s.c_str());
        };
        list("pulled by (row, green)", gPullers);
        list("pulls (column, magenta)", gPulled);
    }
}

} // namespace

int main(int argc, char ** argv)
{
    if (argc < 2) { std::fprintf(stderr, "usage: %s subdiv_graph_<stem>.slg\n", argv[0]); return 2; }
    std::string err;
    if (!read_slg(argv[1], G, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    build_segments();
    std::fprintf(stderr, "graph: %llu vertices, %llu entries | S<->S %lld, C<->C %lld, S<-C %lld, S<-J %lld, C<-J %lld, UNEXPECTED %lld\n",
                 (unsigned long long)G.Vs, (unsigned long long)G.nnz, (long long)kindCount[K_SS], (long long)kindCount[K_CC],
                 (long long)kindCount[K_SC], (long long)kindCount[K_SJ], (long long)kindCount[K_CJ], (long long)kindCount[K_BAD]);
    if (argc > 2 && std::string(argv[2]) == "--check") return kindCount[K_BAD] ? 1 : 0;  // headless parse check

    polyscope::options::programName = "subdiv graph viewer";
    polyscope::init();

    std::vector<glm::vec3> pts(G.Vs), col(G.Vs);
    std::vector<float> rad(G.Vs);
    for (uint64_t i = 0; i < G.Vs; ++i) {
        pts[i] = P((int64_t)i);
        col[i] = kRoleColor[G.role[i]];
        rad[i] = kRoleRadius[G.role[i]];
    }
    auto * pc = polyscope::registerPointCloud("vertices", pts);
    pc->addColorQuantity("role", col)->setEnabled(true);
    pc->addScalarQuantity("radius by role", rad);
    pc->setPointRadiusQuantity("radius by role", true);
    pc->setPointRadius(0.004, true);

    // Everything but the (large) S<->S kind is registered and shown at start.
    for (int k = 0; k < K_N; ++k)
        if (wantShown[k]) register_kind(k);

    polyscope::state::userCallback = ui;
    polyscope::show();
    return 0;
}
