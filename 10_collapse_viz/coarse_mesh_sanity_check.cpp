#include "coarse_mesh_sanity_check.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <vector>

using Vec3 = std::array<double,3>;

namespace {

bool read_obj_vertices(const std::string & path, std::vector<Vec3> & out)
{
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.size() < 2 || line[0] != 'v' || line[1] != ' ') continue;
        std::istringstream ss(line.substr(2));
        Vec3 v;
        if (!(ss >> v[0] >> v[1] >> v[2])) continue;
        out.push_back(v);
    }
    return true;
}

// Reads 'l' polylines (OBJ is 1-based; converted to 0-based here).
bool read_obj_l_lines(const std::string & path, std::vector<std::vector<int>> & out)
{
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.size() < 2 || line[0] != 'l' || line[1] != ' ') continue;
        std::istringstream ss(line.substr(2));
        std::vector<int> chain;
        int idx;
        while (ss >> idx) chain.push_back(idx - 1);
        if (!chain.empty()) out.push_back(std::move(chain));
    }
    return true;
}

// Minimal targeted parser for the fixed layout simp_viz_tracker_write_json
// writes: one `"position": [x, y, z]` per vertex entry, in order.
bool read_json_vertex_positions(const std::string & path, std::vector<Vec3> & out)
{
    std::ifstream in(path);
    if (!in) return false;
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string s = buf.str();

    const std::string key = "\"position\": [";
    size_t pos = 0;
    while ((pos = s.find(key, pos)) != std::string::npos) {
        pos += key.size();
        size_t end = s.find(']', pos);
        if (end == std::string::npos) break;
        std::string inner = s.substr(pos, end - pos);
        for (char & c : inner) if (c == ',') c = ' ';
        std::istringstream ss(inner);
        Vec3 v;
        if (ss >> v[0] >> v[1] >> v[2]) out.push_back(v);
        pos = end;
    }
    return true;
}

bool read_c2f_vertex_ids(const std::string & path, std::set<int> & ids)
{
    std::ifstream in(path);
    if (!in) return false;
    int n;
    if (!(in >> n)) return false;
    std::string rest_of_first_line;
    std::getline(in, rest_of_first_line);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        int vi;
        if (ss >> vi) ids.insert(vi);
    }
    return true;
}

template <typename T>
bool read_pod(std::ifstream & in, T & v)
{
    in.read(reinterpret_cast<char *>(&v), sizeof(T));
    return (bool)in;
}

// Bundle layout expected here (magic 0xC2F50007):
//   magic, NC, FC, NF, FF, NCE  (header)
//   NCE coarse verts, FC coarse faces, NF fine verts, FF fine faces, NC correspondence entries
//   nV_total, nF_decIM, nFO_u
//   NCE compact->global vertex ids
//   stale-chain section: nStale, then per chain: chainLen + chainLen compact indices
//   (gDecIM/gDecInfo/sheets data follows — not needed here, so not parsed)
bool read_bundle(
    const std::string & path,
    std::vector<Vec3> & coarseV,
    uint32_t & NC,
    std::vector<int32_t> & compactToGlobal,
    std::vector<std::vector<int>> & staleChains)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    uint32_t magic, FC, NF, FF, NCE;
    if (!read_pod(in, magic)) return false;
    if (magic != 0xC2F50007u) {
        fprintf(stderr, "[SANITY] MISMATCH: %s has magic 0x%08X, expected 0xC2F50007 "
            "(bundle format doesn't carry stale-chain data yet)\n", path.c_str(), magic);
        return false;
    }
    if (!read_pod(in, NC)) return false;
    if (!read_pod(in, FC)) return false;
    if (!read_pod(in, NF)) return false;
    if (!read_pod(in, FF)) return false;
    if (!read_pod(in, NCE)) return false;

    coarseV.resize(NCE);
    for (uint32_t i = 0; i < NCE; i++) {
        double xyz[3];
        in.read(reinterpret_cast<char *>(xyz), sizeof(xyz));
        coarseV[i] = {xyz[0], xyz[1], xyz[2]};
    }
    if (!in) return false;

    in.seekg((std::streamoff)FC * 12, std::ios::cur);      // coarseF: FC * 3 * uint32
    in.seekg((std::streamoff)NF * 24, std::ios::cur);      // fine vertices: NF * 3 * double
    in.seekg((std::streamoff)FF * 12, std::ios::cur);      // fine faces: FF * 3 * uint32
    in.seekg((std::streamoff)NC * (24 + 12), std::ios::cur); // correspondence: NC * (3 double + 3 uint32)
    if (!in) return false;

    uint32_t nV_total, nF_decIM, nFO_u;
    if (!read_pod(in, nV_total)) return false;
    if (!read_pod(in, nF_decIM)) return false;
    if (!read_pod(in, nFO_u)) return false;

    compactToGlobal.resize(NCE);
    for (uint32_t i = 0; i < NCE; i++) {
        int32_t gv;
        if (!read_pod(in, gv)) return false;
        compactToGlobal[i] = gv;
    }

    uint32_t nStale;
    if (!read_pod(in, nStale)) return false;
    staleChains.resize(nStale);
    for (uint32_t c = 0; c < nStale; c++) {
        uint32_t chainLen;
        if (!read_pod(in, chainLen)) return false;
        staleChains[c].resize(chainLen);
        for (uint32_t k = 0; k < chainLen; k++) {
            uint32_t idx;
            if (!read_pod(in, idx)) return false;
            staleChains[c][k] = (int)idx;
        }
    }
    return true;
}

double vec3_maxdiff(const Vec3 & a, const Vec3 & b)
{
    double d = 0.0;
    for (int k = 0; k < 3; k++) d = std::max(d, std::abs(a[k] - b[k]));
    return d;
}

// Compares chain sets up to reversal (an OBJ/bundle chain and its exact reverse
// describe the same polyline) and reports the first few mismatches.
void diff_chain_sets(
    const std::string & aName, const std::vector<std::vector<int>> & a,
    const std::string & bName, const std::vector<std::vector<int>> & b,
    bool & ok)
{
    auto normalize = [](std::vector<int> c) {
        std::vector<int> rev(c.rbegin(), c.rend());
        return std::min(c, rev);
    };
    std::vector<std::vector<int>> na, nb;
    for (auto & c : a) na.push_back(normalize(c));
    for (auto & c : b) nb.push_back(normalize(c));
    std::sort(na.begin(), na.end());
    std::sort(nb.begin(), nb.end());
    if (na != nb) {
        fprintf(stderr, "[SANITY] MISMATCH: stale-chain topology differs between %s (%zu chains) and %s (%zu chains)\n",
            aName.c_str(), na.size(), bName.c_str(), nb.size());
        ok = false;
    }
}

} // namespace

bool verify_coarse_mesh_outputs(
    const std::string & obj_path,
    const std::string & bundle_path,
    const std::string & json_path,
    const std::string & c2f_path,
    double eps)
{
    std::vector<Vec3> objV, jsonV, bundleV;
    std::vector<std::vector<int>> objChains, bundleChains;
    uint32_t NC = 0;
    std::vector<int32_t> compactToGlobal;
    std::set<int> c2fIds;

    bool ok = true;

    if (!read_obj_vertices(obj_path, objV) || !read_obj_l_lines(obj_path, objChains)) {
        fprintf(stderr, "[SANITY] MISMATCH: could not read %s\n", obj_path.c_str());
        ok = false;
    }
    if (!read_bundle(bundle_path, bundleV, NC, compactToGlobal, bundleChains)) {
        fprintf(stderr, "[SANITY] MISMATCH: could not read/parse %s\n", bundle_path.c_str());
        ok = false;
    }
    if (!read_json_vertex_positions(json_path, jsonV)) {
        fprintf(stderr, "[SANITY] MISMATCH: could not read %s\n", json_path.c_str());
        ok = false;
    }
    if (!read_c2f_vertex_ids(c2f_path, c2fIds)) {
        fprintf(stderr, "[SANITY] MISMATCH: could not read %s\n", c2f_path.c_str());
        ok = false;
    }
    if (!ok) return false;

    const size_t NCE = bundleV.size(); // full extended count: coarse mesh + stale-chain verts

    if (objV.size() != NCE) {
        fprintf(stderr, "[SANITY] MISMATCH: %s has %zu verts, bundle NCE=%zu\n",
            obj_path.c_str(), objV.size(), NCE);
        ok = false;
    }
    if (jsonV.size() != NCE) {
        fprintf(stderr, "[SANITY] MISMATCH: %s has %zu verts, bundle NCE=%zu\n",
            json_path.c_str(), jsonV.size(), NCE);
        ok = false;
    }

    const size_t nCheckObj  = std::min(objV.size(), NCE);
    int badObj = 0;
    for (size_t i = 0; i < nCheckObj; i++) {
        double d = vec3_maxdiff(objV[i], bundleV[i]);
        if (d > eps) {
            if (badObj < 5)
                fprintf(stderr, "[SANITY] MISMATCH: obj vs bundle.coarseV[%zu] differ by %.3g\n", i, d);
            badObj++;
        }
    }
    if (badObj > 0) {
        fprintf(stderr, "[SANITY] MISMATCH: %d/%zu vertices differ between %s and bundle.coarseV\n",
            badObj, nCheckObj, obj_path.c_str());
        ok = false;
    }

    const size_t nCheckJson = std::min(jsonV.size(), NCE);
    int badJson = 0;
    for (size_t i = 0; i < nCheckJson; i++) {
        double d = vec3_maxdiff(jsonV[i], bundleV[i]);
        if (d > eps) {
            if (badJson < 5)
                fprintf(stderr, "[SANITY] MISMATCH: json vs bundle.coarseV[%zu] differ by %.3g\n", i, d);
            badJson++;
        }
    }
    if (badJson > 0) {
        fprintf(stderr, "[SANITY] MISMATCH: %d/%zu vertices differ between %s and bundle.coarseV\n",
            badJson, nCheckJson, json_path.c_str());
        ok = false;
    }

    diff_chain_sets(obj_path, objChains, bundle_path, bundleChains, ok);

    // Only the first NC entries are the face-referenced coarse mesh — stale-chain
    // global ids (NC..NCE-1) are never covered by the c2f correspondence file
    // (coarse_fine_compute_and_save only walks face-adjacent vertices), so they
    // must be excluded here rather than compared.
    std::set<int32_t> bundleIds(compactToGlobal.begin(), compactToGlobal.begin() + NC);
    std::set<int> onlyInC2f, onlyInBundle;
    for (int id : c2fIds) if (!bundleIds.count(id)) onlyInC2f.insert(id);
    for (int32_t id : bundleIds) if (!c2fIds.count(id)) onlyInBundle.insert(id);
    if (!onlyInC2f.empty() || !onlyInBundle.empty()) {
        fprintf(stderr, "[SANITY] MISMATCH: %s vertex-id set differs from bundle's compact->global set "
            "(%zu only in c2f file, %zu only in bundle's coarse-mesh range)\n",
            c2f_path.c_str(), onlyInC2f.size(), onlyInBundle.size());
        ok = false;
    }

    if (ok) {
        fprintf(stderr, "[SANITY] OK: obj/json/bundle agree on %zu vertices (%u coarse + %zu stale-chain); "
            "%zu stale chains agree; c2f/bundle coarse-mesh vertex sets match\n",
            NCE, NC, NCE - NC, bundleChains.size());
    }
    return ok;
}
