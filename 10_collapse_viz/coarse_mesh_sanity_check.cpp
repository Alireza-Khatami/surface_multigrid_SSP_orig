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

bool read_bundle(const std::string & path, std::vector<Vec3> & coarseV, std::vector<int32_t> & compactToGlobal)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    uint32_t magic, NC, FC, NF, FF;
    if (!read_pod(in, magic)) return false;
    if (!read_pod(in, NC)) return false;
    if (!read_pod(in, FC)) return false;
    if (!read_pod(in, NF)) return false;
    if (!read_pod(in, FF)) return false;

    coarseV.resize(NC);
    for (uint32_t i = 0; i < NC; i++) {
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

    compactToGlobal.resize(NC);
    for (uint32_t i = 0; i < NC; i++) {
        int32_t gv;
        if (!read_pod(in, gv)) return false;
        compactToGlobal[i] = gv;
    }
    return true;
}

double vec3_maxdiff(const Vec3 & a, const Vec3 & b)
{
    double d = 0.0;
    for (int k = 0; k < 3; k++) d = std::max(d, std::abs(a[k] - b[k]));
    return d;
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
    std::vector<int32_t> compactToGlobal;
    std::set<int> c2fIds;

    bool ok = true;

    if (!read_obj_vertices(obj_path, objV)) {
        fprintf(stderr, "[SANITY] MISMATCH: could not read %s\n", obj_path.c_str());
        ok = false;
    }
    if (!read_bundle(bundle_path, bundleV, compactToGlobal)) {
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

    const size_t NC = bundleV.size();

    if (objV.size() < NC) {
        fprintf(stderr, "[SANITY] MISMATCH: %s has %zu verts, bundle NC=%zu (need >= NC)\n",
            obj_path.c_str(), objV.size(), NC);
        ok = false;
    }
    if (jsonV.size() != NC) {
        fprintf(stderr, "[SANITY] MISMATCH: %s has %zu verts, bundle NC=%zu\n",
            json_path.c_str(), jsonV.size(), NC);
        ok = false;
    }

    const size_t nCheckObj  = std::min(objV.size(), NC);
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

    const size_t nCheckJson = std::min(jsonV.size(), NC);
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

    std::set<int32_t> bundleIds(compactToGlobal.begin(), compactToGlobal.end());
    std::set<int> onlyInC2f, onlyInBundle;
    for (int id : c2fIds) if (!bundleIds.count(id)) onlyInC2f.insert(id);
    for (int32_t id : bundleIds) if (!c2fIds.count(id)) onlyInBundle.insert(id);
    if (!onlyInC2f.empty() || !onlyInBundle.empty()) {
        fprintf(stderr, "[SANITY] MISMATCH: %s vertex-id set differs from bundle's compact->global set "
            "(%zu only in c2f file, %zu only in bundle)\n",
            c2f_path.c_str(), onlyInC2f.size(), onlyInBundle.size());
        ok = false;
    }

    if (ok) {
        fprintf(stderr, "[SANITY] OK: obj/json/bundle agree on %zu coarse vertices; c2f/bundle vertex sets match\n", NC);
    }
    return ok;
}
