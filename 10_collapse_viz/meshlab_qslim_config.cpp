#include "meshlab_qslim_config.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

// ── helpers ──────────────────────────────────────────────────────────────────
static std::string trim(const std::string & s)
{
    const char * ws = " \t\r\n";
    size_t a = s.find_first_not_of(ws);
    if (a == std::string::npos) return {};
    size_t b = s.find_last_not_of(ws);
    return s.substr(a, b - a + 1);
}

static bool parse_bool(const std::string & v)
{
    std::string lv = v;
    std::transform(lv.begin(), lv.end(), lv.begin(), ::tolower);
    return lv == "true" || lv == "1" || lv == "yes";
}

// ── read ─────────────────────────────────────────────────────────────────────
bool MeshlabQEMConfig::read(const std::string & path)
{
    std::ifstream f(path);
    if (!f.is_open()) return false;

    std::string line;
    while (std::getline(f, line))
    {
        // strip inline comment
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);

        line = trim(line);
        if (line.empty() || line.front() == '[') continue;  // section header / blank

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (key.empty() || val.empty()) continue;

        // ── quadric init ────────────────────────────────────────────────────
        if      (key == "boundary_quadric_weight") boundary_quadric_weight = std::stod(val);
        else if (key == "use_area")                use_area                = parse_bool(val);
        else if (key == "scale_independent")       scale_independent       = parse_bool(val);
        else if (key == "quadric_epsilon")         quadric_epsilon         = std::stod(val);
        // ── placement ───────────────────────────────────────────────────────
        else if (key == "optimal_placement")       optimal_placement       = parse_bool(val);
        else if (key == "svd_placement")           svd_placement           = parse_bool(val);
        // ── soft modifiers ──────────────────────────────────────────────────
        else if (key == "quality_check")           quality_check           = parse_bool(val);
        else if (key == "quality_thr")             quality_thr             = std::stod(val);
        else if (key == "normal_check")            normal_check            = parse_bool(val);
        else if (key == "normal_thr_deg")          normal_thr_deg          = std::stod(val);
        // ── hard gates ──────────────────────────────────────────────────────
        else if (key == "hard_quality_check")      hard_quality_check      = parse_bool(val);
        else if (key == "hard_quality_thr")        hard_quality_thr        = std::stod(val);
        else if (key == "hard_normal_check")       hard_normal_check       = parse_bool(val);
        else if (key == "area_check")              area_check              = parse_bool(val);
        // ── topology ────────────────────────────────────────────────────────
        else if (key == "preserve_topology")       preserve_topology       = parse_bool(val);
        else
            std::cerr << "[meshlab_qem] unknown config key: " << key << "\n";
    }
    return true;
}

// ── write ─────────────────────────────────────────────────────────────────────
static const char * b(bool v) { return v ? "true" : "false"; }

void MeshlabQEMConfig::write(const std::string & path) const
{
    std::ofstream f(path);
    if (!f.is_open()) { std::cerr << "[meshlab_qem] cannot write " << path << "\n"; return; }

    f << "# MeshLab QEM simplification settings\n"
      << "# Reference: QEM_meshlab/VcgQuadricSimplifier.h  (VcgQuadricParameter)\n"
      << "# Reload at runtime by restarting; no hot-reload.\n\n";

    f << "[quadric_init]\n"
      << "boundary_quadric_weight = " << boundary_quadric_weight
         << "  # weight on perpendicular plane quadric added at boundary edges\n"
      << "use_area                = " << b(use_area)
         << "  # scale each face quadric by its area (recommended: true)\n"
      << "scale_independent       = " << b(scale_independent)
         << "  # normalise ScaleFactor = 1e8/diag^6 so cost is mesh-size independent\n"
      << "quadric_epsilon         = " << quadric_epsilon
         << "  # minimum cost; below this multiply by edge length\n\n";

    f << "[placement]\n"
      << "optimal_placement       = " << b(optimal_placement)
         << "  # true=solve for QEM minimum; false=use survivor position\n"
      << "svd_placement           = " << b(svd_placement)
         << "  # true=SVD truncation (bounded); false=fullPivLu (can fly off bounding box)\n\n";

    f << "[soft_modifiers]  # raise cost but never veto\n"
      << "quality_check           = " << b(quality_check)
         << "  # divide error by worst post-collapse face quality\n"
      << "quality_thr             = " << quality_thr
         << "  # quality clamped to this floor before dividing\n"
      << "normal_check            = " << b(normal_check)
         << "  # divide error by remapped normal-change cosine\n"
      << "normal_thr_deg          = " << normal_thr_deg
         << "  # normal rotation above this gets maximum penalty\n\n";

    f << "[hard_gates]  # set cost=+inf when triggered\n"
      << "hard_quality_check      = " << b(hard_quality_check)
         << "  # veto if post quality < hard_quality_thr AND < 90% of pre quality\n"
      << "hard_quality_thr        = " << hard_quality_thr << "\n"
      << "hard_normal_check       = " << b(hard_normal_check)
         << "  # veto if any surviving face flips (dihedral > 150 deg or sliver)\n"
      << "area_check              = " << b(area_check)
         << "  # veto if relative area change > 1%\n\n";

    f << "[topology]\n"
      << "preserve_topology       = " << b(preserve_topology)
         << "  # run link-condition check before each collapse\n";
}

// ── print ─────────────────────────────────────────────────────────────────────
void MeshlabQEMConfig::print() const
{
    std::cout << "[meshlab_qem] config:\n"
              << "  placement       : " << (optimal_placement ? "optimal" : "survivor")
              << " / " << (svd_placement ? "SVD" : "fullPivLu") << "\n"
              << "  boundary_weight : " << boundary_quadric_weight << "\n"
              << "  use_area        : " << b(use_area)
              << "  scale_independent: " << b(scale_independent) << "\n"
              << "  quality_check   : " << b(quality_check)
              << " (thr=" << quality_thr << ")\n"
              << "  normal_check    : " << b(normal_check)
              << " (deg=" << normal_thr_deg << ")\n"
              << "  hard_quality    : " << b(hard_quality_check)
              << " (thr=" << hard_quality_thr << ")\n"
              << "  hard_normal     : " << b(hard_normal_check) << "\n"
              << "  area_check      : " << b(area_check) << "\n"
              << "  preserve_topo   : " << b(preserve_topology) << "\n";
}
