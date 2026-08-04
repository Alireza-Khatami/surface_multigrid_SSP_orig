#pragma once
#include <string>

// All tunable parameters for the MeshLab-style QEM simplification.
// Mirrors VcgQuadricParameter (QEM_meshlab/VcgQuadricSimplifier.h) with
// Eigen-friendly types.  Default values match the vcg *source* defaults,
// except svd_placement and optimal_placement which default to true here to
// prevent the fly-off-bounding-box problem that the unconstrained fullPivLu
// solver produces on ill-conditioned quadrics.
//
// Persisted as a plain key=value ini file (meshlab_qem.ini).
struct MeshlabQEMConfig
{
    // ── Quadric initialisation ───────────────────────────────────────────────
    double boundary_quadric_weight = 0.5;    // weight on boundary-edge quadrics
    bool   use_area                = true;   // scale face quadric by face area
    bool   scale_independent       = true;   // ScaleFactor = 1e8 / diag^6
    double quadric_epsilon         = 1e-15;  // below this error, multiply by edge length

    // ── Placement ────────────────────────────────────────────────────────────
    // optimal_placement = true  : solve for QEM minimum (optimal position)
    // optimal_placement = false : use survivor vertex position (no solve)
    bool   optimal_placement       = true;

    // svd_placement = true  : SVD with singular-value truncation (bounded, recommended)
    // svd_placement = false : fullPivLu (faithful vcg/MeshLab default, can fly off)
    bool   svd_placement           = true;

    // ── Soft modifiers — not required by the paper ───────────────────────────
    // Not required by the paper (Neural Subdivision, Hsueh-Ti et al. 2020).
    // Paper uses hard veto gates only; soft modifiers are disabled in the callbacks.
    bool   quality_check           = false;  // was true
    double quality_thr             = 0.3;
    bool   normal_check            = false;
    double normal_thr_deg          = 90.0;

    // ── Hard veto gates (set cost = +inf when triggered) ─────────────────────
    // Paper (Neural Subdivision, Hsueh-Ti et al. 2020): Q > 0.2 for all neighbors.
    // Removed condition: minPostQual < minPreQual * 0.9 — not in the paper.
    bool   hard_quality_check      = true;   // was false
    double hard_quality_thr        = 0.2;    // was 0.1; paper value

    // Paper (Neural Subdivision, Hsueh-Ti et al. 2020): dot(n_pre, n_post) > δ=0.2.
    // Was: veto if dihedral > 150° — too loose; now uses paper threshold δ=0.2.
    bool   hard_normal_check       = true;   // was false

    // area_check: not required by the paper (Neural Subdivision, Hsueh-Ti et al. 2020).
    bool   area_check              = false;

    // ── Topology ─────────────────────────────────────────────────────────────
    // Paper (Neural Subdivision, Hsueh-Ti et al. 2020): link condition required.
    bool   preserve_topology       = true;   // was false

    // ── I/O ──────────────────────────────────────────────────────────────────
    // Read values from an ini-style key=value file.
    // Missing keys keep their current (default) values; unknown keys are ignored.
    // Returns true if the file was opened successfully.
    bool read(const std::string & path);

    // Write all values with inline comments to path.
    void write(const std::string & path) const;

    // Print a summary of active settings to stdout.
    void print() const;
};
