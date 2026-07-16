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

    // ── Soft modifiers (raise cost, never veto collapse) ─────────────────────
    // quality_check: divide error by worst post-collapse face quality
    bool   quality_check           = true;
    double quality_thr             = 0.3;    // clamp quality to this before dividing

    // normal_check: divide error by remapped normal-change cosine
    bool   normal_check            = false;
    double normal_thr_deg          = 90.0;   // maximum acceptable normal rotation (degrees)

    // ── Hard veto gates (set cost = +inf when triggered) ─────────────────────
    // hard_quality_check: veto if post-collapse quality < thr AND < 90% of pre
    bool   hard_quality_check      = false;
    double hard_quality_thr        = 0.1;

    // hard_normal_check: veto if any surviving face flips (dihedral > 150 deg)
    bool   hard_normal_check       = false;

    // area_check: veto if relative area change > 1%
    bool   area_check              = false;

    // ── Topology ─────────────────────────────────────────────────────────────
    // preserve_topology: run link-condition check in pre_fn
    // (SSP already guards manifoldness; enable for an extra layer)
    bool   preserve_topology       = false;

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
