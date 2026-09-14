#pragma once
#include <string>

// Reloads simplified_*.obj, the .c2f bundle, and *_simp_visualize_info.json
// from disk and cross-checks that they agree on the coarse-mesh vertex
// ordering — this is what should catch a regression like the OBJ/bundle/JSON
// index-space mismatch immediately, instead of it surfacing three layers
// downstream in the Python training pipeline.
//
// Also cross-checks c2f_path (the correspondence file coarse_fine_compute_and_save
// writes, consumed by coarse_fine_save_bundle) against the bundle's own
// compact->global vertex id set, to catch the two writers disagreeing on
// which vertices are "live".
//
// Prints a loud "[SANITY] MISMATCH" line (with details) for every problem
// found and returns false; returns true if everything agrees within eps.
bool verify_coarse_mesh_outputs(
    const std::string & obj_path,
    const std::string & bundle_path,
    const std::string & json_path,
    const std::string & c2f_path,
    double eps = 1e-6);
