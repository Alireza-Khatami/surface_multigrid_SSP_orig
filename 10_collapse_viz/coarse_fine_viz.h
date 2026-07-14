#pragma once
#include <string>

// Compute coarse→fine correspondence for all live coarse vertices and write to file.
// Call only after decimation has finished (gFinished == true).
void coarse_fine_compute_and_save(const std::string & path);

// Read a previously saved correspondence file and visualise it in polyscope:
// semi-transparent fine mesh + a line from every coarse vertex to its fine-mesh counterpart.
void coarse_fine_load_and_show(const std::string & path);

// Remove all correspondence structures from polyscope.
void coarse_fine_clear();

// Save a self-contained bundle (compact coarse mesh + fine mesh + correspondence)
// that 11_correspond_viz can load without any SSP library.
// corrPath = existing coarse_to_fine.txt, bundlePath = output .c2f file.
void coarse_fine_save_bundle(const std::string & corrPath, const std::string & bundlePath);

// ImGui collapsing section: Save / Load+Show buttons + vertex picker.
// Call inside ImGui::Begin / ImGui::End.
void coarse_fine_imgui_section();

// Check polyscope pick selection each frame; fires the single-vertex query when a
// coarse vertex point cloud ("c2f_pick_verts") is clicked. Call from ui_callback().
void coarse_fine_pick_check();

// Run immediately after each collapse: traces ring_post vertices back to the fine
// mesh, logs any group that lands on the same fine position, and registers
// polyscope structures (diag_ring_post, diag_fine_corr, diag_c2f_arrows).
void ring_post_c2f_diagnostic();
