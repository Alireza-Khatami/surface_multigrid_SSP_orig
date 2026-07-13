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

// ImGui collapsing section: Save / Load+Show buttons.
// Call inside ImGui::Begin / ImGui::End.
void coarse_fine_imgui_section();
