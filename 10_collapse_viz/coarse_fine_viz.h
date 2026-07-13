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

// ImGui collapsing section: Save / Load+Show buttons.
// Call inside ImGui::Begin / ImGui::End.
void coarse_fine_imgui_section();
