#pragma once

// Sheet & Seam Inspection — public interface.
// All state is internal to sheet_seam_viz.cpp.

// (Re)register the sheet-colored mesh ("sheet_view") and seam edge curve network
// ("seam_edges").  Call whenever the main mesh changes or checkboxes toggle.
void update_sheet_display();

// Show per-sheet one-ring faces around the currently picked seam edge.
// No-op (and cleans up) when nothing is picked or "Show one-ring" is off.
void update_seam_onering_display();

// Remove all registered one-ring meshes and clear internal name list.
// Call before polyscope::removeAllStructures() so the name list stays consistent.
void clear_seam_onering();

// Must be called at the START of ui_callback() every frame.
// Detects clicks on the "seam_edges" curve network and triggers the one-ring view.
void sheet_seam_pick_check();

// Renders the "Sheet & Seam Inspection" collapsing section.
// Call inside ImGui::Begin / ImGui::End, after the main visibility block.
void sheet_seam_imgui_section();
