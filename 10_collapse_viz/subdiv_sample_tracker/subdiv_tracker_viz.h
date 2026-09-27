#pragma once
// Polyscope display + ImGui panel for the subdivided-mesh tracker.
// Only compiled with C2F_VIZ_DIAGNOSTIC (it is added with visualizer.cpp).
//
// Shows the subdivided mesh on the fine surface ("subdiv_fine") and the same
// connectivity with every vertex at its tracked position on the decimating
// mesh ("subdiv_deformed"), plus the same vertices as point clouds
// ("subdiv_pts_fine" / "subdiv_pts_deformed") with one color per struct set.
// Also colorable by structure type, carrier type and struct-set id. Large
// meshes are shown at a coarser subdivision level (the levels are nested, so
// this is an exact sub-sampling of the same samples).

// Register / refresh the polyscope structures. Call from update_display().
void subdiv_tracker_viz_update();

// Panel section. Call inside the visualizer's ImGui window.
void subdiv_tracker_viz_ui();
