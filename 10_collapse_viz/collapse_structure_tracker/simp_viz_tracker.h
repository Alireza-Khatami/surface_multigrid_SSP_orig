#pragma once
#include "../coarse_mesh_compaction.h"
#include <string>

// Initialize tracking: call once after init_ssp() and after load_matstruct().
// matstruct_path: the same .ma_struct file used for struct IDs — used here to
//   derive per-vertex topo types (seam degree, boundary degree, junction membership).
// n_initial: gVO.rows() at initialization time.
void simp_viz_tracker_init(const std::string& matstruct_path, int n_initial);

// Call after each successful collapse: s = surviving vertex, d = absorbed vertex.
// Unions d's ancestors + struct IDs into s.
void simp_viz_tracker_on_collapse(int s, int d);

// Write *_simp_visualize_info.json to `path`; vertices[i] is cmc vertex i.
void simp_viz_tracker_write_json(const CoarseMeshCompaction& cmc, const std::string& path);

#ifdef C2F_VIZ_DIAGNOSTIC
// Register / refresh the "simp_viz_verts" point cloud (colored by topo type).
// Call from update_display() whenever the mesh changes.
void simp_viz_tracker_update_display();

// Check polyscope pick each frame; when "Show Ancestors" checkbox is active and
// the user clicks a simplified vertex, highlights ancestor positions on gVO.
// Call at the start of ui_callback(), alongside the other pick_check() calls.
void simp_viz_tracker_pick_check();

// Render the "Simp Viz Info" collapsing ImGui section.
// Call inside ui_callback() after the other imgui sections.
void simp_viz_tracker_imgui_section();
#endif
