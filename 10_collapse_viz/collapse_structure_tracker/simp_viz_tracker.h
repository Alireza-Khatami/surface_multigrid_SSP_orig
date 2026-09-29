#pragma once
#include "../coarse_mesh_compaction.h"
#include "../load_matstruct.h"
#include <set>
#include <string>
#include <unordered_set>

// Initialize tracking: call once after init_ssp() and after load_matstruct().
// ms: the parsed .ma_struct (nullptr = none) — used here to derive per-vertex
//   topo types (seam degree, boundary degree, junction membership).
// n_initial: gVO.rows() at initialization time.
void simp_viz_tracker_init(const MatStruct* ms, int n_initial);

// Call after each successful collapse: s = surviving vertex, d = absorbed vertex.
// Unions d's ancestors + struct IDs into s.
void simp_viz_tracker_on_collapse(int s, int d);

// Write *_simp_visualize_info.json to `path`; vertices[i] is cmc vertex i.
void simp_viz_tracker_write_json(const CoarseMeshCompaction& cmc, const std::string& path);

// What the json writes for gV vertex gv: the original (fine) vertices merged
// into it, and its struct IDs. Valid for any gV id (empty sets if never seen).
const std::unordered_set<int>& simp_viz_tracker_ancestors(int gv);
const std::set<int>&           simp_viz_tracker_struct_ids(int gv);

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
