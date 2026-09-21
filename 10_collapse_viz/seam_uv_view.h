#pragma once

// Seam multi-sheet UV overlay — standalone view.
//
// For a seam collapse, every active sheet solves its own independent
// joint_lscm (Case 2/3, double cover), each pinning its own UV frame. Nothing
// forces sheet A's UV frame to agree with sheet B's — even though vi/vj are
// the SAME global vertices shared by every sheet. This view overlays each
// sheet's RAW (unaligned, un-rotated) UV_pre solution in one shared UV frame
// (u, v, 0) so any misalignment at the shared vi/vj seam vertices is visible
// directly.
//
// Deliberately pure UV space: no 3D ring transform (rotation/centroid/scale)
// is applied here. Mixing this with 3D canonical-ring-space geometry (e.g.
// V_pre/V_post point clouds, which are true mesh scale) is a scale mismatch —
// keep this view and the canonical 3D view separate.
//
// See md_files/seam_multisheet_uv_overlay.md for the full design writeup.

#include <single_collapse_data.h>
#include <vector>

// Registers polyscope structures for the seam UV overlay from the given
// sheets (pass gAllSheets — every active sheet of the currently-inspected
// collapse). Caller is responsible for calling polyscope::removeAllStructures()
// before the first call for a given collapse.
//
// show_pre / show_post independently toggle the pre-collapse (UV_pre/FUV_pre)
// and post-collapse (UV_post/FUV_post) overlays — both are the direct,
// unmodified joint_lscm output for that sheet, just like UV_pre.
void show_seam_uv_view(const std::vector<SheetData> & sheets,
                        bool show_pre  = true,
                        bool show_post = true);
